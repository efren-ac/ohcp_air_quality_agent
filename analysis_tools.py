from __future__ import annotations

import calendar
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from config import (
    ANNUAL_MIN_VALID_DAYS, O3_8H_DAILY_LIMIT, OUTPUT_DIR, PM10_ANNUAL_LIMIT,
    PM10_DAILY_LIMIT, PM25_ANNUAL_LIMIT, PM25_DAILY_LIMIT,
)
from data_loader import load_daily_data, load_province_mapping, normalize_city


def configure_chinese_font() -> None:
    """优先选择当前系统真实安装的中文字体，避免固定字体名在不同系统失效。"""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    preferred = (
        "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK SC",
        "Source Han Sans SC", "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans",
    )
    chosen = next((name for name in preferred if name in installed), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False


def _year_days(year: int) -> int:
    return 366 if calendar.isleap(int(year)) else 365


def _load_year(year: int, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    if df is not None:
        frame = df.copy()
        if "date" not in frame.columns or "city" not in frame.columns:
            raise ValueError("传入的数据必须包含 date 和 city 字段。")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date", "city"])
        frame = frame[frame["date"].dt.year.eq(int(year))].copy()
        for column in ("pm25_daily_mean", "pm10_daily_mean", "o3_8h_daily_max", "aqi_daily_max"):
            if column not in frame.columns:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["month"] = frame["date"].dt.month
        frame["city_key"] = frame["city"].map(normalize_city)
        return frame, frame.attrs.get("load_report", {})
    return load_daily_data(year=year)


def validate_pm25_result(summary: pd.DataFrame, detail: pd.DataFrame, year: int) -> dict:
    issues: list[str] = []
    checks: list[str] = []
    if summary.empty:
        return {"passed": False, "checks": checks, "issues": ["PM2.5 排名结果为空。"]}
    vals = pd.to_numeric(summary["年均PM25估算值"], errors="coerce")
    if vals.isna().any() or (vals < 0).any():
        issues.append("排名表存在缺失、非数值或负浓度。")
    else:
        checks.append("排名浓度均为非负有效数值")
    if len(vals) > 1 and not vals.is_monotonic_decreasing:
        issues.append("TOP 结果未按年均估算值降序排列。")
    else:
        checks.append("排名顺序为降序")
    if (pd.to_numeric(summary["有效天数"], errors="coerce") > _year_days(year)).any():
        issues.append("有效天数超过该年度自然日数量。")
    else:
        checks.append("有效天数未超过年度自然日数量")
    source = detail.dropna(subset=["pm25_daily_mean"])
    recomputed = source.groupby("city")["pm25_daily_mean"].agg(["mean", "count"])
    for _, row in summary.iterrows():
        city = row["city"]
        if city not in recomputed.index:
            issues.append(f"排名城市 {city} 无法在明细数据中复核。")
            continue
        if abs(round(float(recomputed.loc[city, "mean"]), 2) - float(row["年均PM25估算值"])) > 0.011:
            issues.append(f"{city} 的年均值与明细复算不一致。")
        if int(recomputed.loc[city, "count"]) != int(row["有效天数"]):
            issues.append(f"{city} 的有效天数与明细复算不一致。")
    if not issues:
        checks.append("TOP 城市年均值和有效天数已由明细独立复算")
    return {"passed": not issues, "checks": checks, "issues": issues}


def validate_ozone_result(result: dict) -> dict:
    issues: list[str] = []
    checks: list[str] = []
    valid_days, exceed_days = int(result["valid_days"]), int(result["exceed_days"])
    monthly = result["monthly"]
    if not (0 <= exceed_days <= valid_days <= 92):
        issues.append("臭氧有效天数或超标天数超出合理范围。")
    else:
        checks.append("有效天数与超标天数范围合理")
    if int(monthly["有效天数"].sum()) != valid_days:
        issues.append("臭氧月度有效天数之和与总有效天数不一致。")
    if int(monthly["超标天数"].sum()) != exceed_days:
        issues.append("臭氧月度超标天数之和与总超标天数不一致。")
    if not any("月度" in issue for issue in issues):
        checks.append("月度分项之和与总计一致")
    expected = round(exceed_days / valid_days * 100, 2) if valid_days else 0.0
    if abs(expected - float(result["exceed_rate"])) > 0.011:
        issues.append("臭氧超标比例与天数复算不一致。")
    else:
        checks.append("超标比例与天数复算一致")
    return {"passed": not issues, "checks": checks, "issues": issues}


def validate_province_result(table: pd.DataFrame, coverage: dict) -> dict:
    issues: list[str] = []
    checks: list[str] = []
    if table.empty:
        return {"passed": False, "checks": checks, "issues": ["省份统计结果为空。"]}
    if (table["有效城市日数"] <= 0).any() or (table["超标城市日数"] < 0).any() or (table["超标城市日数"] > table["有效城市日数"]).any():
        issues.append("省份计数不满足 0 ≤ 超标城市日数 ≤ 有效城市日数。")
    else:
        checks.append("省份计数边界正确")
    rates = pd.to_numeric(table["超标城市日占比(%)"], errors="coerce")
    if rates.isna().any() or (rates < 0).any() or (rates > 100).any():
        issues.append("省份占比存在缺失或超出 0%—100% 范围。")
    else:
        checks.append("省份占比均处于 0%—100% 范围")
    match_rate = float(coverage.get("match_rate", -1))
    if not (0 <= match_rate <= 100):
        issues.append("城市映射覆盖率超出 0%—100% 范围。")
    else:
        checks.append("城市映射覆盖率处于 0%—100% 范围")
    return {"passed": not issues, "checks": checks, "issues": issues}


def top_pm25_cities(year: int = 2024, top_n: int = 5, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    detail, load_report = _load_year(year, df)
    valid = detail.dropna(subset=["pm25_daily_mean"])
    if valid.empty:
        raise ValueError(f"{year} 年没有有效 PM2.5 日均数据。")
    summary = valid.groupby("city", as_index=False).agg(
        年均PM25估算值=("pm25_daily_mean", "mean"), 有效天数=("pm25_daily_mean", "count")
    )
    days = _year_days(year)
    summary["覆盖率(%)"] = (summary["有效天数"] / days * 100).round(1)
    summary["参考判定"] = summary.apply(
        lambda r: ("超过35 μg/m³" if r["年均PM25估算值"] > PM25_ANNUAL_LIMIT else "未超过35 μg/m³")
        if r["有效天数"] >= ANNUAL_MIN_VALID_DAYS else "覆盖不足：仅作样本均值参考", axis=1
    )
    summary["年均PM25估算值"] = summary["年均PM25估算值"].round(2)
    summary = summary.sort_values("年均PM25估算值", ascending=False).head(max(1, min(int(top_n), 20))).reset_index(drop=True)
    validation = validate_pm25_result(summary, detail, year)
    return summary, detail, {"load_report": load_report, "validation": validation}


def plot_pm25_monthly_trend(year: int = 2024, top_n: int = 5, df: pd.DataFrame | None = None, output_dir: str | Path | None = None) -> str:
    summary, detail, _ = top_pm25_cities(year, top_n, df)
    cities = summary["city"].tolist()
    monthly = detail[detail["city"].isin(cities)].dropna(subset=["pm25_daily_mean"]).groupby(["month", "city"], as_index=False)["pm25_daily_mean"].mean()
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for city in cities:
        part = monthly[monthly["city"].eq(city)].sort_values("month")
        ax.plot(part["month"], part["pm25_daily_mean"], marker="o", linewidth=1.8, label=city)
    ax.set_title(f"{year}年 PM2.5 年均高值城市月度变化")
    ax.set_xlabel("月份")
    ax.set_ylabel("PM2.5 月均浓度（μg/m³）")
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.text(0.01, 0.01, "说明：曲线为各月有效日均值的平均；35 μg/m³ 为年均二级参考限值，不作为月均判定线。缺失日未补造。", ha="left", va="bottom", fontsize=7.5, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = out / f"pm25_monthly_trend_{year}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def beijing_summer_o3(year: int = 2024, df: pd.DataFrame | None = None, output_dir: str | Path | None = None) -> dict:
    detail, load_report = _load_year(year, df)
    beijing = detail[detail["city_key"].eq(normalize_city("北京市"))].copy()
    if beijing.empty:
        raise ValueError("本地数据中没有北京市记录。请检查城市名称字段。")
    summer = beijing[beijing["month"].isin([6, 7, 8])].dropna(subset=["o3_8h_daily_max"]).copy()
    if summer.empty:
        raise ValueError(f"{year} 年北京夏季没有有效 O3 日最大8小时均值数据。注意：原始小时 O3 不能直接代替日最大8小时均值。")
    summer["exceed"] = summer["o3_8h_daily_max"] > O3_8H_DAILY_LIMIT
    monthly = summer.groupby("month", as_index=False).agg(
        有效天数=("o3_8h_daily_max", "count"),
        超标天数=("exceed", "sum"),
        O3_8h平均值=("o3_8h_daily_max", "mean"),
    )
    monthly = monthly.set_index("month").reindex([6, 7, 8], fill_value=0).rename_axis("month").reset_index()
    monthly["超标天数"] = monthly["超标天数"].astype(int)
    valid_days = int(len(summer))
    exceed_days = int(summer["exceed"].sum())
    result = {
        "year": int(year), "valid_days": valid_days, "exceed_days": exceed_days,
        "exceed_rate": round(exceed_days / valid_days * 100, 2) if valid_days else 0.0,
        "monthly": monthly, "load_report": load_report,
    }
    result["validation"] = validate_ozone_result(result)
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(monthly["month"].astype(str) + "月", monthly["超标天数"])
    ax.set_title(f"北京 {year} 年夏季 O3 日最大8小时均值超标天数")
    ax.set_ylabel("超标天数（天）")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, monthly["超标天数"]):
        ax.annotate(f"{int(value)}天", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    fig.text(0.01, 0.01, f"参考阈值：O3 日最大8小时均值 > {O3_8H_DAILY_LIMIT:g} μg/m³；仅统计有效观测日。", ha="left", va="bottom", fontsize=7.5, wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = out / f"beijing_o3_summer_{year}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    result["chart"] = str(path.resolve())
    return result


def province_pm25_exceedance(year: int = 2024, df: pd.DataFrame | None = None, mapping_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    detail, load_report = _load_year(year, df)
    valid = detail.dropna(subset=["pm25_daily_mean"]).copy()
    if valid.empty:
        raise ValueError(f"{year} 年没有有效 PM2.5 日均数据。")

    # 数据自带省份时优先使用；若省份字段部分缺失且外部映射文件存在，则补齐可映射项。
    if "province" in valid.columns:
        valid["province"] = valid["province"].astype("string").str.strip()
    else:
        valid["province"] = pd.NA
    needs_mapping = valid["province"].isna().any() or valid["province"].eq("").any()
    if needs_mapping:
        source_path = Path(mapping_path) if mapping_path else None
        if source_path is not None or __import__("config").PROVINCE_CSV.exists():
            mapping = load_province_mapping(mapping_path)
            valid = valid.merge(mapping[["city_key", "province"]].drop_duplicates("city_key"), on="city_key", how="left", suffixes=("", "_map"))
            if "province_map" in valid.columns:
                valid["province"] = valid["province"].replace("", pd.NA).fillna(valid["province_map"])
                valid = valid.drop(columns=["province_map"])
    merged = valid
    total_cities = int(merged["city"].nunique())
    matched_cities = int(merged.loc[merged["province"].notna(), "city"].nunique())
    matched = merged.dropna(subset=["province"]).copy()
    if matched.empty:
        raise ValueError("没有城市成功匹配到省份。请检查 city_province.csv 的城市名称。")
    matched["exceed"] = matched["pm25_daily_mean"] > PM25_DAILY_LIMIT
    table = matched.groupby("province", as_index=False).agg(
        有效城市日数=("pm25_daily_mean", "count"), 超标城市日数=("exceed", "sum")
    )
    table["超标城市日占比(%)"] = (table["超标城市日数"] / table["有效城市日数"] * 100).round(2)
    table = table.sort_values("超标城市日占比(%)", ascending=False).reset_index(drop=True)
    coverage = {
        "total_cities": total_cities, "matched_cities": matched_cities,
        "match_rate": round(matched_cities / total_cities * 100, 1) if total_cities else 0.0,
        "valid_city_days": int(len(matched)), "load_report": load_report,
    }
    coverage["validation"] = validate_province_result(table, coverage)
    return table, coverage


def city_monthly_trend(city: str, year: int = 2024, pollutant: str = "pm25", df: pd.DataFrame | None = None, output_dir: str | Path | None = None) -> dict:
    detail, load_report = _load_year(year, df)
    key = normalize_city(city)
    matched = detail[detail["city_key"].eq(key)].copy()
    if matched.empty:
        raise ValueError(f"本地数据中没有匹配到城市“{city}”。")
    pollutant_map = {
        "pm25": ("pm25_daily_mean", "PM2.5", "μg/m³"),
        "pm10": ("pm10_daily_mean", "PM10", "μg/m³"),
        "o3": ("o3_8h_daily_max", "O3 日最大8小时均值", "μg/m³"),
        "aqi": ("aqi_daily_max", "AQI", "指数"),
    }
    code = str(pollutant).strip().lower().replace(".", "")
    if code not in pollutant_map:
        raise ValueError("pollutant 只支持 pm25、pm10、o3 或 aqi。")
    column, label, unit = pollutant_map[code]
    monthly = matched.dropna(subset=[column]).groupby("month", as_index=False)[column].agg("mean")
    if monthly.empty:
        raise ValueError(f"城市“{city}”在 {year} 年没有有效的 {label} 数据。")
    monthly[column] = monthly[column].round(2)
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(monthly["month"], monthly[column], marker="o", linewidth=1.8)
    ax.set_title(f"{city} {year} 年 {label} 月度变化")
    ax.set_xlabel("月份")
    ax.set_ylabel(f"{label}（{unit}）")
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = out / f"city_trend_{key}_{year}_{code}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {
        "city": city, "year": int(year), "pollutant": label,
        "monthly": monthly.rename(columns={column: "月均值"}).to_dict(orient="records"),
        "chart": str(path.resolve()), "valid_days": int(matched[column].notna().sum()),
        "load_report": load_report,
    }


def data_quality_report(year: int | None = None) -> dict:
    frame, report = load_daily_data(year=year)
    columns = [c for c in ("pm25_daily_mean", "pm10_daily_mean", "o3_8h_daily_max", "aqi_daily_max") if c in frame]
    missing = {c: round(float(frame[c].isna().mean() * 100), 2) for c in columns}
    return {
        "source_file": report["source_file"], "rows": int(len(frame)),
        "cities": int(frame["city"].nunique()),
        "start_date": str(frame["date"].min().date()), "end_date": str(frame["date"].max().date()),
        "years": sorted(frame["year"].unique().astype(int).tolist()),
        "missing_percent_by_field": missing,
        "invalid_values_changed_to_missing": report["cleaning"]["total_invalid_values"],
        "date_parse_failed_rows": report["date_parse_failed"],
        "duplicate_city_date_rows_collapsed": report["duplicate_city_date_rows_collapsed"],
        "note": "缺失日期与缺失测量值均未被虚构填补。",
    }


def as_json(data: Any) -> str:
    """将工具结果转成标准 JSON 字符串，缺失数值转为 null。"""
    def sanitize(value: Any):
        if isinstance(value, dict):
            return {str(k): sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value) if math.isfinite(float(value)) else None
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if value is pd.NA or value is pd.NaT:
            return None
        return value
    return json.dumps(sanitize(data), ensure_ascii=False, indent=2, allow_nan=False)
