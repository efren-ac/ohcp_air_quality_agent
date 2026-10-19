from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR
from data_loader import normalize_city


def _key(value: object) -> str:
    return re.sub(r"[\s_\-\.·（）()\[\]{}:：/\\]+", "", str(value)).casefold()

ALIASES = {
    "city": ["city", "城市", "城市名称", "cityname", "城市名"],
    "province": ["province", "省份", "省", "province_name", "省级行政区"],
    "timestamp": ["timestamp", "datetime", "日期时间", "监测时间", "时间戳", "time", "时间", "date_time"],
    "date_only": ["date", "日期", "监测日期", "统计日期"],
    "hour": ["hour", "小时", "小时数", "时"],
    "pm25": ["pm25", "pm2.5", "pm2.5浓度", "pm25浓度", "细颗粒物"],
    "pm10": ["pm10", "pm10浓度", "可吸入颗粒物"],
    "o3": ["o3", "臭氧", "臭氧浓度"],
    "aqi": ["aqi", "空气质量指数"],
}


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别输入 CSV 编码，请另存为 UTF-8 或 GB18030。")


def _find_column(columns, aliases: list[str]):
    lookup = {_key(c): c for c in columns}
    for alias in aliases:
        found = lookup.get(_key(alias))
        if found is not None:
            return found
    return None


def _combine_date_hour(date_values: pd.Series, hour_values: pd.Series) -> pd.Series:
    numeric_hour = pd.to_numeric(hour_values, errors="coerce")
    if numeric_hour.notna().all() and numeric_hour.between(0, 23).all():
        hour_text = numeric_hour.astype(int).map(lambda value: f"{value:02d}:00:00")
    else:
        hour_text = hour_values.astype(str).str.strip()
        hour_text = hour_text.where(hour_text.str.contains(":", regex=False), hour_text + ":00:00")
    return pd.to_datetime(date_values.astype(str).str.strip() + " " + hour_text.astype(str), errors="coerce")


def preprocess_hourly_csv(input_path: str | Path, output_path: str | Path | None = None) -> dict:
    """将真实小时监测记录转成城市-日汇总。不会补造缺失小时或日期。"""
    source = Path(input_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    raw = _read_csv(source)
    if raw.empty:
        raise ValueError("输入 CSV 为空。")

    city_col = _find_column(raw.columns, ALIASES["city"])
    if city_col is None:
        raise ValueError(f"找不到城市字段。当前列名：{list(raw.columns)}")
    province_col = _find_column(raw.columns, ALIASES["province"])
    timestamp_col = _find_column(raw.columns, ALIASES["timestamp"])
    date_col = _find_column(raw.columns, ALIASES["date_only"])
    hour_col = _find_column(raw.columns, ALIASES["hour"])

    # 若有独立“日期”和“小时”字段，优先组合，避免把 0—23 小时数字误识别成日期。
    if date_col is not None and hour_col is not None:
        date_parsed = pd.to_datetime(raw[date_col], errors="coerce")
        has_real_clock = date_parsed.notna().any() and date_parsed.dt.hour.ne(0).any()
        if has_real_clock and timestamp_col == date_col:
            timestamps = date_parsed
        else:
            timestamps = _combine_date_hour(raw[date_col], raw[hour_col])
    elif timestamp_col is not None:
        timestamps = pd.to_datetime(raw[timestamp_col], errors="coerce")
    elif date_col is not None:
        timestamps = pd.to_datetime(raw[date_col], errors="coerce")
    else:
        raise ValueError("找不到时间字段。需要时间戳，或“日期”与“小时”字段。")

    def pollutant_column(kind: str):
        return _find_column(raw.columns, ALIASES[kind])

    selected = pd.DataFrame({
        "timestamp": timestamps,
        "city": raw[city_col].astype("string").str.strip(),
    })
    if province_col is not None:
        selected["province"] = raw[province_col].astype("string").str.strip()
    for kind in ("pm25", "pm10", "o3", "aqi"):
        col = pollutant_column(kind)
        selected[kind] = pd.to_numeric(raw[col], errors="coerce") if col is not None else np.nan

    selected = selected.dropna(subset=["timestamp", "city"])
    selected = selected[selected["city"].ne("")].copy()
    if selected.empty:
        raise ValueError("时间和城市字段清洗后没有有效记录。")

    # 负值及明显超出业务合理范围的数值转缺失，不填零、不插值。
    bounds = {"pm25": 1000, "pm10": 2000, "o3": 1000, "aqi": 500}
    invalid_counts = {}
    for col, upper in bounds.items():
        bad = selected[col].notna() & ((selected[col] < 0) | (selected[col] > upper))
        invalid_counts[col] = int(bad.sum())
        selected.loc[bad, col] = np.nan

    selected = selected.sort_values(["city", "timestamp"])
    # 同一城市同一时刻重复记录先按数值列均值合并，防止重复记录放大权重。
    group_cols = ["city", "timestamp"]
    agg = {col: "mean" for col in ("pm25", "pm10", "o3", "aqi")}
    if "province" in selected.columns:
        agg["province"] = "first"
    selected = selected.groupby(group_cols, as_index=False).agg(agg)
    selected["day"] = selected["timestamp"].dt.floor("D")

    # 每个城市按时间排序计算尾随 8 小时均值，至少需要 6 个有效小时值（75%）。
    selected["o3_8h_mean"] = np.nan
    for _, city_rows in selected.groupby("city", sort=False):
        ordered = city_rows.sort_values("timestamp")
        rolling = ordered.set_index("timestamp")["o3"].rolling("8h", min_periods=6).mean()
        selected.loc[ordered.index, "o3_8h_mean"] = rolling.to_numpy()

    records = []
    keys = ["city", "day"]
    if "province" in selected.columns:
        keys.append("province")
    for group_key, group in selected.groupby(keys, dropna=False, sort=True):
        if "province" in selected.columns:
            city, day, province = group_key
        else:
            city, day = group_key
            province = None
        pm25_valid = group["pm25"].dropna()
        pm10_valid = group["pm10"].dropna()
        o3_valid = group["o3_8h_mean"].dropna()
        aqi_valid = group["aqi"].dropna()
        records.append({
            "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
            "city": city,
            **({"province": province} if province is not None else {}),
            "pm25_daily_mean": float(pm25_valid.mean()) if len(pm25_valid) >= 20 else np.nan,
            "pm10_daily_mean": float(pm10_valid.mean()) if len(pm10_valid) >= 20 else np.nan,
            "o3_8h_daily_max": float(o3_valid.max()) if not o3_valid.empty else np.nan,
            "aqi_daily_max": float(aqi_valid.max()) if not aqi_valid.empty else np.nan,
            "pm25_valid_hours": int(len(pm25_valid)),
            "pm10_valid_hours": int(len(pm10_valid)),
            "o3_8h_valid_windows": int(len(o3_valid)),
        })

    daily = pd.DataFrame(records)
    destination = Path(output_path).expanduser() if output_path else PROCESSED_DIR / "daily_city_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(destination, index=False, encoding="utf-8-sig")
    return {
        "input_file": str(source.resolve()),
        "output_file": str(destination.resolve()),
        "input_rows": int(len(raw)),
        "hourly_rows_after_cleaning": int(len(selected)),
        "daily_city_rows": int(len(daily)),
        "invalid_values_changed_to_missing": invalid_counts,
        "pm25_days_below_20_valid_hours": int((daily["pm25_valid_hours"] < 20).sum()),
        "pm10_days_below_20_valid_hours": int((daily["pm10_valid_hours"] < 20).sum()),
        "note": "PM2.5/PM10 日均至少要求 20 个有效小时；O3 使用至少含 6 个有效值的尾随 8 小时滑动均值取日最大。缺失值未补造。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="将真实小时空气质量 CSV 预处理为城市-日汇总")
    parser.add_argument("--input", required=True, help="真实小时级空气质量 CSV 文件路径")
    parser.add_argument("--output", default=None, help="输出路径，默认 data/processed/daily_city_summary.csv")
    args = parser.parse_args()
    report = preprocess_hourly_csv(args.input, args.output)
    print(pd.Series(report).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
