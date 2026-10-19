from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import DAILY_CSV, PROVINCE_CSV

# 对输入列名进行轻量归一化；不把原始小时 O3 自动当作 O3 日最大 8 小时均值。
def _key(value: object) -> str:
    return re.sub(r"[\s_\-\.·（）()\[\]{}:：/\\]+", "", str(value)).casefold()

ALIASES = {
    "date": ["date", "日期", "监测日期", "统计日期", "datetime", "time", "日期时间"],
    "city": ["city", "城市", "城市名称", "cityname", "城市名"],
    "province": ["province", "省份", "省", "province_name", "省级行政区"],
    "pm25_daily_mean": [
        "pm25_daily_mean", "pm25", "pm2.5", "pm2.5日均值", "pm25日均值",
        "pm2.5浓度", "pm25浓度", "pm25_24h", "pm2.5日平均浓度", "细颗粒物日均值",
    ],
    "pm10_daily_mean": [
        "pm10_daily_mean", "pm10", "pm10日均值", "pm10浓度", "pm10_24h", "可吸入颗粒物日均值",
    ],
    "o3_8h_daily_max": [
        "o3_8h_daily_max", "o3_8h", "o3-8h", "o3_8h_max", "o3日最大8小时均值",
        "臭氧日最大8小时均值", "臭氧8小时日最大值", "o3_8h日最大值",
    ],
    "aqi_daily_max": ["aqi_daily_max", "aqi", "空气质量指数", "日最大aqi"],
}
ALIAS_LOOKUP = {_key(alias): canonical for canonical, aliases in ALIASES.items() for alias in aliases}
NUMERIC_COLUMNS = ("pm25_daily_mean", "pm10_daily_mean", "o3_8h_daily_max", "aqi_daily_max")
UPPER_BOUNDS = {
    "pm25_daily_mean": 1000.0,
    "pm10_daily_mean": 2000.0,
    "o3_8h_daily_max": 1000.0,
    "aqi_daily_max": 500.0,
}


def normalize_city(name: object) -> str:
    value = str(name).strip()
    for suffix in ("特别行政区", "自治州", "自治县", "地区", "市", "盟", "州"):
        if value.endswith(suffix) and len(value) > len(suffix):
            value = value[:-len(suffix)]
    return value.strip()


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """把常见中英文列名映射为项目内部统一字段名。"""
    rename: dict[object, str] = {}
    for column in frame.columns:
        canonical = ALIAS_LOOKUP.get(_key(column))
        if canonical and canonical not in [v for v in rename.values()]:
            rename[column] = canonical
    return frame.rename(columns=rename).copy()


def clean_invalid_measurements(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """把负数、非数值和超出配置合理上限的测量值转为缺失，不进行猜测填补。"""
    cleaned = frame.copy()
    report = {"invalid_by_column": {}, "total_invalid_values": 0}
    for column, upper in UPPER_BOUNDS.items():
        if column not in cleaned.columns:
            continue
        values = pd.to_numeric(cleaned[column], errors="coerce")
        invalid = values.notna() & ((values < 0) | (values > upper))
        count = int(invalid.sum())
        report["invalid_by_column"][column] = count
        report["total_invalid_values"] += count
        cleaned[column] = values.mask(invalid)
    return cleaned, report


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到本地日汇总数据：{path}\n"
            "请把真实空气质量数据整理为 daily_city_summary.csv，或在 .env 中设置 AQ_DAILY_CSV。"
        )
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("无法识别 CSV 文件编码。请另存为 UTF-8 或 GB18030 编码。\n" + "\n".join(errors))


def load_daily_data(year: int | None = None, path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """加载并规范化本地城市-日汇总数据。要求每行代表一个城市的一天。"""
    source = Path(path).expanduser() if path else DAILY_CSV
    raw = _read_csv(source)
    if raw.empty:
        raise ValueError(f"本地数据文件为空：{source}")

    frame = canonicalize_columns(raw)
    missing = [column for column in ("date", "city") if column not in frame.columns]
    if missing:
        raise ValueError(
            "数据缺少必要字段：" + ", ".join(missing) +
            f"。识别到的字段为：{list(frame.columns)}。请检查数据是否为城市-日汇总表。"
        )
    if not any(column in frame.columns for column in NUMERIC_COLUMNS):
        raise ValueError("数据中未识别到 PM2.5、PM10、O3 日最大8小时均值或 AQI 字段。")

    report: dict = {
        "source_file": str(source.resolve()),
        "raw_rows": int(len(frame)),
        "date_parse_failed": 0,
        "missing_city_rows": 0,
        "duplicate_city_date_rows_collapsed": 0,
    }
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    report["date_parse_failed"] = int(frame["date"].isna().sum())
    report["missing_city_rows"] = int(frame["city"].isna().sum())
    frame["city"] = frame["city"].astype("string").str.strip()
    frame = frame.dropna(subset=["date", "city"])
    frame = frame[frame["city"].ne("")].copy()

    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = float("nan")

    frame, cleaning = clean_invalid_measurements(frame)
    report["cleaning"] = cleaning
    frame["city_key"] = frame["city"].map(normalize_city)
    frame["month"] = frame["date"].dt.month
    frame["year"] = frame["date"].dt.year

    # 每行应是一个城市的一天；若输入有重复城市-日期，数值列取均值并记录处理数量。
    duplicated_mask = frame.duplicated(subset=["city_key", "date"], keep=False)
    if duplicated_mask.any():
        duplicated_rows = int(duplicated_mask.sum())
        report["duplicate_city_date_rows_collapsed"] = duplicated_rows
        agg = {col: "mean" for col in NUMERIC_COLUMNS}
        agg["city"] = "first"
        if "province" in frame.columns:
            agg["province"] = "first"
        frame = frame.groupby(["city_key", "date"], as_index=False).agg(agg)
        frame["month"] = frame["date"].dt.month
        frame["year"] = frame["date"].dt.year

    if year is not None:
        frame = frame[frame["year"].eq(int(year))].copy()
        if frame.empty:
            # 以规范化后的日期为准重新计算可用年份，避免原始字符串解析造成误报。
            parsed_dates = pd.to_datetime(canonicalize_columns(raw).get("date"), errors="coerce")
            available_years = sorted(parsed_dates.dropna().dt.year.unique().tolist())
            suffix = f"可用年份：{available_years}" if available_years else "无法从数据中识别可用年份。"
            raise ValueError(f"本地数据中没有 {year} 年的记录。{suffix}")

    frame.attrs["load_report"] = report
    return frame.reset_index(drop=True), report


def load_province_mapping(path: str | Path | None = None) -> pd.DataFrame:
    """读取城市-省份映射；优先使用数据自身的 province 字段。"""
    source = Path(path).expanduser() if path else PROVINCE_CSV
    mapping = _read_csv(source)
    mapping = canonicalize_columns(mapping)
    if "city" not in mapping.columns or "province" not in mapping.columns:
        raise ValueError("城市-省份映射文件必须包含 city/城市 与 province/省份 两列。")
    mapping = mapping[["city", "province"]].dropna().copy()
    mapping["city_key"] = mapping["city"].map(normalize_city)
    mapping["province"] = mapping["province"].astype(str).str.strip()
    return mapping.drop_duplicates("city_key", keep="first")
