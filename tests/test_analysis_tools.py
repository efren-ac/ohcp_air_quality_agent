from __future__ import annotations

import pandas as pd
import pytest

from data_loader import canonicalize_columns, clean_invalid_measurements, normalize_city
from analysis_tools import (
    beijing_summer_o3, city_monthly_trend, province_pm25_exceedance,
    top_pm25_cities, validate_ozone_result, validate_pm25_result,
    validate_province_result,
)


def frame_for_tests():
    # 仅用于单元测试的构造数据，不会作为项目业务数据或演示数据。
    rows = []
    for day in range(1, 16):
        rows.append({
            "date": f"2024-06-{day:02d}", "city": "北京市", "province": "北京市",
            "pm25_daily_mean": 30 + day, "pm10_daily_mean": 55 + day,
            "o3_8h_daily_max": 150 + (day % 4) * 10, "aqi_daily_max": 60 + day,
        })
        rows.append({
            "date": f"2024-06-{day:02d}", "city": "成都市", "province": "四川省",
            "pm25_daily_mean": 45 + day, "pm10_daily_mean": 75 + day,
            "o3_8h_daily_max": 140 + (day % 3) * 15, "aqi_daily_max": 80 + day,
        })
    for month in (7, 8):
        for day in range(1, 5):
            rows.append({
                "date": f"2024-{month:02d}-{day:02d}", "city": "北京市", "province": "北京市",
                "pm25_daily_mean": 25 + day, "pm10_daily_mean": 50 + day,
                "o3_8h_daily_max": 165 + day, "aqi_daily_max": 50 + day,
            })
    return pd.DataFrame(rows)


def test_canonicalize_chinese_columns():
    frame = pd.DataFrame({"日期": ["2024-01-01"], "城市": ["北京市"], "PM2.5日均值": [10]})
    result = canonicalize_columns(frame)
    assert "date" in result and "city" in result and "pm25_daily_mean" in result


def test_normalize_city():
    assert normalize_city("北京市") == "北京"
    assert normalize_city("石家庄市") == "石家庄"


def test_invalid_values_are_missing_not_filled():
    frame = pd.DataFrame({"pm25_daily_mean": [-1, 35, 1001], "aqi_daily_max": [50, 501, 80]})
    cleaned, report = clean_invalid_measurements(frame)
    assert pd.isna(cleaned.loc[0, "pm25_daily_mean"])
    assert cleaned.loc[1, "pm25_daily_mean"] == 35
    assert pd.isna(cleaned.loc[2, "pm25_daily_mean"])
    assert report["total_invalid_values"] == 3


def test_pm25_top_cities_and_independent_validation():
    df = frame_for_tests()
    summary, detail, meta = top_pm25_cities(2024, 2, df=df)
    assert len(summary) == 2
    assert summary.iloc[0]["年均PM25估算值"] >= summary.iloc[1]["年均PM25估算值"]
    validation = validate_pm25_result(summary, detail, 2024)
    assert validation["passed"], validation["issues"]


def test_ozone_monthly_sums_and_plot(tmp_path):
    df = frame_for_tests()
    result = beijing_summer_o3(2024, df=df, output_dir=tmp_path)
    assert result["valid_days"] == 23
    assert result["exceed_days"] > 0
    assert validate_ozone_result(result)["passed"]
    assert (tmp_path / "beijing_o3_summer_2024.png").exists()


def test_ozone_validator_catches_inconsistent_totals():
    result = {
        "valid_days": 3, "exceed_days": 2, "exceed_rate": 66.67,
        "monthly": pd.DataFrame({"有效天数": [1, 1], "超标天数": [1, 1]}),
    }
    assert not validate_ozone_result(result)["passed"]


def test_province_exceedance_and_coverage():
    df = frame_for_tests()
    table, coverage = province_pm25_exceedance(2024, df=df)
    assert not table.empty
    assert 0 <= coverage["match_rate"] <= 100
    assert validate_province_result(table, coverage)["passed"]


def test_city_monthly_trend_generates_chart(tmp_path):
    result = city_monthly_trend("北京", 2024, "pm25", df=frame_for_tests(), output_dir=tmp_path)
    assert result["monthly"]
    assert (tmp_path / "city_trend_北京_2024_pm25.png").exists()


def test_province_validator_rejects_out_of_range_percentage():
    table = pd.DataFrame({
        "有效城市日数": [10], "超标城市日数": [12], "超标城市日占比(%)": [120.0]
    })
    result = validate_province_result(table, {"match_rate": 95.0})
    assert not result["passed"]
