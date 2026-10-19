from __future__ import annotations

import pandas as pd

from preprocess import preprocess_hourly_csv


def test_preprocess_hourly_real_shape_to_daily(tmp_path):
    # 仅测试转换算法的临时构造数据，不作为项目业务数据。
    rows = []
    for hour in range(24):
        rows.append({
            "时间": f"2024-06-01 {hour:02d}:00:00",
            "城市": "北京市",
            "省份": "北京市",
            "PM2.5": 20 + hour,
            "PM10": 40 + hour,
            "O3": 100 + hour,
            "AQI": 50 + hour,
        })
    input_path = tmp_path / "hourly.csv"
    output_path = tmp_path / "daily.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False, encoding="utf-8-sig")

    report = preprocess_hourly_csv(input_path, output_path)
    daily = pd.read_csv(output_path)
    assert report["daily_city_rows"] == 1
    assert daily.loc[0, "pm25_daily_mean"] == 31.5
    assert daily.loc[0, "pm10_daily_mean"] == 51.5
    assert daily.loc[0, "o3_8h_daily_max"] > 100
    assert daily.loc[0, "pm25_valid_hours"] == 24


def test_preprocess_does_not_fill_missing_pm25_day(tmp_path):
    rows = []
    for hour in range(12):
        rows.append({
            "日期": "2024-06-01", "小时": hour,
            "城市": "北京市", "PM2.5": 25, "O3": 120,
        })
    input_path = tmp_path / "hourly.csv"
    output_path = tmp_path / "daily.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False, encoding="utf-8-sig")
    preprocess_hourly_csv(input_path, output_path)
    daily = pd.read_csv(output_path)
    assert pd.isna(daily.loc[0, "pm25_daily_mean"])
    assert daily.loc[0, "pm25_valid_hours"] == 12
