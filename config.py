from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_DAILY_CSV = PROCESSED_DIR / "daily_city_summary.csv"
DEFAULT_PROVINCE_CSV = PROCESSED_DIR / "city_province.csv"
OUTPUT_DIR = Path(os.getenv("AQ_OUTPUT_DIR", str(BASE_DIR / "outputs"))).expanduser()
DAILY_CSV = Path(os.getenv("AQ_DAILY_CSV", str(DEFAULT_DAILY_CSV))).expanduser()
PROVINCE_CSV = Path(os.getenv("AQ_PROVINCE_CSV", str(DEFAULT_PROVINCE_CSV))).expanduser()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat").strip()

# GB 3095—2012 二级标准参考值。这里使用“超过”判断，即严格大于阈值。
PM25_ANNUAL_LIMIT = 35.0
PM25_DAILY_LIMIT = 75.0
PM10_ANNUAL_LIMIT = 70.0
PM10_DAILY_LIMIT = 150.0
O3_8H_DAILY_LIMIT = 160.0
ANNUAL_MIN_VALID_DAYS = 324
