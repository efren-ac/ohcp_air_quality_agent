from __future__ import annotations

import importlib
import json
import sys
from urllib.request import urlopen
from urllib.error import URLError

from config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL


def main() -> int:
    print("Python:", sys.version.split()[0])
    modules = ["pandas", "numpy", "matplotlib", "pytest", "langchain", "langchain_ollama", "langchain_openai", "dotenv"]
    failed = False
    for name in modules:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "可导入")
            print(f"[OK] {name}: {version}")
        except Exception as exc:
            failed = True
            print(f"[FAIL] {name}: {exc}")
    try:
        from langchain.agents import create_agent
        print("[OK] langchain.agents.create_agent 可导入")
    except Exception as exc:
        failed = True
        print("[FAIL] create_agent:", exc)

    if LLM_PROVIDER == "ollama":
        tags_url = OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
        try:
            with urlopen(tags_url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            names = [item.get("name", "") for item in payload.get("models", [])]
            print("[OK] Ollama 服务可访问:", tags_url)
            print("已安装模型:", names)
            if not any(name == OLLAMA_MODEL or name.startswith(OLLAMA_MODEL + ":") for name in names):
                print(f"[WARN] 配置模型 {OLLAMA_MODEL!r} 未在 Ollama 模型列表中精确匹配，请检查 .env。")
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            failed = True
            print("[FAIL] Ollama 服务不可访问:", exc)
            print("可在另一个终端运行 ollama serve，或检查 Ollama 桌面程序是否正在运行。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
