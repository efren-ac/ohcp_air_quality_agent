from __future__ import annotations

import sys

from config import DAILY_CSV, LLM_PROVIDER, OLLAMA_MODEL
from agent import invoke_query

EXAMPLES = [
    "找出 2024 年 PM2.5 年均浓度最高的 5 个城市，画出月度变化曲线",
    "统计北京市 2024 年夏季 O3 超标总天数和月度分布",
    "统计 2024 年各省份 PM2.5 日均超标城市日占比，按占比排序",
    "检查本地数据集的年份范围、城市数和缺失值情况",
    "画出成都市 2024 年 PM2.5 月度变化曲线",
]


def main() -> int:
    print("=" * 68)
    print("OHCP 空气质量时序分析 Agent")
    print(f"模型提供方：{LLM_PROVIDER} | 本地模型：{OLLAMA_MODEL}")
    print(f"本地数据文件：{DAILY_CSV}")
    print("=" * 68)
    if not DAILY_CSV.exists():
        print("\n[数据提醒] 默认数据文件尚不存在。")
        print("请将真实的城市-日汇总 CSV 放到 data/processed/daily_city_summary.csv，")
        print("或在 .env 中通过 AQ_DAILY_CSV 指向现有数据文件。不会使用模拟业务数据。\n")
    print("示例问题：")
    for i, example in enumerate(EXAMPLES, 1):
        print(f"  {i}. {example}")
    print("\n输入问题开始分析；输入 exit / quit / q 退出。")
    cached_agent = None
    while True:
        try:
            query = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0
        if query.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            return 0
        if not query:
            continue
        try:
            # 同一个会话复用已构建的 Agent，减少重复初始化模型。
            if cached_agent is None:
                from agent import build_agent
                cached_agent = build_agent()
            result = invoke_query(query, agent=cached_agent)
            if result["trace"]:
                print("\n【Agent 工具调用轨迹】")
                for item in result["trace"]:
                    if item["kind"] == "call":
                        args = item.get("args", {})
                        print(f"  [Action] {item['name']}({args})")
                    else:
                        print(f"  [Observation] {item['name']}: {item['summary']}")
                        if item.get("chart"):
                            print(f"  [图表] {item['chart']}")
                        if item.get("validation_passed") is not None:
                            print(f"  [校验] {'通过' if item['validation_passed'] else '未通过'}")
            else:
                print("\n[提示] 本轮未检测到工具调用；请确认模型是否按 ReAct 方式使用工具。")
            print("\nAgent：\n" + result["answer"])
        except Exception as exc:
            print(f"\n[运行错误] {exc}")
            print("请确认：DB 环境已激活、Ollama 服务运行、模型名称正确、本地真实数据已配置。")
            # 初始化失败时不缓存失败状态，允许用户修复后重试。
            cached_agent = None
    return 0


if __name__ == "__main__":
    sys.exit(main())
