from __future__ import annotations

import re
import json
from typing import Any

from config import (
    LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_BASE_URL,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
)
from analysis_tools import (
    as_json, beijing_summer_o3, city_monthly_trend, data_quality_report,
    plot_pm25_monthly_trend, province_pm25_exceedance, top_pm25_cities,
)

SYSTEM_PROMPT = """你是“空气质量时序分析 Agent”，使用 ReAct 工具调用完成真实数据分析。

硬性规则：
1. 必须根据用户问题自主选择工具。不要声称做过没有实际调用的工具。
2. 所有统计数字、排名、占比和日期范围都必须来自工具返回结果。严禁自行编造、补算或猜测数据。
3. 数据缺失、城市映射缺失、覆盖率不足或工具报错时，要如实说明，不得把缺失值当作零，也不得生成虚假结论。
4. PM2.5 年均二级参考限值为 35 μg/m³，日均二级参考限值为 75 μg/m³；O3 日最大 8 小时平均浓度二级参考限值为 160 μg/m³。超标判断采用严格大于阈值。
5. PM2.5 年均值是本地有效日均值的平均估算；有效天数少于 324 天时，必须说明覆盖不足，不作正式年度达标判断。
6. 省份 PM2.5 超标城市日占比 = 超标有效城市日数 / 有效城市日数。明确说明分母和数据口径。
7. 仅将 O3 日最大 8 小时平均浓度用于对应判定，不能将原始小时 O3 直接当作日最大 8 小时均值。
8. 最终用中文输出，优先采用“分析任务、统计结果、国标参考判定、数据口径、图表路径、环境解读”的结构。生态环境解读保持简短自然。
9. 如果问题超出当前工具能力，坦诚说明目前支持的分析范围，并给出可用问题示例，不要硬套不相关工具。
"""


def _build_tools():
    """延迟导入 LangChain，便于在未安装模型框架的环境中单独运行纯计算单元测试。"""
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise RuntimeError("缺少 LangChain 工具依赖，请在 DB 环境执行：pip install -r requirements.txt") from exc

    @tool
    def inspect_air_quality_dataset() -> str:
        """检查本地真实空气质量数据集的年份、城市数、记录数、缺失比例和异常清洗情况。适用于用户询问数据范围、数据质量、数据是否就绪。无参数。返回数据概况 JSON。"""
        try:
            return as_json(data_quality_report())
        except Exception as exc:
            return as_json({"error": str(exc), "tool": "inspect_air_quality_dataset"})

    @tool
    def query_top_pm25_cities(year: int = 2024, top_n: int = 5) -> str:
        """查询指定年份 PM2.5 年均浓度估算值最高的城市，并生成这些城市的月度变化折线图。参数 year 为年份，top_n 为排名数量（1—20）。适用于 PM2.5 年均排名、TOP 城市、月度变化曲线等问题。返回真实统计表、覆盖率、独立校验结果和图表路径。"""
        try:
            summary, detail, meta = top_pm25_cities(year, top_n)
            chart = plot_pm25_monthly_trend(year, top_n, df=detail)
            return as_json({
                "analysis": f"{year} 年 PM2.5 年均估算值 TOP {min(max(int(top_n), 1), 20)} 城市",
                "rows": summary.to_dict(orient="records"),
                "annual_reference_limit_ug_m3": 35,
                "annual_valid_day_minimum_for_formal_assessment": 324,
                "validation": meta["validation"],
                "cleaning": meta["load_report"].get("cleaning", {}),
                "data_note": "年均值由有效日均值计算；有效天数不足 324 天时仅作样本参考。35 μg/m³ 为年均限值，不作为月均曲线判定线。",
                "chart_path": chart,
            })
        except Exception as exc:
            return as_json({"error": str(exc), "tool": "query_top_pm25_cities"})

    @tool
    def query_beijing_summer_o3(year: int = 2024) -> str:
        """统计北京市指定年份夏季（6—8 月）O3 日最大 8 小时平均浓度超过二级参考限值的天数、比例和月度分布，并生成图表。参数 year 为年份。适用于北京夏季臭氧超标天数、臭氧月度分布等问题。"""
        try:
            result = beijing_summer_o3(year)
            serializable = dict(result)
            serializable["monthly"] = result["monthly"].to_dict(orient="records")
            serializable["data_note"] = "仅统计有效 O3 日最大 8 小时均值记录；阈值为 160 μg/m³，缺失日不计入分母。"
            return as_json(serializable)
        except Exception as exc:
            return as_json({"error": str(exc), "tool": "query_beijing_summer_o3"})

    @tool
    def query_province_pm25_exceedance(year: int = 2024, top_n: int = 15) -> str:
        """按省份统计指定年份 PM2.5 日均值超过 75 μg/m³ 的超标城市日占比，并按占比降序返回排名。参数 year 为年份，top_n 为返回省份数量。适用于各省 PM2.5 超标占比、区域比较等问题。需要数据中有 province 字段或本地城市-省份映射文件。"""
        try:
            table, coverage = province_pm25_exceedance(year)
            n = max(1, min(int(top_n), 40))
            return as_json({
                "analysis": f"{year} 年各省 PM2.5 超标城市日占比",
                "rows": table.head(n).to_dict(orient="records"),
                "daily_reference_limit_ug_m3": 75,
                "province_mapping_coverage": {k: v for k, v in coverage.items() if k not in ("load_report", "validation")},
                "validation": coverage.get("validation", {}),
                "data_note": "超标城市日占比 = 超标有效城市日数 / 有效城市日数；未匹配省份的城市不纳入省份汇总，并报告映射覆盖率。",
            })
        except Exception as exc:
            return as_json({"error": str(exc), "tool": "query_province_pm25_exceedance"})

    @tool
    def query_city_monthly_trend(city: str, year: int = 2024, pollutant: str = "pm25") -> str:
        """查询指定城市某一年某污染物的月度变化并生成折线图。参数 city 为城市名称；year 为年份；pollutant 取 pm25、pm10、o3 或 aqi。适用于用户指定城市的月度趋势、季节变化等问题。O3 指标为日最大 8 小时均值的月度平均。"""
        try:
            result = city_monthly_trend(city=city, year=year, pollutant=pollutant)
            return as_json(result)
        except Exception as exc:
            return as_json({"error": str(exc), "tool": "query_city_monthly_trend"})

    return [
        inspect_air_quality_dataset,
        query_top_pm25_cities,
        query_beijing_summer_o3,
        query_province_pm25_exceedance,
        query_city_monthly_trend,
    ]


def _build_model():
    provider = LLM_PROVIDER
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError("未安装 langchain-ollama。请执行 pip install -r requirements.txt") from exc
        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    if provider in ("openai", "cloud"):
        if not OPENAI_API_KEY:
            raise RuntimeError("LLM_PROVIDER=openai，但 .env 中没有设置 OPENAI_API_KEY。")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("未安装 langchain-openai。请执行 pip install -r requirements.txt") from exc
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL or None,
            temperature=0,
        )
    raise ValueError("LLM_PROVIDER 只支持 ollama 或 openai。")


def build_agent():
    try:
        from langchain.agents import create_agent
    except ImportError as exc:
        raise RuntimeError("当前 LangChain 不支持 create_agent 或未安装。需要 LangChain 1.x。") from exc
    model = _build_model()
    tools = _build_tools()
    return create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            elif isinstance(item, str):
                chunks.append(item)
        text = "\n".join(chunks)
    else:
        text = str(content)
    # DeepSeek-R1 有时会把推理标签混入 content；界面只显示最终答复。
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return text


def invoke_query(query: str, agent: Any | None = None) -> dict:
    """执行一次问题，返回最终回答和可供现场演示的工具调用轨迹。"""
    if not query or not query.strip():
        return {"answer": "请输入一个空气质量分析问题。", "trace": []}
    active_agent = agent if agent is not None else build_agent()
    result = active_agent.invoke({"messages": [{"role": "user", "content": query.strip()}]})
    messages = result.get("messages", []) if isinstance(result, dict) else []
    trace: list[dict[str, Any]] = []
    for message in messages:
        message_type = getattr(message, "type", None) or getattr(message, "role", None)
        calls = getattr(message, "tool_calls", None) or []
        for call in calls:
            trace.append({
                "kind": "call",
                "name": call.get("name", "unknown_tool"),
                "args": call.get("args", {}),
            })
        if message_type == "tool":
            raw_content = getattr(message, "content", "")
            content = _message_text(raw_content)
            observation: dict[str, Any] = {"kind": "observation", "name": getattr(message, "name", "tool"), "summary": "工具已返回结果"}
            try:
                payload = json.loads(content)
                if isinstance(payload, dict):
                    if payload.get("error"):
                        observation["summary"] = "工具返回错误：" + str(payload["error"])
                    else:
                        keys = [key for key in payload.keys() if key not in ("load_report",)]
                        observation["summary"] = "返回字段：" + "、".join(keys[:8])
                        if payload.get("chart_path") or payload.get("chart"):
                            observation["chart"] = payload.get("chart_path") or payload.get("chart")
                        if payload.get("validation"):
                            observation["validation_passed"] = payload["validation"].get("passed")
            except Exception:
                observation["summary"] = content[:180].replace("\n", " ")
            trace.append(observation)

    answer = "Agent 已执行，但没有返回可显示的最终文本。请检查 Ollama 模型的工具调用兼容性。"
    for message in reversed(messages):
        role = getattr(message, "type", None) or getattr(message, "role", None)
        if role in ("ai", "assistant"):
            candidate = _message_text(getattr(message, "content", ""))
            if candidate:
                answer = candidate
                break
    return {"answer": answer, "trace": trace}


def run_query(query: str, agent: Any | None = None) -> str:
    """简洁接口：只返回最终答复。"""
    return invoke_query(query, agent=agent)["answer"]
