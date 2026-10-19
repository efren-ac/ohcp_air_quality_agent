# OHCP 空气质量时序分析 Agent

一个面向 OHCP 一小时智能编程大赛的本地优先项目。使用 LangChain 1.x `create_agent` 构建 ReAct Agent，默认调用本地 Ollama 模型；数据清洗、统计、GB 3095—2012 参考判定和绘图由 Python 工具执行，模型不能自行生成统计数字。

## 1. 项目结构

```text
ohcp_air_quality_agent/
├─ agent.py                 # LangChain ReAct Agent、工具定义、模型切换
├─ analysis_tools.py        # 可独立测试的统计、校验、图表函数
├─ data_loader.py           # CSV 字段归一化、清洗、重复城市日处理
├─ preprocess.py            # 可选：真实小时数据转城市-日汇总
├─ config.py                # 本地数据路径、模型与标准限值
├─ main.py                  # 命令行入口
├─ check_environment.py     # Python / LangChain / Ollama 环境检查
├─ requirements.txt
├─ .env.example
├─ start.bat
├─ data/processed/          # 放真实的城市-日汇总 CSV 和城市省份映射
├─ outputs/                 # 自动生成的 PNG 图表
└─ tests/                   # 不依赖业务数据的单元测试
```

## 2. Windows 环境启动

你已确认 Anaconda 环境名为 `DB`，并安装了 Ollama 与 `deepseek-r1:7b`。在 Anaconda Prompt 中执行：

```powershell
conda activate DB
Set-Location "E:\你的项目目录\ohcp_air_quality_agent"
```

首次使用时复制 `.env.example` 为 `.env`（`start.bat` 会自动复制）。确认 Ollama 正在运行后执行：

```powershell
python check_environment.py
python -m pytest -q
python main.py
```

也可以双击 `start.bat`，但推荐从已激活 `DB` 环境的 Anaconda Prompt 启动，以免调用到其他 Python 环境。

## 3. 真实数据要求（重要）

项目不包含、不生成模拟业务数据。必须将你已有的真实空气质量数据放到本地，并且每一行应代表“一个城市的一天”。默认数据文件为：

`data/processed/daily_city_summary.csv`

至少需要字段：

- `date`：日期
- `city`：城市名
- 至少一个污染物字段：`pm25_daily_mean`、`pm10_daily_mean`、`o3_8h_daily_max`、`aqi_daily_max`
- 可选 `province`：省份

中文字段也支持常见别名，例如 `日期`、`城市`、`PM2.5日均值`、`PM10日均值`、`O3日最大8小时均值`。O3 字段必须确实表示“日最大 8 小时平均浓度”，不能把原始小时 O3 列直接改名代替。

如果你拿到的是原始小时级 CSV，可先运行预处理程序（需要提供实际数据文件路径）：

```powershell
python preprocess.py --input "E:\你的数据目录\真实小时数据.csv"
```

程序按城市与时间排序，PM2.5/PM10 每日有效小时数不少于 20 才计算日均值；O3 按时间窗口计算日最大 8 小时均值（窗口至少有 6 个有效小时值）。不补造缺失值。不同数据集字段命名可能不同，若无法识别，先检查报错中的字段列表并调整字段映射。

如果现有日汇总 CSV 在其他位置，在项目根目录 `.env` 中配置绝对路径，例如：

```ini
AQ_DAILY_CSV=E:/chop/air_quality_agent_project/data/processed/daily_city_summary.csv
AQ_PROVINCE_CSV=E:/chop/air_quality_agent_project/data/processed/city_province.csv
```

如果主 CSV 自带完整的 `province` 列，则可以不单独配置映射文件；否则省份分析需要一个包含 `city,province` 两列的真实映射 CSV。没有真实数据时，程序会明确提示，不会构造演示数据来冒充结果。

## 4. 支持的自然语言任务

1. `找出 2024 年 PM2.5 年均浓度最高的 5 个城市，画出月度变化曲线`
2. `统计北京市 2024 年夏季 O3 超标总天数和月度分布`
3. `统计 2024 年各省份 PM2.5 日均超标城市日占比，按占比排序`
4. `检查本地数据集的年份范围、城市数和缺失值情况`
5. `画出成都市 2024 年 PM2.5 月度变化曲线`

Agent 的工具选择由 LLM 完成；工具参数有明确的类型和说明。业务函数返回可核验的统计表、数据口径、校验结果与图表路径。

## 5. 数据口径

- PM2.5 年均参考限值：35 μg/m³；PM2.5 日均参考限值：75 μg/m³。
- PM10 年均参考限值：70 μg/m³；PM10 日均参考限值：150 μg/m³。
- O3 日最大 8 小时平均浓度参考限值：160 μg/m³。
- 超标按严格大于阈值判断。
- PM2.5 年均值由有效日均值求平均；有效天数少于 324 天时标记覆盖不足，不作正式年度达标判定。
- 省份超标城市日占比 = 超标有效城市日数 / 有效城市日数。省份映射覆盖率单独报告。
- 负值及超过配置合理上限的测量值转为缺失，不进行猜测填充；缺失日期不补造。
- 这里的阈值是项目采用的 GB 3095—2012 二级标准参考口径，正式环境评价应同时核对标准适用范围、数据有效性和官方评价规则。

## 6. 单元测试

```powershell
python -m pytest -q
```

测试使用临时构造的 DataFrame 验证函数正确性；这些构造数据仅用于软件测试，不会作为比赛分析结果或真实业务数据。

## 7. 可选云端模型兜底

默认 `LLM_PROVIDER=ollama`，不需要 API Key。只有在本地模型工具调用不稳定且允许联网时，才在 `.env` 中配置：

```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

这里的 `openai` 指 LangChain 的 OpenAI 兼容客户端，并不要求 API 服务一定来自 OpenAI。密钥只放本地 `.env`，不要上传到代码仓库或发到聊天中。比赛现场仍建议优先使用本地模型，避免网络依赖。

## 8. 已知边界

- `preprocess.py` 可将常见字段结构的真实小时级 CSV 转为城市-日汇总；若来源数据列名或时间格式特殊，需要根据实际数据调整字段映射。预处理不补造缺失值。
- 自动工具调用表现取决于本地模型。若 `deepseek-r1:7b` 无法稳定按工具 schema 调用，先用同一套工具测试其他已下载的 Ollama 模型，不要修改真实统计函数来迎合模型。
- 图表输出到 `outputs/`，结果中的统计值始终由工具计算产生。
