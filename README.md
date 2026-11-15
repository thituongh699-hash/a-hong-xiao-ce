<div align="center">
  <img src="./logo.png" alt="A红小策 Logo" width="132" />
  <h1>A红小策</h1>
  <p>面向 A 股的量化研究工作台：策略回测、因子研究、风控审核与模拟执行</p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white" />
    <img alt="API" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" />
    <img alt="UI" src="https://img.shields.io/badge/UI-Single--page-0A66C2" />
    <img alt="License" src="https://img.shields.io/badge/License-NonCommercial-blue" />
    <img alt="Market" src="https://img.shields.io/badge/Market-A--Share-c0392b" />
  </p>
</div>

## 这是什么

一个本地运行的量化工作台。把「策略生成 → 风控审核 → 执行清算 → 绩效复核」拆成四层，每一层都能单独验证。

三条设计取向，决定了它和「拉个框架跑个回测」的区别：

- **口径先于结论**：前向收益从 T+1 起算、按持有期不重叠取样、涨停与停牌样本按不可交易处理。收益算不准，后面所有分析都是噪音。
- **风控前置**：信号先过风险审核，止损、回撤、仓位约束不通过就不下单，违规留痕可查。
- **结论可复核**：回测与因子研究都输出结构化报告（含输入指纹），渲染层不重算、不挑最优参数。

## 核心能力

| 模块 | 能力 | 入口 |
|---|---|---|
| 策略回测 | 11 个内置策略 + 自定义策略，单票 / 批量 / 组合回测 | `/` 工作台 |
| 因子研究 | RankIC、ICIR、分层回测、IC 衰减、样本外检验，四道关卡给准入结论 | `/factor` |
| 通达信公式 | 公式编译成可执行策略（27 个常用函数），支持副图指标改造 | `/api/tdx/compile` |
| 参数优化 | 拐点参数网格搜索 + 样本内/外验证 | `/optimization` |
| 策略进化 | 事件驱动的多 Agent 生成 / 评估 / 迭代 | `/` → 操作工作台 |
| 风控与结算 | 一票否决风控、期末结算快照、现金持仓对账 | 引擎内 |
| AI 条件筛选 | 自然语言 → 结构化筛选条件 + 执行规则 | `/` |
| 数据接入 | 8 种数据源可切换，统一 OHLCV 校验 | 配置中心 |

## 界面导览

| 路径 | 用途 |
|---|---|
| `/` | 主工作台：选策略、填参数、跑回测、看日志与结算 |
| `/report` | 回测报告页（绩效、交易明细、一致性对比） |
| `/factor` | 因子研究：选因子、填样本、四道关卡结论、完整报告 |
| `/optimization` | 参数优化任务列表与结果 |

## 快速开始

**1. 环境**　Windows x64 / CPython 3.14（见 `.python-version`）。建议独立虚拟环境。

**2. 安装**

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.lock.txt
```

**3. 配置**　复制 `config.example.json` 为 `config.json`，按本机情况改数据源路径。密钥只放 `config.private.json`：

```json
{
  "data_provider": {
    "tushare_token": "你的token",
    "default_api_key": "你的api_key",
    "llm_api_key": "你的llm_key"
  }
}
```

读取优先级：环境变量 > `config.private.json` > `config.json`。

**4. 启动**

```bash
python server.py                # 默认 http://127.0.0.1:8000
python server.py --port 8001    # 换端口
```

一键脚本：`scripts/win一键启动.bat`、`bash "scripts/linux&macOS系统启动.sh"`。

首次启动只检查依赖，不会自动 pip 安装；确需自动安装时设 `AHXC_AUTO_INSTALL_DEPS=1`。

## 因子研究引擎

`src/factor/` 把「假设 → 数据 → 加工 → 检验 → 证伪 → 合成 → 组合 → 结论」固化成一条可复用管线，替代「一个因子一份脚本」。

四道关卡，任何一关不过就不进组合池：

| 关卡 | 检查项 | 阈值 |
|---|---|---|
| 预测力 | \|IC\| / ICIR / IC 同号率 / t 值 | 0.03 / 0.5 / 55% / 2.0 |
| 单调性 | 分层单调性 / 相邻逆序 / 多空 t / 胜率 | 0.8 / ≤1 / 1.5 / 55% |
| 稳健性 | 样本外保持率 / 参数扰动 / 分域稳定性 / 退市样本覆盖 | 0.6 / 1.0 / 0.6 / 必须 |
| 可实现性 | 扣费后多头超额 / 换手 | >0 / ≤85% |

结论分三态：`通过` / `待补证` / `淘汰`。缺样本外证据或股票池不含退市标的时，最高只能是「待补证」——把数据缺口摆在明面上，而不是给一个看起来通过的结论。

```bash
# 合成数据验证链路（数值不能当作有效性证据）
python scripts/run_factor_lab.py --demo

# 真实行情
python scripts/run_factor_lab.py --codes 600519,000001 --start 2021-01-01 --end 2025-12-31 --horizon 20
```

产物在 `reports/factor-studies/<研究编号>/`：`report.json`（严格 JSON，含 SHA256 指纹）+ `index.html`（离线单文件报告）。细节见 [因子研究框架](docs/因子研究框架.md)。

## 通达信公式

把公式编译成继承 `BaseImplementedStrategy` 的策略类，支持 27 个常用函数：`MA / EMA / SMA / WMA / HHV / LLV / SUM / REF / COUNT / STD / CROSS / IF / FILTER / EXIST / EVERY / BETWEEN / VALUEWHEN / BARSLAST / BARSSINCE / BARSCOUNT / HHVBARS / LLVBARS / UP / DOWN / ABS / MAX / MIN`。

```bash
curl -X POST http://127.0.0.1:8000/api/tdx/compile \
  -H "Content-Type: application/json" \
  -d '{"formula_text":"X:=MA(C,20);\nCROSS(C,X)","strategy_id":"TDX001","strategy_name":"均线突破"}'
```

三条口径必须注意：

1. **最后一条语句是持仓条件**，不是触发条件：为真则持有，转假则卖出。`CROSS(...)` 放最后会买一天卖一天。
2. **`:=` 是赋值，单个 `=` 是比较**。`A:=C>O` 赋值，`A=1` 是判断。
3. **绘图语句会被忽略**：`DRAWTEXT / DRAWICON / STICKLINE` 与 `COLOR / LINETHICK` 之类的显示属性不进计算。

## 回测口径

| 项 | 口径 |
|---|---|
| 时序 | 只用已完成 K 线；不足完整窗口不触发 |
| 成交 | 信号后下一根基础 K 线成交，不使用当根未来价格 |
| 边界 | 保留涨跌停与停牌字段；缺失时报告明确披露，不假装已验证 |
| 成本 | 佣金 + 印花税 + 滑点，按配置计；印花税卖出单边 |
| 结算 | 期末输出现金、持仓、已实现/未实现损益、费用与对账差额 |
| 范围 | 沪深主板 / 创业板 / 科创板 / 北交所，可独立勾选普通股、ST 与退市整理 |

价格口径与复权规则见 [复权说明](docs/price-policy.md)，交易范围见 [交易范围筛选](docs/交易范围筛选.md)。

## 数据源

| 来源 | 说明 |
|---|---|
| TDX | 本地通达信 `vipdoc` 日线/分钟线，默认选项，离线可用 |
| DuckDB / MySQL / PostgreSQL | 自建行情库，适合批量研究 |
| AkShare / BaoStock / Tushare / JQData | 网络数据源，需 token 或账号 |

统一入口 `DataFactory`，切换只改配置。字段与表结构见 [数据源说明](docs/DATA_PROVIDER_README.md)。

## 目录结构

```text
.
├─src/
│  ├─core/            # 意图规划、策略引擎、风控、执行、实盘
│  ├─services/        # 账户、撮合、报告、行情、策略注册
│  ├─strategies/      # 内置策略与自定义策略
│  ├─factor/          # 因子研究引擎（预处理 / 评估 / 关卡 / 合成）
│  ├─tdx/             # 通达信公式编译与终端桥接
│  ├─strategy_intent/ # 自然语言 → 策略意图
│  ├─evolution/       # 策略进化多 Agent
│  ├─optimization/    # 参数优化任务
│  └─utils/           # 配置、指标、数据源封装
├─data/               # 本地行情与策略库（不入库）
├─docs/               # 口径、接口与方案文档
├─static/             # 前端静态资源与子页面
├─tests/              # 单元测试与前端契约测试
├─dashboard.html      # 主面板
├─server.py           # FastAPI 服务入口
├─main.py             # 命令行回测入口
└─scripts/            # 启动、构建、研究脚本
```

## 文档

| 文档 | 内容 |
|---|---|
| [API 接口文档](docs/API_DOCS.md) | HTTP 接口、请求参数与响应示例 |
| [因子研究框架](docs/因子研究框架.md) | 八步流程、四道关卡、已知限制 |
| [组合回测](docs/组合回测.md) | 多策略信号的 or / and / vote 合成 |
| [策略进化](docs/策略进化.md) | 事件驱动多 Agent 的选种、评分与入库 |
| [参数优化模块](docs/参数优化模块.md) | 网格搜索与样本内外验证 |
| [批量回测操作指南](docs/批量回测操作指南.md) | 批量任务、覆盖率门禁与并发 |
| [回测交易范围](docs/交易范围筛选.md) | 板块与 ST 的筛选口径 |
| [复权说明](docs/price-policy.md) | 指标价、成交价与原价的关系 |
| [数据源说明](docs/DATA_PROVIDER_README.md) | `DataProvider` 封装与表结构 |
| [前端说明](docs/frontend.md) | 主题、本地静态资源与页面结构 |

研究记录（单因子检验的过程与结论）：[低吸反转](docs/low-reversal-factor-study.md)、[OBV 共振](docs/obv-resonance-factor.md)、[OBV 价值检验](docs/obv-alpha-study.md)。

## 桌面端

`desktop.spec` 配置 PyInstaller 打包，`desktop_launcher.py` 负责托盘、单实例与首次配置初始化。

```bash
bash scripts/build_desktop.sh          # macOS / Linux
python -m PyInstaller desktop.spec     # Windows
```

产物：`dist/A红小策/A红小策.exe`、`dist/A红小策.app`。

## 测试

```bash
python -m pytest tests/unit --ignore=tests/unit/akshare_test.py -q
node --test tests/frontend/test_price_readiness_ui.cjs tests/frontend/test_trading_scope_ui.cjs
```

当前基线：单元测试 706 项、前端契约测试 29 项通过。`akshare_test.py` 需联网，离线 CI 不运行。

## 已知限制

- **无历史时点股票池**：只有当前名单快照，缺逐日 ST 状态与退市样本，存在幸存者偏差。因子结论因此最高只能是「待补证」。
- **涨跌停按板块近似**：主板 10%、双创 20%、北交所 30%；ST 的历史 5% 幅度未逐日还原。
- **成本是一阶近似**：换手率 × 费率，没有冲击成本模型，容量需另行估算。
- **未做多重检验校正**：一次检验的因子越多，「最好的那个」越可能是运气，建议单次不超过 10 个。
- **实盘为模拟执行**：不接券商真实下单，仅做信号与撮合演练。

## 许可与免责

**个人非商业免费，商业使用需授权。** 完整条款见 [LICENSE](LICENSE)。

- 本项目仅用于量化学习、历史回测与技术研究，**不构成任何投资建议**，不是荐股软件，也不是投资咨询软件。
- 不提供买卖点位推荐，不预测价格走势，不推荐具体标的。
- 历史回测结果不代表未来收益，不得作为实盘交易依据。市场有风险，投资决策请独立判断。
- 任何第三方以本项目名义开展的收费荐股、代客理财、收益承诺等行为，均与开发者无关。
