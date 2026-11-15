# 冲高回落与近期低位因子研究

研究入口是 `scripts/research_low_reversal.py`，因子实现是
`src/strategies/low_reversal_factor.py`。它实际复用项目的 TDX 读取、
复权缓存、订单撮合、执行及账户核算，不改动运行配置或策略库。

用户信号为最高涨幅严格大于 A，且 A3 < 收盘涨幅 < A2，默认
7.2 / 4.8 / 1。近期低位分别用区间位置、回撤加接近低点、额外
MA5 初拐三个因果定义验证，定义只读取信号当日和以前的价格。

研究最近一年，随后四个市场交易日内比较收盘涨幅 > 3% 的同收盘
理论成交、最高涨幅 > 3% 的盘中突破成交，并补充收盘确认后次日开盘。
盘中成交以严格超过 3% 的第一档 0.01 元报价为触发价，跳空按开盘价。

训练 2025-10-09 至 2026-04-30，之后冻结测试；数据截止本地完整行情末日。
每段留 30 个交易日完成观察、入场、最长持仓和退出延迟缓冲。整体对照
另包含训练分段剔除的信号，缓存按评价区间隔离。末端未平仓和持仓跨
除权事件不用于收益排名。因子 459 组中 27 组是无低位条件的控制，
只有 432 组组合因子可被选中；各入场训练第一名再搜索 21 组退出。

运行与报告渲染：

```powershell
python scripts/research_low_reversal.py
python scripts/render_low_reversal_shapes.py
```

默认产物在 `reports/low-reversal-factor-2026-10-09`：

- `结果摘要.md`：总体、测试、事后配对解释及训练参数。
- `因子回测报告.html`：交互对照和下载；`低位形态核对.html` 只画信号日以前。
- `因子研究与回测.md`：完整定义、费用、时序、调参与限制。
- `comparisons.csv`、`test-trades.csv`、`paired-entry.csv`、`common-cohort.csv`。
- `training-grid.csv`、`training-exit-grid.csv`、`frozen-parameters.json`。
- `summary.json`、`data-coverage.csv`、`predeclared-design.json`：样本与指纹。

`verification.json` 是本轮额外复核记录，不由主入口自动生成。

定向验证：

```powershell
python -m pytest tests/unit/test_low_reversal_factor.py tests/unit/test_turning_point_backtest.py tests/unit/test_turning_point_execution.py tests/unit/test_price_adjustment.py tests/unit/test_price_execution_boundary.py tests/unit/test_execution_accounting_regression.py tests/unit/test_backtest_causal_execution.py -q
```

回测是独立现金试验，不能累加为组合年化。现名单缺历史 ST 和退市变更，
涨跌停价按普通主板假设，日线无法验证盘口成交，持仓公司行动核算尚未
实现。盘中指数时点只是近似，因此报告不声称精确盘中超额收益。
