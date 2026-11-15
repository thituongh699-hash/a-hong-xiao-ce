# 量化研究工作台

界面统一使用中性深灰背景、分层面板、蓝色操作按钮及系统字体。涨跌维持 A 股常见的红涨绿跌；错误、风险和成功提示使用独立状态色。布局支持桌面、笔记本和窄屏，键盘焦点与减少动态效果设置保留。

视觉依据为 Apple 的 [Dark Mode](https://developer.apple.com/design/human-interface-guidelines/dark-mode) 与 [Color](https://developer.apple.com/design/human-interface-guidelines/color) 指南。Web 主题是本项目的实现，不依赖原生系统动态颜色。

## 功能与代码

| 职责 | 模块 |
| --- | --- |
| 策略信号 | `src/core/strategy_engine.py` |
| 交易风险审核 | `src/core/risk_manager.py` |
| 订单执行 | `src/core/execution_engine.py` |
| 数据前置校验 | `src/core/intent_planner.py` |
| 历史回测 | `src/core/backtest_engine.py` |
| 实时监测 | `src/core/live_trading_engine.py` |
| 策略注册、账户、撮合、报告、风险日志、行情 | `src/services/` |

前后端事件名称为 `strategy_signal`、`risk_review`、`order_execution`、`module_tick`。运行状态使用 `active_engine_type`。持久化旧配置和历史报告的读取兼容统一放在 `src/utils/legacy_contracts.py`，对外输出使用现行字段；不修改用户保存的原始记录。

## 价格口径反馈

- 配置中心检测当前表单草稿；更换源或复权设置后，过期响应不能覆盖新草稿。
- 启动前检测已保存配置，结果写入运行记录。此检测不能改写草稿验收结果或已完成报告的实际价格口径。
- 历史回测检测请求的日期区间。实时监测对全部标的检查原始价格、需要的复权因子及分钟时效，不能凭第一只股票的结果启动整组。
- 多标的检测最多并发两个工作进程，并有总时间预算。未通过时显示失败标的，不提交任务。
- 等待检测时终止单票回测会取消待启动请求。后端也会使停止或配置更改之前的实时启动请求失效。
- 报告和 K 线窗口分别说明指标价、成交价及图表原价。数据库仅按配置声明原始价时，明确显示“外部口径未独立验证”。旧报告缺少元数据时显示未确认。

完整价格规则见 [price-policy.md](price-policy.md)。分红和送转自动入账仍未实现；复权模式下持仓跨复权因子变化日会终止，不能把调整后的价格涨跌当作完整账户收益。

## 本地静态资源

仪表盘与报告共用 `static/css/workspace-theme.css` 和 `static/js/workspace-theme.js`。报告的 Tailwind、图标、K 线、雷达图和 Markdown 库使用本地资源，避免 CDN 不可达导致空白或缺失功能。

- Chart.js 4.5.1：`static/vendor/chartjs/`，附上游许可证。
- marked 18.0.14：`static/vendor/marked/`，附上游许可证。
- Lightweight Charts 5.1.0：复用项目现有版本。

前端回归测试：`node --test tests/frontend/test_price_readiness_ui.cjs`。
后端启动与快照回归测试：`python -m pytest tests/unit/test_start_price_readiness.py tests/unit/test_legacy_report_contracts.py`。

## 2026-10-02 验证记录

- 前端回归 18 项通过；针对价格、执行边界、启动取消、配置快照与历史报告的后端回归 134 项通过。
- 全量测试最终结果：187 通过、1 失败、1 跳过、1 警告，耗时 139.50 秒。失败项为 `test_api_test_tdx_connectivity_no_dir_error`：用例只替换服务层的目录探测，仍调用真实 TDX provider；当前本地/网络镜像环境可连通，返回 success，与用例假定的无目录必失败不符。本次未将此环境相关失败改成通过。
- 同日首轮全量测试为 186 通过、2 失败。另一失败是 AkShare 对固定历史日期的分钟接口返回空表；末轮该网络依赖用例被跳过。外部接口可用性存在波动，不能据此宣称全量测试全绿。
- 保留的测试警告：`scripts/test_tdx.py::test_tdx_provider` 返回 bool，触发 PytestReturnNotNoneWarning。
- 使用绝对路径启动本地服务，`/api/status/light`、工作台、报告页均返回 HTTP 200。实时监测未开启。
- TDX 样本 `000001.SZ`、`600000.SH`、`600036.SH` 在 2026-09-01 至 2026-10-02 区间均取得 20 根日线，最新日期均为 2026-09-29；000001.SZ 本地分钟文件最新记录为 2026-09-30 15:00。统一后复权因子可用，历史区间和当前数据检测均通过；这些日期仅说明本地已有数据范围。
- 浏览器验证：1280 像素桌面、390 像素窄屏均无页面横向溢出；复权草稿自动重测；历史报告 K 线成功加载，并标注当前 TDX 原始价格与历史回测指标口径的区别。浏览器未捕获 JavaScript 错误，运行 DOM 未发现重复 ID。
- 空复盘摘要不再伪装成缓存结果，模拟成交文字不再表述为实盘成交。用户配置及已有文件删除未加入本次提交。

详细日志与截图保存在本机 `reports/frontend-refresh/`，不纳入版本控制。
