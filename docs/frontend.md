# 前端说明

主面板是单文件 `dashboard.html`，通过 `server.py` 直接返回；因子研究和参数优化是 `static/` 下的独立子页面。

## 视觉与交互

- **底色**：中性深灰（`#121212` 背景 / `#e2e8f0` 正文），分层面板用更浅一档的灰，避免大面积纯黑。
- **涨跌色**：遵循 A 股习惯，**红涨绿跌**，与欧美惯例相反。
- **状态色**：错误 / 风险 / 成功各有独立色，不与涨跌色混用。
- **布局**：桌面、笔记本、窄屏三档；关键容器都设了 `min-width: 0` 与 `overflow-wrap`，长代码和长标的列表不会撑破页面。
- **可访问性**：保留键盘焦点样式；尊重系统「减少动态效果」设置。

## 本地静态资源

面板与报告共用 `static/css/workspace-theme.css` 与 `static/js/workspace-theme.js`。第三方库全部放本地 `static/vendor/`，不依赖 CDN——行情软件经常在离线环境跑，CDN 不通会直接白屏。

| 资源 | 位置 | 用途 |
|---|---|---|
| Tailwind CSS | `static/vendor/tailwindcss/` | 原子化样式 |
| Font Awesome 6.4 | `static/vendor/font-awesome/` | 图标 |
| Chart.js 4.5 | `static/vendor/chartjs/` | 净值、分布图 |
| Lightweight Charts 5.1 | 现有版本 | K 线 |
| marked | `static/vendor/marked/` | Markdown 渲染 |

各目录下附上游许可证，升级时一并替换。

## 价格口径的界面反馈

前端不猜价格，只显示后端给的口径：

- 更换数据源或复权设置后，过期响应不会覆盖新草稿。
- 报告与 K 线窗口分别标注**指标价**、**成交价**与**图表原价**；数据库仅声明原始价时显示「外部口径未独立验证」。
- 旧报告缺元数据时显示「未确认」，不默认按当前口径解读。

完整规则见 [复权说明](price-policy.md)。

## 子页面

| 页面 | 文件 | 入口 |
|---|---|---|
| 因子研究 | `static/factor-lab.html` | `/factor` |
| 参数优化 | `static/parameter-optimization.html` | `/optimization` |

两个子页面都是独立单文件：内联样式 + 原生 JS，不引入构建链。

## 测试

```bash
node --test tests/frontend/test_price_readiness_ui.cjs tests/frontend/test_trading_scope_ui.cjs
```

测试直接从 `dashboard.html` 抽取函数片段在 `vm` 沙箱里跑，校验存储键、请求参数与渲染分支，不起浏览器。
