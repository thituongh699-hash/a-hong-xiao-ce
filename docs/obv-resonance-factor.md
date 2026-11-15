# OBV 日周共振因子

实现：`src/strategies/obv_resonance_factor.py`。按用户确认的“包含本周”口径，
每个交易日收盘计算一次，本周周线只使用截至该日已经发生的数据。

## 精确定义

保留原公式的 `MAOBV=MA(OBV,18)`。日线为 18 日简单均线，周线为
18 根周 K 的简单均线。默认使用严格大于，不将相等视为满足条件。

```text
日条件 = 日OBV > 日MAOBV18
周上涨 = 本周临时OBV > 上一根周OBV > 上上根周OBV
周条件 = 本周临时OBV > 本周临时MAOBV18
因子 = 1（日条件 AND 周上涨 AND 周条件），否则为0
```

“连续两个周期上涨”指两次上涨，需要三个周 OBV 数值。日线只要求
OBV 高于其均线，没有额外要求日 OBV 连涨、刚刚金叉、均线上行，
也不包含原副图里的芙蓉出水、60 周期趋势线或 210 周期比较条件。

原公式 `REFX` 仅用于绘制预测成交量，因子计算不使用它。

## 包含本周，且不使用未来数据

按自然交易周（周一至周五）分组，不用滚动五日替代周 K。

```text
本周成交量(t) = 本周首个有效交易日至t的原始成交量之和
本周收盘价(t) = t日复权信号收盘价
本周OBV(t) = 上一根已结束周OBV
             + SIGN(本周收盘价(t) - 上一根已结束周收盘价)
               × 本周成交量(t) / 10000
本周MAOBV18(t) = (前17根已结束周OBV之和 + 本周OBV(t)) / 18
```

本周收盘高于前周则加上整段本周成交量，低于则减去，相等则不增减。
每一天都重新确定本周成交量的方向；周 OBV 不是日 OBV 的每周末抽样，
也不是把各日带符号的成交量简单相加。

全周没有有效 K 线时不创建虚构周 K；节假日短周按实际数据累计。
停牌后比较最近有数据的周 K，这与个股周线图的有效周期一致。
调用者需确认缺失日是休市/停牌而非缺失数据，并只传入已经完成的日 K。

周中信号可以 1→0→1。例如周一收盘高于前周，周二回到前周之下，
周三又回到前周之上，临时周 OBV 的方向会随之变化。历史每一天保存的是
当时可见的值，不能把后来周五的最终值回填给周一至周四。

## 价格与数据口径

输入单只股票的原始日 OHLCV，列为 `dt/open/high/low/close/vol`，`code`
可选但若存在必须唯一。输入排序后计算，每日最多一根 K 线。
原价来源必须通过项目 `assert_raw_prices` 检查；复权因子须具有项目
`AdjustmentPlan` 要求的已验证来源、1990 基准和覆盖范围，缺少证据时报错。

先逐日使用当时生效的后复权因子得到信号价格，再构造周收盘价；
成交量不复权，输出的原始 OHLCV 保留。不要先合成未复权周 K，再按
周末因子整体复权，这会错误处理周中除权。`vol` 请统一单位：缩放常数
只改变 OBV 显示值，不改变这些比较条件。初始 OBV 取 0；OBV 的共同
起点偏移不影响与均线的差或连续上涨判断。

默认预热需要至少 18 根有数据的周 K（包括本周）和 18 根日 K。
不足时 `history_ready=False`，因子输出 0。

## Python 使用

```python
from src.strategies.obv_resonance_factor import features, signal_mask
from src.utils.price_adjustment import assert_raw_prices, load_adjustment_factors

# raw 为单只股票已完成的原始日K；provider/config 为项目数据源和配置。
assert_raw_prices(provider, "tdx", raw, config)
raw = raw.copy()
raw.attrs["raw_price_verified_external"] = True  # 仅在上面的检查成功后设置
adjustment = load_adjustment_factors(
    raw.code.iloc[0], raw.dt.min(), raw.dt.max(), config
)
work = features(raw, adjustment)  # 日/周MA均为18，可用ma_period显式修改
all_signals = work.loc[signal_mask(work)]
new_signals = work.loc[signal_mask(work, first_only=True)]
factor = work["obv_resonance_factor"]
```

| 输出列 | 含义 |
|---|---|
| `daily_obv / daily_maobv` | 日 OBV / 日 MAOBV |
| `week_start` | 当前交易周周一日期 |
| `weekly_close / weekly_volume` | 截至当天的周信号收盘价 / 本周累计原始成交量 |
| `weekly_obv / weekly_maobv` | 截至当天的临时周 OBV / 周 MAOBV |
| `weekly_prev_obv / weekly_prev2_obv` | 前两根有数据的已结束周 OBV |
| `daily_above_maobv` | 日条件是否满足 |
| `weekly_above_maobv` | 周均线条件是否满足 |
| `weekly_rising_two` | 周 OBV 是否连续上涨两次 |
| `history_ready` | 日、周历史是否足够 |
| `obv_resonance_factor` | 0/1 共振因子；满足期间每天为 1 |
| `resonance_trigger` | 因子由 0 变为 1 的当日，包括失效后的再次触发 |
| `signal_close / adjustment_factor / signal_multiplier` | 复权信号价及审计明细 |

这是选股状态因子。收盘后产生的信号可用于下一交易日的交易决策；
买入方式、持仓数量、退出规则和收益评价尚未定义，不自动注册为买卖策略。

## 同花顺条件说明

官方选股平台支持设置分析周期和使用自编选股条件：
[同花顺选股平台说明](https://www.10jqka.com.cn/ad_mar/man/07-3.htm)。
分别创建两个条件，再对同一股票池取交集。实际操作须采用包含本周的
实时周线；如果保存历史信号，需要逐日保存当时的周线状态。

**周线条件，分析周期选周线：**

```text
VA:=IF(CLOSE>REF(CLOSE,1),VOL,-VOL);
OBVX:=SUM(IF(CLOSE=REF(CLOSE,1),0,VA),0)/10000;
MAOBVX:=MA(OBVX,18);
SELECT:OBVX>REF(OBVX,1) AND REF(OBVX,1)>REF(OBVX,2) AND OBVX>MAOBVX;
```

**日线条件，分析周期选日线：**

```text
VA:=IF(CLOSE>REF(CLOSE,1),VOL,-VOL);
OBVX:=SUM(IF(CLOSE=REF(CLOSE,1),0,VA),0)/10000;
MAOBVX:=MA(OBVX,18);
SELECT:OBVX>MAOBVX;
```

两者交集就是目标共振条件。没有确认本机同花顺版本的跨周期语法，
因此这里提供分周期条件，不将其他软件的 `#WEEK` 写法当成已验证语法。
上述条件尚未在同花顺客户端编译或与实盘图逐条核对；核对时请统一
复权方式、日/周历史长度与成交量单位。软件预热阶段的均线行为可能不同，
比较应在双方都有完整 18 根周 K 历史以后进行。

## 验证

```powershell
python -m pytest tests/unit/test_obv_resonance_factor.py tests/unit/test_price_adjustment.py -q
```

定向测试覆盖逐日独立周线重算、前缀不变性、周中失效/再触发、严格比较、
日条件拦截、短周/空周、周中除权、原始成交量保持和不合格输入拒绝。
这些验证确认因子实现与指定规则一致，不代表已经验证预测能力或回测收益。

2026-10-10 本轮运行：57 项定向测试通过；000001.SZ、600036.SH、600519.SH
各读取 485 根本地日 K（2024-10-08 至 2026-09-30），33 个截断时点通过
前缀一致性与独立周线重算核验。该本地快照并未覆盖 2026-10-09。
完整输入文件的 SHA256、复权缓存覆盖日期/抓取时间和核验结果保存在
`reports/obv-resonance-factor-2026-10-10/verification.json`，
逐日明细及选股记录为同目录的 `sample-features.csv`、`sample-signals.csv`。
`verify_local_samples.py` 可重跑此三股票核验；它不联网刷新输入。
