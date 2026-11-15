"""Build a readable report from completed, frozen event-study outputs."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "历史研究结果_20261017"
CORE = ["A_原版", "E_V2量价共振", "F_V3突破", "G_V3非突破"]
LABELS = {CORE[0]: "原版重建", CORE[1]: "V2", CORE[2]: "V3突破", CORE[3]: "V3非突破"}


def pct(v):
    return "—" if pd.isna(v) else f"{100*v:+.2f}%"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def main():
    meta = json.loads((OUT / "运行元数据.json").read_text(encoding="utf8"))
    summary = pd.read_csv(OUT / "分段与年度统计.csv")
    cov = pd.read_csv(OUT / "股票覆盖审计.csv")
    annual = pd.read_csv(OUT / "逐年覆盖.csv")
    retired = pd.read_csv(OUT / "退市分组覆盖.csv")
    paired = pd.read_csv(OUT / "同日比较.csv")
    exclusions = pd.read_csv(OUT / "事件排除原因.csv")
    source = pd.read_csv(OUT / "跨来源核对.csv")
    st = pd.read_csv(OUT / "历史ST标签覆盖.csv", dtype={"historical_isST": str})
    full = summary.loc[(summary.scope == "all") & summary.strategy.isin(CORE)]
    h20 = full.loc[full.horizon.eq(20)].set_index("strategy").loc[CORE]
    main_rows = [[LABELS[k], f"{meta['signal_counts'][k]:,}", f"{int(r.events):,}", pct(r.net30_mean),
                  pct(r.excess300_mean), pct(r.excess1000_mean), f"{r.net30_win_rate:.1%}"]
                 for k, r in h20.iterrows()]
    yearly20 = summary.loc[summary.scope.eq("year") & summary.horizon.eq(20)]
    v2_years = yearly20.loc[yearly20.strategy.eq(CORE[1])]
    original_years = yearly20.loc[yearly20.strategy.eq(CORE[0])].set_index("label")
    net_delta_by_year = v2_years.set_index("label").net30_mean - original_years.net30_mean
    positive_1000 = int(v2_years.excess1000_mean.gt(0).sum())
    positive_300 = int(v2_years.excess300_mean.gt(0).sum())
    v2_alpha = h20.loc[CORE[1]]
    interval_text = ("跨越零，未排除零超额" if v2_alpha.excess300_ci_low <= 0 <= v2_alpha.excess300_ci_high
                     else "未跨零，但这没有消除数据选择和风格暴露")
    assessment = (f"20日样本中，V2相对原版的费用后均值变化为{100*(h20.loc[CORE[1], 'net30_mean']-h20.loc[CORE[0], 'net30_mean']):+.2f}个百分点，"
                  f"{int(net_delta_by_year.lt(0).sum())}/{len(net_delta_by_year)}个年份低于原版。"
                  f"V2对沪深300的年度超额为正{positive_300}/{len(v2_years)}年，对中证1000为正{positive_1000}/{len(v2_years)}年；"
                  f"沪深300超额季度块区间{interval_text}。"
                  f"按信号日等权后，V2对沪深300的20日超额为{pct(v2_alpha.date_equal_excess300)}，说明事件拥挤加权会改变判断。"
                  "这些证据不足以确认优化后获得跨年稳定、可成交的增益。不能根据单一平均收益保留或淘汰突破分类。")
    sections = ["# 先锋上车：本地历史事件研究\n\n研究日期：2026-10-17。",
        "## 结论与解释边界\n\n"
        "已完成原版重建、V2、V3突破/非突破及三个逐层过滤对照的分段事件研究。"
        "数据足够描述**当前覆盖样本**，不足以给出消除幸存者偏差后的全A股有效性结论。"
        "本轮没有调参数，也没有据结果另选最优持有期；所有时期都是回顾性描述，不能称为独立样本外检验。\n\n"
        "下表只展示20个市场交易日的事件均值，30bp为往返固定总费用情景；不同事件可重叠，不能累加、年化或解释成资金组合收益。",
        table(["版本", "全部信号", "共同可观察事件", "费用后均值", "超额沪深300", "超额中证1000", "净正收益比例"], main_rows),
        assessment,
        "## 覆盖审计\n\n"
        f"历史上市范围以保存的BaoStock基础信息快照重建，按上市/退市日期约束，每段包含当时上市且本地可核验的沪深A股。"
        f"目标范围共{meta['master_stocks']:,}只；可用原始行情和官方复权因子共{meta['status_counts'].get('ok',0):,}只。"
        f"缺失与异常分类：`{json.dumps(meta['status_counts'], ensure_ascii=False)}`。"
        f"已知截至研究截止日退市的{meta['known_delisted_by_cutoff']}只中，仅{meta['delisted_usable']}只具有两类可用输入；"
        "这些股票的终末交易阶段也未保证完整，因此补入少数退市股票不能消除幸存者偏差。\n\n"
        f"本次从已有BaoStock文件补齐{int(cov.tail_appended_bars.gt(0).sum())}只股票、{int(cov.tail_appended_bars.sum())}条尾部日线，"
        f"补尾重叠区最大成交量差{cov.tail_max_volume_error_shares.max():.0f}股。\n\n"
        "当前`data/stock_list.csv`是2026-08-05快照，只用作偏差对照标签，不作为历史证券筛选名单。"
        "基础信息也只是一个存量档案快照，未独立证明穷尽所有历史退市、代码迁移和证券身份变更。"
        "本地另有两个代码不在该基础信息名单内，已单列并排除："
        + "、".join(meta["local_codes_absent_from_basic"]) + "。\n\n"
        "行情广泛起于2020-01-02，不能补做2016—2019；统一预热120根有效个股日线后，"
        f"首个信号为{meta['signal_start']}，2020年只有后半年的有效信号。"
        f"原始行情虽更晚更新，复权研究统一止于{meta['end']}；"
        f"保留完整20日观察并不跨阶段边界的最后信号日为{meta['last_common_signal_date']}。\n\n"
        "北交所351只证券，以及B股、基金、指数和本轮范围外代码未纳入股票事件样本。北交所存在新浪因子文件，但一致性尚未核验，未混入官方BaoStock因子。",
        f"全信号中，历史日级ST标签未知的比例为{st.loc[st.historical_isST.eq('unknown'), 'signals'].sum()/st.signals.sum():.1%}；"
        "未以今日股票名称推断过去的ST状态，也没有因此宣称已模拟历史涨跌停规则。",
        table(["年份", "历史应有股票", "有原始行情", "有行情及因子", "原始股票日覆盖", "可复权股票日覆盖", "含统一预热后的覆盖"],
              [[int(r.year), int(r.expected_stocks), int(r.raw_stocks), int(r.factored_stocks),
                f"{r.raw_session_coverage:.1%}", f"{r.adjusted_session_coverage:.1%}", f"{r.warmed_session_coverage:.1%}"] for r in annual.itertuples()]),
        "覆盖比例的分母为本轮基础信息范围内、上市至退市期间的市场交易日数；停牌也可能造成原始行情缺口，"
        "所以这是股票日可观察比例，不能当作供应商数据完整率。总体较高的比例会掩盖退市子样本缺失。",
        "",
    ]
    # Dedicated denominator prevents broad-market counts hiding retired stocks.
    retired_rows = retired.loc[retired.delisted_by_cutoff.astype(str).str.lower().eq("true")]
    sections[-1] = table(["年份", "退市股有行情的代码数", "退市股有因子的代码数", "退市股可复权股票日覆盖"],
                         [[int(r.year), int(r.has_raw), int(r.has_factor),
                           f"{r.adjusted_stock_sessions/r.expected_stock_sessions:.1%}"] for r in retired_rows.itertuples()])
    sections += ["## 价格、规则和费用\n\n"
        "原始日线优先取通达信32字节`.day`；没有本地文件时使用已有BaoStock不复权日线（`adjustflag=3`）。"
        "若通达信提前停止而BaoStock保存了更晚的尾部，则先核对两源全部重叠有效日期的OHLC差不超过0.011元，"
        "再保留通达信历史并追加更晚尾部。逐日实查证明本地通达信二进制与BaoStock成交量均为股数，保持原单位；"
        "重叠成交量差须不同时超过100股和0.1%，否则拒绝补尾，避免量纲跳变；"
        "覆盖审计保留尾部追加天数和重叠核对天数。\n\n"
        f"对{len(source)}个重叠股票文件的一个共同日期逐项核对原始OHLC和成交量，最大价格差{source.max_ohlc_abs_error.max():.6f}元，"
        f"最大成交量差{source.volume_abs_error_shares.max():.0f}股；"
        "这是一日抽核，不是全历史逐根复核。\n\n"
        "信号以`原始OHLC × 当日累计backAdjustFactor`计算，0.01价格分母下限同步缩放。"
        "这样在本轮信号计算中使用相对于信号日的前复权等价比例，因子仅向前填充，不用未来因子反向填充。"
        "成交量保持原始量纲，未做股本拆细、送转股量能标准化，可能影响放量判断。"
        "SMA按pandas指数递推并统一预热，仍未在同花顺客户端逐根证明初始化和FILTER语义完全等价。\n\n"
        "A原版是沿用现有Python脚本重建的量价条件：超前9日最大量、收盘不低于开盘及MA15，"
        "并满足原有五组涨幅上限；研究目录没有独立保存最初原公式，本轮不能另行证明与其逐字相符。"
        "V2修正脚本原有`C>=MA15`为公式要求的`C>MA15`；它同时删去原版7/5/4日涨幅上限，"
        "因此V2并非原版信号的严格子集，B/C/D才是按层增加过滤的对照。V3在V2总信号统一FILTER(9)后互斥分类，"
        "突破为收盘超过前20日最高价，非突破为其余；不是分别FILTER后再合并。"
        "所有版本先在完整已有历史上计算9根冷却，再统一隐藏最初120根有效日线的输出，保留预热期的冷却状态。\n\n"
        "T日收盘确认，T+1市场交易日开盘观察进入，T+1+H市场交易日开盘观察退出，H为1/5/10/20。"
        "遇计划进出日没有有效行情，不顺延到下一根个股K线。阶段边界分别为2021、2024年末和2026-10-09；"
        "共同样本要求全部四个期限可观察，且20日期间不跨阶段末日。年度按信号年份划分，同一阶段内持有可跨自然年。\n\n"
        "收益始终按原始开盘价之比计算。持有期`买入日 < 因子变化日 <= 卖出日`的窗口排除，"
        "不将复权因子伪装成分红现金、送股数量或配股资金。"
        "这一排除及完整退出要求使用了事后信息，形成条件样本选择；它只能用于隔离本轮未实现的会计问题，不能直接作为实盘筛选规则。"
        "BaoStock官方说明也指出，其涨跌幅复权算法并不等于投资者实际参加分配后的收益："
        "[复权因子简介](https://www.baostock.com/helpdocs/pdf/BaoStock%E5%A4%8D%E6%9D%83%E5%9B%A0%E5%AD%90%E7%AE%80%E4%BB%8B.pdf)。\n\n"
        "净收益=原始开盘到开盘收益−0.003；60bp情景减0.006。固定费用含义为汇总佣金、税费和冲击的假定往返总扣减，"
        "未按年份税率、最低佣金、持仓规模或真实成交金额逐项计费。"
        "沪深300和中证1000均取相同买卖日期的指数开盘价格收益；不是ETF可成交收益、全收益指数、行业/市值中性收益或风险回归alpha。",
        "## 全期限对照",
        table(["版本", "持有交易日", "共同事件", "净收益30bp", "净收益60bp", "超额沪深300", "超额中证1000"],
              [[LABELS[r.strategy], int(r.horizon), f"{r.events:,}", pct(r.net30_mean), pct(r.net60_mean),
                pct(r.excess300_mean), pct(r.excess1000_mean)] for r in full.itertuples()]),
        "## 分段结果：20日\n\n阶段只是事先固定规则下的描述分组；2020年不完整，2026年也不是全年。",
        table(["信号阶段", "版本", "事件", "净收益", "超额沪深300", "超额中证1000"],
              [[r.label, LABELS[r.strategy], f"{r.events:,}", pct(r.net30_mean), pct(r.excess300_mean), pct(r.excess1000_mean)]
               for r in summary.loc[summary.scope.eq("segment") & summary.horizon.eq(20) & summary.strategy.isin(CORE)].itertuples()]),
        "## 年度稳定性：20日",
    ]
    years = summary.loc[summary.scope.eq("year") & summary.horizon.eq(20) & summary.strategy.isin(CORE)]
    for value, title in [("net30_mean", "费用后收益"), ("excess300_mean", "超额沪深300"), ("excess1000_mean", "超额中证1000")]:
        pivot = years.pivot(index="label", columns="strategy", values=value)
        sections += [f"### {title}", table(["信号年", *[LABELS[k] for k in CORE]],
                                            [[int(y), *[pct(row[k]) for k in CORE]] for y, row in pivot.iterrows()])]
    sections += ["## 相关性、比较与敏感性\n\n"
        "置信区间按信号日期所在自然季度整体重抽样2000次（固定随机种子），同季度内所有股票事件一起抽。"
        "这能保留部分同日拥挤和20日重叠相关性，但无法纠正缺失股票、退市样本选择、季度边界依赖和多重比较。"
        "不能把大事件笔数当成同样数量的独立观测；少于8个季度的分组不计算区间，表内以横线显示。",
        table(["版本", "20日超额沪深300", "季度块95%区间", "同日等权超额", "同日等权95%区间"],
              [[LABELS[k], pct(r.excess300_mean), f"[{pct(r.excess300_ci_low)}, {pct(r.excess300_ci_high)}]",
                pct(r.date_equal_excess300), f"[{pct(r.date_equal_ci_low)}, {pct(r.date_equal_ci_high)}]"] for k, r in h20.iterrows()]),
        "`同日比较.csv`只在同一个信号日两组均有信号时比较**不同股票集合**的日均值，"
        "不是同股配对，也未匹配行业、市值、市场beta或前期涨幅。V2−A或突破−非突破的差不能解释为规则本身的因果增益。",
        table(["阶段", "比较", "共同日期数", "20日均值差", "季度块95%区间"],
              [[r.segment, r.comparison, int(r.paired_dates), pct(r.difference), f"[{pct(r.ci_low)}, {pct(r.ci_high)}]"]
               for r in paired.loc[paired.horizon.eq(20)].itertuples()]),
        "另外已输出三项敏感性：只用2026-08现存名单、名单以外的可覆盖股票、去除进出日一字价格的样本。"
        "一字价格标签仅识别当日高=低，不等于正式涨跌停判定。去除它不等于模拟排队成交。"
        "`diagnostic/horizon_specific_clean`保留相应持有期可观察且不跨除权的事件，"
        "可检查共同20日清洁样本是否扭曲了短期结果；不同期限样本数在该诊断中允许不同。",
        table(["诊断范围", "版本", "20日事件", "净收益", "超额中证1000"],
              [[r.label, LABELS[r.strategy], f"{r.events:,}", pct(r.net30_mean), pct(r.excess1000_mean)]
               for r in summary.loc[summary.scope.isin(["universe", "liquidity_flag", "diagnostic"]) & summary.horizon.eq(20) & summary.strategy.isin(CORE)].itertuples()]),
        "## 尚未模拟的可成交条件\n\n"
        "- 开盘集合竞价是否能获得该开盘价、涨停买入/跌停卖出、封板排队与部分成交。\n"
        "- 历史ST/退市整理期/板块价格限制、IPO特殊交易日规则和除权参考价；没有按当前名称排除ST，已有日级isST只作标签。\n"
        "- 停牌复牌、缺失退出后的真实延迟卖出、退市损失和终末清算；这些退出缺失没有计作零收益。\n"
        "- 持仓期间分红、送转、配股的现金与股数变化；跨因子变化窗口采取排除。\n"
        "- 真实买卖滑点、价差、成交量容量、最低佣金和分年份税费。\n"
        "- 100股整数手、共享资金、仓位上限、多信号拥挤、重复持仓和交易次数约束。\n"
        "- H>=1满足隔日观察的时间安排，但没有实现账户级T+1可卖数量和资金结算。\n\n"
        "因此没有报告组合净值、年化收益或最大回撤；任何正事件均值也不能保证可执行策略有利。",
        "## 文件与复跑\n\n"
        "- `股票覆盖审计.csv`、`逐年覆盖.csv`、`退市分组覆盖.csv`：逐股缺失原因及分年覆盖。\n"
        "- `信号审计.csv.gz`、`逐笔事件.csv.gz`：包含未成熟、除权和无法观察端点的全部事件；主样本用`common_clean=True`筛选。\n"
        "- `分段与年度统计.csv`、`同日比较.csv`：全部7组规则及所有期限，收益字段均为小数（0.01表示1%）。\n"
        "- `事件排除原因.csv`、`历史ST标签覆盖.csv`、`跨来源核对.csv`：观察失败与来源验证。\n"
        "- `运行元数据.json`、`输入指纹.json`：参数、运行库、逐输入SHA-256；复跑时使用归档覆盖日，不触发缓存TTL刷新或网络请求。\n\n"
        "使用项目现有Python、pandas、numpy即可：\n\n"
        "```powershell\n"
        "python '先锋上车指标优化/运行历史事件研究.py'\n"
        "python '先锋上车指标优化/生成研究报告.py'\n"
        "```\n\n"
        f"本次输入与代码综合指纹：`{meta['data_and_code_sha256']}`；输入文件{meta['input_files']:,}个。"
        "`.day`文件若更新，指纹会变化；本轮只保存指纹，不复制行情原件，复现需要保留相同原始文件。",
    ]
    verification_path = OUT / "验收核对.json"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text(encoding="utf8"))
        sections += ["## 验证结果\n\n"
                     f"{verification['unit_tests_passed']}项针对性测试通过；"
                     f"{verification['signal_rows']:,}条版本信号、{verification['event_rows']:,}条持有期事件已完成计数与汇总收益核对。"
                     f"全部{verification['source_fingerprints_checked']:,}个归档输入的指纹未变化；"
                     f"独立重读核对{verification['sampled_price_and_calendar_checks']}个事件的原始进出价格、交易日和双基准收益，"
                     f"其中{verification['supplemental_price_rechecks']}个涉及补充行情。"
                     "已通过V3互斥且穷尽V2、费用单位、共同样本条件、逐期限诊断覆盖及汇总均值核对。"
                     "这些检查验证研究实现与导出一致性，不代表已验证真实成交或消除数据偏差。"]
    (HERE / "研究结果.md").write_text("\n\n".join(sections) + "\n", encoding="utf8")
    print(HERE / "研究结果.md")


if __name__ == "__main__":
    main()
