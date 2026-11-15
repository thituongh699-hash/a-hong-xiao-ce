"""Render exported OBV study metrics; never recompute or select parameters."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    alpha = pd.read_csv(out / "alpha-summary.csv")
    execution = pd.read_csv(out / "execution-summary.csv")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f6f8fb"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    periods = ["2023_Q4", "2024", "2025", "2026_YTD", "recent_year"]
    labels = ["2023 Q4", "2024", "2025", "2026 至9月", "最近一年"]
    for metric, offset, color, title in (("matched_spread_pct", -.13, "#246eb9", "动量/波动率匹配"),
                                        ("conditional_beta_pct", .13, "#da7c30", "控制变量后系数")):
        selected = alpha[(alpha.variant == "resonance") & (alpha.horizon == 10)
                         & (alpha.metric == metric)].set_index("period").reindex(periods)
        mean = selected.mean_pct.to_numpy()
        error = np.vstack([mean - selected.hac_ci_low_pct.to_numpy(),
                           selected.hac_ci_high_pct.to_numpy() - mean])
        axes[0].errorbar(np.arange(len(periods)) + offset, mean, yerr=error,
                         fmt="o", capsize=3, color=color, label=title)
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=25)
    axes[0].set_title("10日增量预测力：均值及95% HAC区间")
    axes[0].set_ylabel("百分点")
    axes[0].legend(loc="best", fontsize=9)

    latest = alpha[(alpha.variant == "resonance") & (alpha.period == "recent_year")
                   & (alpha.metric == "matched_spread_pct")].set_index("horizon").reindex([1, 5, 10, 20])
    mean = latest.mean_pct.to_numpy()
    error = np.vstack([mean - latest.bootstrap_ci_low_pct.to_numpy(),
                       latest.bootstrap_ci_high_pct.to_numpy() - mean])
    axes[1].bar(np.arange(4), mean, color="#246eb9", alpha=.8)
    axes[1].errorbar(np.arange(4), mean, yerr=np.maximum(error, 0), fmt="none", color="#172238", capsize=4)
    axes[1].set_xticks(np.arange(4), ["1日", "5日", "10日主检验", "20日"])
    axes[1].set_title("最近一年：匹配超额及日期分块95%区间")
    axes[1].set_ylabel("百分点")

    rules = ["hold5", "hold10", "hold20", "factor_loss", "hold10_stress"]
    trading = execution[execution.period.eq("recent_year")].set_index("rule").reindex(rules)
    x = np.arange(len(rules))
    axes[2].bar(x-.18, trading.mean_net_pct, width=.36, color="#da7c30", label="每笔净收益")
    axes[2].bar(x+.18, trading.mean_excess_csi300_pct, width=.36, color="#246eb9", label="相同持有区间超额")
    axes[2].set_xticks(x, ["5日", "10日", "20日", "失效退出", "10日滑点压力"], rotation=25)
    axes[2].set_title("最近一年：首次触发后次日开盘买入")
    axes[2].set_ylabel("平均收益 %")
    axes[2].legend(loc="best", fontsize=9)
    for ax in axes:
        ax.axhline(0, color="#576477", linewidth=1)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("OBV 日周共振因子 · 固定18周期 · 包含当周", fontsize=18, fontweight="bold")
    fig.text(.5, .025, "2023-10至2026-09｜当前主板正常股票名单有生存者偏差｜独立交易均值不能累加为组合年化",
             ha="center", fontsize=10, color="#576477")
    fig.tight_layout(rect=[0, .06, 1, .91])
    fig.savefig(out / "study-overview.png", dpi=180)
    plt.close(fig)
    report = out / "因子价值报告.html"
    page = report.read_text(encoding="utf-8")
    if "study-overview.png" not in page:
        page = page.replace("<section><h2>主要10日", '<section><img src="study-overview.png" alt="因子增量预测力和执行收益对照" style="width:100%;height:auto"></section><section><h2>主要10日', 1)
        report.write_text(page, encoding="utf-8")
    print(str(out / "study-overview.png"))


if __name__ == "__main__":
    main()
