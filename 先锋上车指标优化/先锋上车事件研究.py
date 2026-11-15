#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""先锋上车事件研究：原版与逐层改进版的消融对照。
仅执行信号后固定持有期收益的事件研究；不是可实盘执行的投资组合回测。
CSV 列（不区分大小写）：date,symbol,open,high,low,close,volume。
例：python 先锋上车事件研究.py --input prices.csv --outdir output --cost-bps 30
当天收盘信号，T+1日开盘模拟买入，T+1+H日开盘模拟卖出；收益需确保实际可成交。
同花顺 SMA 递推在程序中用 pandas EWM(alpha=1/N,adjust=False)近似，初始化口径可能与终端不同。
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 20)


def suppress_signals(raw: pd.Series, cooldown: int = 9) -> pd.Series:
    keep = np.zeros(len(raw), dtype=bool)
    last = -cooldown - 1
    for i, flag in enumerate(raw.fillna(False).astype(bool).to_numpy()):
        if flag and i > last + cooldown:
            keep[i] = True
            last = i
    return pd.Series(keep, index=raw.index)


def compute_one(g: pd.DataFrame, common_warmup: int = 0) -> pd.DataFrame:
    g = g.sort_values("date").copy().reset_index(drop=True)
    c, o, h, l, v = (g[k].astype(float) for k in ["close", "open", "high", "low", "volume"])
    # A local archival runner can supply raw_price * cumulative_factor. Scaling
    # the floor too gives the same ratios as qfq rebased at the signal date,
    # without using any future factor to normalize historical prices.
    price_floor = g["price_floor"] if "price_floor" in g else 0.01
    ma15 = c.rolling(15, min_periods=15).mean()
    rsv10 = (c - l.rolling(10, min_periods=10).min()) / (
        h.rolling(10, min_periods=10).max() - l.rolling(10, min_periods=10).min()
    ).clip(lower=price_floor) * 100
    rsv30 = (c - l.rolling(30, min_periods=30).min()) / (
        h.rolling(30, min_periods=30).max() - l.rolling(30, min_periods=30).min()
    ).clip(lower=price_floor) * 100
    k10 = rsv10.ewm(alpha=1/5, adjust=False).mean()
    d10 = k10.ewm(alpha=1/3, adjust=False).mean()
    pioneer = 4*k10 - 3*d10
    mid = rsv30.ewm(alpha=1/5, adjust=False).mean().ewm(alpha=1/3, adjust=False).mean()
    g["先锋"] = pioneer
    g["中期"] = mid

    big_volume = v > v.shift(1).rolling(9, min_periods=9).max()
    ratio_13 = v > (v.shift(1).rolling(10, min_periods=10).mean() * 1.30)
    positive_candle = c >= o
    above_ma15 = c >= ma15
    early_returns = pd.Series(True, index=g.index)
    for n, ceil in [(10, 1.50), (7, 1.33), (5, 1.25), (4, 1.22), (3, 1.17)]:
        early_returns &= (c / c.shift(n)) < ceil

    base = big_volume & positive_candle & above_ma15 & early_returns
    close_pos = ((c - l) / (h-l).clip(lower=price_floor)) >= 0.65
    candle_quality = (c > o) & (c > c.shift(1)) & close_pos
    slope = ma15 > ma15.shift(3)
    momentum = (pioneer > 25) & (pioneer > pioneer.shift(1)) & (mid >= mid.shift(1))
    no_overheat = (c / c.shift(3) < 1.17) & (c / c.shift(10) < 1.50)
    valid = pd.Series(np.arange(len(g)) >= 60, index=g.index)

    rules = {
        "A_原版": base,
        "B_原版加均量": base & ratio_13,
        "C_再加收盘质量": base & ratio_13 & candle_quality,
        "D_再加趋势斜率": base & ratio_13 & candle_quality & slope,
        "E_V2量价共振": big_volume & ratio_13 & candle_quality & (c > ma15)
                      & slope & momentum & no_overheat & valid,
    }
    for name, raw in rules.items():
        # Warmup hides outputs; it must not erase earlier FILTER state.
        g[name] = suppress_signals(raw, 9) & (np.arange(len(g)) >= common_warmup)

    prior20h = h.shift(1).rolling(20, min_periods=20).max()
    g["F_V3突破"] = g["E_V2量价共振"] & (c > prior20h)
    g["G_V3非突破"] = g["E_V2量价共振"] & (c <= prior20h)
    return g


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="日线CSV：date,symbol,open,high,low,close,volume")
    parser.add_argument("--outdir", default="research_results")
    parser.add_argument("--cost-bps", type=float, default=30.0,
                        help="假定双边总成本，基点；仅作敏感性情景，不代表实际费率")
    args = parser.parse_args()
    if args.cost_bps < 0:
        raise ValueError("--cost-bps 必须 >=0")
    df = pd.read_csv(args.input)
    df.columns = [c.lower().strip() for c in df.columns]
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError("缺少列：" + ", ".join(sorted(required-set(df.columns))))
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df.duplicated(["symbol", "date"]).any():
        raise ValueError("发现同一股票同一天的重复行，请先去重")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df["volume"] < 0).any():
        raise ValueError("发现非法价格或成交量")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    processed = []
    for sym, one in df.groupby("symbol", sort=False):
        processed.append(compute_one(one))
    g_all = pd.concat(processed, ignore_index=True)
    signal_cols = [c for c in g_all.columns if c.startswith(("A_", "B_", "C_", "D_", "E_", "F_", "G_"))]
    signal_rows = []
    event_rows = []
    for sym, g in g_all.groupby("symbol", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        op = g["open"].to_numpy()
        for strategy in signal_cols:
            indices = np.flatnonzero(g[strategy].fillna(False).to_numpy())
            for t in indices:
                signal_rows.append({"date": g.loc[t,"date"], "symbol":sym,
                                    "strategy":strategy, "pioneer":g.loc[t,"先锋"],
                                    "mid":g.loc[t,"中期"], "close":g.loc[t,"close"]})
                for horizon in HORIZONS:
                    entry, exit_t = t+1, t+1+horizon
                    if exit_t >= n:
                        continue
                    if op[entry] <= 0 or op[exit_t] <= 0:
                        continue
                    gross = op[exit_t] / op[entry] - 1.0
                    event_rows.append({
                        "symbol":sym,"strategy":strategy,"signal_date":g.loc[t,"date"],
                        "buy_date":g.loc[entry,"date"],"sell_date":g.loc[exit_t,"date"],
                        "holding_sessions":horizon,"gross_return":gross,
                        "net_return_fixed_cost":gross - args.cost_bps/10000.0,
                    })

    pd.DataFrame(signal_rows, columns=["date","symbol","strategy","pioneer","mid","close"]).to_csv(
        outdir/"signals.csv", index=False, encoding="utf-8-sig")
    events = pd.DataFrame(event_rows, columns=["symbol","strategy","signal_date","buy_date","sell_date",
                                              "holding_sessions","gross_return","net_return_fixed_cost"])
    events.to_csv(outdir/"events.csv", index=False, encoding="utf-8-sig")
    if len(events):
        group = events.groupby(["strategy","holding_sessions"])
        summary = group.agg(events=("gross_return", "size"),
                            gross_mean=("gross_return", "mean"),
                            gross_median=("gross_return", "median"),
                            net_mean=("net_return_fixed_cost", "mean"),
                            positive_rate=("net_return_fixed_cost", lambda x: float((x>0).mean()))).reset_index()
        summary.to_csv(outdir/"summary.csv", index=False, encoding="utf-8-sig")
        print(summary.to_string(index=False))
    else:
        pd.DataFrame(columns=["strategy","holding_sessions","events","gross_mean","gross_median",
                              "net_mean","positive_rate"]).to_csv(outdir/"summary.csv",index=False,encoding="utf-8-sig")
        print("没有可完整观察到持有期的信号；请检查数据历史长度与筛选强度")
    print("\n输出路径：", outdir.resolve())
    print("注意：这不是完整组合回测，未处理无法成交、停牌、动态费率、指数超额收益或资金占用。")


if __name__ == "__main__":
    main()
