# -*- coding: utf-8 -*-
"""复核：用朴素循环逐日重算 600938 / 002424 / 600066 的公式指标，与向量化结果对比；并输出信号月度分布。"""
import os
import datetime
import numpy as np
import pandas as pd

TDXDIR = r"D:\software\tongxinda"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(PROJECT_ROOT, "outputs", "support-cross-2026-10-29", "支撑价上穿_近一年股票名单_20261029.csv")


def read_day(path):
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 32
    arr = np.frombuffer(raw[: n * 32], dtype=np.dtype([
        ("date", "<i4"), ("open", "<i4"), ("high", "<i4"),
        ("low", "<i4"), ("close", "<i4"), ("amount", "<f4"),
        ("vol", "<i4"), ("res", "<i4"),
    ]))
    return (arr["date"].astype(np.int64),
            arr["open"].astype(np.float64) / 100.0,
            arr["high"].astype(np.float64) / 100.0,
            arr["low"].astype(np.float64) / 100.0,
            arr["close"].astype(np.float64) / 100.0)


def manual_check(code):
    sym = code.split(".")[0]
    exch = "sh" if code.endswith(".SH") else "sz"
    p = os.path.join(TDXDIR, "vipdoc", exch, "lday", f"{exch}{sym}.day")
    dates, o, h, l, c = read_day(p)
    n = len(dates)
    x2b = np.full(n, np.nan); x1b = np.full(n, np.nan)
    support = np.full(n, np.nan)
    for i in range(n):
        if i >= 29:
            x2b[i] = c[i - 29:i + 1].min()
            x1b[i] = c[i - 29:i + 1].max()
        if i >= 18:
            support[i] = l[i - 18:i + 1].min()
    ema30 = np.full(n, np.nan); ema6 = np.full(n, np.nan)
    ema30[0] = c[0]; ema6[0] = c[0]
    for i in range(1, n):
        ema30[i] = (2 * c[i] + 29 * ema30[i - 1]) / 31.0
        ema6[i] = (2 * c[i] + 5 * ema6[i - 1]) / 7.0
    dist1 = (ema30 - support) / (ema30 + support) * 200.0
    dist2 = (x1b - x2b) / (x1b + x2b) * 200.0
    up = np.zeros(n, dtype=bool)
    up[1:] = (support[:-1] <= x2b[:-1]) & (x2b[1:] < support[1:])
    sig = up & (dist1 < 11.2) & (dist2 < 25.0) & (dates >= 20251029) & (dates <= 20261028)
    hit = np.where(sig)[0]
    return dates, c, x2b, x1b, support, ema30, ema6, dist1, dist2, hit


def main():
    # 1) 抽查对比（取每只股票最后一次命中日）
    df = pd.read_csv(OUT_CSV, dtype={"股票代码": str})
    for code in ["600938.SH", "002424.SZ", "600066.SH", "000001.SZ"]:
        dates, c, x2b, x1b, support, ema30, ema6, d1, d2, hit = manual_check(code)
        if len(hit) == 0:
            print(f"{code}: 手动循环无命中（或不在窗口）")
            continue
        i = hit[-1]
        d = datetime.datetime.strptime(str(int(dates[i])), "%Y%m%d").date().isoformat()
        # 查向量化输出中的对应行
        sub = df[(df["股票代码"] == code) & (df["信号日期"] == d)]
        if sub.empty:
            print(f"{code}: {d} 手动命中但向量化输出中无此行 -> 不一致!")
            continue
        r = sub.iloc[0]
        tol = 1e-3
        diffs = {
            "收盘": abs(r["收盘价"] - c[i]),
            "X2B": abs(r["X2B_30日最低收盘"] - x2b[i]),
            "X1B": abs(r["X1B_30日最高收盘"] - x1b[i]),
            "支撑价": abs(r["支撑价_19日最低低点"] - support[i]),
            "EMA30": abs(r["EMA30_清仓"] - ema30[i]),
            "距离1": abs(r["距离1_%"] - d1[i]),
            "距离2": abs(r["距离2_%"] - d2[i]),
        }
        ok = all(v < tol for v in diffs.values())
        print(f"{code} {d}: 手动 vs 向量化 最大偏差={max(diffs.values()):.5f} -> {'一致' if ok else '不一致!'}")
        # 同时验证上穿两个子条件
        print(f"   昨天: 支撑价(19日最低低点)={support[i-1]:.3f} <= X2B={x2b[i-1]:.3f}  {support[i-1] <= x2b[i-1]}")
        print(f"   今天: X2B={x2b[i]:.3f} < 支撑价={support[i]:.3f}  {x2b[i] < support[i]}")
        print(f"   距离1={d1[i]:.3f} (<11.2)  距离2={d2[i]:.3f} (<25)")

    # 2) 月度分布
    df["月"] = pd.to_datetime(df["信号日期"]).dt.strftime("%Y-%m")
    print("\n信号月度分布:")
    print(df.groupby("月").size().sort_index().to_string())
    print("\n总信号:", len(df), " 去重股票:", df["股票代码"].nunique())


if __name__ == "__main__":
    main()
