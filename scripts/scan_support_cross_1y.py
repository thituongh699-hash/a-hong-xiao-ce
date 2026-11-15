# -*- coding: utf-8 -*-
"""
全市场扫描：支撑价上穿（近一年）—— a-hong-xiao-ce 项目

公式（通达信/同花顺语法）:
    X2B:=LLV(C,30);               30日最低收盘
    X1B:=HHV(C,30);               30日最高收盘
    支撑价:=LLV(LOW,19);          19日最低低点
    稳起:=EMA(CLOSE,6);
    清仓:=EMA(CLOSE,30);
    距离1:=(清仓-支撑价)/(清仓+支撑价)*2*100;
    距离2:=(X1B-X2B)/(X1B+X2B)*2*100;
    上穿:=REF(支撑价,1)<=REF(X2B,1) AND X2B<支撑价;
    选股: 上穿 AND 距离1<11.2 AND 距离2<25;

数据源: 本地通达信 vipdoc 日线 (.day 二进制, 32字节/记录, 价格×100, 不复权)
扫描范围: 2025-10-29 ~ 2026-10-28 (本地数据最新交易日)

输出: outputs/support-cross-2026-10-29/支撑价上穿_近一年股票名单_20261029.csv
"""
import os
import datetime
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TDXDIR = r"D:\software\tongxinda"
STOCK_LIST = os.path.join(PROJECT_ROOT, "data", "stock_list.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "support-cross-2026-10-29")
OUT_CSV = os.path.join(OUT_DIR, "支撑价上穿_近一年股票名单_20261029.csv")

WINDOW_START = 20251029  # 含
WINDOW_END = 20261028    # 含（本地数据最新日期）

N_LLV_C = 30    # X2B 窗口
N_HHV_C = 30    # X1B 窗口
N_LLV_L = 19    # 支撑价窗口
N_EMA6 = 6
N_EMA30 = 30
LIMIT_D1 = 11.2
LIMIT_D2 = 25.0


def read_day(path):
    """解析通达信 .day 日线文件。返回 (dates, open, high, low, close)。"""
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 32
    if n == 0:
        return None
    arr = np.frombuffer(raw[: n * 32], dtype=np.dtype([
        ("date", "<i4"), ("open", "<i4"), ("high", "<i4"),
        ("low", "<i4"), ("close", "<i4"), ("amount", "<f4"),
        ("vol", "<i4"), ("res", "<i4"),
    ]))
    dates = arr["date"].astype(np.int64)
    # 该通达信安装日线价格存储为 ×100
    open_ = arr["open"].astype(np.float64) / 100.0
    high = arr["high"].astype(np.float64) / 100.0
    low = arr["low"].astype(np.float64) / 100.0
    close = arr["close"].astype(np.float64) / 100.0
    return dates, open_, high, low, close


def day_path(code):
    sym = str(code).strip().upper()
    raw = sym.split(".")[0]
    if sym.endswith(".SH"):
        return os.path.join(TDXDIR, "vipdoc", "sh", "lday", f"sh{raw}.day")
    return os.path.join(TDXDIR, "vipdoc", "sz", "lday", f"sz{raw}.day")


def compute_signal(code, dates, open_, high, low, close):
    """逐日计算公式指标，返回窗口内命中行列表。"""
    s_close = pd.Series(close)
    s_low = pd.Series(low)

    x2b = s_close.rolling(N_LLV_C, min_periods=N_LLV_C).min().to_numpy()
    x1b = s_close.rolling(N_HHV_C, min_periods=N_HHV_C).max().to_numpy()
    support = s_low.rolling(N_LLV_L, min_periods=N_LLV_L).min().to_numpy()

    ema6 = s_close.ewm(alpha=2.0 / (N_EMA6 + 1), adjust=False).mean().to_numpy()
    ema30 = s_close.ewm(alpha=2.0 / (N_EMA30 + 1), adjust=False).mean().to_numpy()

    denom1 = ema30 + support
    denom2 = x1b + x2b
    with np.errstate(divide="ignore", invalid="ignore"):
        dist1 = np.where(np.abs(denom1) > 1e-9,
                         (ema30 - support) / denom1 * 2.0 * 100.0, np.nan)
        dist2 = np.where(np.abs(denom2) > 1e-9,
                         (x1b - x2b) / denom2 * 2.0 * 100.0, np.nan)

    n = len(dates)
    up_cross = np.zeros(n, dtype=bool)
    if n >= 2:
        up_cross[1:] = (support[:-1] <= x2b[:-1]) & (x2b[1:] < support[1:])

    signal = up_cross & (dist1 < LIMIT_D1) & (dist2 < LIMIT_D2)

    idx = np.where(signal & (dates >= WINDOW_START) & (dates <= WINDOW_END))[0]
    rows = []
    for i in idx:
        d = int(dates[i])
        rows.append({
            "信号日期": datetime.datetime.strptime(str(d), "%Y%m%d").date().isoformat(),
            "股票代码": code,
            "收盘价": round(float(close[i]), 3),
            "X2B_30日最低收盘": round(float(x2b[i]), 3),
            "X1B_30日最高收盘": round(float(x1b[i]), 3),
            "支撑价_19日最低低点": round(float(support[i]), 3),
            "EMA30_清仓": round(float(ema30[i]), 3),
            "EMA6_稳起": round(float(ema6[i]), 3),
            "距离1_%": round(float(dist1[i]), 3),
            "距离2_%": round(float(dist2[i]), 3),
        })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_list = pd.read_csv(STOCK_LIST, dtype={"code": str, "name": str})

    all_rows = []
    missing = []
    too_short = []
    scanned = 0

    for code, name in zip(df_list["code"], df_list["name"]):
        p = day_path(code)
        if not os.path.exists(p):
            missing.append(code)
            continue
        parsed = read_day(p)
        if parsed is None:
            missing.append(code)
            continue
        dates, o, h, l, c = parsed
        # 排除异常价（如全0或负数）
        valid = (l > 0) & (h > 0) & (c > 0)
        dates, o, h, l, c = dates[valid], o[valid], h[valid], l[valid], c[valid]
        if len(dates) < max(N_LLV_C, N_LLV_L) + 1:
            too_short.append(code)
            continue
        if dates[-1] < WINDOW_START:
            continue  # 数据未覆盖窗口
        scanned += 1
        rows = compute_signal(code, dates, o, h, l, c)
        for r in rows:
            r["股票名称"] = name
        all_rows.extend(rows)

    if not all_rows:
        print("未命中任何股票")
        return

    out = pd.DataFrame(all_rows)
    out = out.sort_values(["信号日期", "股票代码"], ascending=[False, True]).reset_index(drop=True)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"扫描股票数(覆盖窗口): {scanned}")
    print(f"缺文件/无法解析: {len(missing)}")
    print(f"K线过短(<31根): {len(too_short)}")
    print(f"命中信号总数: {len(out)}")
    print(f"命中股票数(去重): {out['股票代码'].nunique()}")
    print(f"输出: {OUT_CSV}")
    print("\n近5个最近信号:")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
