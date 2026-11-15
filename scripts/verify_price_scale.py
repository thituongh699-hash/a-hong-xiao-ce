# -*- coding: utf-8 -*-
"""验证价格比例：用贵州茅台确定 .day 价格存储倍数。"""
import numpy as np
import os
import datetime

p = r"D:\software\tongxinda\vipdoc\sh\lday\sh600519.day"
with open(p, "rb") as f:
    raw = f.read()
n = len(raw) // 32
arr = np.frombuffer(raw[:n * 32], dtype=np.dtype([
    ("date", "<i4"), ("open", "<i4"), ("high", "<i4"),
    ("low", "<i4"), ("close", "<i4"), ("amount", "<f4"),
    ("vol", "<i4"), ("res", "<i4"),
]))
print("records:", n, "range:", int(arr["date"][0]), "->", int(arr["date"][-1]))

def d2s(d):
    return datetime.datetime.strptime(str(int(d)), "%Y%m%d").date().isoformat()

for i in range(n - 5, n):
    print(d2s(arr["date"][i]),
          "raw_close_int=", int(arr["close"][i]),
          "vol=", int(arr["vol"][i]),
          "amount=", float(arr["amount"][i]),
          "close/1000=", round(float(arr["close"][i]) / 1000.0, 3),
          "close/100=", round(float(arr["close"][i]) / 100.0, 2))

# 用缓存 CSV（akshare/tdx 网络源）交叉验证 600519 某日收盘
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = os.path.join(PROJECT_ROOT, "data", "history", "cache", "tdx_raw_v2_600519_SH_1min.csv")
if os.path.exists(cache):
    import pandas as pd
    df = pd.read_csv(cache)
    print("cache cols:", list(df.columns))
    print(df.tail(3))
