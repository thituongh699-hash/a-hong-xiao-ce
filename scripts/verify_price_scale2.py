# -*- coding: utf-8 -*-
"""验证价格比例在历史段一致：茅台首日 + 浦发 2026-09-29 与缓存CSV交叉核对。"""
import numpy as np

def read(p):
    with open(p, "rb") as f:
        raw = f.read()
    n = len(raw) // 32
    arr = np.frombuffer(raw[:n * 32], dtype=np.dtype([
        ("date", "<i4"), ("open", "<i4"), ("high", "<i4"),
        ("low", "<i4"), ("close", "<i4"), ("amount", "<f4"),
        ("vol", "<i4"), ("res", "<i4"),
    ]))
    return arr

# 茅台首日
arr = read(r"D:\software\tongxinda\vipdoc\sh\lday\sh600519.day")
i = 0
print("茅台首日", int(arr["date"][i]), "close_int=", int(arr["close"][i]),
      "-> /100 =", round(int(arr["close"][i]) / 100.0, 2))

# 浦发 2026-09-29
arr = read(r"D:\software\tongxinda\vipdoc\sh\lday\sh600000.day")
idx = np.where(arr["date"] == 20260929)[0]
if len(idx):
    i = idx[0]
    print("浦发20260929 close_int=", int(arr["close"][i]),
          "-> /100 =", round(int(arr["close"][i]) / 100.0, 2),
          "(缓存CSV为9.18)  high_int=", int(arr["high"][i]),
          "low_int=", int(arr["low"][i]))
else:
    print("浦发 20260929 不在文件中")
