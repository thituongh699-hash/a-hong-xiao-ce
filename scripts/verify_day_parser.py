# -*- coding: utf-8 -*-
"""验证通达信 .day 日线文件解析（32字节定长记录）。"""
import numpy as np
import os

TDXDIR = r"D:\software\tongxinda"


def read_day_file(path):
    """解析通达信 .day 二进制文件。返回 (dates, open, high, low, close)。"""
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 32
    arr = np.frombuffer(raw[: n * 32], dtype=np.dtype([
        ("date", "<i4"), ("open", "<i4"), ("high", "<i4"),
        ("low", "<i4"), ("close", "<i4"), ("amount", "<f4"),
        ("vol", "<i4"), ("res", "<i4"),
    ]))
    dates = arr["date"].astype(np.int64)
    open_ = arr["open"].astype(np.float64) / 1000.0
    high = arr["high"].astype(np.float64) / 1000.0
    low = arr["low"].astype(np.float64) / 1000.0
    close = arr["close"].astype(np.float64) / 1000.0
    return dates, open_, high, low, close


if __name__ == "__main__":
    for rel in [os.path.join("sh", "lday", "sh600000.day"),
                os.path.join("sz", "lday", "sz000001.day")]:
        p = os.path.join(TDXDIR, "vipdoc", rel)
        if not os.path.exists(p):
            print("MISSING", p)
            continue
        dates, o, h, l, c = read_day_file(p)
        print("=" * 60)
        print(rel, "records:", len(dates),
              "date range:", int(dates[0]), "->", int(dates[-1]))
        # 转换为 YYYY-MM-DD
        import datetime
        def d2s(d):
            return datetime.datetime.strptime(str(int(d)), "%Y%m%d").date().isoformat()
        for i in range(max(0, len(dates) - 5), len(dates)):
            print(d2s(dates[i]), "O=%.3f H=%.3f L=%.3f C=%.3f" % (o[i], h[i], l[i], c[i]))
