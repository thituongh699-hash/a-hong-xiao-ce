"""Completed, end-stamped A-share strategy bars derived from one base feed.

Minute inputs use exchange/end timestamps: (09:30, 11:30] and
(13:00, 15:00]. A partial or gapped window is never delivered as complete.
Daily input bars are handled directly by the engine, not by this helper.
"""
import re

import pandas as pd


def aggregate_completed_bars(frame, timeframe, source_interval="1min"):
    if frame.empty:
        return frame.copy()
    source = re.fullmatch(r"(\d+)min", str(source_interval))
    target = re.fullmatch(r"(\d+)min", str(timeframe))
    if source is None or (target is None and timeframe != "D"):
        raise ValueError("completed aggregation requires minute source and minute/day target")
    source_minutes = int(source.group(1))
    target_minutes = int(target.group(1)) if target else 240
    if source_minutes <= 0 or target_minutes < source_minutes or target_minutes % source_minutes:
        raise ValueError("target timeframe must be a positive multiple of the source timeframe")
    work = frame.copy()
    work["dt"] = pd.to_datetime(work["dt"], format="mixed", errors="raise")
    work = work.sort_values("dt", kind="stable")
    if work["dt"].duplicated().any():
        raise ValueError("completed aggregation rejects duplicate base timestamps")
    if "vol" not in work and "volume" in work:
        work["vol"] = work["volume"]
    if "amount" not in work and "turnover" in work:
        work["amount"] = work["turnover"]
    rows = []
    step = pd.Timedelta(minutes=source_minutes)
    for day, daily in work.groupby(work["dt"].dt.normalize(), sort=True):
        sessions = [(day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=11, minutes=30)),
                    (day + pd.Timedelta(hours=13), day + pd.Timedelta(hours=15))]
        if timeframe == "D":
            expected = sessions[0][0] + pd.timedelta_range(step, periods=120 // source_minutes, freq=step)
            expected = expected.append(sessions[1][0] + pd.timedelta_range(step, periods=120 // source_minutes, freq=step))
            selected = daily[daily["dt"].isin(expected)]
            if len(selected) == len(expected) and set(selected["dt"]) == set(expected):
                rows.append(_aggregate(selected, sessions[-1][1], sessions[0][0]))
            continue
        for start, end in sessions:
            selected = daily[(daily["dt"] > start) & (daily["dt"] <= end)]
            if selected.empty:
                continue
            # Session-relative windows avoid an hour bar spanning the lunch break.
            numbers = ((selected["dt"] - start).dt.total_seconds() / 60)
            buckets = ((numbers - 1e-9) // target_minutes).astype(int)
            for bucket, group in selected.groupby(buckets, sort=True):
                close_at = start + pd.Timedelta(minutes=(int(bucket) + 1) * target_minutes)
                if close_at > end:
                    continue
                expected = pd.date_range(close_at - pd.Timedelta(minutes=target_minutes) + step,
                                         close_at, freq=step)
                if len(group) != len(expected) or set(group["dt"]) != set(expected):
                    continue
                rows.append(_aggregate(group, close_at, close_at - pd.Timedelta(minutes=target_minutes)))
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=list(work.columns) + ["bar_start", "bar_end"])
    result.attrs.update(frame.attrs)
    result.attrs["bar_timestamp"] = "end"
    result.attrs["aggregation_policy"] = "completed_a_share_sessions"
    return result.sort_values("dt", kind="stable").reset_index(drop=True)


def _aggregate(group, close_at, start_at):
    result = group.iloc[-1].to_dict()
    result.update(dt=close_at, bar_start=start_at, bar_end=close_at,
                  open=float(group["open"].iloc[0]), high=float(group["high"].max()),
                  low=float(group["low"].min()), close=float(group["close"].iloc[-1]),
                  vol=float(group["vol"].sum()), amount=float(group["amount"].sum()))
    result["volume"] = result["vol"]
    result["turnover"] = result["amount"]
    return result
