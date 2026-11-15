"""Conservative A-share intraday cache coverage helpers."""
from __future__ import annotations

from datetime import time

import pandas as pd


_SESSIONS = ((time(9, 30), time(11, 30)), (time(13, 0), time(15, 0)))


def _naive_timestamp(value) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp


def expected_a_share_bar_times(start_time, end_time, interval_minutes: int) -> pd.DatetimeIndex:
    """Build expected bar-end timestamps for mainland continuous sessions.

    Weekends are excluded.  Exchange holidays and suspensions are deliberately
    not inferred: without calendar or suspension evidence they remain cache
    misses rather than being treated as complete data.
    """
    start = _naive_timestamp(start_time)
    end = _naive_timestamp(end_time)
    try:
        step = int(interval_minutes)
    except (TypeError, ValueError):
        step = 0
    if pd.isna(start) or pd.isna(end) or start > end or step <= 0:
        return pd.DatetimeIndex([])

    expected: list[pd.Timestamp] = []
    for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        if day.weekday() >= 5:
            continue
        for session_start, session_end in _SESSIONS:
            left = day + pd.Timedelta(
                hours=session_start.hour, minutes=session_start.minute + step
            )
            right = day + pd.Timedelta(hours=session_end.hour, minutes=session_end.minute)
            if left > right:
                continue
            slots = pd.date_range(left, right, freq=f"{step}min")
            expected.extend(slot for slot in slots if start <= slot <= end)
    return pd.DatetimeIndex(expected)


def minute_cache_covers_range(
    datetimes,
    start_time,
    end_time,
    *,
    interval_minutes: int,
) -> bool:
    """Return true only when every expected in-session bar is present."""
    expected = expected_a_share_bar_times(start_time, end_time, interval_minutes)
    if expected.empty:
        return False
    actual = pd.DatetimeIndex([_naive_timestamp(value) for value in datetimes]).dropna()
    if actual.empty:
        return False
    return expected.difference(actual.unique()).empty


def merge_market_data_frames(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge cache frames so later, newly fetched rows replace old revisions."""
    frames = [frame for frame in (old, new) if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged["dt"] = pd.to_datetime(merged["dt"], errors="coerce")
    merged = merged.dropna(subset=["dt"])
    return (
        merged.drop_duplicates(subset=["dt"], keep="last")
        .sort_values("dt", kind="stable")
        .reset_index(drop=True)
    )
