import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('turning_point_scan', Path(__file__).resolve().parents[2] / 'scripts/scan_turning_points.py')
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


def bar(high, close):
    return pd.DataFrame({'dt': pd.date_range('2026-04-07', periods=2),
                         'open': [100., 100.], 'high': [100., high],
                         'low': [100., 99.], 'close': [100., close]})


@pytest.mark.parametrize('high,close,expected', [
    (107.2, 102., False), (107.21, 102., True),
    (108., 101., False), (108., 101.01, True),
    (108., 104.8, False), (108., 104.79, True),
    (107.19, 102., False), (108., 100.99, False),
    (108., 104.81, False),
])
def test_user_formula_strict_boundaries(high, close, expected):
    frame = scan.features(bar(high, close))
    assert not frame.signal.iloc[0]
    assert bool(frame.signal.iloc[1]) is expected


def test_invalid_price_record_must_not_create_a_signal():
    frame = bar(108, 102)
    frame.loc[0, 'close'] = 0
    with pytest.raises(ValueError, match='invalid OHLC'):
        scan.features(frame)


def test_no_lookahead_even_if_future_has_extreme_prices():
    close = np.r_[np.linspace(10, 18, 70), np.linspace(18, 11, 25), np.linspace(11, 12, 8)]
    frame = pd.DataFrame({'dt': pd.date_range('2026-01-01', periods=len(close)), 'open': close - .05, 'high': close + .3, 'low': close - .3, 'close': close})
    index = len(frame) - 1
    base = scan.features(frame)
    future = pd.DataFrame({'dt': [frame.dt.iloc[-1] + pd.Timedelta(days=1)], 'open': [500.], 'high': [1000.], 'low': [400.], 'close': [900.]})
    extended = scan.features(pd.concat([frame, future], ignore_index=True))
    assert scan.describe_shape(base, index) == scan.describe_shape(extended, index)
    pd.testing.assert_series_equal(base.signal, extended.signal.iloc[:len(base)])


def test_signal_can_be_present_without_turning_shape():
    work = scan.features(bar(108, 102))
    assert work.signal.iloc[-1]
    assert not scan.describe_shape(work, 1)['turn_confirmed']


def test_reject_inverted_close_thresholds():
    with pytest.raises(ValueError):
        scan.features(bar(108, 102), a2=1, a3=4.8)
