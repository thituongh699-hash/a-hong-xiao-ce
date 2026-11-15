import pandas as pd
import pytest

from scripts.backtest_turning_points import (
    COSTS, PROFILES, choose_exit, daily_bars, exit_levels, find_entry,
    first_tick_above_pct, in_scope, run_trial,
)
from src.utils.runtime_params import parameter_overrides


def fixture_bars():
    days = [f'2026-08-{d:02d}' for d in [3, 4, 5, 6, 7, 10, 11]]
    rows = [dict(code='000001.SZ', dt=pd.Timestamp(day), open=10., high=10.2,
                 low=9.8, close=10., vol=100000., amount=1000000.) for day in days]
    bars = daily_bars(pd.DataFrame(rows))
    for b in bars.values():
        b['history_sessions'] = 100
    return days, bars


def test_fractional_threshold_and_exact_boundary():
    assert first_tick_above_pct(10, 3) == 10.31
    assert first_tick_above_pct(10, 4.5) == 10.46
    assert first_tick_above_pct(7.13, 4.5) == 7.46
    days, bars = fixture_bars()
    bars[days[1]].update(high=10.45, close=10.45)
    bars[days[2]].update(high=10.46)
    with parameter_overrides(COSTS):
        found = find_entry('000001.SZ', days[0], 'intraday', bars, days, threshold=4.5)
    assert found['entry_date'] == days[2]
    assert found['entry_fill'] == pytest.approx(10.46 * 1.001)


def test_four_market_days_exclude_signal_and_missing_does_not_extend():
    days, bars = fixture_bars()
    bars[days[0]].update(high=10.8, close=10.5)
    bars.pop(days[1])
    bars[days[5]].update(high=10.8, close=10.5)
    found = find_entry('000001.SZ', days[0], 'intraday', bars, days)
    assert found['status'] == 'no_entry' and found['observed_days'] == 4
    assert found['attempts'][0]['reason'] == 'missing_stock_bar'


def test_fourth_day_allowed_fifth_day_not_and_censored_window():
    days, bars = fixture_bars()
    bars[days[4]].update(high=10.6, close=10.5)
    assert find_entry('000001.SZ', days[0], 'close', bars, days)['entry_date'] == days[4]
    assert find_entry('000001.SZ', days[-2], 'close', bars, days)['status'] == 'observation_incomplete'


def test_close_up_rejected_then_keeps_observing():
    days, bars = fixture_bars()
    bars[days[1]].update(high=11., close=11., is_limit_up=True)
    bars[days[2]].update(high=10.6, close=10.5)
    found = find_entry('000001.SZ', days[0], 'close', bars, days)
    assert found['entry_date'] == days[2]
    assert found['attempts'][0]['reason'] == 'close_limit_up'


def test_opening_gaps_before_ambiguous_intraday_ordering():
    assert choose_exit(dict(open=12., high=12.5, low=8.), 9., 11.) == ('sell_limit',11.,'gap_take_profit',False)
    assert choose_exit(dict(open=10., high=12., low=8.), 9., 11.) == ('sell_stop',9.,'stop_loss_or_trailing',True)


def test_trailing_stop_depends_only_on_prior_peak():
    p = next(x for x in PROFILES if x['id']=='trailing')
    assert exit_levels(p, 10., 10.5)[0] == 9.66
    assert exit_levels(p, 10., 11.5)[0] == 10.92


def test_entry_day_stop_not_sold_t_plus_one_and_final_position_censored():
    days, bars = fixture_bars()
    bars[days[1]].update(high=10.6, close=10.5, low=8.5)
    with parameter_overrides(COSTS):
        found = find_entry('000001.SZ', days[0], 'intraday', bars, days)
        trade = run_trial('000001.SZ', 'intraday', found, PROFILES[0], bars, days[:2])
    assert trade['status'] == 'open_at_data_end'
    assert trade['net_return_pct'] is None and trade['exit_date'] is None


@pytest.mark.parametrize('code,name,expected', [
    ('600001.SH','普通公司',True), ('001234.SZ','普通公司',True),
    ('300001.SZ','普通公司',False),('688001.SH','普通公司',False),
    ('600001.SH','*ST公司',False),('600001.SH','退市公司',False),('830001.BJ','普通公司',False),
])
def test_universe(code,name,expected):
    assert in_scope(code,name) is expected
