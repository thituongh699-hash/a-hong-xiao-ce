import numpy as np
import pandas as pd
import pytest

from src.strategies.low_reversal_factor import features, parameter_grid, signal_mask
from src.utils.price_adjustment import _factor_frame
from src.utils.runtime_params import parameter_overrides
from scripts.research_low_reversal import (
    CAPITAL, FIXED_HOLD, STUDY_COSTS, Evaluator, execution_bars, period_cohorts, summarize,
)


def raw_frame(count=180):
    days = pd.bdate_range("2025-01-01", periods=count)
    c = 10 + np.sin(np.arange(count)/10)
    return pd.DataFrame(dict(code="000001.SZ", dt=days, open=c, high=c+.2,
                             low=c-.2, close=c, vol=100000., amount=1000000.))


def factor_frame(raw, events=()):
    return _factor_frame([dict(dt=d, factor=f) for d,f in events], raw.dt.max())


@pytest.mark.parametrize("high,close,expected", [
    (10.72,10.20,False), (10.73,10.48,False), (10.73,10.10,False),
    (10.73,10.11,True), (10.71,10.20,False), (10.73,10.47,True),
])
def test_formula_strict_cents_boundaries(high, close, expected):
    raw = raw_frame()
    raw.loc[178,["open","high","low","close"]] = [10.,10.2,9.8,10.]
    raw.loc[179,["open","high","low","close"]] = [10.,high,9.9,close]
    work = features(raw,factor_frame(raw))
    assert bool(signal_mask(work,dict(low_mode="none")).iloc[-1]) is expected


def test_features_prefix_invariant_with_later_factor_and_extreme_peak():
    raw = raw_frame()
    raw.loc[155:,["open","high","low","close"]] /= 2
    raw.loc[170,"high"] = 500.
    factors = factor_frame(raw,[(str(raw.dt.iloc[155].date()),2.)])
    full = features(raw,factors)
    prefix = features(raw.iloc[:150],factors)
    cols = ["high_pct","close_pct","fade_pct","range_pos20","range_pos60", "drawdown60_pct", "peak_age60", "ma5_turn"]
    pd.testing.assert_frame_equal(full.loc[:149,cols],prefix[cols])
    pd.testing.assert_series_equal(signal_mask(full).iloc[:150],signal_mask(prefix))


def test_factor_price_continuity_and_execution_raw():
    raw = raw_frame()
    raw.loc[149,["open","high","low","close"]] = [10.,10.2,9.8,10.]
    raw.loc[150,["open","high","low","close"]] = [5.,5.4,4.95,5.1]
    factors = factor_frame(raw,[(str(raw.dt.iloc[150].date()),2.)])
    work = features(raw,factors)
    assert work.loc[150,"high_pct"] == pytest.approx(8.)
    assert work.loc[150,"close_pct"] == pytest.approx(2.)
    bars = execution_bars(work)
    bar = bars[str(raw.dt.iloc[150].date())]
    assert bar["close"] == 5.1
    assert bar["factor_changed"] and not bar["price_reference_valid"]


def test_no_raw_fallback_on_unverified_factors():
    raw = raw_frame()
    with pytest.raises(Exception,match="not verified"):
        features(raw,pd.DataFrame(dict(dt=[raw.dt.iloc[0]],factor=[1.])))


def test_grid_and_low_requirement():
    grid = parameter_grid()
    assert len(grid)==459
    assert sum(p["low_mode"]!="none" for p in grid)==432
    with pytest.raises(ValueError):
        signal_mask(features(raw_frame(),factor_frame(raw_frame())),dict(a3=5.,a2=4.8))


def trial_fixture():
    raw = raw_frame(45)
    raw[["open","high","low","close"]] = [10.,10.2,9.8,10.]
    raw["factor"] = 1.
    # High crosses 3%, close does not; close crosses on third observation day.
    raw.loc[1,["high","close"]] = [10.5,10.2]
    raw.loc[2,["high","close"]] = [10.3,10.]
    raw.loc[3,["high","close"]] = [10.6,10.5]
    bars = execution_bars(raw)
    for b in bars.values(): b["history_sessions"]=120
    calendar = list(bars)
    panel = pd.DataFrame([dict(code="000001.SZ",name="普通公司",date=calendar[0],period="train")])
    cohorts = dict(train=dict(calendar=calendar),test=dict(calendar=calendar))
    benchmark = {d:dict(open=10.,close=10.) for d in calendar}
    return Evaluator(panel,{"000001.SZ":bars},cohorts,benchmark),panel.iloc[0].to_dict(),calendar,bars


def test_four_day_high_vs_close_and_next_open_cash_denominator():
    ev,setup,calendar,bars = trial_fixture()
    with parameter_overrides(STUDY_COSTS):
        intra=ev.outcome(setup,"intraday",FIXED_HOLD,5)
        close=ev.outcome(setup,"close",FIXED_HOLD,5)
        following=ev.outcome(setup,"next_open",FIXED_HOLD,5)
    assert intra["entry_date"]==calendar[1]
    assert close["entry_date"]==calendar[3]
    assert following["entry_date"]==calendar[4]
    assert intra["entry_price"]==pytest.approx(10.31*1.001)
    assert intra["net"]==pytest.approx(intra["realized_pnl"]/CAPITAL*100)
    assert intra["exit_date"]==calendar[6]


def test_corporate_action_censored_not_used_to_reject_prior_entry():
    ev,setup,calendar,bars = trial_fixture()
    bars[calendar[2]]["price_reference_valid"]=False
    with parameter_overrides(STUDY_COSTS):
        row=ev.outcome(setup,"intraday",FIXED_HOLD,5)
    assert row["entered"]
    assert row["entry_date"]==calendar[1]
    assert row["net"] is None
    m=summarize([dict(code=setup["code"],signal_date=setup["date"],**row)])
    assert m["censored"]==1 and m["trades"]==0


def test_no_entry_contributes_zero():
    ev,setup,calendar,bars = trial_fixture()
    for day in calendar[1:5]: bars[day].update(high=10.2,close=10.,high_gt3=False,close_gt3=False)
    with parameter_overrides(STUDY_COSTS): row=ev.outcome(setup,"intraday",FIXED_HOLD,5)
    m=summarize([dict(code=setup["code"],signal_date=setup["date"],**row)])
    assert m["signals"]==m["labels"]==m["no_entry"]==1
    assert m["mean_signal_net_pct"]==0.


def test_split_purge_never_crosses_train_end():
    calendar = pd.bdate_range("2025-09-01","2026-09-30").strftime("%Y-%m-%d").tolist()
    cohorts=period_cohorts(calendar)
    for c in cohorts.values():
        assert c["calendar"].index(c["end"])-c["calendar"].index(c["last_signal"])==30


def test_terminal_position_censored_and_full_period_cache_isolated():
    ev,setup,calendar,bars=trial_fixture()
    ev.cohorts["train"]={"calendar":calendar[:5]}
    ev.cohorts["full"]={"calendar":calendar}
    with parameter_overrides(STUDY_COSTS):
        short=ev.outcome(setup,"intraday",FIXED_HOLD,10)
        full=ev.outcome({**setup,"period":"full"},"intraday",FIXED_HOLD,10)
        cached=ev.outcome(setup,"intraday",FIXED_HOLD,10)
    assert short["status"]=="open_at_data_end" and short["net"] is None
    assert full["status"]=="closed" and full["net"] is not None
    assert cached["net"] is None and len(ev.entries)==2
    m=summarize([dict(code=setup["code"],signal_date=setup["date"],**short)])
    assert m["open_at_end"]==m["censored"]==1 and m["labels"]==0
