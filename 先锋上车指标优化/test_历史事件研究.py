"""Tests protect calendar timing, corporate accounting and causal formula rules."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("historical_runner", HERE / "运行历史事件研究.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
signal = runner.SIGNAL


def prices(n=260):
    rng = np.random.default_rng(73)
    close = 20 * np.exp(np.cumsum(rng.normal(.002, .025, n)))
    op = close * np.exp(rng.normal(-.002, .006, n))
    frame = pd.DataFrame(dict(date=pd.bdate_range("2020-01-02", periods=n).strftime("%Y-%m-%d"),
                             open=op, high=np.maximum(op, close) * 1.006,
                             low=np.minimum(op, close) * .994, close=close,
                             volume=rng.lognormal(10, .7, n)))
    c = frame.high.iloc[-21:-1].max() * 1.01
    frame.loc[n - 1, ["open", "high", "low", "close", "volume"]] = [c * .98, c * 1.002, c * .979, c, frame.volume.max() * 5]
    return frame


class SignalTests(unittest.TestCase):
    def test_filter_cooldown_is_nine_following_stock_bars(self):
        out = signal.suppress_signals(pd.Series([True] * 24), 9)
        self.assertEqual(np.flatnonzero(out).tolist(), [0, 10, 20])

    def test_warmup_hides_output_without_erasing_filter_state(self):
        df = prices()
        reference = signal.compute_one(df, common_warmup=0)
        masked = signal.compute_one(df, common_warmup=120)
        pd.testing.assert_frame_equal(reference.loc[120:, runner.STRATEGIES], masked.loc[120:, runner.STRATEGIES])

    def test_future_changes_cannot_rewrite_past_signals(self):
        df = prices()
        before = signal.compute_one(df, common_warmup=120)
        df.loc[200:, ["open", "high", "low", "close", "volume"]] *= 10
        after = signal.compute_one(df, common_warmup=120)
        pd.testing.assert_frame_equal(before.loc[:199, runner.STRATEGIES], after.loc[:199, runner.STRATEGIES])

    def test_v3_is_exhaustive_disjoint_partition_of_v2(self):
        g = signal.compute_one(prices(), common_warmup=120)
        self.assertFalse((g[runner.CORE[2]] & g[runner.CORE[3]]).any())
        self.assertTrue((g[runner.CORE[1]] == (g[runner.CORE[2]] | g[runner.CORE[3]])).all())
        self.assertFalse(g.loc[:119, runner.STRATEGIES].any().any())
        self.assertTrue(g[runner.CORE[1]].any())

    def test_qfq_and_cumulative_units_agree_with_scaled_floor(self):
        df = prices()
        df.loc[140:, ["open", "high", "low", "close"]] /= 2
        factor = np.where(np.arange(len(df)) < 140, 1., 2.)
        hfq = df.copy()
        qfq = df.copy()
        for col in ("open", "high", "low", "close"):
            hfq[col] *= factor
            qfq[col] *= factor / 2
        hfq["price_floor"] = factor * .01
        qfq["price_floor"] = factor / 2 * .01
        a = signal.compute_one(hfq, common_warmup=120)
        b = signal.compute_one(qfq, common_warmup=120)
        pd.testing.assert_frame_equal(a[runner.STRATEGIES], b[runner.STRATEGIES])
        np.testing.assert_allclose(a["先锋"], b["先锋"], equal_nan=True)

    def test_v2_requires_strictly_above_ma15(self):
        df = prices()
        # Equality at last bar is representable to floating precision. Inspect
        # neighboring below/above cases while keeping all other last-bar rules.
        t = len(df) - 1
        ma = df.loc[t - 14:t - 1, "close"].mean()
        df.loc[t, "close"] = ma - .000001
        df.loc[t, "open"] = ma - .1
        df.loc[t, "low"] = ma - .15
        df.loc[t, "high"] = ma + .01
        df.loc[t, "volume"] = df.volume.max() * 5
        self.assertFalse(signal.compute_one(df).loc[t, runner.CORE[1]])


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.cal = np.array(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
        self.position = {d: i for i, d in enumerate(self.cal)}
        self.raw = pd.DataFrame(dict(open=[10, 11, 12, 13, 14], high=[11, 12, 13, 14, 15],
                                     low=[9, 10, 11, 12, 13]), index=self.cal)
        self.bench = pd.DataFrame(dict(csi300=[100, 102, 104, 106, 108],
                                       csi1000=[100, 101, 103, 104, 106]), index=self.cal)

    def outcome(self, h=1, dates=()):
        return runner.scheduled_outcome(self.raw, self.cal, self.position, self.cal[0], h, np.array(dates), self.bench)

    def test_next_market_open_to_open_with_t_plus_one_minimum(self):
        o = self.outcome()
        self.assertEqual((o["buy_date"], o["sell_date"]), ("2024-01-03", "2024-01-04"))
        self.assertAlmostEqual(o["gross_return"], 12 / 11 - 1)
        self.assertAlmostEqual(o["csi300_return"], 104 / 102 - 1)

    def test_missing_next_day_is_never_skipped(self):
        self.raw = self.raw.drop("2024-01-03")
        self.assertEqual(self.outcome()["status"], "entry_missing_or_suspended")

    def test_missing_scheduled_exit_is_never_delayed(self):
        self.raw = self.raw.drop("2024-01-04")
        self.assertEqual(self.outcome()["status"], "exit_missing_or_suspended_or_delisted")

    def test_corporate_change_while_holding_excluded(self):
        self.assertEqual(self.outcome(dates=["2024-01-04"])["status"], "corporate_action_excluded")
        self.assertEqual(self.outcome(dates=["2024-01-03"])["status"], "observable_raw_endpoints")

    def test_no_future_exit_is_censored(self):
        self.assertEqual(self.outcome(h=5)["status"], "right_censored")

    def test_invalid_ohlc_and_duplicates_fail_closed(self):
        df = prices()
        df.loc[0, "high"] = 0.1
        with self.assertRaises(ValueError):
            runner.validate_bars(df)
        df = prices()
        with self.assertRaises(ValueError):
            runner.validate_bars(pd.concat([df, df.tail(1)], ignore_index=True))

    def test_terminal_history_is_appended_after_overlap_check(self):
        df = prices(260)
        primary, bao = df.iloc[:200].copy(), df.iloc[190:].copy()
        merged = runner.append_verified_tail(primary, bao)
        self.assertEqual(len(merged), len(df))
        self.assertEqual(merged.attrs["tail_overlap_bars"], 10)
        self.assertEqual(merged.attrs["tail_appended_bars"], 60)
        bao.loc[bao.index[0], "close"] += .02
        with self.assertRaises(ValueError):
            runner.append_verified_tail(primary, bao)

    def test_tail_volume_unit_mismatch_is_rejected(self):
        df = prices(260)
        primary, bao = df.iloc[:200].copy(), df.iloc[190:].copy()
        bao["volume"] /= 100
        with self.assertRaises(ValueError):
            runner.append_verified_tail(primary, bao)


if __name__ == "__main__":
    unittest.main()
