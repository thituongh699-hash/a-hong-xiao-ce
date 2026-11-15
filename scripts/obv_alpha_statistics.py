"""Cross-sectional daily alpha summaries and calendar-aware mean inference.

This module deliberately does not load data, construct signals, purge study
splits, or model trading execution. Callers own those choices and pass a
date-indexed panel with forward labels already constructed.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ("mom10", "mom20", "mom60", "vol20")
DAILY_INDEX = ("dt", "horizon")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _date_bins(values: pd.Series, q: int) -> pd.Series:
    """Deterministic quantile bins, with a single-bin constant fallback."""
    x = _numeric(values)
    result = pd.Series(index=values.index, dtype="float64")
    valid = x.notna()
    if not valid.any():
        return result
    if x.loc[valid].nunique() <= 1:
        result.loc[valid] = 0
        return result
    bins = pd.qcut(x.loc[valid], q=q, labels=False, duplicates="drop")
    result.loc[valid] = bins.astype(float)
    return result


def _matched_spread(day: pd.DataFrame, selected: pd.Series,
                    *, min_per_side: int) -> tuple[float, float]:
    """Return selected-weighted matched spread and selected coverage."""
    mom_bin = _date_bins(day["mom20"], 5)
    vol_bin = _date_bins(day["vol20"], 3)
    groups = pd.DataFrame({"mom": mom_bin, "vol": vol_bin}, index=day.index)
    groups["factor"] = selected.astype(bool)
    groups["y"] = _numeric(day["_y"])
    groups = groups.dropna(subset=["mom", "vol", "y"])
    if groups.empty:
        return np.nan, np.nan
    cell = groups.mom.to_numpy(dtype=int) * 3 + groups.vol.to_numpy(dtype=int)
    selected_rows = groups.factor.to_numpy(dtype=bool)
    y = groups.y.to_numpy(dtype=float)
    ns = np.bincount(cell[selected_rows], minlength=15)
    nc = np.bincount(cell[~selected_rows], minlength=15)
    ss = np.bincount(cell[selected_rows], weights=y[selected_rows], minlength=15)
    sc = np.bincount(cell[~selected_rows], weights=y[~selected_rows], minlength=15)
    retained = (ns >= min_per_side) & (nc >= min_per_side)
    matched_selected = int(ns[retained].sum())
    weighted_sum = float(np.sum(ns[retained] * (ss[retained] / ns[retained] - sc[retained] / nc[retained])))
    total_selected = int(selected.sum())
    if total_selected == 0 or matched_selected == 0:
        return np.nan, (matched_selected / total_selected if total_selected else np.nan)
    return weighted_sum / matched_selected, matched_selected / total_selected


def _conditional_beta(day: pd.DataFrame, *, min_n: int) -> float:
    """Per-date OLS factor coefficient, in percentage points."""
    controls = [*FEATURE_COLUMNS]
    if "log_amount20" in day.columns:
        controls.append("log_amount20")
    needed = ["_y", "factor", *controls]
    xdf = day[needed].copy()
    for col in needed:
        xdf[col] = _numeric(xdf[col])
    xdf = xdf.dropna()
    if len(xdf) < min_n or xdf.factor.nunique() < 2:
        return np.nan

    design_parts = [np.ones(len(xdf))]
    for col in controls:
        x = xdf[col].to_numpy(dtype=float)
        lo, hi = np.quantile(x, [0.01, 0.99])
        x = np.clip(x, lo, hi)
        sd = float(x.std(ddof=0))
        if sd <= np.finfo(float).eps:
            x = np.zeros_like(x)
        else:
            x = (x - x.mean()) / sd
        design_parts.append(x)
    factor = xdf.factor.to_numpy(dtype=float)
    controls_matrix = np.column_stack(design_parts)
    outcomes = np.column_stack([factor, xdf["_y"].to_numpy(dtype=float)])
    fitted, _, _, _ = np.linalg.lstsq(controls_matrix, outcomes, rcond=None)
    residual = outcomes - controls_matrix @ fitted
    norm = float(residual[:, 0] @ residual[:, 0])
    # Frisch-Waugh-Lovell gives the same factor coefficient as full OLS, while
    # one projection handles redundant controls and tests factor identification.
    if norm <= np.finfo(float).eps * len(xdf):
        return np.nan
    return float(residual[:, 0] @ residual[:, 1] / norm)


def daily_effects(
    panel: pd.DataFrame,
    horizons: Iterable[int] = (1, 5, 10, 20),
    *,
    min_selected: int = 10,
    min_controls: int = 10,
    min_match_per_side: int = 3,
    min_ols: int = 100,
) -> pd.DataFrame:
    """Calculate per-date unconditional, matched, and conditional effects.

    ``fwd_close_h`` is interpreted as a return in percentage points. Rows are
    filtered to same-date eligible, finite outcomes before every estimate.
    Matching uses five ``mom20`` quantiles by three ``vol20`` quantiles; cells
    without sufficient observations on both sides are omitted. The final
    matched spread weights each retained cell by its selected count.
    """
    required = {"dt", "eligible", "factor", "mom20", "vol20", *FEATURE_COLUMNS}
    if not isinstance(panel, pd.DataFrame) or required - set(panel.columns):
        missing = sorted(required - set(panel.columns)) if isinstance(panel, pd.DataFrame) else sorted(required)
        raise ValueError(f"panel lacks required columns: {missing}")
    hs = tuple(int(h) for h in horizons)
    if not hs or any(h <= 0 for h in hs):
        raise ValueError("horizons must be positive integers")
    if min(min_selected, min_controls, min_match_per_side, min_ols) < 1:
        raise ValueError("minimum sample sizes must be positive")

    source = panel.copy()
    source["dt"] = pd.to_datetime(source.dt, errors="coerce")
    source["eligible"] = source.eligible.fillna(False).astype(bool)
    source["factor"] = _numeric(source.factor)
    rows: list[dict] = []
    for dt, date_panel in source.loc[source.dt.notna()].groupby("dt", sort=True):
        for h in hs:
            label = f"fwd_close_{h}"
            if label not in date_panel:
                raise ValueError(f"panel lacks required outcome column: {label}")
            day = date_panel.loc[date_panel.eligible].copy()
            day["_y"] = _numeric(day[label])
            day = day.loc[day._y.notna() & day.factor.isin([0, 1])].copy()
            selected = day.factor.eq(1)
            controls = day.factor.eq(0)
            ns, nc = int(selected.sum()), int(controls.sum())
            selected_mean = float(day.loc[selected, "_y"].mean()) if ns >= min_selected else np.nan
            control_mean = float(day.loc[controls, "_y"].mean()) if nc >= min_controls else np.nan
            spread = selected_mean - control_mean if np.isfinite(selected_mean + control_mean) else np.nan
            matched, coverage = _matched_spread(day, selected, min_per_side=min_match_per_side)
            if ns < min_selected or nc < min_controls:
                matched, coverage = np.nan, np.nan
            beta = _conditional_beta(day, min_n=min_ols) if ns and nc else np.nan
            rows.append({
                "dt": dt, "horizon": h,
                "all_selected": ns, "all_control": nc,
                "selected_mean": selected_mean, "control_mean": control_mean,
                "spread_pct": spread, "matched_spread_pct": matched,
                "matched_selected_coverage": coverage,
                "conditional_beta_pct": beta,
            })
    columns = ["all_selected", "all_control", "selected_mean", "control_mean",
               "spread_pct", "matched_spread_pct", "matched_selected_coverage",
               "conditional_beta_pct"]
    if not rows:
        empty = pd.DataFrame(columns=columns, index=pd.MultiIndex.from_arrays([[], []], names=DAILY_INDEX))
        return empty
    return pd.DataFrame(rows).set_index(list(DAILY_INDEX))[columns].sort_index()


def mean_inference(
    values: Iterable[float],
    maxlags: int,
    block_length: int = 20,
    draws: int = 2000,
    seed: int = 20261010,
) -> dict[str, float | int]:
    """HAC and circular moving-block bootstrap inference on a full calendar.

    Missing calendar slots stay in place for both estimators. The HAC estimator
    uses Bartlett weights and an observed-count finite-sample correction.
    Bootstrap blocks are sampled from contiguous positions in the supplied
    calendar, retaining both values and the missing-value mask.
    """
    if isinstance(maxlags, bool) or int(maxlags) != maxlags or maxlags < 0:
        raise ValueError("maxlags must be a nonnegative integer")
    if isinstance(block_length, bool) or int(block_length) != block_length or block_length < 1:
        raise ValueError("block_length must be a positive integer")
    if isinstance(draws, bool) or int(draws) != draws or draws < 1:
        raise ValueError("draws must be a positive integer")
    x = np.asarray(list(values), dtype=float).reshape(-1)
    x[~np.isfinite(x)] = np.nan
    valid = np.isfinite(x)
    n = int(valid.sum())
    result: dict[str, float | int] = {
        "n": n, "mean_pct": np.nan, "hac_se_pct": np.nan, "hac_t": np.nan,
        "hac_ci_low_pct": np.nan, "hac_ci_high_pct": np.nan,
        "bootstrap_ci_low_pct": np.nan, "bootstrap_ci_high_pct": np.nan,
    }
    if n == 0:
        return result
    mean = float(np.nanmean(x))
    result["mean_pct"] = mean
    if n > 1:
        residual = np.where(valid, x - mean, 0.0)
        t_len = len(x)
        var_sum = float(residual @ residual)
        for lag in range(1, min(int(maxlags), t_len - 1) + 1):
            weight = 1.0 - lag / (int(maxlags) + 1.0)
            var_sum += 2.0 * weight * float(residual[lag:] @ residual[:-lag])
        variance = max(0.0, var_sum) / (n * n) * (n / (n - 1))
        se = float(np.sqrt(variance))
        result["hac_se_pct"] = se
        result["hac_t"] = mean / se if se > 0 else np.nan
        result["hac_ci_low_pct"] = mean - 1.96 * se
        result["hac_ci_high_pct"] = mean + 1.96 * se

    if len(x) == 0:
        return result
    rng = np.random.default_rng(seed)
    t_len = len(x)
    block = min(int(block_length), t_len)
    blocks_needed = int(np.ceil(t_len / block))
    boot = np.empty(int(draws), dtype=float)
    offsets = np.arange(block)
    for first in range(0, int(draws), 256):
        count = min(256, int(draws) - first)
        starts = rng.integers(0, t_len, size=(count, blocks_needed))
        indices = ((starts[:, :, None] + offsets) % t_len).reshape(count, -1)[:, :t_len]
        sample = x[indices]
        observed = np.isfinite(sample).sum(axis=1)
        boot[first:first+count] = np.divide(np.nansum(sample, axis=1), observed,
                                           out=np.full(count, np.nan), where=observed > 0)
    boot = boot[np.isfinite(boot)]
    if len(boot):
        lo, hi = np.quantile(boot, [0.025, 0.975])
        result["bootstrap_ci_low_pct"] = float(lo)
        result["bootstrap_ci_high_pct"] = float(hi)
    return result
