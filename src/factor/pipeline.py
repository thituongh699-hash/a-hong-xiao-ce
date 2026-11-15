"""研究编排：从行情面板一路跑到「通过 / 待补证 / 淘汰」的结论。

流程对应八步研究法，每一步都留下可核验的中间证据：

1. 立题：因子目录里每个因子都带经济假设，没写假设的注册不上。
2. 数据：面板构建阶段整只剔除坏数据，剔除原因进报告。
3. 加工：去极值 → 标准化 → 行业市值中性化 → 再标准化。
4. 检验：不重叠采样下的 RankIC / ICIR / 分层 / 衰减 / 换手。
5. 证伪：样本内外切分、参数扰动、样本覆盖度检查。
6. 合成：相关性去冗余后按 ICIR 加权。
7. 组合：合成得分作为组合因子重新走一遍评估。
8. 结论：四道关卡逐条给出通过与未通过的原因。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from .combine import build_combination
from .evaluate import EvalConfig, evaluate_factor, ic_series, ic_summary
from .gates import PASSED, GateConfig, judge
from .panel import (
    PanelBuild,
    attach_labels,
    compute_factor,
    industry_series,
    rebalance_dates,
    size_proxy,
)
from .preprocess import PreprocessConfig, preprocess_cross_section
from .spec import FactorSpec, get_factor

SCHEMA = "factor-study/v1"
MIN_PERIODS = 6

COMBINED_SPEC = FactorSpec(
    factor_id="combined",
    name="多因子合成",
    category="合成",
    direction=1,
    hypothesis="多个通过预测力与单调性检查的低相关因子加权后，信号比单因子更稳。",
    lookback=1,
    compute=lambda frame, params: pd.Series(dtype=float),
)


@dataclass(frozen=True)
class StudyConfig:
    factor_ids: tuple[str, ...] = ("rev_5",)
    horizon: int = 10
    quantiles: int = 5
    entry_lag: int = 1
    min_names: int = 20
    min_names_per_group: int = 3
    start: str | None = None
    end: str | None = None
    universe: str = "未命名股票池"
    survivorship_complete: bool = False
    industry: Mapping[str, str] | None = None
    neutralize: bool = True
    winsor_n: float = 3.0
    standardize_method: str = "zscore"
    oos_ratio: float = 0.3
    combine_method: str = "icir"
    max_abs_corr: float = 0.7
    round_trip_cost_bps: float = 20.0
    cost_multiple: float = 2.0
    param_grids: Mapping[str, tuple[Mapping[str, object], ...]] = field(default_factory=dict)
    gates: GateConfig = field(default_factory=GateConfig)
    demo: bool = False

    def eval_config(self) -> EvalConfig:
        return EvalConfig(
            horizon=self.horizon,
            quantiles=self.quantiles,
            entry_lag=self.entry_lag,
            min_names=self.min_names,
            min_names_per_group=self.min_names_per_group,
            round_trip_cost_bps=self.round_trip_cost_bps,
            cost_multiple=self.cost_multiple,
        )

    def preprocess_config(self) -> PreprocessConfig:
        return PreprocessConfig(
            winsor_n=self.winsor_n,
            method=self.standardize_method,
            neutralize=self.neutralize,
            min_valid=max(5, self.min_names),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "factor_ids": list(self.factor_ids),
            "horizon": self.horizon,
            "quantiles": self.quantiles,
            "entry_lag": self.entry_lag,
            "min_names": self.min_names,
            "start": self.start,
            "end": self.end,
            "universe": self.universe,
            "survivorship_complete": self.survivorship_complete,
            "industry_provided": bool(self.industry),
            "neutralize": self.neutralize,
            "winsor_n": self.winsor_n,
            "standardize_method": self.standardize_method,
            "oos_ratio": self.oos_ratio,
            "combine_method": self.combine_method,
            "max_abs_corr": self.max_abs_corr,
            "round_trip_cost_bps": self.round_trip_cost_bps,
            "cost_multiple": self.cost_multiple,
            "param_grids": {k: [dict(p) for p in v] for k, v in self.param_grids.items()},
        }


def _fingerprint(config: StudyConfig, build: PanelBuild) -> str:
    payload = json.dumps(
        {"config": config.to_dict(), "quality": build.quality},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _preprocess_frame(
    frame: pd.DataFrame,
    column: str,
    config: PreprocessConfig,
    has_industry: bool,
) -> pd.Series:
    pieces: list[pd.Series] = []
    for _, sub in frame.groupby("dt", sort=True):
        pieces.append(
            preprocess_cross_section(
                sub[column],
                industry=sub["industry"] if has_industry else None,
                size=sub["size"],
                config=config,
            )
        )
    if not pieces:
        return pd.Series(dtype=float, index=frame.index)
    return pd.concat(pieces).reindex(frame.index)


def _param_stability(
    panel: pd.DataFrame,
    frame: pd.DataFrame,
    spec: FactorSpec,
    grid: tuple[Mapping[str, object], ...],
    config: StudyConfig,
    base_ic: float,
    has_industry: bool,
) -> float | None:
    if not grid or not np.isfinite(base_ic) or base_ic == 0:
        return None
    hits: list[bool] = []
    for params in grid:
        values = compute_factor(panel, spec, params)
        probe = frame[["dt", "code", "fwd_ret", "size"]].copy()
        if has_industry:
            probe["industry"] = frame["industry"]
        probe["raw"] = values.reindex(frame.index)
        probe["score"] = _preprocess_frame(probe, "raw", config.preprocess_config(), has_industry)
        ic = ic_series(probe, min_names=config.min_names)
        if ic.empty:
            continue
        variant = float(ic.mean()) * spec.direction
        hits.append(np.sign(variant) == np.sign(base_ic))
    if not hits:
        return None
    return float(np.mean(hits))


def _universe_stability(
    work: pd.DataFrame,
    spec: FactorSpec,
    config: StudyConfig,
    base_ic: float,
    folds: int = 3,
) -> float | None:
    """标的池等分后，各子池是否仍同向且强度不低于基准的一半。

    这是「换池检验」的轻量替代：真换池需要另一套成分股历史，这里至少排除
    「信号全部来自少数几只股票」的情况。
    """
    if not np.isfinite(base_ic) or base_ic == 0:
        return None
    codes = sorted(work["code"].astype(str).unique())
    if len(codes) < folds * 8:
        return None
    min_names = max(8, config.min_names // 2)
    hits: list[bool] = []
    for fold in range(folds):
        subset = set(codes[fold::folds])
        block = work.loc[work["code"].astype(str).isin(subset)]
        ic = ic_series(block, min_names=min_names)
        if ic.empty:
            continue
        value = float(ic.mean()) * spec.direction
        hits.append(bool(np.sign(value) == np.sign(base_ic) and abs(value) >= 0.5 * abs(base_ic)))
    if not hits:
        return None
    return float(np.mean(hits))


def run_study(build: PanelBuild, config: StudyConfig) -> dict[str, object]:
    """执行一轮完整因子研究，返回可直接序列化的报告。"""
    panel = build.panel
    if config.start or config.end:
        calendar = pd.to_datetime(panel["dt"])
        mask = pd.Series(True, index=panel.index)
        if config.start:
            mask &= calendar >= pd.Timestamp(config.start)
        if config.end:
            mask &= calendar <= pd.Timestamp(config.end)
        panel = panel.loc[mask].reset_index(drop=True)
    if panel.empty:
        raise ValueError("筛选后没有行情数据")

    labelled = attach_labels(panel, config.horizon, config.entry_lag)
    dates = rebalance_dates(panel, step=config.horizon, start=config.start, end=config.end)
    if len(dates) < MIN_PERIODS:
        raise ValueError(
            f"可用截面仅 {len(dates)} 期，至少需要 {MIN_PERIODS} 期；"
            "请缩短 horizon 或延长数据区间"
        )

    frame = labelled.loc[labelled["dt"].isin(dates)].copy()
    frame["size"] = size_proxy(panel).reindex(frame.index)
    industry = industry_series(panel, config.industry)
    has_industry = industry is not None
    if has_industry:
        frame["industry"] = industry.reindex(frame.index)

    warnings: list[str] = []
    if config.demo:
        warnings.append("本轮使用合成数据，仅用于验证链路，任何数值都不能作为因子有效性证据。")
    if not config.survivorship_complete:
        warnings.append("股票池只含当前存续标的，未覆盖退市与历史 ST，存在幸存者偏差，不能据此认定完整 alpha。")
    if not has_industry:
        warnings.append("未提供行业分类，只做了市值中性，行业暴露未被剥离。")
    warnings.append(
        f"截面按每 {config.horizon} 个交易日取样，避免重叠收益把 t 值抬高；"
        "持有期内的实际换手成本另按换手率估算。"
    )
    warnings.append("涨跌停幅度按板块前缀近似，ST 历史涨跌停状态未逐日还原。")

    oos_cut = int(len(dates) * (1.0 - float(config.oos_ratio)))
    in_sample_dates = dates[:oos_cut]
    out_sample_dates = dates[oos_cut:]
    enough_oos = len(in_sample_dates) >= MIN_PERIODS and len(out_sample_dates) >= 3

    eval_cfg = config.eval_config()
    pcfg = config.preprocess_config()
    base_columns = ["dt", "code", "fwd_ret", "close", "vol", "amount", "size"]
    if has_industry:
        base_columns.append("industry")

    factor_results: list[dict[str, object]] = []
    score_columns: dict[str, pd.Series] = {}
    metrics_map: dict[str, dict[str, object]] = {}

    for factor_id in config.factor_ids:
        try:
            spec = get_factor(factor_id)
            raw = compute_factor(panel, spec)
        except ValueError as exc:
            factor_results.append({
                "factor_id": factor_id, "name": factor_id, "category": "-",
                "direction": 1, "hypothesis": "", "error": str(exc), "gates": None,
            })
            warnings.append(f"{factor_id} 无法检验：{exc}")
            continue

        work = frame[base_columns].copy()
        work["raw"] = raw.reindex(frame.index)
        work["score"] = _preprocess_frame(work, "raw", pcfg, has_industry)
        metrics = evaluate_factor(
            work, spec, config=eval_cfg, factor_col="score", decay_panel=work
        )

        oos_block: dict[str, object] | None = None
        if enough_oos:
            is_ic = ic_series(work.loc[work["dt"].isin(in_sample_dates)], min_names=config.min_names)
            oos_ic = ic_series(work.loc[work["dt"].isin(out_sample_dates)], min_names=config.min_names)
            if not is_ic.empty and not oos_ic.empty:
                is_summary = ic_summary(is_ic, spec.direction)
                oos_block = {
                    "base_mean_adj": is_summary["mean_adj"],
                    "ic": ic_summary(oos_ic, spec.direction),
                    "periods": int(oos_ic.count()),
                }

        base_ic = float((metrics.get("ic") or {}).get("mean_adj") or 0.0)
        grid = tuple(config.param_grids.get(factor_id) or spec.param_grid)
        stability = _param_stability(panel, work, spec, grid, config, base_ic, has_industry)
        universe_stability = _universe_stability(work, spec, config, base_ic)
        verdict = judge(
            metrics,
            config.gates,
            oos=oos_block,
            param_stability=stability,
            universe_stability=universe_stability,
            survivorship_complete=config.survivorship_complete,
            survivorship_note="" if config.survivorship_complete else "股票池只有存续标的",
        )

        gate_status = {g["gate"]: g["status"] for g in verdict["gates"]}
        eligible = gate_status.get(1) == PASSED and gate_status.get(2) == PASSED
        if eligible:
            # 合成前先把方向对齐成「值越大越好」，否则反向因子会把组合信号抵消。
            score_columns[factor_id] = (work["score"] * spec.direction).rename(factor_id)
            metrics_map[factor_id] = metrics

        factor_results.append({
            **{k: v for k, v in metrics.items() if k != "layered"},
            "layered": {
                k: v for k, v in metrics["layered"].items() if k != "nav"
            },
            "nav": metrics["layered"]["nav"],
            "gates": verdict,
            "oos": oos_block,
            "param_stability": stability,
            "universe_stability": universe_stability,
            "eligible_for_combination": eligible,
        })

    combination: dict[str, object] | None = None
    if len(score_columns) >= 1:
        score_frame = pd.DataFrame(score_columns, index=frame.index)
        score_frame.index = pd.MultiIndex.from_arrays(
            [frame["dt"].to_numpy(), frame["code"].to_numpy()]
        )
        combo = build_combination(
            score_frame, metrics_map,
            method=config.combine_method, max_abs_corr=config.max_abs_corr,
        )
        combined_scores = combo.pop("scores")
        if len(score_columns) == 1:
            warnings.append("只有一个因子通过预测力与单调性检查，合成结果等同该单因子。")
        combo_frame = frame[base_columns].copy()
        combo_frame["score"] = combined_scores.reindex(score_frame.index).to_numpy()
        combo_metrics = evaluate_factor(
            combo_frame, COMBINED_SPEC, config=eval_cfg, factor_col="score"
        )
        combo_oos = None
        if enough_oos:
            is_ic = ic_series(combo_frame.loc[combo_frame["dt"].isin(in_sample_dates)], min_names=config.min_names)
            oos_ic = ic_series(combo_frame.loc[combo_frame["dt"].isin(out_sample_dates)], min_names=config.min_names)
            if not is_ic.empty and not oos_ic.empty:
                combo_oos = {
                    "base_mean_adj": ic_summary(is_ic, 1)["mean_adj"],
                    "ic": ic_summary(oos_ic, 1),
                    "periods": int(oos_ic.count()),
                }
        # 组合继承其成分因子的稳健性下限：成分撑不住的扰动，组合也撑不住。
        kept = [f for f in combo.get("kept", [])]
        stability_by_factor = {r["factor_id"]: r for r in factor_results if r.get("gates") is not None}
        combo_param = min(
            (float(stability_by_factor[f]["param_stability"]) for f in kept
             if stability_by_factor.get(f, {}).get("param_stability") is not None),
            default=None,
        )
        combo_universe = min(
            (float(stability_by_factor[f]["universe_stability"]) for f in kept
             if stability_by_factor.get(f, {}).get("universe_stability") is not None),
            default=None,
        )
        combo_verdict = judge(
            combo_metrics, config.gates, oos=combo_oos,
            param_stability=combo_param,
            universe_stability=combo_universe,
            survivorship_complete=config.survivorship_complete,
            survivorship_note="" if config.survivorship_complete else "股票池只有存续标的",
        )
        combination = {
            **combo,
            "metrics": {k: v for k, v in combo_metrics.items() if k != "layered"},
            "nav": combo_metrics["layered"]["nav"],
            "layered": {k: v for k, v in combo_metrics["layered"].items() if k != "nav"},
            "gates": combo_verdict,
        }

    summary = _summarize(factor_results, combination)
    return _json_safe({
        "schema": SCHEMA,
        "study_id": f"fa-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "fingerprint": _fingerprint(config, build),
        "universe": config.universe,
        "config": config.to_dict(),
        "preprocess": pcfg.describe(),
        "gates_config": config.gates.describe(),
        "data_quality": build.quality,
        "sample": {
            "rebalance_dates": len(dates),
            "date_start": pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
            "date_end": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
            "in_sample_periods": len(in_sample_dates),
            "out_sample_periods": len(out_sample_dates),
            "out_of_sample_available": enough_oos,
            "avg_names": float(frame.groupby("dt").size().mean()),
        },
        "factors": factor_results,
        "combination": combination,
        "summary": summary,
        "warnings": warnings,
        "disclaimer": (
            "本报告为量化研究工具输出，历史统计不代表未来收益，不构成任何投资建议。"
            "结论仅在所列数据口径与样本范围内成立。"
        ),
    })


def _json_safe(value: object) -> object:
    """把 NaN/inf 与 numpy 标量转成严格 JSON 可表示的值。

    报告要能被外部逐字复核，不能让 ``json.dumps`` 靠 ``default=str`` 蒙混过关。
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(float(value)) else None
    return value


def _summarize(results: list[dict[str, object]], combination: dict[str, object] | None) -> dict[str, object]:
    counts = {"通过": 0, "待补证": 0, "淘汰": 0, "计算失败": 0}
    best: dict[str, object] | None = None
    best_icir = -np.inf
    for item in results:
        if item.get("gates") is None:
            counts["计算失败"] += 1
            continue
        verdict = str(item["gates"]["verdict"])
        counts[verdict] = counts.get(verdict, 0) + 1
        icir = float((item.get("ic") or {}).get("icir_adj") or -np.inf)
        if np.isfinite(icir) and icir > best_icir:
            best_icir = icir
            best = {
                "factor_id": item["factor_id"],
                "name": item["name"],
                "ic": float((item.get("ic") or {}).get("mean_adj") or 0.0),
                "icir": icir,
                "verdict": verdict,
            }
    total = len(results)
    headline = (
        f"共检验 {total} 个因子：通过 {counts['通过']} 个、待补证 {counts['待补证']} 个、"
        f"淘汰 {counts['淘汰']} 个"
    )
    if counts["计算失败"]:
        headline += f"、计算失败 {counts['计算失败']} 个"
    if best:
        headline += f"。方向调整后 ICIR 最高的是 {best['name']}（IC {best['ic']:.4f}，ICIR {best['icir']:.2f}）。"
    if combination and combination.get("gates"):
        headline += f" 合成组合结论：{combination['gates']['verdict']}。"
    return {"counts": counts, "best": best, "headline": headline}
