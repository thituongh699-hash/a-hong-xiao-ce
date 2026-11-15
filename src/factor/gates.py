"""四道关卡与结论判定。

关卡不是打分，是门槛：任何一条硬性检查不过，因子就不能进组合池。
区分两类失败，避免把「数据不够」和「因子无效」混为一谈：

- ``failed``：因子本身不过关，结论直接淘汰。
- ``weak`` / ``incomplete``：证据不足（样本外缺失、样本未覆盖退市股），
  结论是「待补证」，可以用，但必须在报告里写明限制。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PASSED = "passed"
WEAK = "weak"
INCOMPLETE = "incomplete"
FAILED = "failed"

_STATUS_LABEL = {
    PASSED: "通过",
    WEAK: "证据偏弱",
    INCOMPLETE: "证据不足",
    FAILED: "未通过",
}

GATE_NAMES = ("预测力", "单调性", "稳健性", "可实现性")


@dataclass(frozen=True)
class GateConfig:
    min_abs_ic: float = 0.03
    min_icir: float = 0.5
    min_hit_rate: float = 0.55
    min_abs_t: float = 2.0
    min_monotone: float = 0.8
    max_reversals: int = 1
    min_spread_t: float = 1.5
    min_spread_win: float = 0.55
    min_oos_ic_ratio: float = 0.6
    max_turnover: float = 0.85
    min_net_long_excess: float = 0.0

    def describe(self) -> dict[str, object]:
        return {
            "IC 绝对值下限": self.min_abs_ic,
            "ICIR 下限": self.min_icir,
            "同号率下限": self.min_hit_rate,
            "t 值下限": self.min_abs_t,
            "单调性下限": self.min_monotone,
            "相邻逆序上限": self.max_reversals,
            "多空 t 值下限": self.min_spread_t,
            "样本外 IC 保持率": self.min_oos_ic_ratio,
            "换手上限": self.max_turnover,
        }


@dataclass
class Check:
    name: str
    value: float | None
    threshold: float | None
    passed: bool | None
    blocking: bool = True
    note: str = ""


@dataclass
class Gate:
    gate: int
    name: str
    status: str
    checks: list[Check] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "name": self.name,
            "status": self.status,
            "status_label": _STATUS_LABEL[self.status],
            "reason": self.reason,
            "checks": [
                {
                    "name": c.name,
                    "value": None if c.value is None else float(c.value),
                    "threshold": None if c.threshold is None else float(c.threshold),
                    "passed": c.passed,
                    "blocking": c.blocking,
                    "note": c.note,
                }
                for c in self.checks
            ],
        }


def _finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _gate_status(checks: list[Check]) -> str:
    hard_fail = [c for c in checks if c.passed is False and c.blocking]
    if hard_fail:
        return FAILED
    if any(c.passed is False and not c.blocking for c in checks):
        return WEAK
    if any(c.passed is None for c in checks):
        return INCOMPLETE
    return PASSED


def _summarize(status: str, checks: list[Check]) -> str:
    if status == PASSED:
        return "全部检查通过"
    problems = []
    for c in checks:
        if c.passed is False:
            problems.append(f"{c.name}不达标" + (f"（{c.note}）" if c.note else ""))
        elif c.passed is None:
            problems.append(f"{c.name}未检验" + (f"（{c.note}）" if c.note else ""))
    return "；".join(problems) or _STATUS_LABEL[status]


def judge(
    metrics: dict[str, object],
    config: GateConfig | None = None,
    *,
    oos: dict[str, object] | None = None,
    param_stability: float | None = None,
    universe_stability: float | None = None,
    survivorship_complete: bool = False,
    survivorship_note: str = "",
) -> dict[str, object]:
    """对单个因子的证据包给出四道关卡结论。"""
    cfg = config or GateConfig()
    ic = metrics.get("ic") or {}
    layered = metrics.get("layered") or {}
    cost = metrics.get("cost") or {}

    abs_ic = _finite(ic.get("mean_adj"))
    icir = _finite(ic.get("icir_adj"))
    hit_rate = _finite(ic.get("hit_rate_adj"))
    t_value = _finite(ic.get("t_adj"))

    gate1 = Gate(1, GATE_NAMES[0], PASSED, [
        Check("IC 绝对值", None if abs_ic is None else abs(abs_ic), cfg.min_abs_ic,
              None if abs_ic is None else abs(abs_ic) >= cfg.min_abs_ic),
        Check("ICIR", icir, cfg.min_icir, None if icir is None else icir >= cfg.min_icir),
        Check("IC 同号率", hit_rate, cfg.min_hit_rate,
              None if hit_rate is None else hit_rate >= cfg.min_hit_rate),
        Check("IC t 值", t_value, cfg.min_abs_t, None if t_value is None else t_value >= cfg.min_abs_t),
    ])
    gate1.status = _gate_status(gate1.checks)
    gate1.reason = _summarize(gate1.status, gate1.checks)

    monotone = _finite(layered.get("monotone_adj"))
    reversals = layered.get("reversals")
    spread_t = _finite(layered.get("spread_t"))
    spread_win = _finite(layered.get("spread_win"))
    reversals_value = None if reversals is None else float(reversals)
    gate2 = Gate(2, GATE_NAMES[1], PASSED, [
        Check("分层单调性", monotone, cfg.min_monotone,
              None if monotone is None else monotone >= cfg.min_monotone),
        Check("相邻逆序数", reversals_value, cfg.max_reversals,
              None if reversals_value is None else reversals_value <= cfg.max_reversals),
        Check("多空 t 值", spread_t, cfg.min_spread_t,
              None if spread_t is None else spread_t >= cfg.min_spread_t),
        Check("多空胜率", spread_win, cfg.min_spread_win,
              None if spread_win is None else spread_win >= cfg.min_spread_win),
    ])
    gate2.status = _gate_status(gate2.checks)
    gate2.reason = _summarize(gate2.status, gate2.checks)

    gate3_checks: list[Check] = []
    if oos is None:
        gate3_checks.append(Check("样本外 IC 保持率", None, cfg.min_oos_ic_ratio, None,
                                  note="未做样本内外切分"))
    else:
        oos_ic = _finite((oos.get("ic") or {}).get("mean_adj"))
        base_ic = _finite(oos.get("base_mean_adj"))
        if base_ic is None:
            base_ic = abs_ic
        if oos_ic is None or base_ic is None or base_ic == 0:
            gate3_checks.append(Check("样本外 IC 保持率", None, cfg.min_oos_ic_ratio, None,
                                      note="样本内 IC 为 0，无法计算保持率"))
        else:
            ratio = abs(oos_ic) / abs(base_ic)
            same_sign = (oos_ic > 0) == (base_ic > 0)
            gate3_checks.append(Check("样本外 IC 保持率", ratio, cfg.min_oos_ic_ratio,
                                      ratio >= cfg.min_oos_ic_ratio and same_sign,
                                      note="方向需与样本内一致"))
    gate3_checks.append(
        Check("参数扰动稳定性", param_stability, 1.0,
              None if param_stability is None else param_stability >= 0.6,
              note="因子参数 ±50% 后同向的比例" if param_stability is not None else "未做参数扰动")
    )
    gate3_checks.append(
        Check("分域稳定性", universe_stability, 0.6,
              None if universe_stability is None else universe_stability >= 0.6,
              note="标的池三等分后，各子池同向且强度不低于基准一半的比例"
              if universe_stability is not None else "未做分域检验")
    )
    gate3_checks.append(
        Check("样本覆盖退市与历史 ST", 1.0 if survivorship_complete else 0.0, 1.0,
              survivorship_complete, blocking=False,
              note=survivorship_note
              or ("股票池已覆盖退市与历史 ST" if survivorship_complete else "当前股票池只有存续标的"))
    )
    gate3 = Gate(3, GATE_NAMES[2], _gate_status(gate3_checks), gate3_checks)
    gate3.reason = _summarize(gate3.status, gate3_checks)

    net_excess = _finite(cost.get("net_long_excess"))
    turnover = _finite(cost.get("turnover"))
    gate4 = Gate(4, GATE_NAMES[3], PASSED, [
        Check("扣费后多头超额", net_excess, cfg.min_net_long_excess,
              None if net_excess is None else net_excess > cfg.min_net_long_excess,
              note="成本按换手率 × 双边费率计算，非零成本假设"),
        Check("换手率", turnover, cfg.max_turnover,
              None if turnover is None else turnover <= cfg.max_turnover,
              note="期间组合成员变动比例"),
    ])
    gate4.status = _gate_status(gate4.checks)
    gate4.reason = _summarize(gate4.status, gate4.checks)

    gates = [gate1, gate2, gate3, gate4]
    if any(g.status == FAILED for g in gates):
        verdict = "淘汰"
    elif any(g.status in (WEAK, INCOMPLETE) for g in gates):
        verdict = "待补证"
    else:
        verdict = "通过"

    return {
        "gates": [g.to_dict() for g in gates],
        "status": "淘汰" if verdict == "淘汰" else ("通过" if verdict == "通过" else "待补证"),
        "verdict": verdict,
        "reasons": [f"关{g.gate} {g.name}：{g.reason}" for g in gates if g.status != PASSED],
        "config": cfg.describe(),
    }
