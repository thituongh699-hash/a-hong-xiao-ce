"""Deterministic parameter selection and crash-safe sequential evaluation.

The evaluator owns strategy data and any later validation. This module only
searches the supplied training objective; it does not run a validation period.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Callable, Mapping


MAX_GRID_SIZE = 100_000
_STATUSES = {"ok", "ineligible", "error"}


def _json(value: Any) -> str:
    """Canonical strict JSON; also rejects unsupported parameter/context types."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clean(value: Any) -> Any:
    """Make evaluator metrics strict JSON by replacing nonfinite floats."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(value, int) or math.isfinite(value)


def _copy_params(params: dict[str, Any]) -> dict[str, Any]:
    # JSON values can be nested; never let callbacks mutate the search state.
    return json.loads(_json(params))


def _unique(configurations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for params in configurations:
        if not isinstance(params, dict) or not all(isinstance(key, str) for key in params):
            raise ValueError("each configuration must be a dictionary with string keys")
        signature = _json(params)
        if signature not in seen:
            seen.add(signature)
            output.append(json.loads(signature))
    return output


def _candidates(space: dict[str, list[Any]] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(space, list):
        if len(space) > MAX_GRID_SIZE:
            raise ValueError(f"parameter space exceeds {MAX_GRID_SIZE} configurations")
        return _unique(space)
    if not isinstance(space, dict) or not space:
        raise ValueError("parameter_space must be a nonempty grid or a list of configurations")
    if not all(isinstance(key, str) and isinstance(values, list) and values for key, values in space.items()):
        raise ValueError("grid dimensions must have string keys and nonempty value lists")
    names = list(space)
    dimensions: list[list[Any]] = []
    total = 1
    for name in names:
        values = []
        seen = set()
        for value in space[name]:
            signature = _json(value)
            if signature not in seen:
                seen.add(signature)
                values.append(json.loads(signature))
        dimensions.append(values)
        total *= len(values)
        if total > MAX_GRID_SIZE:
            raise ValueError(f"parameter grid exceeds {MAX_GRID_SIZE} configurations")
    return [dict(zip(names, combination)) for combination in product(*dimensions)]


def _append(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_json(item) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_checkpoint(path: Path, fingerprint: str, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.endswith(b"\n"):
                raise ValueError(f"checkpoint has a truncated line at {line_number}")
            try:
                item = json.loads(raw.decode("utf-8"))
                _json(item)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"malformed checkpoint line {line_number}") from exc
            if line_number == 1:
                if not isinstance(item, dict) or item.get("type") != "header" or item.get("fingerprint") != fingerprint:
                    raise ValueError("checkpoint fingerprint mismatch or invalid header")
                continue
            if not isinstance(item, dict) or item.get("type") != "trial" or not isinstance(item.get("record"), dict):
                raise ValueError(f"invalid checkpoint trial at line {line_number}")
            record = item["record"]
            index = len(records)
            if (index >= len(selected) or type(record.get("trial_id")) is not int
                    or record["trial_id"] != index + 1 or _json(record.get("params")) != _json(selected[index])):
                raise ValueError(f"checkpoint trial sequence mismatch at line {line_number}")
            if (record.get("status") not in _STATUSES or not isinstance(record.get("metrics"), dict)
                    or not isinstance(record.get("error"), (str, type(None)))
                    or not (record.get("score") is None or _finite_number(record.get("score")))):
                raise ValueError(f"invalid checkpoint record at line {line_number}")
            if record["status"] == "ok" and record["score"] is None:
                raise ValueError(f"invalid checkpoint score at line {line_number}")
            records.append(record)
    if not records and path.stat().st_size == 0:
        raise ValueError("checkpoint is empty")
    return records


@dataclass(frozen=True)
class OptimizationResult:
    records: list[dict[str, Any]]
    best: dict[str, Any] | None
    ranked: list[dict[str, Any]]
    metadata: dict[str, Any]


class ParameterOptimizer:
    """Run at most ``max_trials`` unique configurations in deterministic order."""

    def run(
        self,
        parameter_space: dict[str, list[Any]] | list[dict[str, Any]],
        evaluate: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        objective: str = "score",
        maximize: bool = True,
        max_trials: int | None = None,
        seed: int = 0,
        checkpoint: str | os.PathLike[str] | None = None,
        context: Mapping[str, Any] | None = None,
        evaluator_version: str | None = None,
        eligibility_version: str | None = None,
        constraint_version: str | None = None,
        eligible: Callable[[dict[str, Any]], bool] | None = None,
        constraint: Callable[[dict[str, Any]], bool] | None = None,
        progress: Callable[[dict[str, Any], int, int], None] | None = None,
    ) -> OptimizationResult:
        if not isinstance(objective, str) or not objective:
            raise ValueError("objective must be a nonempty metric key")
        if not isinstance(maximize, bool):
            raise ValueError("maximize must be a boolean")
        if max_trials is not None and (isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials < 1):
            raise ValueError("max_trials must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        candidates = _candidates(parameter_space)
        if not candidates:
            raise ValueError("parameter_space has no configurations")
        limit = min(len(candidates), max_trials or len(candidates))
        if limit == len(candidates):
            selected = candidates
        else:
            selected = [candidates[index] for index in random.Random(seed).sample(range(len(candidates)), limit)]
        path = Path(checkpoint) if checkpoint is not None else None
        if path is not None and (not isinstance(context, Mapping) or not context):
            raise ValueError("checkpoint requires explicit nonempty context identifying data and evaluator")
        versions = {}
        for name, callback in (("evaluator_version", evaluate),
                               ("eligibility_version", eligible),
                               ("constraint_version", constraint)):
            supplied = {"evaluator_version": evaluator_version,
                        "eligibility_version": eligibility_version,
                        "constraint_version": constraint_version}[name]
            inherited = context.get(name) if isinstance(context, Mapping) else None
            version = supplied if supplied is not None else inherited
            if version is not None and (not isinstance(version, str) or not version.strip()):
                raise ValueError(f"{name} must be a nonempty string")
            if path is not None and callback is not None and version is None:
                raise ValueError(f"checkpoint requires {name} for its callback")
            versions[name] = version
        signature = _json({"parameter_space": candidates, "objective": objective, "maximize": maximize,
                           "context": dict(context) if context is not None else None,
                           "callback_versions": versions, "max_trials": max_trials, "seed": seed})
        fingerprint = sha256(signature.encode("utf-8")).hexdigest()
        records: list[dict[str, Any]] = []
        if path is not None:
            if path.exists():
                records = _load_checkpoint(path, fingerprint, selected)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                _append(path, {"type": "header", "fingerprint": fingerprint})
        resumed = len(records)
        for index in range(resumed, limit):
            params = selected[index]
            record: dict[str, Any] = {"trial_id": index + 1, "params": params, "metrics": {},
                                      "status": "error", "score": None, "error": None}
            try:
                if constraint is not None and not constraint(_copy_params(params)):
                    record["status"] = "ineligible"
                    record["error"] = "constraint rejected"
                else:
                    metrics = evaluate(_copy_params(params))
                    if not isinstance(metrics, dict):
                        raise TypeError("evaluator must return a metrics dictionary")
                    original_score = metrics.get(objective)
                    cleaned = _clean(metrics)
                    _json(cleaned)
                    record["metrics"] = cleaned
                    if objective not in metrics:
                        raise KeyError(f"missing objective metric: {objective}")
                    if isinstance(original_score, bool) or not isinstance(original_score, (int, float)):
                        raise TypeError(f"objective metric {objective} must be numeric")
                    if not _finite_number(original_score):
                        record["status"] = "ineligible"
                        record["error"] = "nonfinite objective"
                    elif eligible is not None and not eligible(_clean(cleaned)):
                        record["status"] = "ineligible"
                        record["error"] = "eligibility rejected"
                    else:
                        record["status"] = "ok"
                        record["score"] = original_score
            except Exception as exc:
                record["status"] = "error"
                record["score"] = None
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            if path is not None:
                _append(path, {"type": "trial", "record": record})
            if progress is not None:
                progress(json.loads(_json(record)), len(records), limit)
        ranked = sorted((record for record in records if record["status"] == "ok"),
                        key=lambda record: record["score"], reverse=maximize)
        metadata = {"total_candidates": len(candidates), "selected": limit, "completed": len(records),
                    "attempted": sum(record["status"] != "ineligible" or record["error"] != "constraint rejected"
                                     for record in records),
                    "skipped": sum(record["status"] == "ineligible" and record["error"] == "constraint rejected"
                                   for record in records),
                    "resumed": resumed, "succeeded": len(ranked),
                    "ineligible": sum(record["status"] == "ineligible" for record in records),
                    "errors": sum(record["status"] == "error" for record in records),
                    "objective": objective, "maximize": maximize, "seed": seed, "fingerprint": fingerprint}
        return OptimizationResult(records=records, best=ranked[0] if ranked else None,
                                  ranked=ranked, metadata=metadata)
