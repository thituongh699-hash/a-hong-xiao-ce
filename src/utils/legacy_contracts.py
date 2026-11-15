"""Read-only compatibility for persisted contracts from older releases."""

from typing import Any


LEGACY_EVENT_TYPE_ALIASES = {
    "zhongshu": "strategy_signal",
    "menxia": "risk_review",
    "shangshu": "order_execution",
}
LEGACY_REPORT_FIELD_ALIASES = {
    "buy_points": ("硅基节点", "硅基节点分析", "买点分析", "买入节点分析"),
    "sell_points": ("流码节点", "流码节点分析", "卖点分析", "卖出节点分析"),
}


def canonical_event_type(value: Any) -> str:
    text = str(value or "").strip()
    return LEGACY_EVENT_TYPE_ALIASES.get(text, text)


def canonical_event_types(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(x for x in (canonical_event_type(v) for v in values) if x))


def read_legacy_field(payload: Any, canonical_key: str, default: Any = None) -> Any:
    if not isinstance(payload, dict):
        return default
    if payload.get(canonical_key) is not None:
        return payload[canonical_key]
    for legacy_key in LEGACY_REPORT_FIELD_ALIASES.get(canonical_key, ()):
        if payload.get(legacy_key) is not None:
            return payload[legacy_key]
    return default


def canonicalize_report_fields(payload: Any) -> dict[str, Any]:
    result = dict(payload) if isinstance(payload, dict) else {}
    for key, legacy_keys in LEGACY_REPORT_FIELD_ALIASES.items():
        result[key] = read_legacy_field(payload, key, result.get(key))
        for legacy_key in legacy_keys:
            result.pop(legacy_key, None)
    return result


def canonicalize_review_markdown(value: Any) -> str:
    """Read historical point headings using the current parser vocabulary."""
    text = str(value or "")
    for old, current in (("硅基节点", "买入节点"), ("流码节点", "卖出节点"),
                         ("买点分析", "买入节点分析"), ("卖点分析", "卖出节点分析")):
        text = text.replace(old, current)
    return text
