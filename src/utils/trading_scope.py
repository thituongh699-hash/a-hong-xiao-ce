"""Trading-universe normalization and conservative stock classification.

The stock list used by the application is a current snapshot.  It can support
current name-based ST/delisting filtering, but it cannot establish historical
point-in-time status.  This module keeps that limitation explicit and fails
closed whenever a restricted stock-type selection cannot be verified.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any


BOARD_ORDER = ("main_board", "chinext", "star", "beijing")
STOCK_TYPE_ORDER = ("normal", "st", "delisting")

_SCOPE_KEYS = {"boards", "stock_types"}
_EXCHANGES = {"SH", "SZ", "BJ"}
_CODE_WITH_PREFIX = re.compile(r"^(SH|SZ|BJ)[.\-_:]?([0-9]{1,6})$")
_CODE_WITH_SUFFIX = re.compile(r"^([0-9]{1,6})[.\-_:]?(SH|SZ|BJ)$")
_DIGITS_ONLY = re.compile(r"^[0-9]{1,6}$")


def scope_options() -> dict[str, list[dict[str, str]]]:
    """Return stable UI options for supported boards and stock types."""

    return {
        "boards": [
            {"value": "main_board", "label": "沪深主板"},
            {"value": "chinext", "label": "创业板"},
            {"value": "star", "label": "科创板"},
            {"value": "beijing", "label": "北交所"},
        ],
        "stock_types": [
            {"value": "normal", "label": "普通股票"},
            {"value": "st", "label": "ST 股票"},
            {"value": "delisting", "label": "退市整理/退市股票"},
        ],
    }


def normalize_trading_scope(scope: Any) -> dict[str, list[str]] | None:
    """Validate and canonicalize a trading-scope payload.

    ``None`` deliberately means compatibility mode for older clients: no
    board or stock-type filtering is applied.  An explicit payload must be
    complete so a misspelled or omitted field never silently widens scope.
    """

    if scope is None:
        return None
    if not isinstance(scope, Mapping):
        raise ValueError("trading scope must be an object or null")

    keys = set(scope.keys())
    if keys != _SCOPE_KEYS:
        missing = sorted(_SCOPE_KEYS - keys)
        unknown = sorted(keys - _SCOPE_KEYS, key=str)
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(map(str, unknown))}")
        raise ValueError("invalid trading scope (" + "; ".join(details) + ")")

    boards = _normalize_enum_list(scope["boards"], "boards", BOARD_ORDER)
    stock_types = _normalize_enum_list(
        scope["stock_types"], "stock_types", STOCK_TYPE_ORDER
    )
    return {"boards": boards, "stock_types": stock_types}


def normalize_stock_code(value: Any) -> str:
    """Return a conservative canonical security code.

    Canonical exchange-qualified codes use ``NNNNNN.EX``.  Bare historical
    43/83/87 codes stay unqualified because those prefixes also cover NEEQ
    securities; only an explicit BJ prefix/suffix establishes their exchange.
    """

    if isinstance(value, bool) or value is None:
        raise ValueError("stock code must be a number or string")

    if isinstance(value, int):
        if value < 0 or value > 999999:
            raise ValueError(f"invalid stock code: {value!r}")
        raw = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"invalid stock code: {value!r}")
        integer = int(value)
        if integer < 0 or integer > 999999:
            raise ValueError(f"invalid stock code: {value!r}")
        raw = str(integer)
    elif isinstance(value, str):
        raw = value.strip().upper()
    else:
        raise ValueError("stock code must be a number or string")

    if not raw:
        raise ValueError("stock code must not be empty")

    exchange: str | None = None
    digits: str | None = None
    prefix_match = _CODE_WITH_PREFIX.fullmatch(raw)
    suffix_match = _CODE_WITH_SUFFIX.fullmatch(raw)
    if prefix_match:
        exchange, digits = prefix_match.groups()
    elif suffix_match:
        digits, exchange = suffix_match.groups()
    elif _DIGITS_ONLY.fullmatch(raw):
        digits = raw
    else:
        raise ValueError(f"invalid stock code: {value!r}")

    digits = digits.zfill(6)
    if exchange is None:
        exchange = _infer_exchange(digits)
    if exchange is None:
        return digits
    if exchange not in _EXCHANGES:  # defensive; regex currently enforces it
        raise ValueError(f"unsupported stock exchange: {exchange}")
    return f"{digits}.{exchange}"


def classify_stock(code: Any, record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Classify a code by supported A-share board and current-list name."""

    normalized = normalize_stock_code(code)
    digits, exchange = _split_normalized_code(normalized)
    board, board_basis = _classify_board(digits, exchange)
    name = _record_name(record)
    stock_type, type_basis = _classify_stock_type(name)

    return {
        "stock_code": normalized,
        "board": board,
        "stock_type": stock_type,
        "name": name,
        "classification_basis": {
            "board": board_basis,
            "stock_type": type_basis,
        },
        "historical_st_status_verified": False,
    }


def select_trading_scope(
    codes: Iterable[Any],
    scope: Any,
    records: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize, deduplicate, and optionally filter a sequence of securities."""

    if isinstance(codes, (str, bytes)) or not isinstance(codes, Iterable):
        raise ValueError("codes must be an iterable of stock codes")

    normalized_scope = normalize_trading_scope(scope)
    normalized_codes: list[str] = []
    seen: set[str] = set()
    for code in codes:
        normalized = normalize_stock_code(code)
        if normalized not in seen:
            seen.add(normalized)
            normalized_codes.append(normalized)

    metadata: dict[str, Any] = {
        "stock_type_source": "current stock-list name snapshot",
        "historical_st_status_verified": False,
        "historical_status_note": (
            "Current names do not verify point-in-time historical ST or delisting status."
        ),
        "unverified_stock_type_codes": [],
    }

    # A missing scope is the explicit old-client compatibility contract.
    if normalized_scope is None:
        metadata["compatibility_mode"] = True
        return {
            "trading_scope": None,
            "eligible_codes": normalized_codes,
            "excluded": [],
            "metadata": metadata,
        }

    metadata["compatibility_mode"] = False
    record_index = _index_records(records or ())
    selected_boards = set(normalized_scope["boards"])
    selected_types = set(normalized_scope["stock_types"])
    all_stock_types_selected = selected_types == set(STOCK_TYPE_ORDER)
    eligible: list[str] = []
    excluded: list[dict[str, Any]] = []

    for code in normalized_codes:
        classification = classify_stock(code, record_index.get(code))
        board = classification["board"]
        stock_type = classification["stock_type"]

        reason: str | None = None
        if board is None:
            reason = "unsupported_or_ambiguous_board"
        elif board not in selected_boards:
            reason = "board_not_selected"
        elif stock_type is None:
            if all_stock_types_selected:
                metadata["unverified_stock_type_codes"].append(code)
            else:
                reason = "stock_type_unverified"
        elif stock_type not in selected_types:
            reason = "stock_type_not_selected"

        if reason is None:
            eligible.append(code)
        else:
            excluded.append({**classification, "reason": reason})

    return {
        "trading_scope": normalized_scope,
        "eligible_codes": eligible,
        "excluded": excluded,
        "metadata": metadata,
    }


def _normalize_enum_list(value: Any, field: str, order: tuple[str, ...]) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} entries must be strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicate entries")
    invalid = [item for item in value if item not in order]
    if invalid:
        raise ValueError(f"invalid {field}: {', '.join(invalid)}")
    selected = set(value)
    return [item for item in order if item in selected]


def _infer_exchange(digits: str) -> str | None:
    if digits.startswith("920"):
        return "BJ"
    if digits.startswith(("4", "8")):
        return None
    if digits.startswith(("5", "6", "7", "9")):
        return "SH"
    if digits.startswith(("0", "1", "2", "3")):
        return "SZ"
    return None


def _split_normalized_code(normalized: str) -> tuple[str, str | None]:
    if "." not in normalized:
        return normalized, None
    digits, exchange = normalized.rsplit(".", 1)
    return digits, exchange


def _classify_board(digits: str, exchange: str | None) -> tuple[str | None, str]:
    if exchange == "SH":
        if digits.startswith(("600", "601", "603", "605")):
            return "main_board", "SH main-board stock prefix"
        if digits.startswith("688"):
            return "star", "SH STAR Market stock prefix 688"
        # 689 records include depositary receipts and are deliberately not
        # treated as ordinary STAR A shares by this scope selector.
        return None, "SH code is outside supported A-share stock prefixes"

    if exchange == "SZ":
        number = int(digits)
        if digits.startswith("000"):
            return "main_board", "SZ main-board stock prefix 000"
        if 1200 <= number <= 1999:
            return "main_board", "SZ main-board stock range 001200-001999"
        if digits.startswith(("002", "003", "004")):
            return "main_board", f"SZ main-board stock prefix {digits[:3]}"
        if 300000 <= number <= 309799:
            return "chinext", "SZ ChiNext stock range 300000-309799"
        return None, "SZ code is outside supported A-share stock prefixes"

    if exchange == "BJ":
        if digits.startswith("920"):
            return "beijing", "BJ current stock prefix 920"
        if digits.startswith(("43", "83", "87")):
            return "beijing", "explicit BJ historical stock prefix"
        return None, "BJ code is outside supported stock prefixes"

    return None, "exchange is ambiguous or unsupported"


def _record_name(record: Mapping[str, Any] | None) -> str | None:
    if not isinstance(record, Mapping):
        return None
    value = record.get("name")
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    name = str(value).strip()
    if not name or name.lower() in {"nan", "none", "null"}:
        return None
    return name


def _classify_stock_type(name: str | None) -> tuple[str | None, str]:
    if name is None:
        return None, "current-list name metadata is missing"
    compact = re.sub(r"\s+", "", name).upper().replace("＊", "*")
    if "退市" in compact or compact.endswith("退"):
        return "delisting", "current-list name indicates delisting"
    if compact.startswith(("ST", "*ST", "SST", "S*ST")):
        return "st", "current-list name has an ST prefix"
    return "normal", "current-list name has no ST or delisting marker"


def _index_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or "code" not in record:
            continue
        try:
            normalized = normalize_stock_code(record["code"])
        except ValueError:
            continue
        index.setdefault(normalized, record)
    return index


__all__ = [
    "classify_stock",
    "normalize_stock_code",
    "normalize_trading_scope",
    "scope_options",
    "select_trading_scope",
]
