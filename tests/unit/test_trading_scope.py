import pytest

from src.utils.trading_scope import (
    classify_stock,
    normalize_stock_code,
    normalize_trading_scope,
    scope_options,
    select_trading_scope,
)


ALL_SCOPE = {
    "boards": ["main_board", "chinext", "star", "beijing"],
    "stock_types": ["normal", "st", "delisting"],
}


def test_normalize_scope_accepts_none_for_old_client_compatibility():
    assert normalize_trading_scope(None) is None


def test_normalize_scope_orders_values_and_does_not_mutate_input():
    original = {
        "boards": ["beijing", "main_board", "star"],
        "stock_types": ["delisting", "normal"],
    }
    assert normalize_trading_scope(original) == {
        "boards": ["main_board", "star", "beijing"],
        "stock_types": ["normal", "delisting"],
    }
    assert original["boards"] == ["beijing", "main_board", "star"]


@pytest.mark.parametrize(
    "scope",
    [
        {},
        {"boards": [], "stock_types": ["normal"]},
        {"boards": ["main_board"], "stock_types": []},
        {"boards": ["main_board", "main_board"], "stock_types": ["normal"]},
        {"boards": ["fund"], "stock_types": ["normal"]},
        {"boards": ["main_board"], "stock_types": ["ordinary"]},
        {"boards": ["main_board"], "stock_types": ["normal"], "extra": True},
        {"boards": ["main_board"]},
    ],
)
def test_normalize_scope_rejects_incomplete_empty_duplicate_or_unknown_values(scope):
    with pytest.raises(ValueError):
        normalize_trading_scope(scope)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1, "000001.SZ"),
        ("000001", "000001.SZ"),
        ("SZ000001", "000001.SZ"),
        ("000001.sz", "000001.SZ"),
        ("sh.600519", "600519.SH"),
        ("600519SH", "600519.SH"),
        ("920001", "920001.BJ"),
        ("BJ430047", "430047.BJ"),
        ("830799.BJ", "830799.BJ"),
        ("430047", "430047"),
        (830799, "830799"),
    ],
)
def test_normalize_stock_code_preserves_leading_zero_and_explicit_exchange(raw, expected):
    assert normalize_stock_code(raw) == expected


@pytest.mark.parametrize("raw", ["600519.HK", "600519.XSHG", "SH600519.SZ", "abc", -1])
def test_normalize_stock_code_rejects_invalid_or_unsupported_suffix(raw):
    with pytest.raises(ValueError):
        normalize_stock_code(raw)


@pytest.mark.parametrize(
    ("code", "board"),
    [
        ("600519.SH", "main_board"),
        ("001299.SZ", "main_board"),
        ("002594.SZ", "main_board"),
        ("300750.SZ", "chinext"),
        ("301227.SZ", "chinext"),
        ("688001.SH", "star"),
        ("920001.BJ", "beijing"),
        ("430047.BJ", "beijing"),
    ],
)
def test_classify_stock_assigns_supported_boards(code, board):
    assert classify_stock(code, {"name": "示例股份"})["board"] == board


@pytest.mark.parametrize(
    "code",
    [
        "430047",
        "830799",
        "600519.SZ",
        "000001.SH",
        "689009.SH",
        "510300.SH",  # fund
        "900901.SH",  # B share
        "200002.SZ",  # B share
        "399001.SZ",  # index
        "920001.SZ",
    ],
)
def test_classify_stock_does_not_admit_ambiguous_or_non_a_share_codes(code):
    assert classify_stock(code, {"name": "示例"})["board"] is None


@pytest.mark.parametrize(
    ("name", "stock_type"),
    [
        ("ST某某", "st"),
        ("*ST某某", "st"),
        ("SST某某", "st"),
        ("S*ST某某", "st"),
        ("某某退市", "delisting"),
        ("某某退", "delisting"),
        ("普通股份", "normal"),
    ],
)
def test_classify_stock_uses_current_name_markers(name, stock_type):
    result = classify_stock("600519", {"name": name})
    assert result["stock_type"] == stock_type
    assert result["historical_st_status_verified"] is False


def test_classify_stock_does_not_assume_normal_without_name_metadata():
    result = classify_stock("600519")
    assert result["stock_type"] is None
    assert result["name"] is None


def test_select_scope_preserves_order_deduplicates_and_matches_integer_record_codes():
    result = select_trading_scope(
        [1, "000001.SZ", "600519", "300750.SZ", "688001.SH", "920001"],
        ALL_SCOPE,
        [
            {"code": 1, "name": "平安银行"},
            {"code": "600519.SH", "name": "贵州茅台"},
            {"code": "300750.SZ", "name": "宁德时代"},
            {"code": "688001.SH", "name": "华兴源创"},
            {"code": "920001.BJ", "name": "北交示例"},
        ],
    )
    assert result["eligible_codes"] == [
        "000001.SZ",
        "600519.SH",
        "300750.SZ",
        "688001.SH",
        "920001.BJ",
    ]
    assert result["excluded"] == []


def test_select_scope_filters_boards_and_stock_types_with_explanations():
    result = select_trading_scope(
        ["600519", "300750", "688001", "920001"],
        {"boards": ["main_board", "chinext"], "stock_types": ["normal"]},
        [
            {"code": "600519.SH", "name": "贵州茅台"},
            {"code": "300750.SZ", "name": "ST宁德"},
            {"code": "688001.SH", "name": "华兴源创"},
            {"code": "920001.BJ", "name": "北交示例"},
        ],
    )
    assert result["eligible_codes"] == ["600519.SH"]
    assert [item["reason"] for item in result["excluded"]] == [
        "stock_type_not_selected",
        "board_not_selected",
        "board_not_selected",
    ]


def test_select_scope_fails_closed_on_missing_name_when_types_are_restricted():
    result = select_trading_scope(
        ["600519", "300750"],
        {"boards": ["main_board", "chinext"], "stock_types": ["normal", "st"]},
        [{"code": "600519.SH", "name": "贵州茅台"}],
    )
    assert result["eligible_codes"] == ["600519.SH"]
    assert result["excluded"][0]["stock_code"] == "300750.SZ"
    assert result["excluded"][0]["reason"] == "stock_type_unverified"


def test_select_scope_all_types_allows_missing_name_but_marks_it_unverified():
    result = select_trading_scope(
        ["600519"],
        ALL_SCOPE,
        [],
    )
    assert result["eligible_codes"] == ["600519.SH"]
    assert result["metadata"]["unverified_stock_type_codes"] == ["600519.SH"]
    assert result["metadata"]["historical_st_status_verified"] is False


def test_select_scope_none_is_unfiltered_compatibility_mode():
    result = select_trading_scope(["510300", "430047", "600519.SZ"], None, [])
    assert result["trading_scope"] is None
    assert result["eligible_codes"] == ["510300.SH", "430047", "600519.SZ"]
    assert result["excluded"] == []
    assert result["metadata"]["compatibility_mode"] is True


def test_select_scope_wrong_exchange_suffix_cannot_bypass_board_filter():
    result = select_trading_scope(
        ["600519.SZ"],
        {"boards": ["main_board"], "stock_types": ["normal"]},
        [{"code": "600519.SZ", "name": "错误后缀"}],
    )
    assert result["eligible_codes"] == []
    assert result["excluded"][0]["reason"] == "unsupported_or_ambiguous_board"


def test_scope_options_exposes_all_supported_enum_values():
    options = scope_options()
    assert [item["value"] for item in options["boards"]] == list(
        ("main_board", "chinext", "star", "beijing")
    )
    assert [item["value"] for item in options["stock_types"]] == list(
        ("normal", "st", "delisting")
    )
