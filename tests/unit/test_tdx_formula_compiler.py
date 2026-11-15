import pytest

from src.tdx.formula_compiler import compile_tdx_formula, get_tdx_compile_capabilities


def test_compile_tdx_formula_generates_strategy_code():
    formula = """
    MA5:=MA(C,5);
    MA10:=MA(C,10);
    CROSS(MA5,MA10)
    """
    payload = compile_tdx_formula(
        formula_text=formula,
        strategy_id="T001",
        strategy_name="TDX_MA_CROSS",
        kline_type="1min",
    )
    code_text = str(payload.get("code", ""))
    assert payload["strategy_id"] == "T001"
    assert payload["class_name"].startswith("GeneratedTdx")
    assert payload["warmup_bars"] >= 60
    assert "Indicators.ma" in code_text
    assert "self._CROSS" in code_text
    assert "class " in code_text


def test_compile_tdx_formula_supports_if_and_ref():
    formula = """
    X:=REF(C,1);
    Y:=IF(C>X,1,0);
    Y=1
    """
    payload = compile_tdx_formula(
        formula_text=formula,
        strategy_id="T002",
        strategy_name="TDX_IF_REF",
        kline_type="D",
    )
    code_text = str(payload.get("code", ""))
    assert "self._REF" in code_text
    assert "self._IF" in code_text
    assert payload["used_functions"]


def test_compile_tdx_formula_rejects_unsupported_functions_in_strict_mode():
    formula = """
    MA5:=MA(C,5);
    Z:=FOO(C,10);
    Z>0
    """
    with pytest.raises(ValueError, match="unsupported tdx functions: FOO"):
        compile_tdx_formula(
            formula_text=formula,
            strategy_id="T003",
            strategy_name="TDX_UNSUPPORTED",
            kline_type="1min",
            strict=True,
        )


def test_tdx_compile_capabilities_contains_core_contract():
    caps = get_tdx_compile_capabilities()
    assert "supported_functions" in caps
    assert "series_alias" in caps
    assert "operators" in caps
    assert "CROSS" in caps["supported_functions"]
    assert "EXIST" in caps["supported_functions"]
    assert "FILTER" in caps["supported_functions"]
    assert "BARSSINCE" in caps["supported_functions"]
    assert caps["series_alias"].get("C") == "close"


def test_compile_tdx_formula_supports_extended_tdx_functions():
    formula = """
    A:=EXIST(C>O,5);
    B:=EVERY(C>=L,3);
    C1:=BETWEEN(C,L,H);
    C2:=VALUEWHEN(C>O,C);
    FILTER(A AND B AND C1 AND (C2>0),2)
    """
    payload = compile_tdx_formula(
        formula_text=formula,
        strategy_id="T004",
        strategy_name="TDX_EXTENDED",
        kline_type="1min",
    )
    code_text = str(payload.get("code", ""))
    assert "self._EXIST" in code_text
    assert "self._EVERY" in code_text
    assert "self._BETWEEN" in code_text
    assert "self._VALUEWHEN" in code_text
    assert "self._FILTER" in code_text
    unsupported = payload.get("compile_meta", {}).get("unsupported_functions", [])
    assert unsupported == []


def test_compile_tdx_formula_supports_barscount_family():
    formula = """
    A:=BARSSINCE(C>O);
    B:=BARSCOUNT(C);
    C1:=HHVBARS(H,10);
    C2:=LLVBARS(L,10);
    (A>=0) AND (B>0) AND (C1>=0) AND (C2>=0)
    """
    payload = compile_tdx_formula(
        formula_text=formula,
        strategy_id="T005",
        strategy_name="TDX_BARS_FAMILY",
        kline_type="1min",
    )
    code_text = str(payload.get("code", ""))
    assert "self._BARSSINCE" in code_text
    assert "self._BARSCOUNT" in code_text
    assert "self._HHVBARS" in code_text
    assert "self._LLVBARS" in code_text
    unsupported = payload.get("compile_meta", {}).get("unsupported_functions", [])
    assert unsupported == []


def _manual_tdx_sma(values, n, m):
    out = []
    prev = None
    for x in values:
        prev = x if prev is None else (m * x + (n - m) * prev) / n
        out.append(prev)
    return out


def test_tdx_sma_follows_native_definition_not_simple_ma():
    import numpy as np
    import pandas as pd

    from src.tdx.formula_compiler import tdx_sma

    values = [10.0, 11.0, 9.5, 12.0, 12.5, 11.2]
    got = tdx_sma(pd.Series(values), 5, 1).tolist()
    assert got == pytest.approx(_manual_tdx_sma(values, 5, 1))
    # 若按简单均线实现，M 会被当成周期，结果退化成 MA(1) == 原值
    assert got != pytest.approx(values)


def test_display_output_and_drawing_attributes_are_stripped():
    formula = """
    X:MA(C,5),COLORRED,LINETHICK2;
    CROSS(C,X)
    """
    payload = compile_tdx_formula(formula_text=formula, strategy_id="T101", strategy_name="strip")
    code_text = str(payload["code"])
    assert "colored" not in code_text.lower()
    assert "linethick" not in code_text.lower()
    assert "x = Indicators.ma(close,5)" in code_text


def test_drawtext_is_ignored_instead_of_failing_strict_mode():
    formula = """
    X:=MA(C,5);
    DRAWTEXT(CROSS(C,X),H,'买'),COLOR00FFFF;
    CROSS(C,X)
    """
    payload = compile_tdx_formula(
        formula_text=formula, strategy_id="T102", strategy_name="drawtext", strict=True
    )
    assert payload["compile_meta"]["unsupported_functions"] == []
    assert payload["last_signal_var"] == "expr_1"


def test_multiline_expression_is_joined_before_translation():
    formula = """
    N:=5;
    V1:4*SMA(C,5,1)-
    3*SMA(C,3.2,1);
    V1>0
    """
    payload = compile_tdx_formula(formula_text=formula, strategy_id="T103", strategy_name="multiline")
    code_text = str(payload["code"])
    assert "self._TDX_SMA(close,5,1)" in code_text
    assert "-\n" not in code_text


def test_chinese_variable_names_get_unique_ascii_names():
    formula = """
    飞溅:=IF(C>O,1,0);
    吸血吃肉:=IF(C<O,1,0);
    飞溅 AND 吸血吃肉
    """
    payload = compile_tdx_formula(formula_text=formula, strategy_id="T104", strategy_name="chinese")
    code_text = str(payload["code"])
    assert "v1 = " in code_text and "v2 = " in code_text
    assert "____" not in code_text
    assert "(v1) & (v2)" in code_text


def test_and_keeps_comparison_precedence():
    formula = """
    A:=C;
    A>O AND A<H
    """
    payload = compile_tdx_formula(formula_text=formula, strategy_id="T105", strategy_name="precedence")
    assert "((a>open_) & (a<high))" in str(payload["code"])


def test_constant_final_statement_is_rejected_with_actionable_message():
    formula = """
    X:=MA(C,5);
    0,COLORFFCC00;
    """
    with pytest.raises(ValueError, match="最后一条语句是常量"):
        compile_tdx_formula(formula_text=formula, strategy_id="T106", strategy_name="baseline")
