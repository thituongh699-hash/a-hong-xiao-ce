"""品牌一致性回归：关键界面必须带当前品牌，且业务代码不得硬编码品牌名。

新增界面或入口一律走 ``src.utils.brand``，不要自己再写一份字符串。
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from src.utils.brand import BRAND_NAME, BRAND_NAME_EN, BRAND_SLUG

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _docstring_nodes(tree: ast.AST) -> set[int]:
    nodes: set[int] = set()
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders) or not getattr(node, "body", None):
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            nodes.add(id(first.value))
    return nodes


def test_brand_constants_are_stable():
    assert BRAND_NAME == "A红小策"
    assert BRAND_NAME_EN == "AHongXiaoCe"
    assert BRAND_SLUG == "a-hong-xiao-ce"


@pytest.mark.parametrize(
    "relative, expected",
    [
        ("dashboard.html", f"<title>{BRAND_NAME} · 量化研究工作台</title>"),
        ("dashboard.html", f'id="app-brand" class="font-bold text-lg tracking-wide text-white">{BRAND_NAME} '),
        ("static/factor-lab.html", f"<title>因子研究 · {BRAND_NAME}</title>"),
        ("static/parameter-optimization.html", f"<title>拐弯买入点参数优化 · {BRAND_NAME}</title>"),
        ("backtest_report.html", f"<title>回测报告 · {BRAND_NAME}</title>"),
        ("server.py", "app = FastAPI(title=f\"{BRAND_NAME}量化研究工作台 AI 交易决策控制台\""),
        ("desktop.spec", f'name="{BRAND_NAME}"'),
        ("desktop.spec", 'bundle_identifier="com.ahxc.app"'),
        ("LICENSE", f"<{BRAND_NAME}智能投研决策系统>"),
        ("README.md", f"<h1>{BRAND_NAME}</h1>"),
    ],
)
def test_key_surfaces_carry_current_brand(relative: str, expected: str):
    text = (ROOT / relative).read_text(encoding="utf-8")
    assert expected in text


def test_frontend_storage_keys_share_one_prefix():
    text = (ROOT / "dashboard.html").read_text(encoding="utf-8")
    assert text.count("'ahxc-") >= 12, "本地存储键必须统一使用品牌前缀"


def test_business_code_does_not_hardcode_brand_literal():
    """docstring 里可以写品牌，业务字符串必须引用 BRAND_NAME。"""
    offenders: list[str] = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name == "brand.py":
            continue
        source = path.read_text(encoding="utf-8")
        if BRAND_NAME not in source:
            continue
        tree = ast.parse(source)
        allowed = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and BRAND_NAME in node.value
                and id(node) not in allowed
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == [], f"硬编码品牌名，请改用 BRAND_NAME：{offenders}"
