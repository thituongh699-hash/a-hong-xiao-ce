"""因子定义与注册表。

把研究纪律写成代码约束，而不是靠口头约定：

1. 每个因子必须携带**可证伪的经济假设**（``hypothesis``），只写公式的因子注册不上。
2. ``compute`` 只允许使用 T 日及之前的数据。实现上依赖 ``rolling``/``shift``，
   禁止 ``shift(-n)``；``tests/unit/test_factor_panel.py`` 有截断未来数据的因果性用例。
3. ``direction`` 只做方向声明，不改写因子值。因子值保持原始经济含义，
   方向在评分、分层解读与关卡判定时统一应用（``-1`` 表示值越小越好）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd


class FactorRegistryError(ValueError):
    """因子定义不合规或注册冲突。"""


FactorCompute = Callable[[pd.DataFrame, Mapping[str, object]], pd.Series]


@dataclass(frozen=True)
class FactorSpec:
    """一个可检验因子的完整定义。"""

    factor_id: str
    name: str
    category: str
    direction: int
    hypothesis: str
    lookback: int
    compute: FactorCompute
    params: Mapping[str, object] = field(default_factory=dict)
    param_grid: tuple[Mapping[str, object], ...] = ()
    required_columns: tuple[str, ...] = ("close",)

    def __post_init__(self) -> None:
        if not self.factor_id or not self.factor_id.replace("_", "").isalnum():
            raise FactorRegistryError("factor_id 必须是字母数字下划线组合")
        if self.direction not in (1, -1):
            raise FactorRegistryError("direction 只能是 1 或 -1")
        if not str(self.hypothesis).strip():
            raise FactorRegistryError(
                f"{self.factor_id}: 缺少经济假设，不允许注册只讲公式的因子"
            )
        if not callable(self.compute):
            raise FactorRegistryError(f"{self.factor_id}: compute 不可调用")
        if int(self.lookback) <= 0:
            raise FactorRegistryError(f"{self.factor_id}: lookback 必须为正整数")


def _close(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["close"], errors="coerce")


def _positive_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """分母非正或缺失时返回 NaN，不用 0 顶替。"""
    den = pd.to_numeric(denominator, errors="coerce")
    return numerator / den.where(den > 0)


def _mom_20(df: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    window = int(params.get("window", 20))
    close = _close(df)
    return _positive_ratio(close, close.shift(window)) - 1


def _rev_5(df: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    window = int(params.get("window", 5))
    close = _close(df)
    return _positive_ratio(close, close.shift(window)) - 1


def _vol_20(df: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    window = int(params.get("window", 20))
    ret = _close(df).pct_change(fill_method=None)
    return ret.rolling(window, min_periods=window).std()


def _amihud_20(df: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    """非流动性：单位成交额推动的绝对涨跌幅，越大越难交易。"""
    window = int(params.get("window", 20))
    ret = _close(df).pct_change(fill_method=None).abs()
    amount = pd.to_numeric(df["amount"], errors="coerce")
    ratio = ret / amount.where(amount > 0) * 1e9
    return ratio.rolling(window, min_periods=window).mean()


def _max_ret_20(df: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    window = int(params.get("window", 20))
    ret = _close(df).pct_change(fill_method=None)
    return ret.rolling(window, min_periods=window).max()


def _net_obv_ratio_20(df: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    """OBV 净流向占同期成交量的比例，无量纲，落在 [-1, 1] 附近。"""
    window = int(params.get("window", 20))
    close = _close(df)
    vol = pd.to_numeric(df["vol"], errors="coerce").where(lambda x: x > 0)
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * vol.fillna(0.0)).cumsum()
    denominator = vol.rolling(window, min_periods=window).sum()
    return (obv - obv.shift(window)) / denominator.where(denominator > 0)


BUILTIN_FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec(
        factor_id="mom_20",
        name="20 日动量",
        category="量价",
        direction=1,
        hypothesis="趋势延续：资金分批建仓导致价格在 1 个月内同向漂移。",
        lookback=21,
        compute=_mom_20,
        param_grid=({"window": 10}, {"window": 40}),
    ),
    FactorSpec(
        factor_id="rev_5",
        name="5 日反转",
        category="量价",
        direction=-1,
        hypothesis="短期过度反应：5 日内涨幅过大后出现均值回复。",
        lookback=6,
        compute=_rev_5,
        param_grid=({"window": 3}, {"window": 8}),
    ),
    FactorSpec(
        factor_id="vol_20",
        name="20 日波动率",
        category="量价",
        direction=-1,
        hypothesis="低波动溢价：高波动股票承担无法分散的风险却未被充分补偿。",
        lookback=21,
        compute=_vol_20,
        param_grid=({"window": 10}, {"window": 40}),
    ),
    FactorSpec(
        factor_id="amihud_20",
        name="20 日非流动性",
        category="量价",
        direction=1,
        hypothesis="流动性补偿：单位成交额冲击更大的股票需要更高预期收益。",
        lookback=21,
        compute=_amihud_20,
        param_grid=({"window": 10}, {"window": 40}),
        required_columns=("close", "amount"),
    ),
    FactorSpec(
        factor_id="max_ret_20",
        name="20 日极端收益",
        category="量价",
        direction=-1,
        hypothesis="彩票偏好：日内极端涨幅吸引偏好小概率大收益的资金，后续收益偏低。",
        lookback=21,
        compute=_max_ret_20,
        param_grid=({"window": 10}, {"window": 40}),
    ),
    FactorSpec(
        factor_id="net_obv_20",
        name="20 日 OBV 净流向",
        category="量价",
        direction=1,
        hypothesis="量价共振：上涨日成交占优说明增量资金在持续进场。",
        lookback=21,
        compute=_net_obv_ratio_20,
        param_grid=({"window": 10}, {"window": 40}),
        required_columns=("close", "vol"),
    ),
)


_REGISTRY: dict[str, FactorSpec] = {spec.factor_id: spec for spec in BUILTIN_FACTORS}


def register(spec: FactorSpec, *, replace: bool = False) -> FactorSpec:
    """注册自定义因子；默认拒绝覆盖同名因子。"""
    if spec.factor_id in _REGISTRY and not replace:
        raise FactorRegistryError(f"因子 {spec.factor_id} 已存在，如需覆盖请显式 replace=True")
    _REGISTRY[spec.factor_id] = spec
    return spec


def get_factor(factor_id: str) -> FactorSpec:
    try:
        return _REGISTRY[factor_id]
    except KeyError as exc:
        raise FactorRegistryError(f"未注册的因子: {factor_id}") from exc


def list_factors(category: str | None = None) -> list[FactorSpec]:
    specs = list(_REGISTRY.values())
    if category:
        specs = [s for s in specs if s.category == category]
    return sorted(specs, key=lambda s: s.factor_id)


def catalog() -> list[dict[str, object]]:
    """给接口/前端用的因子目录，不含可调用对象。"""
    return [
        {
            "factor_id": s.factor_id,
            "name": s.name,
            "category": s.category,
            "direction": s.direction,
            "hypothesis": s.hypothesis,
            "lookback": s.lookback,
            "required_columns": list(s.required_columns),
            "builtin": s in BUILTIN_FACTORS,
        }
        for s in list_factors()
    ]
