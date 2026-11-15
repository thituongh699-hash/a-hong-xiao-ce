"""数据适配：从项目现有数据源取日线，以及用于验证链路的合成面板。

真实取数一律复用 ``src.utils.data_factory.DataFactory``，不另开数据通道；
单只标的取数失败不影响整体，失败原因随报告一起返回。

``synthetic_bars`` 生成的是**合成数据**，只用于验证代码链路是否跑通，
不能作为任何因子的有效性证据，报告里会强制标注。
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

_COLUMN_ALIASES = {
    "vol": ("vol", "volume", "成交量"),
    "amount": ("amount", "turnover", "money", "成交额"),
}


def _normalize_columns(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    work = frame.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]
    for target, aliases in _COLUMN_ALIASES.items():
        if target in work.columns:
            continue
        for alias in aliases:
            if alias.lower() in work.columns:
                work[target] = work[alias.lower()]
                break
    if "dt" not in work.columns:
        if "date" in work.columns:
            work["dt"] = work["date"]
        else:
            work = work.reset_index().rename(columns={"index": "dt"})
    work["code"] = code
    keep = [c for c in ("dt", "code", "open", "high", "low", "close", "vol", "amount") if c in work.columns]
    return work[keep]


def load_bars(
    codes: Iterable[str],
    start: str,
    end: str,
    source: str = "tdx",
    interval: str = "1D",
    factory: object | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """批量取日线，返回 ``(成功数据, 失败原因)``。"""
    from src.utils.data_factory import DataFactory

    provider = factory.get_provider() if factory is not None else DataFactory(source=source).get_provider()
    bars: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for code in codes:
        frame = pd.DataFrame()
        try:
            if hasattr(provider, "fetch_kline_data"):
                frame = provider.fetch_kline_data(code, start, end, interval=interval)
            if (frame is None or len(frame) == 0) and hasattr(provider, "fetch_daily_data"):
                frame = provider.fetch_daily_data(code, start, end)
        except Exception as exc:  # 数据源异常不应中断整轮研究
            errors[str(code)] = f"{type(exc).__name__}: {exc}"
            continue
        if frame is None or len(frame) == 0:
            errors[str(code)] = "无数据返回"
            continue
        normalized = _normalize_columns(frame, str(code))
        if normalized.empty or "close" not in normalized.columns:
            errors[str(code)] = "返回字段缺少 close"
            continue
        bars[str(code)] = normalized
    return bars, errors


def synthetic_bars(
    n_codes: int = 80,
    n_days: int = 400,
    seed: int = 20261110,
    mean_reversion: float = 0.6,
) -> dict[str, pd.DataFrame]:
    """构造带均值回复的合成行情，用于端到端验证。

    收益率按「当期 = 负系数 × 过去 5 日均值 + 噪声」生成，因此 5 日反转类因子
    应当呈现稳定的负 IC。每只标的另有固定的质量参数，同时决定漂移与波动，
    所以低波动因子也应当有效。两个性质都是刻意写进数据生成器的，
    只证明链路能识别信号、能处理相关性，不证明任何真实市场规律。
    """
    generator = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    bars: dict[str, pd.DataFrame] = {}
    quality = generator.random(n_codes)
    for index in range(n_codes):
        drift = 0.0045 * (float(quality[index]) - 0.5)
        sigma = 0.016 * (1.25 - 0.5 * float(quality[index]))
        eps = generator.normal(0, sigma, n_days)
        returns = np.zeros(n_days)
        for t in range(n_days):
            if t >= 5:
                returns[t] = drift - mean_reversion * float(np.mean(returns[t - 5:t])) + eps[t]
            else:
                returns[t] = drift + eps[t]
        close = 10.0 * np.cumprod(1.0 + returns)
        intraday = np.abs(generator.normal(0, 0.008, n_days))
        high = close * (1.0 + intraday)
        low = close * (1.0 - intraday)
        open_ = low + (high - low) * generator.random(n_days)
        vol = generator.integers(200_000, 5_000_000, n_days).astype(float)
        frame = pd.DataFrame({
            "dt": dates,
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "vol": vol,
            "amount": vol * close,
        })
        bars[f"{600000 + index:06d}"] = frame
    return bars
