# src/services/market_data_service.py
import pandas as pd

from src.utils.market_data_validation import MarketDataValidationError, validate_market_data

class MarketDataService:
    """
    行情与运维 (Works): K线清洗、时序排序、缺失值处理、数据对齐
    """
    def __init__(self):
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the K-line data.
        """
        if not isinstance(df, pd.DataFrame):
            raise MarketDataValidationError("market data must be a pandas DataFrame")
        if df.empty:
            return df.copy()
        if "code" not in df.columns:
            raise MarketDataValidationError("market data lacks required column: code")
        cleaned = validate_market_data(df, sort=True)
        cleaned["code"] = cleaned["code"].astype(str)
        return cleaned

    def align_data(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Align multiple stock dataframes (if needed for portfolio level analysis).
        """
        # Placeholder for complex alignment logic
        pass
