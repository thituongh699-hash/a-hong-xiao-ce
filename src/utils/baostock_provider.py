# src/utils/baostock_provider.py
import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
import os
from src.utils.cache_coverage import merge_market_data_frames, minute_cache_covers_range
from src.utils.config_loader import ConfigLoader
from src.utils.indicators import Indicators
from src.utils.market_data_validation import validate_market_data


class BaostockProvider:
    """
    Baostock Data Provider (Open Source, Free)
    数据来源：证券宝 (baostock.com)，完全免费，无需 Token
    注意：不支持 1 分钟 K 线，最小周期为 5 分钟；不支持实时行情
    """

    minimum_interval = "5min"
    _MINUTE_FREQ = "5"

    def __init__(self):
        cfg = ConfigLoader.reload()
        self.last_error = ""
        self.price_mode = "none"
        self._cache_enabled = bool(cfg.get("data_provider.local_cache_enabled", True))
        cache_dir = str(cfg.get("data_provider.local_cache_dir", "data/history/cache") or "data/history/cache")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(base_dir))
        self._cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(project_root, cache_dir)
        os.makedirs(self._cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  代码转换
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_baostock_code(code: str) -> str:
        """600036.SH -> sh.600036, 000001.SZ -> sz.000001"""
        code = str(code).upper().strip()
        if code.startswith("SH.") or code.startswith("SZ."):
            raw = code.split(".")[1]
            prefix = code[:2].lower()
            return f"{prefix}.{raw}"
        if "." in code:
            parts = code.split(".")
            suffix = parts[1].upper()
            prefix = "sh" if suffix == "SH" else "sz"
            return f"{prefix}.{parts[0]}"
        # 纯 6 位数字，根据首位判断市场
        if code.startswith(("6", "9")):
            return f"sh.{code}"
        return f"sz.{code}"

    # ------------------------------------------------------------------ #
    #  缓存（与其他 Provider 保持一致的模式）
    # ------------------------------------------------------------------ #
    def _cache_file_path(self, code, interval="5min"):
        safe_code = str(code).upper().replace(".", "_")
        return os.path.join(self._cache_dir, f"baostock_raw_v2_{safe_code}_{interval}.csv")

    @staticmethod
    def _parse_minute_time(values):
        """BaoStock minute `time` is YYYYMMDDHHMMSSfff."""
        return pd.to_datetime(values, format="%Y%m%d%H%M%S%f", errors="coerce")

    def _query_rows(self, code, fields, start_date, end_date, frequency):
        """Consume the BaoStock cursor before closing its login session."""
        lg = bs.login()
        if lg.error_code != "0":
            self.last_error = f"baostock_login_failed code={code} err={lg.error_msg}"
            return None, []
        try:
            rs = bs.query_history_k_data_plus(
                code, fields, start_date=start_date, end_date=end_date,
                frequency=frequency, adjustflag="3"
            )
            if rs.error_code != "0":
                self.last_error = f"query_failed code={code} err={rs.error_msg}"
                return None, []
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code != "0":
                self.last_error = f"query_failed code={code} err={rs.error_msg}"
                return None, []
            return rs.fields, rows
        finally:
            bs.logout()

    def _normalize_minutes_df(self, df):
        if df is None or df.empty:
            return pd.DataFrame()
        work = df.copy()
        if "code" not in work.columns:
            return pd.DataFrame()
        work = validate_market_data(work, sort=True)
        return work[["code", "dt", "open", "high", "low", "close", "vol", "amount"]]

    def _load_cached_minute_data(self, code, start_time, end_time):
        if not self._cache_enabled:
            return pd.DataFrame(), False
        path = self._cache_file_path(code, "5min")
        if not os.path.exists(path):
            return pd.DataFrame(), False
        try:
            df = pd.read_csv(path)
            df = self._normalize_minutes_df(df)
            if df.empty:
                return pd.DataFrame(), False
            df_range = df[(df["dt"] >= start_time) & (df["dt"] <= end_time)].copy()
            full_coverage = minute_cache_covers_range(
                df["dt"], start_time, end_time, interval_minutes=5
            )
            return df_range, bool(full_coverage)
        except Exception:
            return pd.DataFrame(), False

    def _save_minute_cache(self, code, df):
        if not self._cache_enabled or df is None or df.empty:
            return
        path = self._cache_file_path(code, "5min")
        try:
            df_save = self._normalize_minutes_df(df)
            if df_save.empty:
                return
            if os.path.exists(path):
                try:
                    old_df = self._normalize_minutes_df(pd.read_csv(path))
                except Exception:
                    old_df = pd.DataFrame()
                if not old_df.empty:
                    df_save = self._normalize_minutes_df(
                        merge_market_data_frames(old_df, df_save)
                    )
            df_save.to_csv(path, index=False, encoding="utf-8")
        except Exception:
            return

    # ------------------------------------------------------------------ #
    #  核心接口
    # ------------------------------------------------------------------ #
    def fetch_minute_data(self, code, start_time, end_time):
        """The provider has no native 1-minute bars."""
        self.last_error = (
            "baostock_1min_unsupported: use 5min or longer, "
            "or choose a provider with native 1min data"
        )
        return pd.DataFrame()

    def _fetch_5min_data(self, code, start_time, end_time):
        """
        获取历史 5 分钟 K 线数据
        """
        cached_df, cache_hit = self._load_cached_minute_data(code, start_time, end_time)
        if cache_hit:
            return cached_df

        fetch_start = start_time

        ba_code = self._to_baostock_code(code)
        start_str = fetch_start.strftime("%Y-%m-%d")
        end_str = end_time.strftime("%Y-%m-%d")

        try:
            fields, rows = self._query_rows(
                ba_code, "date,time,code,open,high,low,close,volume,amount",
                start_str, end_str, self._MINUTE_FREQ
            )
            if fields is None:
                return cached_df if not cached_df.empty else pd.DataFrame()
            if not rows:
                self.last_error = f"no_data code={ba_code} range={start_str}->{end_str}"
                return cached_df if not cached_df.empty else pd.DataFrame()

            df = pd.DataFrame(rows, columns=fields)
            df["dt"] = self._parse_minute_time(df["time"])
            df = df.rename(columns={"volume": "vol"})
            df["code"] = code
            df = self._normalize_minutes_df(df[["code", "dt", "open", "high", "low", "close", "vol", "amount"]])
            df = df[(df["dt"] >= start_time) & (df["dt"] <= end_time)]
            if not cached_df.empty:
                df = self._normalize_minutes_df(merge_market_data_frames(cached_df, df))

            self._save_minute_cache(code, df)
            self.last_error = ""
            return df

        except Exception as e:
            self.last_error = f"fetch_5min_data_failed code={ba_code} range={start_str}->{end_str} err={e}"
            print(f"Error fetching Baostock history: {e}")
            return cached_df if not cached_df.empty else pd.DataFrame()

    def fetch_daily_data(self, code, start_time, end_time):
        """Fetch unadjusted daily bars for the backtest interface."""
        ba_code = self._to_baostock_code(code)
        start_str = start_time.strftime("%Y-%m-%d")
        end_str = end_time.strftime("%Y-%m-%d")
        try:
            fields, rows = self._query_rows(
                ba_code, "date,code,open,high,low,close,volume,amount",
                start_str, end_str, "d"
            )
            if fields is None:
                return pd.DataFrame()
            if not rows:
                self.last_error = f"no_data code={ba_code} range={start_str}->{end_str}"
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=fields).rename(
                columns={"date": "dt", "volume": "vol"}
            )
            df["code"] = code
            df = self._normalize_minutes_df(df)
            self.last_error = ""
            return df
        except Exception as e:
            self.last_error = f"fetch_daily_data_failed code={ba_code} range={start_str}->{end_str} err={e}"
            return pd.DataFrame()

    def fetch_kline_data(self, code, start_time, end_time, interval="1min"):
        """Return BaoStock bars only for a supported native or resampled interval."""
        tf = str(interval or "1min")
        if tf == "D":
            return self.fetch_daily_data(code, start_time, end_time)
        if tf == "1min":
            return self.fetch_minute_data(code, start_time, end_time)
        if tf == "5min":
            return self._fetch_5min_data(code, start_time, end_time)
        frequencies = {"15min": "15", "30min": "30", "60min": "60"}
        if tf not in frequencies:
            df_5m = self._fetch_5min_data(code, start_time, end_time)
            return Indicators.resample(df_5m, tf) if not df_5m.empty else pd.DataFrame()

        ba_code = self._to_baostock_code(code)
        start_str = start_time.strftime("%Y-%m-%d")
        end_str = end_time.strftime("%Y-%m-%d")
        try:
            fields, rows = self._query_rows(
                ba_code, "date,time,code,open,high,low,close,volume,amount",
                start_str, end_str, frequencies[tf]
            )
            if fields is None:
                return pd.DataFrame()
            if not rows:
                self.last_error = f"no_data code={ba_code} range={start_str}->{end_str} interval={tf}"
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=fields)
            df["dt"] = self._parse_minute_time(df["time"])
            df = df.rename(columns={"volume": "vol"})
            df["code"] = code
            df = self._normalize_minutes_df(df)
            df = df[(df["dt"] >= start_time) & (df["dt"] <= end_time)]
            self.last_error = ""
            return df
        except Exception as e:
            self.last_error = f"fetch_kline_data_failed code={ba_code} interval={tf} err={e}"
            return pd.DataFrame()

    def get_latest_bar(self, code):
        """
        获取最新行情
        Baostock 不提供实时行情，返回最近一个交易日的最后一根 5 分钟 K 线
        """
        # 请求今天及昨天的数据，以防非交易时段
        ba_code = self._to_baostock_code(code)
        start_str = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        end_str = datetime.now().strftime("%Y-%m-%d")

        try:
            fields, rows = self._query_rows(
                ba_code, "date,time,code,open,high,low,close,volume,amount",
                start_str, end_str, self._MINUTE_FREQ
            )
            if fields is None:
                return None
            if not rows:
                self.last_error = f"no_data code={ba_code}"
                return None

            df = pd.DataFrame(rows, columns=fields)
            df["dt"] = self._parse_minute_time(df["time"])
            df = df.rename(columns={"volume": "vol"})
            df["code"] = code
            df = self._normalize_minutes_df(df)

            row = df.iloc[-1]
            dt = row["dt"]
            return {
                "code": code,
                "dt": dt,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "vol": float(row["vol"]),
                "amount": float(row["amount"])
            }

        except Exception as e:
            self.last_error = f"get_latest_bar_failed code={ba_code} err={e}"
            print(f"Error fetching Baostock latest bar: {e}")
            return None

    def check_connectivity(self, code="sh.600000"):
        """连通性检查"""
        try:
            lg = bs.login()
            if lg.error_code != "0":
                return False, lg.error_msg
            bs.logout()
            self.last_error = ""
            return True, "ok"
        except Exception as e:
            return False, str(e)
