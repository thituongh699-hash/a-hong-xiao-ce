"""Short-lived price readiness probe. stdin and stdout carry one JSON document each."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
import sys


class RuntimeConfig(dict):
    """A config draft with the dotted get interface used by providers."""

    def get(self, key, default=None):
        value = self
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


def _bind_table_names(provider, config, prefix):
    keys = {
        "1min": "1min", "5min": "5min", "10min": "10min", "15min": "15min",
        "30min": "30min", "60min": "60min", "D": "day",
    }
    defaults = dict(getattr(provider, "_table_defaults", {}) or {})
    safe_name = getattr(provider, "_safe_table_name", None)

    def resolve(interval):
        name = str(config.get(f"data_provider.{prefix}_table_{keys.get(interval, '')}", "") or "").strip()
        if callable(safe_name):
            name = str(safe_name(name) or "").strip()
        return name or str(defaults.get(interval, "") or "")

    provider._resolve_table_name = resolve
    return provider


def build_provider(source, config):
    """Construct only the selected provider; importing server would bootstrap the app."""
    source = str(source or "default").strip().lower()
    if source == "tushare":
        from src.utils.tushare_provider import TushareProvider

        token = str(config.get("data_provider.tushare_token", "") or "").strip()
        provider = TushareProvider(token=token)
        provider._tushare_http_url = str(config.get("data_provider.tushare_api_url", "http://tushare.xyz") or "http://tushare.xyz").strip()
        provider.set_token(token)
        return provider
    if source == "akshare":
        from src.utils.akshare_provider import AkshareProvider

        return AkshareProvider()
    if source == "baostock":
        from src.utils.baostock_provider import BaostockProvider

        return BaostockProvider()
    if source == "jqdata":
        from src.utils.jqdata_provider import JqdataProvider

        return JqdataProvider(username=config.get("data_provider.jqdata_username"),
                              password=config.get("data_provider.jqdata_password"))
    if source == "mysql":
        from src.utils.mysql_provider import MysqlProvider

        provider = MysqlProvider(
            host=config.get("data_provider.mysql_host", "127.0.0.1"),
            port=config.get("data_provider.mysql_port", 3306),
            user=config.get("data_provider.mysql_user", ""),
            password=config.get("data_provider.mysql_password", ""),
            database=config.get("data_provider.mysql_database", ""),
            charset=config.get("data_provider.mysql_charset", "utf8mb4"),
        )
        provider.page_size = max(1000, int(config.get("data_provider.mysql_query_page_size", getattr(provider, "page_size", 20000)) or getattr(provider, "page_size", 20000)))
        return _bind_table_names(provider, config, "mysql")
    if source == "postgresql":
        from src.utils.postgres_provider import PostgresProvider

        provider = PostgresProvider(
            host=config.get("data_provider.postgres_host", "127.0.0.1"),
            port=config.get("data_provider.postgres_port", 5432),
            user=config.get("data_provider.postgres_user", ""),
            password=config.get("data_provider.postgres_password", ""),
            database=config.get("data_provider.postgres_database", ""),
            schema=config.get("data_provider.postgres_schema", "public"),
        )
        provider.page_size = max(1000, int(config.get("data_provider.postgres_query_page_size", getattr(provider, "page_size", 20000)) or getattr(provider, "page_size", 20000)))
        return _bind_table_names(provider, config, "postgres")
    if source == "duckdb":
        from src.utils.duckdb_provider import DuckDbProvider

        provider = DuckDbProvider(db_path=config.get("data_provider.duckdb_path", ""))
        provider.page_size = max(1000, int(config.get("data_provider.duckdb_query_page_size", getattr(provider, "page_size", 20000)) or getattr(provider, "page_size", 20000)))
        return _bind_table_names(provider, config, "duckdb")
    if source == "tdx":
        from src.utils.tdx_provider import TdxProvider

        return TdxProvider(
            host=config.get("data_provider.tdx_host", None),
            port=config.get("data_provider.tdx_port", None),
            tdxdir=config.get("data_provider.tdxdir", "") or config.get("data_provider.tdx_dir", ""),
        )
    from src.utils.data_provider import DataProvider

    return DataProvider(api_key=config.get("data_provider.default_api_key", ""),
                        base_url=config.get("data_provider.default_api_url", ""))


def run(request):
    if not isinstance(request, dict) or not isinstance(request.get("config"), dict):
        raise ValueError("invalid request")
    source = request.get("source")
    stock_code = request.get("stock_code")
    if not isinstance(source, str) or not isinstance(stock_code, str):
        raise ValueError("invalid request")
    from src.utils.price_readiness import check_price_readiness

    return check_price_readiness(source, stock_code, RuntimeConfig(request["config"]), build_provider,
                                 start_date=request.get('start_date'), end_date=request.get('end_date'),
                                 require_minute=bool(request.get('require_minute', False)))


def _redact_credentials(value, config):
    secrets = []

    def collect(node):
        if isinstance(node, dict):
            for key, child in node.items():
                if isinstance(child, str) and any(part in key.lower() for part in ("token", "password", "secret", "api_key")) and child:
                    secrets.append(child)
                else:
                    collect(child)

    collect(config)

    def clean(node):
        if isinstance(node, str):
            for secret in secrets:
                node = node.replace(secret, "[redacted]")
        elif isinstance(node, list):
            node = [clean(item) for item in node]
        elif isinstance(node, dict):
            node = {key: clean(item) for key, item in node.items()}
        return node

    return clean(value)


def main():
    try:
        request = json.load(sys.stdin)
        # Provider libraries may print login banners. Only the final JSON uses stdout.
        with contextlib.redirect_stdout(sys.stderr):
            result = run(request)
        if not isinstance(result, dict):
            raise ValueError("invalid result")
        output = json.dumps(_redact_credentials(result, request["config"]), ensure_ascii=False, allow_nan=False)
        sys.stdout.write(output + "\n")
        return 0
    except Exception:
        # Never put provider exception text or a supplied credential on stdout.
        sys.stdout.write('{"status":"error","ok":false,"checks":[]}\n')
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
