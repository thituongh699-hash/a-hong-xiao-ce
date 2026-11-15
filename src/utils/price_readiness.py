"""Read-only, per-source checks for the signal/execution price boundary."""
import hashlib
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.utils.market_data_validation import MarketDataValidationError, validate_market_data
from src.utils.price_adjustment import (
    AdjustmentPlan, PriceAdjustmentError, assert_raw_prices, load_adjustment_factors,
)


def check_price_readiness(source, stock_code, config, provider_factory, *,
                          start_date=None, end_date=None, require_minute=False):
    mode = str(config.get('data_provider.adjustment_mode', 'none') or 'none').lower()
    # Hash the actual draft (including credential changes); never return its values.
    provider_config = config.get('data_provider', {})
    identity = {'source': source, 'stock_code': stock_code, 'provider': provider_config,
                'start_date': start_date, 'end_date': end_date, 'require_minute': require_minute}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:24]
    checks = []
    policy = {
        'requested_mode': mode, 'signal_price_mode': 'none' if mode == 'none' else 'unavailable', 'execution_price_mode': 'none',
        'factor_source': 'baostock', 'factor_available': False, 'raw_data_verified': False,
        'causal_adjustment': False, 'corporate_action_accounting': 'unsupported',
        'limitations': ['分红与送转未自动入账。持仓跨除权日的复权回测会终止；不复权收益也不含公司行动收益。'],
    }
    now = datetime.now(ZoneInfo('Asia/Shanghai')).replace(tzinfo=None)
    historical = start_date is not None or end_date is not None
    response = {'source': source, 'stock_code': stock_code, 'checked_at': now.isoformat(),
                'probe_scope': 'historical' if historical else 'current',
                'fingerprint': fingerprint, 'checks': checks, 'price_policy': policy}

    def add(check_id, title, ok, severity, detail):
        checks.append({'id': check_id, 'title': title, 'ok': bool(ok), 'severity': severity, 'detail': detail})

    def finish():
        add('corporate_actions', '分红及送转账户处理', False, 'warning',
            '未实现分红与送转入账。复权模式下持仓跨除权日会终止回测；不复权收益也不包含公司行动收益。')
        ok = not any(not item['ok'] and item['severity'] == 'error' for item in checks)
        response.update(status='success' if ok else 'error', ok=ok)
        return response

    try:
        end = now
        start = end - timedelta(days=120)
        if historical:
            if require_minute or not start_date or not end_date:
                raise PriceAdjustmentError('历史检测需同时提供开始和结束日期，且不能用于实时分钟验收。')
            try:
                start = datetime.strptime(start_date, '%Y-%m-%d')
                end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1) - timedelta(microseconds=1)
            except (TypeError, ValueError):
                raise PriceAdjustmentError('历史检测日期必须为 YYYY-MM-DD。') from None
            if start > end or end.date() > now.date():
                raise PriceAdjustmentError('历史检测日期区间无效或结束日期在未来。')
            response.update(requested_start=start_date, requested_end=end_date)
        provider = provider_factory(source, config)
        if hasattr(provider, '_cache_enabled'):
            provider._cache_enabled = False  # Probe fresh source data, not old cache coverage.
        daily = provider.fetch_kline_data(stock_code, start, end, 'D')
        if daily is None or daily.empty:
            raise PriceAdjustmentError('未取得请求历史区间的日线，无法验证价格口径。' if historical
                                       else '未取得最近 120 天日线，无法验证价格口径。')
        try:
            daily = validate_market_data(daily, sort=True)
        except MarketDataValidationError as exc:
            raise PriceAdjustmentError(f'日线数据校验失败：{exc}') from exc
        if historical:
            timestamps = pd.to_datetime(daily['dt'])
            if timestamps.dt.tz is not None:
                timestamps = timestamps.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            daily = daily.loc[(timestamps >= start) & (timestamps <= end)]
            if daily.empty:
                raise PriceAdjustmentError('请求历史区间没有有效日线，无法验证价格口径。')
        add('daily', '日线数据', True, 'info', f'取得 {len(daily)} 根日线。')
        latest = pd.to_datetime(daily['dt']).max()
        response['latest_daily_date'] = latest.isoformat()
        if not historical:
            add('daily_freshness', '日线更新时间', 0 <= (now.date() - latest.date()).days <= 7,
                'error' if require_minute else 'warning',
                f'最新日线 {latest.date()}；请确认是否覆盖所需回测日期。')
        assert_raw_prices(provider, source, daily, config)
        policy['raw_data_verified'] = True
        policy['raw_price_evidence'] = 'provider_contract' if source in {'tdx', 'tushare', 'akshare', 'baostock', 'jqdata'} else 'storage_declaration'
        detail = ('已确认数据源原价接口；成交和估值使用不复权价格。' if policy['raw_price_evidence'] == 'provider_contract'
                  else '按用户声明的不复权存储口径使用；未自动识别或外部比对。')
        add('raw_prices', '成交价格口径', True, 'info', detail)
        try:
            factors = load_adjustment_factors(stock_code, start, end, config, timeout=18)
            plan = AdjustmentPlan(daily, factors, mode)
            limitations = policy['limitations']
            policy.update(plan.metadata())
            policy['raw_data_verified'] = True
            policy['raw_price_evidence'] = 'provider_contract' if source in {'tdx', 'tushare', 'akshare', 'baostock', 'jqdata'} else 'storage_declaration'
            policy['limitations'] = list(dict.fromkeys(policy.get('limitations', []) + limitations))
            add('factors', '统一复权因子', True, 'info', '后复权累计因子可用；仅按当时已生效的除权日期计算。')
        except PriceAdjustmentError as exc:
            add('factors', '统一复权因子', False, 'error' if mode != 'none' else 'warning', str(exc))
        if historical:
            # Current minute freshness cannot disqualify a historical backtest.
            # Its execution engine still validates the actual requested bars.
            return finish()
        try:
            # TDX is a local-file source here. Inspect the actual file's newest
            # record even when it falls outside the requested recent window;
            # an unrelated network snapshot cannot prove the file was updated.
            local_reader = getattr(provider, '_reader_minute', None)
            if source == 'tdx' and callable(local_reader):
                minute = provider._normalize_ohlcv_df(local_reader(provider._raw_symbol(stock_code)), code=stock_code)
                scope = '通达信本地分钟文件'
            else:
                interval = getattr(provider, 'minimum_interval', '1min')
                minute = provider.fetch_kline_data(stock_code, end - timedelta(days=30), end, interval)
                scope = f'{interval} 数据'
            if minute is not None and not minute.empty:
                try:
                    minute = validate_market_data(minute, sort=True)
                except MarketDataValidationError as exc:
                    add('minute', '分钟数据', False, 'error', f'{scope}校验失败：{exc}')
                    return finish()
                minute_date = pd.to_datetime(minute['dt']).max()
                response['latest_minute_date'] = minute_date.isoformat()
                # A week-old minute file is useful for historical analysis,
                # but cannot qualify approximate realtime startup. Also require
                # it to cover the source's newest daily session.
                age = (now.date() - minute_date.date()).days
                fresh = (0 <= age <= 3 and minute_date.date() >= latest.date()
                         and minute_date <= pd.Timestamp(now))
                add('minute', '分钟数据', fresh, 'error' if require_minute else 'warning',
                    f'{scope}最新记录 {minute_date}；近似实时要求不早于最新日线且在最近 3 个自然日内。'
                    '日线与分钟数据需分别更新，长假期间需在交易恢复后重新检查。')
                if getattr(provider, 'minimum_interval', '1min') != '1min':
                    add('minute_resolution', '分钟周期', False, 'warning', '该数据源最小周期为 5 分钟，不支持 1 分钟策略。')
            else:
                add('minute', '分钟数据', False, 'error' if require_minute else 'warning', '最近 30 天未取得分钟数据；分钟策略不可据此判定就绪。')
        except Exception:
            add('minute', '分钟数据', False, 'error' if require_minute else 'warning', '分钟数据不可用，请检查该数据源的分钟权限或本地缓存。')
    except Exception as exc:
        # Do not echo provider exceptions that may contain URLs or credentials.
        detail = str(exc) if isinstance(exc, PriceAdjustmentError) else '数据读取失败，请检查数据源配置和网络或本地目录。'
        add('prices', '价格数据检测', False, 'error', detail)
    return finish()
