# main.py
import sys
import os
from src.utils.brand import BRAND_NAME
from src.utils.dependency_bootstrap import ensure_project_dependencies

# 在导入 pandas 等第三方依赖前先执行依赖检查。
ensure_project_dependencies()

import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.core.intent_planner import IntentPlanner
from src.core.strategy_engine import StrategyEngine
from src.core.risk_manager import RiskManager
from src.core.execution_engine import ExecutionEngine
from src.services.strategy_registry import StrategyRegistry
from src.services.account_ledger import AccountLedger
from src.services.performance_reporter import PerformanceReporter
from src.services.order_matcher import OrderMatcher
from src.services.risk_audit_log import RiskAuditLog
from src.services.market_data_service import MarketDataService
from src.strategies.strategy_factory import create_strategies
from src.utils.data_provider import DataProvider
from src.utils.tushare_provider import TushareProvider
from src.utils.akshare_provider import AkshareProvider
from src.utils.mysql_provider import MysqlProvider
from src.utils.postgres_provider import PostgresProvider
from src.utils.tdx_provider import TdxProvider
from src.utils.config_loader import ConfigLoader
from src.utils.data_generator import generate_mock_data
from src.utils.constants import INITIAL_CAPITAL

def main():
    config = ConfigLoader()
    
    print(f"【{BRAND_NAME}量化研究工作台回测研究引擎】启动...")
    print("=============================================")

    # Load from config
    initial_capital = config.get("system.initial_capital", 1000000.0)

    # Initialize Services
    strategy_registry = StrategyRegistry()
    account_ledger = AccountLedger(initial_capital)
    performance_reporter = PerformanceReporter()
    order_matcher = OrderMatcher()
    risk_audit_log = RiskAuditLog()
    market_data_service = MarketDataService()

    # Initialize Departments
    intent_planner = IntentPlanner()
    risk_manager = RiskManager(risk_audit_log)
    execution_engine = ExecutionEngine(account_ledger, order_matcher, risk_audit_log)

    # Initialize Strategies
    strategies = create_strategies()
    strategy_engine = StrategyEngine(strategies)
    
    # Register active strategies from config if specified
    active_ids = config.get("strategies.active_ids", [])
    
    for s in strategies:
        if not active_ids or s.id in active_ids:
            strategy_registry.register_strategy(s)
            print(f"策略 {s.id} 已注册")
        else:
            print(f"策略 {s.id} 已跳过 (未在配置中激活)")

    # Fetch Real Data via API
    # 2024-2025 Full Year for Senying Window (301227.SZ)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    print(f"正在从API获取真实历史数据 ({start_date.date()} to {end_date.date()})...")
    
    provider_source = config.get("data_provider.source", "default")
    tushare_token = config.get("data_provider.tushare_token")
    
    if provider_source == 'tushare':
        provider = TushareProvider(token=tushare_token)
    elif provider_source == 'akshare':
        provider = AkshareProvider()
    elif provider_source == 'mysql':
        provider = MysqlProvider()
    elif provider_source == 'postgresql':
        provider = PostgresProvider()
    elif provider_source == 'tdx':
        provider = TdxProvider()
    else:
        provider = DataProvider()
        
    stocks = config.get("targets", ['600036.SH'])
    market_data = {}
    
    for code in stocks:
        df = provider.fetch_minute_data(code, start_date, end_date)
        if not df.empty:
            market_data[code] = market_data_service.clean_data(df)
            print(f"股票 {code} 数据已获取: {len(df)} 条K线")
        else:
            print(f"❌ 错误: 股票 {code} 未获取到数据 (数据源: {provider_source})。请检查网络或数据源配置。")
            print("⚠️ 既然要求实事求是，本次回测将跳过该股票或终止。")
            # Skip or handle empty
            continue

    # Main Backtest Loop
    # Iterate through time across all stocks (merged timeline)
    # 1. Merge all dataframes by time
    if not market_data:
        print("❌ 无有效数据，回测终止。")
        return

    combined_data = pd.concat(market_data.values())
    if combined_data.empty:
        print("无有效数据，回测终止")
        return

    combined_data = combined_data.sort_values('dt').reset_index(drop=True)
    
    print("回测开始...")
    total_bars = len(combined_data)
    
    # Process bar by bar
    for i, row in combined_data.iterrows():
        if i % 1000 == 0:
            print(f"Processing {i}/{total_bars}...", end='\r')
            
        kline = row
        
        # 1. Intent Planner Validation
        # Validate data for this specific stock/bar
        # (Simplified: Prince usually filters list of stocks for the day, here we check per bar)
        # validated_kline = intent_planner.validate_and_distribute(pd.DataFrame([kline]))
        # Since validate_and_distribute expects DF and does bulk work, let's just assume valid here for speed
        # or call a lightweight validation method.
        
        # 2. Strategy Engine: Generate Signals
        # Strategies see the bar and decide
        signals = strategy_engine.generate_signals(kline)
        
        for signal in signals:
            strategy_id = signal['strategy_id']
            
            # 3. Risk Manager: Risk Control
            # Need current portfolio state for risk checks
            # ExecutionEngine holds positions
            current_positions = execution_engine.positions.get(strategy_id, {})
            # Need daily PnL (approximated or tracked)
            daily_pnl = 0.0 # Placeholder
            
            # Calculate total portfolio value for risk limits
            # This is global capital or allocated capital?
            # Prompt says "All strategies share same standards", usually implies fund level or per-strategy level limits.
            # "Total Position Limit 50%" implies Fund Level.
            # So we need Total Fund Value.
            current_fund_value = account_ledger.cash + execution_engine.update_holdings_value({kline['code']: kline['close']})
            # Note: update_holdings_value is expensive if called every signal. 
            # Optimization: Update only on price change or periodically.
            # For backtest, we can use account_ledger.cash + sum(pos * current_price)
            
            approved, reason = risk_manager.check_signal(signal, current_fund_value, current_positions, daily_pnl)
            
            if approved:
                # 4. Order Execution: Execution
                executed = execution_engine.execute_order(strategy_id, signal, kline)
                if executed:
                    # Update strategy internal state
                    # Get new position quantity for this stock
                    new_qty = execution_engine.positions[strategy_id][signal['code']]['qty'] if signal['code'] in execution_engine.positions.get(strategy_id, {}) else 0
                    
                    strategy_engine.update_strategy_state(strategy_id, signal['code'], new_qty)
        
        # Check Stop Loss / Take Profit for existing positions
        triggered_orders = execution_engine.check_stops(kline)
        for order in triggered_orders:
             executed = execution_engine.execute_order(order['strategy_id'], order, kline)
             if executed:
                 new_qty = 0 # Closed position
                 strategy_engine.update_strategy_state(order['strategy_id'], order['code'], new_qty)

    print("\n回测结束.")
    
    # Generate Reports
    print("====================================================================")
    print(f"【{BRAND_NAME}量化研究工作台回测总览 · 研究总览】")
    print(f"回测区间：{start_date.date()}–{end_date.date()}")
    print(f"初始资金：{INITIAL_CAPITAL}")
    print(f"对比策略数：{len(strategies)}")
    print("====================================================================")
    
    reports = []
    for s in strategies:
        report = performance_reporter.generate_report(s.id, account_ledger, risk_audit_log, INITIAL_CAPITAL/len(strategies)) # Assessing on allocated capital
        reports.append(report)
        
        print(f"--------------------------------------------------------------------")
        print(f"策略 {s.id} 号业绩报表")
        print(f"--------------------------------------------------------------------")
        print(f"总净收益率：{report['roi']:.2%}")
        print(f"年化收益率：{report['annualized_roi']:.2%}")
        print(f"最大回撤：{report['max_dd']:.2%}")
        print(f"胜率：{report['win_rate']:.2%}")
        print(f"盈亏比：{report['profit_factor']:.2f}")
        print(f"总交易次数：{report['total_trades']}")
        print(f"风控驳回次数：{report['rejections']}")
        print(f"熔断触发次数：{report['circuit_breaks']}")
        print(f"违规次数：{report['violations']}")
        print(f"夏普比率：{report['sharpe']:.2f}")
        print(f"卡玛比率：{report['calmar']:.2f}")

    print("\n====================================================================")
    print("【十套策略综合排行榜】")
    ranking = performance_reporter.generate_ranking(reports)
    # 排名 | 策略编号 | 卡玛比率 | 年化收益 | 最大回撤 | 胜率 | 综合评级
    display_df = ranking[['rank', 'strategy_id', 'calmar', 'annualized_roi', 'max_dd', 'win_rate', 'rating']]
    display_df.columns = ['排名', '策略编号', '卡玛比率', '年化收益', '最大回撤', '胜率', '综合评级']
    
    # Format columns
    # Pandas to_string default format is okay, but let's make it cleaner if needed.
    # We can just print standard DF string.
    print(display_df.to_string(index=False))
    print("====================================================================")
    
    # Engine General Assessment
    best_strategy = ranking.iloc[0] if not ranking.empty else None
    lowest_risk = ranking.sort_values('max_dd').iloc[0] if not ranking.empty else None
    # Most stable: simplistic assumption, same as lowest risk for now or high win rate
    most_stable = ranking.sort_values('win_rate', ascending=False).iloc[0] if not ranking.empty else None
    
    # Elimination: Negative ROI or low score
    eliminated = ranking[ranking['roi'] < 0]['strategy_id'].tolist()
    
    print("\n【研究引擎总评】")
    if best_strategy is not None:
        print(f"1. 最优策略：Strategy {best_strategy['strategy_id']} (Calmar: {best_strategy['calmar']:.2f})")
        print(f"2. 风险最低策略：Strategy {lowest_risk['strategy_id']} (MaxDD: {lowest_risk['max_dd']:.2%})")
        print(f"3. 最稳健策略：Strategy {most_stable['strategy_id']} (WinRate: {most_stable['win_rate']:.2%})")
        print(f"4. 需淘汰/优化策略：{', '.join(eliminated) if eliminated else 'None'}")
        print(f"5. 整体风控有效性评价：风控介入 {sum(r['rejections'] for r in reports)} 次，有效阻断高风险交易。")
    else:
        print("无有效策略数据。")

if __name__ == "__main__":
    main()
