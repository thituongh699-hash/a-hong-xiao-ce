const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, '..', '..');
const standalone = fs.readFileSync(path.join(root, 'backtest_report.html'), 'utf8');
const dashboard = fs.readFileSync(path.join(root, 'dashboard.html'), 'utf8');

function section(source, startMarker, endMarker) {
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start + startMarker.length);
    assert.ok(start >= 0 && end > start, `${startMarker} section exists`);
    return source.slice(start, end);
}

function standaloneHarness() {
    const wrap = { innerHTML: '', children: [], appendChild(child) { this.children.push(child); } };
    const context = vm.createContext({
        strategyReportMap: {}, STRATEGY_NAME_MAP: {}, STRATEGY_PERIOD_MAP: {},
        document: {
            getElementById(id) { return id === 'strategy-report-list' ? wrap : null; },
            createElement() { return { id: '', className: '', innerHTML: '' }; }
        },
        cleanupReportRadarCharts() {}, renderReportRadarChart() {}, Chart: null, window: {},
        WorkspaceTheme: { getColors() { return {}; } }
    });
    const code = section(standalone, '        function getStrategyExecutionState(', '        function openTradeDetails(');
    vm.runInContext(code, context);
    context.renderReportRadarChart = () => {};
    return { context, wrap };
}

function dashboardHarness() {
    const wrap = { innerHTML: '', children: [], appendChild(child) { this.children.push(child); } };
    const context = vm.createContext({
        strategyReportsBuffer: [], STRATEGY_GUIDE: { all: { trigger: '--' } },
        toZhTimeframe(value) { return value; },
        document: {
            getElementById(id) { return id === 'bt-strategy-reports' ? wrap : null; },
            createElement() { return { className: '', innerHTML: '' }; }
        }
    });
    const code = section(dashboard, '        function getBacktestExecutionState(', '        function closeBacktest(');
    vm.runInContext(code, context);
    return { context, wrap };
}

const buyOnly = {
    strategy_id: 's', start_date: '2026-01-01', end_date: '2026-01-31',
    init_capital: 100000, end_capital: 99995, total_return: -0.00005, annual_return: -0.001,
    max_drawdown: -0.125, trade_count: 0, execution_count: 1, buy_count: 1, sell_count: 0,
    win_num: 0, loss_num: 0, breakeven_num: 0, win_rate: 0, avg_win: 0, avg_loss: 0,
    profit_ratio: null, trade_details: [{direction: 'BUY'}], scorecard: {}
};

const settlement = {
    state: 'open', cash: 79995, holdings_value: 21000, realized_pnl: 100,
    unrealized_pnl: 900, fees_paid: 5, open_position_count: 1, open_quantity: 100, reconciliation_delta: 0
};

test('standalone distinguishes no execution from a buy-only open position', () => {
    const h = standaloneHarness();
    assert.equal(h.context.getStrategyExecutionState({trade_count: 0, execution_count: 0}).label, '无成交');
    assert.equal(h.context.getStrategyExecutionState(buyOnly).label, '有未平仓 · 暂无已平仓');
    h.context.renderStrategyReports([{...buyOnly, settlement}], null);
    const html = h.wrap.children[0].innerHTML;
    assert.match(html, /有未平仓/);
    assert.doesNotMatch(html, /本次无成交/);
    assert.match(html, /最大回撤：12\.50%/);
    assert.match(html, /盈利因子：--/);
    assert.match(html, /期末结算对账/);
    assert.match(html, /总费用：5\.00 元/);
    assert.match(html, /不重复从已实现与未实现盈亏中扣减/);
});

test('standalone preserves a zero profit factor and exposes missing settlement', () => {
    const h = standaloneHarness();
    h.context.renderStrategyReports([{
        ...buyOnly, trade_count: 1, execution_count: 2, buy_count: 1, sell_count: 1,
        win_num: 0, loss_num: 1, profit_ratio: 0
    }], null);
    const html = h.wrap.children[0].innerHTML;
    assert.match(html, /盈利因子：0\.00/);
    assert.match(html, /期末结算：<\/span>未提供/);
});

test('dashboard renders the same execution, drawdown, factor, and settlement contract', () => {
    const h = dashboardHarness();
    h.context.strategyReportsBuffer.push({...buyOnly, settlement});
    h.context.renderStrategyReports();
    const html = h.wrap.children[0].innerHTML;
    assert.match(html, /有未平仓 · 暂无已平仓/);
    assert.doesNotMatch(html, />无成交</);
    assert.match(html, /最大回撤：12\.50%/);
    assert.match(html, /盈利因子：--/);
    assert.match(html, /未平仓标的数：1/);
    assert.match(html, /未平仓股数：100/);
    assert.match(html, /对账差额：0\.00 元/);
});

test('both real HTML pages use the explicit execution and settlement fields', () => {
    for (const source of [standalone, dashboard]) {
        assert.match(source, /execution_count/);
        assert.match(source, /buy_count/);
        assert.match(source, /sell_count/);
        assert.match(source, /breakeven_num/);
        assert.match(source, /settlement\.reconciliation_delta/);
        assert.match(source, /Math\.abs\(Number\(r\.max_drawdown/);
        assert.doesNotMatch(source, />盈亏比：/);
    }
});
