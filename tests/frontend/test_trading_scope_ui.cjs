const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', '..', 'dashboard.html'), 'utf8');
const scopeStart = html.indexOf("        const TRADING_SCOPE_STORAGE_KEY = 'ahxc-backtest-trading-scope-v1';");
const scopeEnd = html.indexOf('        function openConfigCenter()', scopeStart);
assert.ok(scopeStart >= 0 && scopeEnd > scopeStart, 'trading scope functions are present');
const scopeCode = html.slice(scopeStart, scopeEnd);

function element(initial = {}) {
    const classes = new Set(initial.classes || []);
    return {
        checked: false,
        value: '',
        textContent: '',
        innerText: '',
        innerHTML: '',
        className: '',
        children: [],
        ...initial,
        classList: {
            add(...names) { names.forEach(name => classes.add(name)); },
            remove(...names) { names.forEach(name => classes.delete(name)); },
            contains(name) { return classes.has(name); },
            toggle(name, force) {
                if (force === undefined ? !classes.has(name) : force) classes.add(name);
                else classes.delete(name);
            }
        },
        setAttribute(name, value) { this[name] = String(value); },
        replaceChildren() { this.children = []; },
        appendChild(child) { this.children.push(child); }
    };
}

const checkboxIds = [
    'trading-scope-board-main_board', 'trading-scope-board-chinext',
    'trading-scope-board-star', 'trading-scope-board-beijing',
    'trading-scope-type-normal', 'trading-scope-type-st', 'trading-scope-type-delisting'
];

function harness(stored = null) {
    const elements = new Map(checkboxIds.map(id => [id, element()]));
    elements.set('trading-scope-panel', element({ classes: ['hidden'] }));
    elements.set('trading-scope-toggle', element());
    elements.set('trading-scope-summary', element());
    elements.set('trading-scope-error', element({ classes: ['hidden'] }));
    const storage = new Map();
    if (stored !== null) storage.set('ahxc-backtest-trading-scope-v1', stored);
    const requests = [];
    const logs = [];
    const context = vm.createContext({
        AbortController,
        console,
        normalizeCode(value) { return String(value || '').trim().toUpperCase(); },
        logMessage(module, message, level) { logs.push({ module, message, level }); },
        localStorage: {
            getItem(key) { return storage.has(key) ? storage.get(key) : null; },
            setItem(key, value) { storage.set(key, String(value)); }
        },
        document: {
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, element());
                return elements.get(id);
            },
            createElement() { return element(); }
        },
        setTimeout,
        clearTimeout,
        fetch(url, options = {}) {
            requests.push({ url, options });
            if (url === '/api/backtest/trading_scope/preview') {
                return Promise.resolve({
                    ok: true,
                    json: async () => ({
                        status: 'success',
                        eligible_codes: ['920001.BJ'],
                        excluded: []
                    })
                });
            }
            if (url === '/api/config/test_price_readiness') {
                return Promise.resolve({
                    ok: true,
                    json: async () => ({
                        status: 'success', ok: true, checks: [],
                        price_policy: { requested_mode: 'none', signal_price_mode: 'none', execution_price_mode: 'none' }
                    })
                });
            }
            if (url === '/api/control/start_backtest') {
                return Promise.resolve({ ok: true, json: async () => ({ status: 'success', report_id: 'scope-test' }) });
            }
            throw new Error(`unexpected request: ${url}`);
        }
    });
    vm.runInContext(scopeCode, context);
    vm.runInContext(actualFunction('startBacktestByApi'), context);
    return { context, elements, storage, requests, logs };
}

function plain(value) {
    return JSON.parse(JSON.stringify(value));
}

function actualFunction(name) {
    const match = html.match(new RegExp('        (?:async )?function ' + name + '\\('));
    assert.ok(match, name + ' exists');
    const from = match.index;
    const next = html.slice(from + match[0].length).search(/\n        (?:async )?function /);
    return html.slice(from, from + match[0].length + next);
}

test('uses semantic checkbox groups and serializes the checked values', () => {
    assert.match(html, /<fieldset[^>]*>[\s\S]*?<legend[^>]*>上市板块/);
    assert.match(html, /<input id="trading-scope-board-main_board"[^>]*type="checkbox"[^>]*checked/);
    assert.match(html, /<label[^>]*>[\s\S]*?trading-scope-type-delisting/);
    const h = harness();
    h.context.writeTradingScopeSelection({ boards: ['main_board', 'beijing'], stock_types: ['normal', 'st'] });
    assert.deepEqual(plain(h.context.readTradingScopeSelection()), {
        boards: ['main_board', 'beijing'], stock_types: ['normal', 'st']
    });
    h.context.persistTradingScopePreference();
    assert.deepEqual(JSON.parse(h.storage.get('ahxc-backtest-trading-scope-v1')), {
        boards: ['main_board', 'beijing'], stock_types: ['normal', 'st']
    });
});

test('restores valid storage and falls back from empty or illegal history', () => {
    const valid = harness(JSON.stringify({ boards: ['star'], stock_types: ['st'] }));
    assert.deepEqual(plain(valid.context.restoreTradingScopePreference()), { boards: ['star'], stock_types: ['st'] });
    const empty = harness(JSON.stringify({ boards: [], stock_types: ['normal'] }));
    assert.deepEqual(plain(empty.context.restoreTradingScopePreference()), {
        boards: ['main_board', 'chinext', 'star', 'beijing'], stock_types: ['normal']
    });
    const illegal = harness(JSON.stringify({ boards: ['main_board', 'unknown'], stock_types: ['normal'] }));
    assert.deepEqual(plain(illegal.context.restoreTradingScopePreference()), {
        boards: ['main_board', 'chinext', 'star', 'beijing'], stock_types: ['normal']
    });
});

test('an all-empty selection is blocked with a visible message', () => {
    const h = harness();
    assert.equal(h.context.requireTradingScopeSelection(), null);
    assert.match(h.elements.get('trading-scope-error').textContent, /至少选择 1 项/);
    assert.equal(h.elements.get('trading-scope-panel').classList.contains('hidden'), false);
    assert.match(h.logs[0].message, /交易范围无效/);
});

test('single backtest previews scope before readiness and submits the frozen scope', async () => {
    const h = harness();
    h.context.writeTradingScopeSelection({ boards: ['beijing'], stock_types: ['normal'] });
    const result = await h.context.startBacktestByApi({ stock_code: '920001.BJ', start: '2026-01-01', end: '2026-06-01' });
    assert.equal(result.report_id, 'scope-test');
    assert.deepEqual(h.requests.map(item => item.url), [
        '/api/backtest/trading_scope/preview',
        '/api/config/test_price_readiness',
        '/api/control/start_backtest'
    ]);
    const previewBody = JSON.parse(h.requests[0].options.body);
    const submitBody = JSON.parse(h.requests[2].options.body);
    assert.deepEqual(previewBody.trading_scope, { boards: ['beijing'], stock_types: ['normal'] });
    assert.deepEqual(submitBody.trading_scope, { boards: ['beijing'], stock_types: ['normal'] });
});

test('screener batch filters with preview before checking prices and submits only eligible codes', async () => {
    const events = [];
    let submitted = null;
    const context = vm.createContext({
        document: { getElementById(id) { return element({ value: id.endsWith('start') ? '2026-01-01' : (id.endsWith('end') ? '2026-06-01' : '1000000') }); } },
        getSelectedCodes: () => ['000001.SZ', '920001.BJ'],
        _resolveDisplayStockCode: row => row.code,
        closeScreenerModal() {}, updateScreenerBacktestButtons() {}, openBatchDashboardModal() {},
        showScreenerToast() {}, confirm: () => true, setTimeout() {},
        requireTradingScopeSelection: () => ({ boards: ['beijing'], stock_types: ['normal'] }),
        tradingScopeSummary: () => '1个板块 · 普通股',
        reportTradingScopeExclusions() {},
        previewTradingScopeForCodes: async () => {
            events.push('preview');
            return { eligible_codes: ['920001.BJ'], excluded: [{ stock_code: '000001.SZ', reason: '板块不匹配' }] };
        },
        checkPriceReadinessForTargets: async codes => { events.push('readiness:' + codes.join(',')); return true; },
        fetch: async (url, options) => {
            events.push('submit');
            submitted = JSON.parse(options.body);
            return { json: async () => ({ status: 'success' }) };
        }
    });
    vm.runInContext(`let screenerResults = []; let screenerAiSelectedStrategyIds = []; ${actualFunction('startScreenerBatchBacktest')}`, context);
    await context.startScreenerBatchBacktest();
    assert.deepEqual(events, ['preview', 'readiness:920001.BJ', 'submit']);
    assert.deepEqual(submitted.codes, ['920001.BJ']);
    assert.deepEqual(submitted.trading_scope, { boards: ['beijing'], stock_types: ['normal'] });
});

test('AI batch and generic batch start payloads carry the current scope', () => {
    const aiBatch = actualFunction('runAIScreenerAndStartBatchBacktest');
    assert.ok(aiBatch.indexOf('previewTradingScopeForCodes') < aiBatch.indexOf('checkPriceReadinessForTargets'));
    assert.match(aiBatch, /trading_scope:\s*tradingScope/);
    for (const name of ['startBatchRunFromDashboard', 'startBatchAiAnalyzeFromDashboard', 'startBatchRunFromUi', 'startBatchAiAnalyzeFromUi']) {
        assert.match(actualFunction(name), /payload\.trading_scope = tradingScope/, name);
    }
    assert.match(html, /if \(v\.startsWith\('920'\)\) return `\$\{v\}\.BJ`/);
});
