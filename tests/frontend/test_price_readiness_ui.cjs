const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', '..', 'dashboard.html'), 'utf8');
const start = html.indexOf("        const TRADING_SCOPE_STORAGE_KEY = 'ahxc-backtest-trading-scope-v1';");
const end = html.indexOf('        function openConfigCenter()', start);
assert.ok(start >= 0 && end > start, 'price readiness functions are present');
const code = html.slice(start, end);

function node() {
    return {
        textContent: '', className: '', innerHTML: '', children: [],
        classList: {add() {}, remove() {}},
        replaceChildren() { this.children = []; },
        appendChild(child) { this.children.push(child); }
    };
}

function harness() {
    const elements = new Map();
    for (const id of [
        'trading-scope-board-main_board', 'trading-scope-board-chinext',
        'trading-scope-board-star', 'trading-scope-board-beijing', 'trading-scope-type-normal'
    ]) {
        const el = node();
        el.checked = true;
        elements.set(id, el);
    }
    const timers = new Map();
    const requests = [];
    let nextTimer = 1;
    let draft = { data_provider: { source: 'tushare', adjustment_mode: 'qfq', tushare_token: 'secret-token' } };
    const context = vm.createContext({
        AbortController,
        logMessage() {},
        escapeOnboardingHtml(value) { return String(value); },
        document: {
            getElementById(id) { if (!elements.has(id)) elements.set(id, node()); return elements.get(id); },
            createElement: node
        },
        getConfigDraftStrict() { return draft; },
        setTimeout(fn, delay) { const id = nextTimer++; timers.set(id, { fn, delay }); return id; },
        clearTimeout(id) { timers.delete(id); },
        fetch(url, options) {
            return new Promise((resolve, reject) => {
                options.signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
                requests.push({ url, options, resolve, reject });
            });
        }
    });
    vm.runInContext(code, context);
    return {
        context, elements, requests,
        setDraft(value) { draft = value; },
        fire(delay) {
            for (const [id, timer] of [...timers]) {
                if (timer.delay === delay) { timers.delete(id); timer.fn(); }
            }
        }
    };
}

function allowTradingScopePreview(h) {
    vm.runInContext(`previewTradingScopeForCodes = async (codes, tradingScope) => ({
        status: 'success', trading_scope: tradingScope, eligible_codes: codes, excluded: []
    })`, h.context);
}

const ready = (mode, detail = '原始日线可用') => ({
    status: 'success', ok: true, source: 'tushare', stock_code: '000001.SZ',
    checks: [{ id: 'raw', title: '原始价格', ok: true, severity: 'info', detail }],
    price_policy: {
        requested_mode: mode, signal_price_mode: mode, execution_price_mode: 'none',
        factor_source: 'adj_factor', factor_available: true, raw_data_verified: true,
        causal_adjustment: true, corporate_action_accounting: 'unsupported', limitations: []
    }
});

test('checks unsaved source draft and shows actual policy without persisting secrets', async () => {
    const h = harness();
    h.context.schedulePriceReadiness(true);
    h.fire(0);
    assert.equal(h.requests.length, 1);
    assert.equal(h.requests[0].url, '/api/config/test_price_readiness');
    assert.equal(h.requests[0].options.method, 'POST');
    const payload = JSON.parse(h.requests[0].options.body);
    assert.equal(payload.source, 'tushare');
    assert.equal(payload.stock_code, '000001.SZ');
    assert.equal(payload.config.data_provider.adjustment_mode, 'qfq');
    h.requests[0].resolve({ ok: true, json: async () => ready('qfq', 'secret-token 已通过') });
    await new Promise(setImmediate);
    assert.match(h.elements.get('price-readiness-policy').textContent, /前复权.*成交\/估值：不复权/);
    assert.match(h.elements.get('price-readiness-checks').children[0].textContent, /\[已隐藏\]/);
    assert.doesNotMatch(h.elements.get('price-readiness-checks').children[0].textContent, /secret-token/);
});

test('late response from earlier draft cannot overwrite current result', async () => {
    const h = harness();
    h.context.schedulePriceReadiness(true);
    h.fire(0);
    h.setDraft({ data_provider: { source: 'tushare', adjustment_mode: 'hfq' } });
    h.context.schedulePriceReadiness();
    h.fire(450);
    assert.equal(h.requests.length, 2);
    h.requests[1].resolve({ ok: true, json: async () => ready('hfq') });
    await new Promise(setImmediate);
    h.requests[0].resolve({ ok: true, json: async () => ready('qfq') });
    await new Promise(setImmediate);
    assert.match(h.elements.get('price-readiness-policy').textContent, /后复权/);
    assert.doesNotMatch(h.elements.get('price-readiness-policy').textContent, /前复权/);
});

test('35 second timeout reports failure and retry starts a fresh check', async () => {
    const h = harness();
    h.context.schedulePriceReadiness(true);
    h.fire(0);
    h.fire(35000);
    await new Promise(setImmediate);
    assert.match(h.elements.get('price-readiness-status').textContent, /超时（35 秒）/);
    h.context.retryPriceReadiness();
    h.fire(0);
    assert.equal(h.requests.length, 2);
});

test('report view uses actual baseline metadata', () => {
    const h = harness();
    h.context.renderBacktestPricePolicy({ baseline_meta: { price_policy: ready('hfq').price_policy } });
    assert.match(h.elements.get('bt-price-policy').textContent, /后复权.*不复权/);
    assert.match(h.elements.get('bt-price-policy').textContent, /持仓跨复权因子变化日会终止回测/);
    h.context.renderBacktestPricePolicy(ready('none').price_policy);
    assert.match(h.elements.get('backtest-price-policy-live').textContent, /不保证识别公司行动事件/);
});

test('storage declaration is not presented as independently verified', () => {
    const h = harness();
    const result = ready('qfq');
    result.source = 'mysql';
    result.price_policy.raw_price_evidence = 'storage_declaration';
    result.price_policy.signal_price_mode = 'unavailable';
    h.context.renderPriceReadiness(result, { data_provider: { source: 'mysql' } });
    assert.match(h.elements.get('price-readiness-policy').textContent, /实际信号价：未确认/);
    assert.match(h.elements.get('price-readiness-policy').textContent, /按配置声明，外部口径未独立验证/);
    h.context.renderBacktestPricePolicy({ baseline_meta: { price_policy: result.price_policy } });
    assert.match(h.elements.get('bt-price-policy').textContent, /按配置声明为原始价，外部口径未独立验证/);
});

test('first actual use probes saved source and blocks failed readiness', async () => {
    const h = harness();
    const attempt = h.context.checkPriceReadinessBeforeUse('600000.SH');
    assert.equal(h.requests.length, 1);
    assert.deepEqual(JSON.parse(h.requests[0].options.body), {stock_code: '600000.SH'});
    h.requests[0].resolve({ok: true, json: async () => ({...ready('hfq'), ok: false, status: 'error'})});
    assert.equal(await attempt, false);
    assert.match(html, /await checkPriceReadinessBeforeUse\(payload.stock_code, false,/);
    assert.match(html, /await checkPriceReadinessForTargets\(targets, true\)/);
});

test('stale minute data is shown and prevents approximate realtime use', async () => {
    const h = harness();
    const attempt = h.context.checkPriceReadinessBeforeUse('000001.SZ', true);
    const result = ready('hfq');
    result.checks.push({id: 'minute', title: '分钟数据', ok: false, severity: 'warning', detail: '本地分钟数据过期'});
    h.requests[0].resolve({ok: true, json: async () => result});
    assert.equal(await attempt, false);
});

test('old report cannot inherit a new runs price policy', () => {
    const h = harness();
    vm.runInContext(`latestBacktestPricePolicy = ${JSON.stringify(ready('hfq').price_policy)}`, h.context);
    h.context.renderBacktestPricePolicy({baseline_meta: {}});
    assert.match(h.elements.get('bt-price-policy').textContent, /未返回价格口径元数据/);
});

function actualFunction(name, source = html) {
    const match = source.match(new RegExp('        (?:async )?function ' + name + '\\('));
    assert.ok(match, name + ' exists');
    const from = match.index;
    const next = source.slice(from + match[0].length).search(/\n        (?:async )?function /);
    return source.slice(from, from + match[0].length + next);
}

test('a saved-source probe preserves both completed report coverage and the draft result', async () => {
    const h = harness();
    h.context.renderBacktestPricePolicy({baseline_meta: {
        price_policy: ready('hfq').price_policy,
        data_coverage: {first_bar:'2020-01-02', last_bar:'2020-12-31', rows:243, requested_start:'2020-01-01', requested_end:'2020-12-31'}
    }});
    h.context.renderPriceReadiness({...ready('hfq'), source:'duckdb'}, {data_provider:{source:'duckdb'}});
    const reportBefore = h.elements.get('bt-price-policy').textContent;
    const draftBefore = h.elements.get('price-readiness-policy').textContent;
    const attempt = h.context.checkPriceReadinessBeforeUse('000001.SZ', true);
    const result = ready('qfq');
    result.checks.push({id:'minute',ok:true});
    h.requests[0].resolve({ok:true,json:async()=>result});
    assert.equal(await attempt, true);
    assert.equal(h.elements.get('bt-price-policy').textContent, reportBefore);
    assert.equal(h.elements.get('price-readiness-policy').textContent, draftBefore);
    assert.match(reportBefore, /243 根/);
});

test('a later draft result remains independent of a pending saved-config check', async () => {
    const h = harness();
    const saved = h.context.checkPriceReadinessBeforeUse('000001.SZ');
    h.context.schedulePriceReadiness(true);
    h.fire(0);
    h.requests[1].resolve({ok:true,json:async()=>ready('hfq')});
    await new Promise(setImmediate);
    h.requests[0].resolve({ok:true,json:async()=>ready('none')});
    assert.equal(await saved, true);
    assert.match(h.elements.get('price-readiness-policy').textContent, /后复权/);
});

test('historical startup sends the actual requested range', async () => {
    const h = harness();
    const attempt = h.context.checkPriceReadinessBeforeUse('000001.SZ', false, {start:'2020-01-01',end:'2021-01-01'});
    assert.deepEqual(JSON.parse(h.requests[0].options.body), {stock_code:'000001.SZ',start_date:'2020-01-01',end_date:'2021-01-01'});
    h.requests[0].resolve({ok:true,json:async()=>ready('hfq')});
    assert.equal(await attempt,true);
});

test('live startup checks both symbols and never submits after a stale second symbol', async () => {
    const h = harness();
    let starts = 0;
    Object.assign(h.context, {
        liveEnabled:true, selectedStrategyIds:[], strategyCatalog:[],
        resolveLiveStartTargets:()=>['000001.SZ','600000.SH'],
        readLiveAllocationSettings:()=>({mode:'equal',weights:{}}),
        fetchJsonWithTimeout:async()=>{starts++; return {res:{ok:true},data:{status:'success'}};},
        syncStatus:async()=>{}, renderLiveStatus(){}
    });
    vm.runInContext(actualFunction('startSimulation'),h.context);
    const attempt = h.context.startSimulation();
    assert.equal(h.requests.length,2);
    for(const request of h.requests) assert.equal(JSON.parse(request.options.body).require_minute,true);
    const fresh = ready('hfq'); fresh.checks.push({id:'minute',ok:true});
    const stale = ready('hfq'); stale.checks.push({id:'minute',ok:false,severity:'warning'});
    h.requests[0].resolve({ok:true,json:async()=>fresh});
    h.requests[1].resolve({ok:true,json:async()=>stale});
    assert.equal(await attempt,false);
    assert.equal(starts,0);
});

test('stopping during readiness aborts the check and never starts a backtest', async () => {
    const h = harness();
    allowTradingScopePreview(h);
    vm.runInContext(`
        const BACKTEST_BTN_STATE={running:'running',idle:'idle'};
        let backtestBtnState='running', backtestStopInFlight=false;
        function closeBacktestStream() {}
        function setBacktestButtonState(state) { backtestBtnState=state; }
        backtestStartupController=new AbortController();
        backtestStartupPhase='readiness';
        ${actualFunction('startBacktestByApi')}
        ${actualFunction('stopBacktest')}
    `,h.context);
    const attempt = vm.runInContext(`startBacktestByApi({stock_code:'000001.SZ',start:'2020-01-01',end:'2021-01-01'}, {
        signal:backtestStartupController.signal, isCurrent:()=>backtestStartupGeneration===0
    }).catch(error=>error.name)`,h.context);
    await new Promise(setImmediate);
    assert.equal(h.requests.length,1);
    await h.context.stopBacktest();
    assert.equal(await attempt,'AbortError');
    assert.equal(vm.runInContext('backtestBtnState',h.context),'idle');
    h.requests[0].resolve({ok:true,json:async()=>ready('hfq')});
    await new Promise(setImmediate);
    assert.equal(h.requests.length,1);
    assert.ok(!h.requests.some(request=>request.url==='/api/control/start_backtest'));
});

test('a superseded start cannot submit even if the provider ignores abort', async () => {
    const h = harness();
    allowTradingScopePreview(h);
    vm.runInContext(actualFunction('startBacktestByApi'),h.context);
    let current=true;
    h.context.isCurrent=()=>current;
    const attempt=h.context.startBacktestByApi({stock_code:'000001.SZ'}, {isCurrent:h.context.isCurrent}).catch(error=>error.name);
    await new Promise(setImmediate);
    current=false;
    h.requests[0].resolve({ok:true,json:async()=>ready('hfq')});
    assert.equal(await attempt,'AbortError');
    assert.equal(h.requests.length,1);
});

test('batch validation is bounded to two workers and stops after failure', async () => {
    const h=harness();
    const attempt=h.context.checkPriceReadinessForTargets(['000001.SZ','600000.SH','600036.SH'],false,{start:'2020-01-01',end:'2021-01-01'});
    assert.equal(h.requests.length,2);
    h.requests[0].resolve({ok:true,json:async()=>({...ready('hfq'),ok:false})});
    h.requests[1].resolve({ok:true,json:async()=>ready('hfq')});
    assert.equal(await attempt,false);
    assert.equal(h.requests.length,2);
});

test('batch validation has one total budget and cancels active probes', async () => {
    const h=harness();
    const attempt=h.context.checkPriceReadinessForTargets(['000001.SZ','600000.SH','600036.SH']);
    assert.equal(h.requests.length,2);
    h.fire(30000);
    assert.equal(await attempt,false);
    assert.ok(h.requests.every(request=>request.options.signal.aborted));
    assert.equal(h.requests.length,2);
});

const reportHtml = fs.readFileSync(path.join(__dirname,'..','..','backtest_report.html'),'utf8');
test('standalone reports disclose actual execution and storage declarations', () => {
    const h=harness();
    vm.runInContext(actualFunction('renderSummary',reportHtml),h.context);
    h.context.renderSummary({stock:'000001.SZ',baseline_meta:{price_policy:{...ready('hfq').price_policy,execution_price_mode:'unavailable',raw_price_evidence:'storage_declaration'}}});
    const note=h.elements.get('summary-card').children[0].textContent;
    assert.match(note,/后复权.*成交及估值：未确认/);
    assert.match(note,/按配置声明，外部口径未独立验证/);
});

test('standalone raw charts explain adjusted strategy indicators', () => {
    const h=harness();
    vm.runInContext(actualFunction('renderReportKlinePricePolicy',reportHtml),h.context);
    h.context.renderReportKlinePricePolicy({}, {baseline_meta:{price_policy:ready('qfq').price_policy}});
    const note=h.elements.get('report-kline-price-policy').textContent;
    assert.match(note,/不复权真实价格.*前复权/);
    assert.match(note,/复权指标数值可能与图中原价不同/);
});
