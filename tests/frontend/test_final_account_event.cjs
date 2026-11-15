const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('final account snapshot is ordered before completion in the critical queue', () => {
  const html = fs.readFileSync(path.join(__dirname,'../../dashboard.html'),'utf8');
  const start = html.indexOf('        function enqueueIncomingEvent(payload) {');
  const end = html.indexOf('        function closeBacktestStream()',start);
  assert.ok(start>=0 && end>start);
  const high=[], normal=[];
  const ctx = vm.createContext({backtestStreamClosed:false,
    BACKTEST_STREAM_TYPES:new Set(),RENDER_CRITICAL_TYPES:new Set(['backtest_result']),
    RENDER_COALESCE_TYPES:new Set(),renderHighQueue:high,renderNormalQueue:normal,
    renderCoalesceMap:new Map(),RENDER_QUEUE_MAX:1500,ensureRenderLoop(){}});
  vm.runInContext(html.slice(start,end),ctx);
  ctx.enqueueIncomingEvent({type:'account',data:{assets:100000}});
  ctx.enqueueIncomingEvent({type:'account',data:{assets:100100,is_final:true}});
  ctx.enqueueIncomingEvent({type:'backtest_result',data:{}});
  assert.equal(normal.length,1);
  assert.deepEqual(high.map(x=>x.type),['account','backtest_result']);
  assert.equal(high[0].data.assets,100100);
});
