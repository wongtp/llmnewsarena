"""Regression tests for the arena chart's candle-loading behavior.

Runs the REAL inline script from arena.html under Node with a stubbed DOM /
fetch / LightweightCharts, and drives the exact race that showed up live:
clicking a news ticker while the candle proxy is rate-limited used to park a
permanent "no candle data" message over a chart that then trickled in a lone
live candle and, a minute later, full history.

Covered invariants:
- a failed /api/candles fetch (429/502/network) retries with backoff instead
  of declaring "no candle data"; the message only appears for a genuine
  OK-but-empty response, and even that is re-asked before it's believed
- a live WS candle landing on an empty chart clears the stale message
- switching coins aborts a stale retry loop (no clobbering the new chart)
- newsPxAt never caches transient failures (only clean empties cache as null)
- feed price-enrichment lookups skip intervals whose retention can't reach the
  news time, and hold off entirely while a foreground chart load is in flight
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

ARENA = pathlib.Path(__file__).resolve().parent.parent / "src" / "hlbot" / "ui" / "static" / "arena.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _inline_script() -> str:
    html = ARENA.read_text(encoding="utf-8")
    m = re.search(r"<script>\n(.*)\n</script>", html, re.S)
    assert m, "arena.html inline script not found"
    return m.group(1)


# Minimal browser environment: enough DOM for $()/innerHTML/classList writes, a
# controllable fetch, silenced intervals, and timer delays capped so the retry
# backoff paths run in milliseconds. Names are __-prefixed to dodge collisions.
PRELUDE = r"""
const __realSetTimeout=global.setTimeout;
const setTimeout=(fn,ms)=>__realSetTimeout(fn,Math.min(ms||0,15));
const setInterval=()=>0;
const __els={};
function __mkEl(id){
  const el={id,style:{},dataset:{},hidden:false,textContent:'',className:'',value:'',
    offsetWidth:0,offsetHeight:0,offsetLeft:0,offsetTop:0,clientWidth:800,clientHeight:400,
    addEventListener(){},removeEventListener(){},setPointerCapture(){},focus(){},
    classList:{toggle(){},add(){},remove(){},contains:()=>false},
    querySelector:()=>null,querySelectorAll:()=>[],appendChild(){},replaceWith(){}};
  let inner='';Object.defineProperty(el,'innerHTML',{get:()=>inner,set:v=>{inner=String(v);}});
  return el;
}
const document={getElementById:id=>__els[id]||(__els[id]=__mkEl(id)),
  addEventListener(){},createElement:t=>__mkEl(t),querySelectorAll:()=>[]};
const localStorage={getItem:()=>null,setItem(){},removeItem(){}};
class WebSocket{constructor(){this.readyState=0;}send(){}close(){}}
class Image{constructor(){this.complete=false;this.naturalWidth=0;}}
const location={protocol:'http:',host:'test'};
const requestAnimationFrame=fn=>__realSetTimeout(fn,0);
const __priceScale={applyOptions(){},options:()=>({autoScale:true})};
const __series=()=>({applyOptions(){},priceScale:()=>__priceScale,setData(){},update(){},
  attachPrimitive(){},createPriceLine:()=>({applyOptions(){}}),removePriceLine(){},
  priceToCoordinate:()=>100,setMarkers(){}});
const __timeScale={subscribeVisibleLogicalRangeChange(){},setVisibleLogicalRange(){},
  setVisibleRange(){},getVisibleLogicalRange:()=>null,timeToCoordinate:()=>100};
const window={LightweightCharts:{CrosshairMode:{Normal:0},
  createChart:()=>({addCandlestickSeries:__series,addHistogramSeries:__series,
    priceScale:()=>__priceScale,timeScale:()=>__timeScale,
    subscribeClick(){},subscribeCrosshairMove(){}})}};
let __fetchLog=[];
let __fetchHandler=async()=>({ok:false,status:503});
const fetch=async url=>{__fetchLog.push(String(url));return __fetchHandler(String(url));};
"""

TESTS = r"""
function assert(cond,label){if(!cond){console.error('ASSERT FAIL: '+label);process.exit(1);}}
const __sleep=ms=>new Promise(r=>__realSetTimeout(r,ms));
const __mkCandles=(n,endMs)=>{endMs=endMs||Date.now();
  return Array.from({length:n},(_,i)=>({t:endMs-(n-i)*300000,o:'1',h:'2',l:'0.5',c:'1.5',v:'10'}));};
__realSetTimeout(()=>{console.error('HARNESS TIMEOUT');process.exit(2);},60000);
(async()=>{
  assert(chartReady,'chart initialized under stub LWC');
  curCoin='BTC';curSym='BTC';curDex='';curIval='5m';
  const cmsg=document.getElementById('cmsg');

  // T1: transient 429s retry and succeed — never "no candle data"
  let fails=2;
  __fetchHandler=async url=>{
    if(!url.includes('/api/candles'))return {ok:false,status:404};
    if(fails-->0)return {ok:false,status:429};
    return {ok:true,json:async()=>__mkCandles(5)};
  };
  await loadCandles();
  assert(bars.length===5,'T1: history loaded after retries');
  assert(cmsg.style.display==='none','T1: loading message cleared');
  assert(!cmsg.textContent.startsWith('no candle data'),'T1: no false no-data verdict');

  // T2: genuine OK-empty shows "no candle data" after a bounded re-ask
  __fetchLog=[];
  __fetchHandler=async()=>({ok:true,json:async()=>[]});
  await loadCandles();
  assert(bars.length===0,'T2: no bars');
  assert(cmsg.textContent.startsWith('no candle data'),'T2: no-data message shown');
  const t2Fetches=__fetchLog.filter(u=>u.includes('/api/candles')).length;
  assert(t2Fetches===3,'T2: empty re-asked exactly 3 times, got '+t2Fetches);

  // T3: a live WS candle arriving on the empty chart clears the stale message
  applyCandle({s:'BTC',i:'5m',t:Date.now(),o:1,h:1,l:1,c:1,v:2});
  assert(bars.length===1,'T3: live candle applied');
  assert(cmsg.style.display==='none','T3: stale no-data message cleared by live data');

  // T4: newsPxAt does NOT cache transient failures...
  __fetchHandler=async()=>({ok:false,status:429});
  const ms4=Date.now()-120000,key4='BTC|'+Math.floor(ms4/60000);
  assert(await newsPxAt('BTC',ms4)===null,'T4: null on failure');
  assert(!newsPxCache[key4],'T4: failed lookup evicted from cache');
  __fetchHandler=async()=>({ok:true,json:async()=>[{t:ms4-60000,o:'42',h:'43',l:'41',c:'42.5',v:'1'}]});
  assert(await newsPxAt('BTC',ms4)===42,'T4: retry after failure succeeds');
  assert(!!newsPxCache[key4],'T4: success cached');
  // ...but a clean OK-empty result DOES cache as null (genuinely no history)
  const ms4b=Date.now()-180000,key4b='BTC|'+Math.floor(ms4b/60000);
  __fetchHandler=async()=>({ok:true,json:async()=>[]});
  assert(await newsPxAt('BTC',ms4b)===null,'T4b: empty -> null');
  __fetchHandler=async()=>({ok:true,json:async()=>[{t:ms4b,o:'7',h:'7',l:'7',c:'7',v:'1'}]});
  assert(await newsPxAt('BTC',ms4b)===null,'T4b: clean empty stays cached as null');

  // T5: intervals whose retention can't reach the news time are skipped
  __fetchLog=[];
  const ms5=Date.now()-30*86400000;   // 30d old: 1m retention (~3d) can't reach
  __fetchHandler=async()=>({ok:true,json:async()=>[{t:ms5-3600000,o:'9',h:'9',l:'9',c:'9',v:'1'}]});
  assert(await newsPxAt('SOL',ms5)===9,'T5: old lookup resolves');
  assert(!__fetchLog.some(u=>u.includes('interval=1m')),'T5: no wasted 1m request for 30d-old news');

  // T6: switching coins aborts the stale retry loop
  __fetchHandler=async url=>{
    if(decodeURIComponent(url).includes('coin=BTC'))return {ok:false,status:429};
    return {ok:true,json:async()=>__mkCandles(4)};
  };
  curCoin='BTC';curSym='BTC';
  loadCandles();                     // left retrying (BTC keeps failing)
  await __sleep(30);
  curCoin='ETH';curSym='ETH';
  await loadCandles();
  assert(barsKey==='ETH|5m'&&bars.length===4,'T6: new coin loaded while old retried');
  __fetchHandler=async()=>({ok:true,json:async()=>__mkCandles(9)});
  await __sleep(150);                // stale BTC loop wakes, must abort on reqId guard
  assert(bars.length===4&&barsKey==='ETH|5m','T6: stale retry loop did not clobber new chart');

  // T7: background feed lookups hold while a foreground chart load is in flight
  fgLoads=1;__fetchLog=[];
  const ms7=Date.now()-240000;
  __fetchHandler=async()=>({ok:true,json:async()=>[{t:ms7,o:'5',h:'5',l:'5',c:'5',v:'1'}]});
  const bg=newsPxAt('DOGE',ms7,true);
  await __sleep(80);
  assert(__fetchLog.length===0,'T7: background lookup held while foreground busy');
  fgLoads=0;
  assert(await bg===5,'T7: background lookup proceeds once foreground idle');

  console.log('ALL OK');process.exit(0);
})().catch(e=>{console.error('FAIL',e&&e.stack||e);process.exit(1);});
"""


def test_arena_chart_candle_loading_behavior(tmp_path):
    harness = tmp_path / "arena_chart_harness.js"
    harness.write_text(PRELUDE + "\n" + _inline_script() + "\n" + TESTS, encoding="utf-8")
    proc = subprocess.run(["node", str(harness)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "ALL OK" in proc.stdout
