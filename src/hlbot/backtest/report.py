"""Render backtest results to a self-contained HTML file (data embedded inline,
no server needed). Open it in a browser to review every classified news item, the
trade decision, exit and PnL."""
from __future__ import annotations

import json
import pathlib

from .engine import BTTrade
from .metrics import summary_metrics

# Price sheet lives in analysis/pricing.py (shared with live per-analysis cost tracking);
# re-exported here because scripts/token_report.py imports it from this module.
from ..analysis.pricing import PRICING, _DEFAULT_PRICE  # noqa: F401,E402


def _cost(usage: dict) -> dict:
    """Estimate $ cost + token totals from per-model usage counts."""
    per, total = [], 0.0
    tin = tout = tcache = calls = 0
    for model, u in (usage or {}).items():
        pin, pout, pcw, pcr = PRICING.get(model, _DEFAULT_PRICE)
        c = (u["input"] * pin + u["output"] * pout
             + u["cache_creation"] * pcw + u["cache_read"] * pcr) / 1e6
        total += c
        calls += u["calls"]
        tin += u["input"]
        tout += u["output"]
        tcache += u["cache_read"] + u["cache_creation"]
        per.append({"model": model, "calls": u["calls"], "input": u["input"],
                    "output": u["output"], "cached": u["cache_read"] + u["cache_creation"],
                    "cost": c})
    per.sort(key=lambda p: -p["cost"])
    return {"total": total, "calls": calls, "input": tin, "output": tout,
            "cached": tcache, "per_model": per}


def _summary(trades: list[BTTrade]) -> dict:
    wins = [t for t in trades if t.pnl > 0]
    total = sum(t.pnl for t in trades)
    deployed = sum(t.notional for t in trades)
    by_reason: dict[str, int] = {}
    for t in trades:
        by_reason[t.reason] = by_reason.get(t.reason, 0) + 1
    return {
        "n_trades": len(trades),
        "wins": len(wins),
        "win_rate": (len(wins) / len(trades) * 100) if trades else 0.0,
        "total_pnl": total,
        "deployed": deployed,
        "return_pct": (total / deployed * 100) if deployed else 0.0,
        "by_reason": by_reason,
    }


def build_payload(results: dict) -> dict:
    w = results.get("window", (0, 0))
    return {
        "window": [w[0], w[1]],
        "model": results.get("model", ""),
        "threshold": results.get("threshold", 0),
        "n_news": results.get("n_news", 0),
        "n_candidates": results.get("n_candidates", 0),
        "n_analyzed": results.get("n_analyzed", 0),
        "n_signals": results.get("n_signals", 0),
        "exit_config": results.get("exit_config", {}),
        "cost": _cost(results.get("usage", {})),
        "summary": _summary(results.get("trades", [])),
        "metrics": summary_metrics(results.get("trades", []),
                                   results.get("account_size_usd", 0.0)),
        "n_no_data": results.get("n_no_data", 0),
        "funding_stats": results.get("funding_stats", {}),
        "rows": results.get("rows", []),
    }


def write_html(results: dict, path: str = "data/backtest_report.html") -> str:
    payload = build_payload(results)
    blob = json.dumps(payload).replace("</", "<\\/")  # safe to embed in <script>
    html = _TEMPLATE.replace("/*__DATA__*/", blob)
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return str(p.resolve())


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>hlbot · backtest review</title>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--line:#21262d;--fg:#c9d1d9;--muted:#8b949e;
        --green:#3fb950;--red:#f85149;--blue:#58a6ff;--yellow:#d29922;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
  header{padding:16px 20px;background:var(--panel);border-bottom:1px solid var(--line)}
  h1{font-size:18px;margin:0 0 8px}
  .sub{color:var(--muted);font-size:13px}
  .stats{display:flex;gap:26px;margin-top:12px;flex-wrap:wrap}
  .stat b{display:block;font-size:21px}.stat span{color:var(--muted);font-size:12px}
  .pos{color:var(--green)}.neg{color:var(--red)}
  .legend{margin-top:10px;padding:9px 13px;background:#0d1117;border:1px solid var(--line);border-radius:8px;font-size:12.5px;color:var(--muted)}
  .legend b{color:var(--fg)} .legend code{color:var(--blue)}
  .controls{display:flex;gap:9px;align-items:center;padding:12px 20px;flex-wrap:wrap}
  button{background:#21262d;color:var(--fg);border:1px solid var(--line);padding:7px 13px;border-radius:6px;cursor:pointer;font-size:13px}
  button.active{border-color:var(--blue);color:var(--blue)}
  input,select{background:#0d1117;color:var(--fg);border:1px solid var(--line);padding:7px 11px;border-radius:6px;font-size:13px}
  input{flex:1;min-width:200px}
  .tablewrap{overflow:auto;max-height:74vh;border-top:1px solid var(--line)}
  table{width:100%;border-collapse:collapse}
  thead th{position:sticky;top:0;background:#0d1117;z-index:2;box-shadow:0 1px 0 var(--line);
           color:var(--muted);font-weight:500;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;cursor:pointer}
  th,td{padding:10px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
  tr.sum{cursor:pointer} tr.sum:hover{background:#161b22}
  tr.sum.traded td:first-child{border-left:3px solid var(--yellow)}
  tr.sum.rej td:first-child{border-left:3px solid #30363d}
  td.news{max-width:380px;white-space:normal;line-height:1.4}
  .long{color:var(--green);font-weight:600}.short{color:var(--red);font-weight:600}
  .pill{font-size:11px;padding:2px 7px;border-radius:5px;background:#21262d;color:var(--muted)}
  .stale{background:#d2992233;color:var(--yellow)}
  .bar{display:inline-block;width:50px;height:8px;background:#21262d;border-radius:4px;overflow:hidden;vertical-align:middle;margin-right:6px}
  .bar>i{display:block;height:100%;background:var(--blue)}
  td.tnum{font-variant-numeric:tabular-nums;white-space:nowrap}
  a{color:var(--blue);text-decoration:none} a:hover{text-decoration:underline}
  .muted{color:var(--muted)}
  .detail{display:none;background:#0d1117}
  tr.detail.open{display:table-row}
  .dwrap{padding:8px 18px 16px;display:grid;grid-template-columns:1.3fr 1fr;gap:16px}
  .card{background:#161b22;border:1px solid var(--line);border-radius:8px;padding:12px 14px}
  .card h4{margin:0 0 7px;font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
  .card .txt{white-space:pre-wrap;font-size:13.5px;line-height:1.5}
  .kv{display:flex;flex-wrap:wrap;gap:6px 20px;font-size:13px} .kv div{white-space:nowrap}
  .kv b{color:var(--fg)} .kv span{color:var(--muted)}
</style></head>
<body>
<header>
  <h1>🌳 hlbot — backtest review</h1>
  <div class="sub" id="meta"></div>
  <div class="stats" id="stats"></div>
  <div class="legend" id="cost"></div>
  <div class="legend" id="legend"></div>
</header>
<div class="controls">
  <button data-f="traded" class="active">Traded</button>
  <button data-f="signals">Passed gate</button>
  <button data-f="all">All reads</button>
  <button data-f="wins">Wins</button>
  <button data-f="losses">Losses</button>
  <button data-f="missed">Missed (would-be)</button>
  <input id="q" placeholder="filter by ticker / news / reason…"/>
  <select id="sort">
    <option value="time">sort: time</option>
    <option value="conf">sort: confidence</option>
    <option value="pnl">sort: PnL</option>
    <option value="would">sort: would-PnL</option>
  </select>
  <span class="muted" style="font-size:12px">▶ click a row for full news + AI reasoning</span>
</div>
<div class="tablewrap"><table>
  <thead><tr>
    <th data-s="time">Time</th><th>Source</th><th>News</th><th>Ticker</th><th>Dir</th>
    <th data-s="conf">Conf</th><th>Horizon</th><th>Size·Lvg</th>
    <th class="tnum">Entry → Exit</th><th class="tnum" data-s="pnl">PnL</th>
    <th class="tnum" data-s="would">Would&nbsp;PnL</th><th>Outcome</th>
  </tr></thead>
  <tbody id="rows"></tbody>
</table></div>
<script>
const DATA = /*__DATA__*/;
const $ = s => document.querySelector(s);
let filter="traded", sortKey="time";

function ts(ms){return new Date(ms).toISOString().slice(5,16).replace('T',' ')}
function num(x,d=4){return x==null?'':(+x).toLocaleString(undefined,{maximumFractionDigits:d})}
function esc(s){return (s==null?'':String(s)).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function hlUrl(r){ return 'https://app.hyperliquid.xyz/trade/' + (r.market || r.ticker || ''); }
function fmtDur(ms){ if(ms==null||ms<=0) return ''; const h=ms/3600000;
  if(h<1) return Math.round(h*60)+'m'; if(h<48) return h.toFixed(1)+'h'; return (h/24).toFixed(1)+'d'; }

function sparkline(curve){
  if(!curve||curve.length<2) return '';
  const W=160,H=36,vals=curve.map(p=>p[1]);
  const lo=Math.min(0,...vals), hi=Math.max(0,...vals), span=(hi-lo)||1;
  const x=i=>(i/(curve.length-1))*(W-2)+1, y=v=>H-1-((v-lo)/span)*(H-2);
  const pts=vals.map((v,i)=>`${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const last=vals[vals.length-1];
  return `<svg width="${W}" height="${H}" style="vertical-align:middle">
    <line x1="1" y1="${y(0)}" x2="${W-1}" y2="${y(0)}" stroke="#30363d" stroke-width="1"/>
    <polyline points="${pts}" fill="none" stroke="${last>=0?'#3fb950':'#f85149'}" stroke-width="1.5"/></svg>`;
}

function header(){
  const d=DATA, s=d.summary, m=d.metrics||{};
  $('#meta').innerHTML = `${ts(d.window[0])} → ${ts(d.window[1])} UTC &nbsp;·&nbsp; model <b>${esc(d.model)}</b>`
    + ` &nbsp;·&nbsp; gate <b>${Math.round(d.threshold*100)}%</b> &nbsp;·&nbsp; ${d.n_news} news, ${d.n_analyzed} analyzed`
    + (d.n_no_data?` &nbsp;·&nbsp; <b class="neg">⚠ ${d.n_no_data} signal(s) dropped: no candle data</b>`:'');
  const cls = s.total_pnl>=0?'pos':'neg';
  const co=d.cost||{total:0,calls:0,input:0,output:0,cached:0,per_model:[]};
  const missed=d.rows.filter(r=>r.status!=='traded'&&r.would_pnl!=null);
  const missedSum=missed.reduce((a,r)=>a+r.would_pnl,0), mcls=missedSum>=0?'pos':'neg';
  const pf=m.profit_factor==null?'∞':num(m.profit_factor,2);
  const ddPct=m.max_dd_pct_of_account!=null?` (${num(m.max_dd_pct_of_account,1)}%)`:'';
  $('#stats').innerHTML = `
    <div class="stat"><b>${s.n_trades}</b><span>trades</span></div>
    <div class="stat"><b>${d.n_signals}</b><span>passed gate</span></div>
    <div class="stat"><b class="${cls}">$${num(s.total_pnl,2)}</b><span>total PnL</span></div>
    <div class="stat"><b class="${cls}">${s.return_pct>=0?'+':''}${num(s.return_pct,2)}%</b><span>on $${num(s.deployed,0)} deployed</span></div>
    <div class="stat"><b>${s.wins}/${s.n_trades} (${Math.round(s.win_rate)}%)</b><span>win rate</span></div>
    <div class="stat"><b>${pf}</b><span>profit factor</span></div>
    <div class="stat"><b class="neg">-$${num(m.max_dd_usd||0,0)}${ddPct}</b><span>max realized DD</span></div>
    <div class="stat"><b>${num(m.median_hold_h||0,1)}h</b><span>median hold</span></div>
    <div class="stat">${sparkline(m.equity_curve)}<span>equity (realized, daily)</span></div>
    <div class="stat"><b class="${mcls}">$${num(missedSum,0)}</b><span>would-be PnL · ${missed.length} skipped</span></div>
    <div class="stat"><b>$${num(co.total,3)}</b><span>est. API cost</span></div>`;
  const kf=n=>n>=1000?(n/1000).toFixed(1)+'k':String(n);
  const sm=m=>m.replace('claude-','').replace(/-\d.*$/,'');
  $('#cost').innerHTML = `<b>API cost</b> (this run): <b>$${num(co.total,4)}</b> over ${co.calls} calls`
    + ` &nbsp;·&nbsp; tokens ${kf(co.input)} in / ${kf(co.output)} out / ${kf(co.cached)} cached`
    + ` &nbsp;·&nbsp; ~$${num(co.total/Math.max(1,s.n_trades),4)}/trade · $${num(co.total/Math.max(1,d.n_analyzed),5)}/news`
    + (co.per_model.length?' &nbsp;|&nbsp; '+co.per_model.map(p=>`<code>${sm(p.model)}</code> ${p.calls}x $${num(p.cost,4)}`).join(' · '):'')
    + ' &nbsp;<span>(calls made this run; cached re-runs cost ~$0)</span>';
  const ec=d.exit_config||{};
  const parts=['immediate','hours','days'].filter(k=>ec[k]).map(k=>{
    const c=ec[k]; const exit = c.trail_pct>0 ? `trailing ${(c.trail_pct*100).toFixed(1)}% (let it run)` : `take-profit ${(c.tp_pct*100).toFixed(1)}%`;
    return `<b>${k}</b>: hold ≤${c.hold_h}h · stop ${(c.stop_pct*100).toFixed(1)}% · ${exit}`;
  });
  $('#legend').innerHTML = '<b>Horizon</b> (the AI\'s read of how long the edge lasts → sets the exit): &nbsp; '
    + parts.join(' &nbsp;|&nbsp; ')
    + '. &nbsp; <span>Outcome shows the exit reason and how long the trade was held.</span>';
}

function visible(){
  let rows = DATA.rows.slice();
  if(filter==='traded') rows=rows.filter(r=>r.status==='traded');
  else if(filter==='signals') rows=rows.filter(r=>r.status==='traded'||r.passed_gate);
  else if(filter==='wins') rows=rows.filter(r=>r.pnl!=null&&r.pnl>0);
  else if(filter==='losses') rows=rows.filter(r=>r.pnl!=null&&r.pnl<=0);
  else if(filter==='missed') rows=rows.filter(r=>r.status!=='traded' && r.would_pnl!=null);
  const q=$('#q').value.toLowerCase();
  if(q) rows=rows.filter(r=>((r.ticker||'')+' '+(r.body||'')+' '+(r.reason||'')+' '+(r.rationale||'')).toLowerCase().includes(q));
  rows.sort((a,b)=> sortKey==='time'? b.time_ms-a.time_ms
    : sortKey==='conf'? b.confidence-a.confidence
    : sortKey==='would'? ((b.would_pnl==null?-1e9:b.would_pnl)-(a.would_pnl==null?-1e9:a.would_pnl))
    : ((b.pnl==null?-1e9:b.pnl)-(a.pnl==null?-1e9:a.pnl)));
  return rows;
}

function detail(r){
  const exitRule = r.status!=='traded' ? '' :
    (r.trail_pct>0 ? `trailing ${(r.trail_pct*100).toFixed(1)}%` : `take-profit ${num(r.tp_px)}`);
  const tradeCard = r.status==='traded' ? `
    <div class="card"><h4>Trade</h4><div class="kv">
      <div><span>side</span> <b class="${r.side}">${r.side}</b></div>
      <div><span>notional</span> <b>$${num(r.notional,0)}</b></div>
      <div><span>margin</span> <b>$${num(r.notional/(r.leverage||1),0)}</b></div>
      <div><span>leverage</span> <b>${r.leverage}x</b></div>
      <div><span>size</span> <b>${num(r.size,4)}</b></div>
      <div><span>entry</span> <b>${num(r.entry_px)}</b></div>
      <div><span>stop</span> <b>${num(r.stop_px)}</b></div>
      <div><span>target</span> <b>${exitRule}</b></div>
      <div><span>horizon</span> <b>${esc(r.time_sensitivity)}</b></div>
      <div><span>held</span> <b>${fmtDur(r.exit_ms-r.time_ms)}</b></div>
      <div><span>exit</span> <b>${num(r.exit_px)}</b> (${esc(r.exit_reason||'')})</div>
      <div><span>pnl</span> <b class="${r.pnl>=0?'pos':'neg'}">$${num(r.pnl,2)}</b></div>
      ${r.mae_pct!=null?`<div><span>MAE</span> <b>-${num(r.mae_pct*100,2)}%</b></div>`:''}
      ${r.mfe_pct!=null?`<div><span>MFE</span> <b>+${num(r.mfe_pct*100,2)}%</b></div>`:''}
      ${r.funding_usd?`<div><span>funding</span> <b>$${num(r.funding_usd,2)}</b></div>`:''}
      ${r.candle_interval?`<div><span>bars</span> <b>${esc(r.candle_interval)}</b></div>`:''}
      ${r.pre_move_pct!=null?`<div><span>pre-move</span> <b>${r.pre_move_pct>=0?'+':''}${num(r.pre_move_pct*100,2)}%</b></div>`:''}
    </div></div>` :
    `<div class="card"><h4>Decision</h4><div class="txt">Not traded — ${esc(r.reason)}</div></div>`;
  return `<td colspan="12"><div class="dwrap">
    <div class="card"><h4>News — ${esc(r.source)}</h4><div class="txt">${esc(r.body||r.title)}</div>
      ${r.link?`<div style="margin-top:8px"><a href="${esc(r.link)}" target="_blank">↗ open source</a></div>`:''}</div>
    <div>
      <div class="card"><h4>AI analysis — ${esc(r.direction)} @ confidence ${Math.round((r.confidence||0)*100)}%</h4>
        <div class="txt">${esc(r.rationale||'(none)')}</div></div>
      ${r.confirm_confidence!=null?`<div class="card" style="margin-top:12px"><h4>Skeptic confirmation — ${r.confirm_agree?'agreed':'VETOED'} @ ${Math.round(r.confirm_confidence*100)}%</h4>
        <div class="txt">${esc(r.confirm_risk||'')}</div></div>`:''}
      ${tradeCard?`<div style="margin-top:12px">${tradeCard}</div>`:''}
    </div>
  </div></td>`;
}

function render(){
  const rows=visible();
  const html = rows.map((r,i)=>{
    const dir=`<span class="${r.direction}">${r.direction}</span>`;
    const conf=`<span class="bar"><i style="width:${Math.round(r.confidence*100)}%"></i></span>${Math.round(r.confidence*100)}%`;
    const stale=r.is_stale?'<span class="pill stale">stale</span>':'';
    const sizelvg = r.status==='traded' ? `$${num(r.notional,0)} · ${r.leverage}x` : '';
    const px = r.status==='traded' ? `${num(r.entry_px)} → ${num(r.exit_px)}` : '';
    const pnl = r.pnl==null?'' : `<b class="${r.pnl>=0?'pos':'neg'}">${r.pnl>=0?'+':''}${num(r.pnl,2)}</b>`;
    const would = (r.status!=='traded' && r.would_pnl!=null)
      ? `<b class="${r.would_pnl>=0?'pos':'neg'}">${r.would_pnl>=0?'+':''}${num(r.would_pnl,2)}</b>`
      : '<span class="muted">—</span>';
    let outcome;
    if(r.status==='traded'){
      const dur=fmtDur(r.exit_ms-r.time_ms);
      outcome = `${esc(r.exit_reason)}${dur?` <span class="muted">· ${dur}</span>`:''}`;
    } else { outcome = `<span class="muted">${esc(r.reason)}</span>`; }
    const tickerLink = `<a href="${hlUrl(r)}" target="_blank" title="open on Hyperliquid"><b>${esc(r.ticker)}</b></a>`;
    return `<tr class="sum ${r.status==='traded'?'traded':'rej'}">
      <td class="tnum">${ts(r.time_ms)}</td>
      <td>${esc(r.source)}</td>
      <td class="news">${esc((r.body||r.title||'').slice(0,160))}</td>
      <td>${tickerLink}</td><td>${dir}</td>
      <td class="tnum">${conf}</td>
      <td><span class="pill">${esc(r.time_sensitivity)}</span> ${stale}</td>
      <td class="tnum">${sizelvg}</td>
      <td class="tnum">${px}</td>
      <td class="tnum">${pnl}</td><td class="tnum">${would}</td><td>${outcome}</td>
    </tr><tr class="detail">${detail(r)}</tr>`;
  }).join('') || '<tr><td colspan="12" class="muted" style="padding:24px">No rows match.</td></tr>';
  $('#rows').innerHTML = html;
  document.querySelectorAll('#rows tr.sum').forEach(tr=>tr.onclick=e=>{
    if(e.target.tagName==='A') return;
    const d=tr.nextElementSibling; if(d&&d.classList.contains('detail')) d.classList.toggle('open');
  });
}

document.querySelectorAll('.controls button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.controls button').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); filter=b.dataset.f; render();
});
$('#q').oninput=render;
$('#sort').onchange=e=>{sortKey=e.target.value;render()};
document.querySelectorAll('th[data-s]').forEach(th=>th.onclick=()=>{sortKey=th.dataset.s;$('#sort').value=sortKey;render()});

header(); render();
</script>
</body></html>"""
