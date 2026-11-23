// Control-room screen (D-052). The server shapes everything that is shown (view.py); this file
// only lays it out. No build step: Preact + htm are one vendored module.
import { html, render, useEffect, useRef, useState } from './vendor/htm-preact.module.js';
import { createMap } from './map.js';
import { createCharts, drawAnswerChart, drawResultsCharts } from './charts.js';

const ICONS = {
  normal: 'M5 12l4.5 4.5L19 7',
  alert: 'M12 2.5 22.5 20.5H1.5ZM12 9v5.5M12 17.3v.3',
  waiting: 'M12 3a9 9 0 1 0 9 9M12 7v5l3.5 2',
  unknown: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18M12 7.5v6M12 16.6v.3',
};

function Icon({ kind }) {
  const stroke = kind === 'alert' || kind === 'unknown' ? 'var(--amber)' : 'var(--ink-2)';
  return html`<svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
    <path d=${ICONS[kind] || ICONS.unknown} fill="none" stroke=${stroke} stroke-width="2"
      stroke-linecap="round" stroke-linejoin="round" />
  </svg>`;
}

// One request; the answer is always { ok, message?, ...data }.
async function call(method, url, body) {
  try {
    const res = await fetch(url, {
      method,
      cache: 'no-store',
      // The custom header marks the request as the page's own; the server refuses commands without it.
      headers: { 'X-Requested-With': 'leakwatch', ...(body ? { 'Content-Type': 'application/json' } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return { ok: false, message: data.message || 'The server refused the request.' };
    return { ok: true, ...data };
  } catch {
    return { ok: false, message: 'The dashboard server did not answer.' };
  }
}

// Poll the snapshot. `lost` is true while the dashboard server itself does not answer.
function useLive() {
  const [vm, setVm] = useState(null);
  const [lost, setLost] = useState(false);
  const [settings, setSettings] = useState(null);
  useEffect(() => {
    let stop = false;
    let timer = 0;
    let pollMs = 1000;
    const tick = async () => {
      const res = await call('GET', '/api/live');
      if (res.ok) setVm(res);
      setLost(!res.ok);
      if (!stop) timer = setTimeout(tick, pollMs);
    };
    // The snapshot does not wait for the settings: the page shows the runner's state at once.
    // Settings that did not arrive (the server was still starting) are asked for again.
    const settle = () => call('GET', '/api/settings').then((s) => {
      if (stop) return;
      if (s.ok) { pollMs = s.poll_ms; setSettings(s); } else setTimeout(settle, 5000);
    });
    settle();
    tick();
    return () => { stop = true; clearTimeout(timer); };
  }, []);
  return { vm, lost, settings };
}

function useNetwork() {
  const [net, setNet] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let stop = false;
    const load = async () => {
      const res = await call('GET', '/api/network');
      if (stop) return;
      if (res.ok) { setNet(res); setError(null); } else { setError(res.message); setTimeout(load, 5000); }
    };
    load();
    return () => { stop = true; };
  }, []);
  return { net, error };
}

/* ------------------------------------------------------------------------------ header */

function Controls({ settings, clock, runner, size, setSize, notify }) {
  const [speed, setSpeed] = useState(null);
  const [armed, setArmed] = useState(false);
  if (!settings) return null;
  const current = clock && clock.speed ? Number(clock.speed.replace(/[^0-9]/g, '')) : settings.speeds[0];
  const chosen = speed ?? current;
  const send = async (url, body) => { const res = await call('POST', url, body); notify(res.ok ? null : res.message); };
  // While the runner starts it does not read its commands yet: a speed or restart sent then is lost.
  const starting = runner.code === 'starting';
  const down = starting || runner.code === 'offline' || runner.code === 'stale' || runner.code === 'error';
  return html`<details class="controls">
    <summary>Demo controls</summary>
    <div class="controls-panel">
      <label for="leak-size">Leak size <output>${size} mm</output></label>
      <input id="leak-size" type="range" min=${settings.leak_mm.min} max=${settings.leak_mm.max} step="5"
        value=${size} onInput=${(e) => setSize(Number(e.target.value))} aria-describedby=${settings.detector ? 'leak-size-note' : undefined} />
      ${settings.detector && html`<p class="fine" id="leak-size-note">${settings.detector.sentence}</p>`}
      <label for="replay-speed">Replay speed</label>
      <div class="row">
        <select id="replay-speed" value=${chosen} onChange=${(e) => setSpeed(Number(e.target.value))}>
          ${settings.speeds.map((s) => html`<option value=${s}>${s.toLocaleString('en')}×</option>`)}
        </select>
        <button class="btn ghost" disabled=${down || chosen === current}
          onClick=${() => send('/api/demo/speed', { speed: chosen })}>Set speed</button>
      </div>
      <button class="btn ghost" disabled=${down}
        onClick=${() => { if (armed) { send('/api/demo/restart'); setArmed(false); } else setArmed(true); }}
        onBlur=${() => setArmed(false)}>
        ${armed ? 'Click again to restart the replay' : 'Restart replay'}
      </button>
      <p class="fine">${starting ? 'Speed and restart work once the replay is warming up. '
        : ''}Restart clears the leaks and alerts of this run and replays the week from Monday.</p>
    </div>
  </details>`;
}

function Header({ runner, clock, analysed, tabs, children }) {
  const width = clock && clock.replay_fraction != null ? `${(clock.replay_fraction * 100).toFixed(1)}%` : '0';
  return html`<header class="header">
    <div class="brand"><h1>LeakWatch</h1><span>L-Town, the BattLeDIM benchmark network</span></div>
    ${tabs}
    <div class="meta">
    <dl class="facts">
      ${clock && clock.sim_time && html`<div class="fact"><dt>Simulated time</dt><dd>${clock.sim_time}</dd></div>`}
      ${clock && clock.sim_time && analysed && html`<div class="fact" title="Newest window the stream has scored; the replay runs ahead of it"><dt>Analysed up to</dt><dd>${analysed}</dd></div>`}
      ${clock && clock.speed && html`<div class="fact"><dt>Replay speed</dt><dd>${clock.speed}</dd></div>`}
      ${clock && clock.replay_left && runner.code !== 'finished' && runner.code !== 'finishing' && html`<div class="fact"><dt>Replay ends in</dt><dd>${clock.replay_left}</dd></div>`}
    </dl>
    <div class="status" data-code=${runner.code} title=${runner.detail}><span class="dot" aria-hidden="true"></span>${runner.label}</div>
    ${children}
    </div>
    <div class="replay" role="presentation"><i style=${{ width }}></i></div>
  </header>`;
}

/* --------------------------------------------------------------------------------- map */

function Banner({ banner, runner, onRestart }) {
  return html`<div class="banner" data-kind=${banner.kind} role="status" aria-live="polite">
    <${Icon} kind=${banner.kind} />
    <div><b>${banner.title}</b>${banner.sub && html`<small>${banner.sub}</small>`}</div>
    ${(runner.code === 'finished' || runner.code === 'finishing') && html`<button class="btn" onClick=${onRestart}>Restart replay</button>`}
  </div>`;
}

function MapView({ net, selected, drops, incident, truth, label, onReady, onPick, onError }) {
  const el = useRef(null);
  const map = useRef(null);
  const handlers = useRef({});
  handlers.current = { onPick, onError, onReady };
  useEffect(() => {
    map.current = createMap(el.current, net, {
      onPick: (node, at) => handlers.current.onPick(node, at),
      onError: (error) => handlers.current.onError(error),
    });
    handlers.current.onReady(map.current);
    return () => { if (map.current) map.current.destroy(); map.current = null; handlers.current.onReady(null); };
  }, [net]);
  useEffect(() => { if (map.current) map.current.set({ selected, drops, incident, truth }); }, [selected, drops, incident, truth]);
  return html`<div class="deck" ref=${el} role="img" aria-label=${label}></div>`;
}

// What the map shows, in words (the canvas itself says nothing to a screen reader).
function mapLabel(net, incident, drops, truth) {
  const parts = [`Map of L-Town: ${net.counts.junctions} junctions, ${net.counts.pressure_sensors} pressure sensors.`];
  if (!incident) parts.push('No open alert.');
  else {
    parts.push(`Alert in Area ${incident.area}.`);
    if (incident.candidates.length) parts.push(`Candidates, best first: ${incident.candidates.join(', ')}.`);
  }
  if (drops && drops.shown) parts.push(`${drops.shown} sensor${drops.shown === 1 ? ' shows' : 's show'} a pressure drop of ${drops.floor_m} m or more.`);
  if (truth) parts.push(`True leak at ${truth.node_id}.`);
  return parts.join(' ');
}

const metres = (m) => `${Number(m.toFixed(1))} m`;

function Legend({ net, drops, incident, floor, truth }) {
  return html`<div class="legend" aria-label="Map legend">
    <span><svg width="16" height="16" aria-hidden="true"><circle cx="8" cy="8" r="4.2" fill="var(--sheet)" stroke="var(--ink)" stroke-width="1.5"/></svg>
      Pressure sensor (${net.counts.pressure_sensors})</span>
    <span><svg width="26" height="26" aria-hidden="true"><circle cx="13" cy="13" r="11" fill="var(--amber)" fill-opacity=".22" stroke="var(--amber)" stroke-width="1.5"/>
      <circle cx="13" cy="13" r="4.2" fill="var(--sheet)" stroke="var(--ink)" stroke-width="1.5"/></svg>
      Measured drop, ring size (${metres(floor)} or more)</span>
    <span><i class="ramp" aria-hidden="true"></i>Drop along the pipes, interpolated between sensors (0 to ${metres(net.tint_full_m)})</span>
    ${incident && incident.candidates.length > 0 && html`<span><svg width="18" height="18" aria-hidden="true"><circle cx="9" cy="9" r="7.5" fill="var(--amber)"/>
      <text x="9" y="13" text-anchor="middle" font-size="11" font-weight="700" fill="var(--sheet)">1</text></svg>Candidate, by rank</span>`}
    ${incident && incident.order && incident.order.pipes && incident.order.pipes.length > 0 && html`<span><svg width="26" height="12" aria-hidden="true"><path d="M2 6H24" stroke="var(--ink)" stroke-width="3.5" stroke-linecap="round"/></svg>Pipe to inspect, numbered in order when zoomed in</span>`}
    ${truth && html`<span><svg width="20" height="20" aria-hidden="true"><circle cx="10" cy="10" r="6.5" fill="none" stroke="var(--coral)" stroke-width="2.4"/><path d="M10 0v3.5M10 16.5V20M0 10h3.5M16.5 10H20" stroke="var(--coral)" stroke-width="2"/></svg>True leak (after the reveal)</span>`}
    ${drops && html`<span class="when">Readings of the window ending ${drops.time}</span>`}
  </div>`;
}

function PickCard({ pick, net, size, canStart, why, busy, onStart, onClose }) {
  const left = Math.min(Math.max(pick.x + 16, 12), pick.maxX - 292);
  const top = Math.min(Math.max(pick.y - 20, 84), pick.maxY - 150);
  return html`<div class="pick" style=${{ left: `${left}px`, top: `${top}px` }} role="dialog" aria-label="Start a leak">
    <p><b>Junction ${pick.node}</b>, Area ${net.areas[pick.node]}</p>
    <button class="btn" disabled=${!canStart || busy} onClick=${onStart}>
      ${busy ? 'Sending…' : `Start a ${size} mm leak here`}
    </button>
    ${!canStart && html`<p class="fine">${why}</p>`}
    <button class="close" aria-label="Close" onClick=${onClose}>×</button>
  </div>`;
}

/* -------------------------------------------------------------------------- instruments */

function ChartPanel({ id, title, note, table, empty, chartRef }) {
  const [asTable, setAsTable] = useState(false);
  const showTable = asTable && table;
  return html`<section class="panel" aria-labelledby=${id}>
    <header>
      <h3 id=${id}>${title}${note && html` <span>${note}</span>`}</h3>
      ${table && html`<button class="link" aria-pressed=${asTable ? 'true' : 'false'} onClick=${() => setAsTable(!asTable)}>
        Table</button>`}
    </header>
    <div class="chart" ref=${chartRef} aria-hidden="true"></div>
    ${empty && html`<p class="chart-empty">${empty}</p>`}
    ${showTable && html`<div class="table-view" tabindex="0" role="region" aria-label=${`${title}, as a table`}><table>
      <thead><tr>${table.columns.map((c) => html`<th scope="col">${c}</th>`)}</tr></thead>
      <tbody>${table.rows.map((r) => html`<tr>${r.map((v, i) => (i === 0 ? html`<th scope="row">${v}</th>` : html`<td>${v}</td>`))}</tr>`)}</tbody>
    </table></div>`}
  </section>`;
}

function Instruments({ charts }) {
  const scoreEl = useRef(null);
  const pressureEl = useRef(null);
  const drawn = useRef(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    try {
      drawn.current = createCharts(scoreEl.current, pressureEl.current);
    } catch {
      setFailed(true);
    }
    return () => { if (drawn.current) drawn.current.destroy(); drawn.current = null; };
  }, []);
  const key = charts ? charts.key : null;
  useEffect(() => { if (drawn.current) drawn.current.update(charts); }, [key]);
  const broken = failed && 'The charts could not be drawn in this browser.';
  const scores = charts && charts.score.points.length > 0;
  const pressure = charts && charts.pressure.series.length > 0;
  return html`<section class="strip" aria-label="Instruments">
    <${ChartPanel} id="score-title" title="Anomaly score against the alert threshold" note="log scale"
      table=${scores ? charts.score.table : null} chartRef=${scoreEl}
      empty=${broken || (!scores && 'The score appears with the first scored windows of the replay.')} />
    <${ChartPanel} id="pressure-title" title="Pressure against usual, m" note=${pressure ? charts.pressure.why : ''}
      table=${pressure ? charts.pressure.table : null} chartRef=${pressureEl}
      empty=${broken || (!pressure && 'Pressure appears with the first windows of the replay.')} />
  </section>`;
}

/* -------------------------------------------------------------------------------- rail */

const STEPS = ['Start a leak', 'Alert', 'Located', 'Work order', 'Check against the true leak', 'Ask the copilot'];

// Step 1 while no alert of this leak is open: still waiting for one, or why its alert closed.
function leakAfter(alert, detector, days) {
  if (!alert) return `Waiting for an alert.${detector ? ` ${detector.sentence}` : ''}`;
  if (alert.status === 'open') return '';
  const ran = `Its alert in Area ${alert.area} ran from ${alert.started}${alert.ended ? ` to ${alert.ended}` : ''} and closed when the score fell back below the threshold.`;
  // A leak that has run for about `days` is inside the detector's own reference (D-038).
  const folded = days && alert.leak_days != null && alert.leak_days >= days - 1;
  return folded ? `${ran} The detector compares each window with the median of a week of earlier windows; `
    + `after about ${days} days a running leak is part of that median.` : ran;
}

// "300", "N300" and " n300 " all name junction n300; null when no junction has that name.
function junctionNamed(net, typed) {
  if (!net || !typed) return null;
  const name = /^\d+$/.test(typed) ? `n${typed}` : typed.toLowerCase();
  return net.junctions.includes(name) ? name : null;
}

function LeakStep({ leaks, net, size, canStart, why, busy, onStart, inc, detector, days }) {
  const [node, setNode] = useState('');
  const active = leaks.active[leaks.active.length - 1];
  const pending = leaks.pending[leaks.pending.length - 1];
  const failed = leaks.failed[leaks.failed.length - 1];
  if (pending) {
    return { on: true, title: 'Simulating the leak', body: html`
      <p>${pending.diameter_mm} mm leak at <b>${pending.node_id}</b>: EPANET is computing its pressure and flow signature.</p>
      <div class="working" role="progressbar" aria-label="Simulating the leak"></div>` };
  }
  if (active) {
    return { on: true, title: `Leak started at ${active.node_id}`, when: active.started, short: `${active.diameter_mm} mm`, body: html`
      <p><b>${active.diameter_mm} mm</b> hole, simulated with EPANET${leaks.active.length > 1 ? `. ${leaks.active.length - 1} earlier leak${leaks.active.length > 2 ? 's' : ''} still running.` : ''}</p>
      ${!inc && html`<p class="fine">${leakAfter(active.alert, detector, days)}</p>`}` };
  }
  const known = junctionNamed(net, node);
  const wrong = !!node && !!net && !known && !/^n$/i.test(node);  // a lone "n" is a name being typed
  return { on: true, open: true, title: 'Start a leak', body: html`
    ${failed && html`<p class="problem" role="alert">The leak at ${failed.node_id} could not be simulated. Pick another junction.</p>`}
    <p>Click any junction on the map, or name one here. EPANET simulates the leak and adds it to the replayed readings.</p>
    <form class="row" onSubmit=${(e) => { e.preventDefault(); if (known) onStart(known); }}>
      <input list="junctions" value=${node} placeholder="Junction, for example n470" aria-label="Junction"
        aria-invalid=${wrong ? 'true' : undefined} aria-describedby="junction-note"
        onInput=${(e) => setNode(e.target.value.trim())} />
      <button class="btn" type="submit" disabled=${!known || !canStart || busy}>${busy ? 'Sending…' : `Start a ${size} mm leak`}</button>
    </form>
    ${net && html`<datalist id="junctions">${net.junctions.map((j) => html`<option value=${j} />`)}</datalist>`}
    <p class="fine" id="junction-note" aria-live="polite">${wrong
      ? `${node} is not a junction of L-Town. Junctions are named n1 to n${net.junctions.length}.` : !canStart ? why : ''}</p>` };
}

const COUNT = ['No', 'One', 'Two', 'Three', 'Four', 'Five'];

function AlertStep({ inc }) {
  if (!inc) return { on: false, title: STEPS[1] };
  let timing;
  if (inc.after_click_s != null) {
    timing = html`<b>${inc.after_click_s} s</b> after the click${inc.after_leak_min != null
      ? html`, <b>${inc.after_leak_min} simulated minutes</b> after the leak began` : ''}.`;
  } else {
    timing = 'It began before any leak started from this screen.';
  }
  const short = `Area ${inc.area}${inc.after_click_s != null ? `, ${inc.after_click_s} s after the click` : ''}`;
  return { on: true, title: STEPS[1], when: inc.started, short, body: html`
    <p>Area ${inc.area}. ${timing}${inc.ratio != null ? html` Score <b>${inc.ratio} times</b> the threshold.` : ''}</p>
    ${inc.sensors.length > 0 && html`<p>Moved most, share of the score:
      ${inc.sensors.map((s, i) => html`${i ? ', ' : ' '}<b>${s.label}</b>${s.share_pct != null ? ` ${s.share_pct}%` : ''}`)}</p>`}` };
}

// The localizer names junctions to inspect, not one answer; the benchmark says how good the list is.
function LocatedStep({ inc, bench, net }) {
  if (!inc) return { on: false, title: STEPS[2] };
  if (!inc.candidates.length) {
    return { on: false, title: STEPS[2], body: html`<p>The localizer has not answered for this alert yet.</p>` };
  }
  const n = inc.candidates.length;
  const weak = inc.led_by ? ', weak pressure signal' : '';  // the note below, in the folded line
  return { on: true, title: STEPS[2], short: `${n} to inspect, ${inc.candidates[0]} first${weak}`, body: html`
    <p>${COUNT[n] || n} junction${n === 1 ? '' : 's'} to inspect, best first${inc.located_after_s != null
      ? html`, <b>${inc.located_after_s} s</b> after the click` : ''}.</p>
    <ol class="cands" aria-label="Candidate junctions, best first">${inc.candidates.map((c, i) => html`<li><i>${i + 1}</i>${c}</li>`)}</ol>
    ${inc.led_by && html`<p class="note">Most of this alert's score came from <b>${inc.led_by.label}</b>${inc.led_by.share_pct != null
      ? `, ${inc.led_by.share_pct}%` : ''}. The localizer reads only the ${net ? net.counts.pressure_sensors : ''} pressure sensors, so this list rests on a weaker signal.</p>`}
    ${bench && html`<p class="fine">${bench.sentence}</p>`}` };
}

// A number that counts to its new value (the loss when the model's order replaces the template's).
function useTicker(value) {
  const [shown, setShown] = useState(value);
  const last = useRef(value);
  useEffect(() => {
    const from = last.current;
    last.current = value;
    if (value == null || from == null || from === value || matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setShown(value);
      return undefined;
    }
    let frame = 0;
    const t0 = performance.now();
    const step = (t) => {
      const p = Math.min(1, (t - t0) / 700);
      setShown(from + (value - from) * (1 - Math.pow(1 - p, 3)));
      if (p < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [value]);
  return shown;
}

function OrderCard({ order, inc }) {
  const loss = useTicker(order.loss_m3h);
  const more = order.actions.length - 1;
  return html`<div class="wo" data-state=${order.state}>
    <div class="wo-top">
      ${order.priority && html`<span class="pri">Priority ${order.priority}</span>`}
      <span class="gen">${order.version}</span>
    </div>
    <dl class="wo-body">
      <div class="wo-loss">
        <dt>Estimated loss${order.loss_at ? ` at ${order.loss_at}` : ''}</dt>
        <dd class="big">${loss == null ? html`<span class="fine">not in the order</span>` : loss.toFixed(1)}${loss != null && html` <small>m³ per hour</small>`}</dd>
        ${inc.loss_now_m3h != null && html`<dd class="now">${inc.loss_now_m3h.toFixed(1)} m³/h now</dd>`}
      </div>
      <div class="wo-pipes">
        <dt>Pipes to inspect, in order</dt>
        <dd>${order.pipes.length ? html`<ol class="chips">${order.pipes.map((p, i) => html`<li><i>${i + 1}</i>${p}</li>`)}</ol>`
          : html`<span class="fine">none named</span>`}</dd>
      </div>
    </dl>
    ${order.actions.length > 0 && html`<details class="wo-act">
      <summary>1. ${order.actions[0]}${more > 0 && html` <small>and ${more} more step${more === 1 ? '' : 's'}</small>`}</summary>
      <ol start="2">${order.actions.slice(1).map((step) => html`<li>${step}</li>`)}</ol>
    </details>`}
    ${order.sources.length > 0 && html`<p class="wo-src">Source: ${order.sources.join('; ')}</p>`}
  </div>`;
}

function OrderStep({ inc }) {
  const title = STEPS[3];
  if (!inc) return { on: false, title };
  const o = inc.order;
  if (!o || o.state === 'writing') {
    return { on: false, title, body: html`<p>Writing the work order for this alert.</p>
      <div class="working" role="progressbar" aria-label="Writing the work order"></div>` };
  }
  if (o.state === 'unreadable') {
    return { on: false, title, body: html`<p class="problem">The work order in the lake could not be read.</p>` };
  }
  return { on: true, title, short: o.short, body: html`<${OrderCard} order=${o} inc=${inc} />` };
}

const thousands = (m) => m.toLocaleString('en');

function RevealStep({ inc, revealed, onReveal, bench }) {
  const title = STEPS[4];
  if (!inc) return { on: false, title };
  const t = inc.truth;
  if (!t) {
    const why = !inc.leak_id ? 'No leak started from this screen belongs to this alert, so there is nothing to check it against.'
      : !inc.candidates.length ? 'This needs the candidates of step 3.'
        : 'The pipe distances are being prepared.';
    return { on: false, title, body: html`<p>${why}</p>` };
  }
  if (!revealed) {
    return { on: false, title, body: html`<button class="btn ghost" onClick=${onReveal}>Reveal the true leak</button>` };
  }
  let where;
  if (t.rank === 1) where = html`Candidate 1 is the leak junction itself.`;
  else if (t.to_first_m != null) {
    where = html`Candidate 1 is <b>${thousands(t.to_first_m)} m</b> away along the pipes${bench
      ? ` (benchmark median ${thousands(bench.first_median_m)} m)` : ''}.`;
  }
  else where = 'No pipe path joins it to candidate 1.';
  const nearest = t.rank == null && t.nearest_rank != null && t.nearest_rank !== 1
    ? html` The nearest, candidate ${t.nearest_rank}, is <b>${thousands(t.nearest_m)} m</b> away.` : '';
  // Does the crew's list reach the leak? The first work-order pipe that ends at the leak junction.
  const pipe = !t.order_checked ? ''
    : t.order_pipe ? html` Pipe <b>${t.order_pipe.rank}</b> of the work order, ${t.order_pipe.name}, ends at the leak.`
      : ' No pipe in the work order ends at the leak.';
  return { on: true, title, short: t.short, body: html`
    <p>The leak is at <b>${t.node_id}</b>, ${t.rank != null ? html`candidate <b>${t.rank}</b> of ${t.of}.` : html`not one of the ${t.of} candidates.`}${' '}${where}${nearest}${pipe}</p>
    ${(t.flow_m3h != null || t.loss_now_m3h != null) && html`<p>${t.flow_m3h != null && html`Simulated leak <b>${t.flow_m3h.toFixed(1)} m³/h</b>`}${t.flow_m3h != null && t.loss_now_m3h != null ? ', ' : ''}${t.loss_now_m3h != null && html`current estimate <b>${t.loss_now_m3h.toFixed(1)} m³/h</b>`}.</p>`}` };
}

// The model writes light markdown: **bold**, `code` and "- " bullets. Rendered as elements, never as HTML.
function rich(text) {
  return text.split('\n').map((line, n) => {
    const bullet = /^\s*[-*]\s+/.test(line);
    const body = bullet ? line.replace(/^\s*[-*]\s+/, '') : line;
    const parts = body.split(/(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g).filter(Boolean).map((p) => {
      if (p.startsWith('**') && p.endsWith('**')) return html`<b>${p.slice(2, -2)}</b>`;
      if (p.length > 2 && p.startsWith('*') && p.endsWith('*')) return html`<i>${p.slice(1, -1)}</i>`;
      if (p.startsWith('`') && p.endsWith('`')) return html`<code>${p.slice(1, -1)}</code>`;
      return p;
    });
    return html`${n ? '\n' : ''}${bullet ? '• ' : ''}${parts}`;
  });
}

function AnswerChart({ chart, tz, tzName }) {
  const el = useRef(null);
  useEffect(() => (window.echarts ? drawAnswerChart(el.current, chart, tz, tzName) : undefined), [chart, tz]);
  return html`<figure class="answer-chart"><figcaption>${chart.title}</figcaption><div class="answer-plot" ref=${el} aria-hidden="true"></div></figure>`;
}

function Answer({ qa, tz, tzName }) {
  if (qa.state === 'asking') {
    return html`<li class="answer"><p class="q">${qa.question}</p><p>Waiting for the copilot.</p>
      <div class="working" role="progressbar" aria-label="Waiting for the copilot"></div></li>`;
  }
  const how = qa.how;
  const summary = how && [
    how.tools.length ? `tools: ${how.tools.join(', ')}` : 'no tools',
    how.model ? `model ${how.model}` : 'no model',
    `${how.seconds} s`,
  ].join(' · ');
  return html`<li class="answer" data-state=${qa.state}>
    <p class="q">${qa.question}</p>
    <p class="a">${rich(qa.text || '')}</p>
    ${(qa.charts || []).map((c) => html`<${AnswerChart} chart=${c} tz=${tz} tzName=${tzName} />`)}
    ${how && html`<details class="how"><summary>How this was answered: ${summary}</summary>
      ${qa.sent && how.model && html`<p>Sent to the model before the question: ${qa.sent}</p>`}
      ${how.sql.length > 0 && html`<p>SQL that ran:</p>${how.sql.map((q) => html`<pre class="sql">${q}</pre>`)}`}
      ${how.checked.length > 0 && html`<p>Checked against the tool results (times in ${tzName}): ${how.checked.join('; ')}</p>`}
      ${how.sql.length > 0 && html`<p>The lake keeps UTC; the answer's times are shown in ${tzName}.</p>`}
      ${!how.model && html`<p>${how.prefilter
        ? 'Refused by the scope filter before any model call: the question is off-topic or tries to change the copilot\'s instructions.'
        : 'Answered by the estimate_loss tool directly; no language model was involved.'}</p>`}
    </details>`}
  </li>`;
}

// A conversation: every question and answer stays above the question box, oldest first, until a
// new leak is started or the replay restarts (useCopilot). The newest one is scrolled into view.
function CopilotStep({ inc, items, asking, ask, settings, about }) {
  const title = STEPS[5];
  const [text, setText] = useState('');
  const last = useRef(null);
  const loss = settings ? settings.loss_question : null;
  const tz = settings ? settings.tz : 'UTC';
  const tzName = settings ? settings.tz_name : 'UTC';
  const newest = items.length ? `${items[items.length - 1].key}:${items[items.length - 1].state}` : '';
  useEffect(() => {
    if (!last.current || !newest) return;  // nothing asked yet: leave the rail where it is
    const smooth = !matchMedia('(prefers-reduced-motion: reduce)').matches;
    last.current.scrollIntoView({ block: 'nearest', behavior: smooth ? 'smooth' : 'auto' });
  }, [newest]);
  const submit = (e) => {
    e.preventDefault();
    if (text.trim() && !asking) { ask({ question: text.trim() }); setText(''); }
  };
  return { on: items.some((q) => q.state !== 'asking'), title, body: html`
    ${items.length > 0 && html`<ol class="chat" aria-label="Questions and answers" aria-live="polite">
      ${items.map((q) => html`<${Answer} key=${q.key} qa=${q} tz=${tz} tzName=${tzName} />`)}
    </ol>`}
    <div class="ask-box" ref=${last}>
      ${inc && loss && html`<button class="chipq" disabled=${asking} onClick=${() => ask({ direct: true, question: loss })}>${loss}</button>`}
      <form class="row ask" onSubmit=${submit}>
        <input aria-label="Question for the copilot" maxlength="500" value=${text}
          placeholder=${inc ? 'Ask about this alert, a sensor or a procedure' : 'Ask about a sensor, an alert or a procedure'}
          onInput=${(e) => setText(e.target.value)} />
        <button class="btn" type="submit" disabled=${asking || !text.trim()}>Ask</button>
      </form>
      ${about && html`<p class="fine about">${about}</p>`}
    </div>` };
}

// One question at a time. The conversation is kept until `session` changes: a new leak (its id)
// or a restarted replay. An answer that arrives after that is dropped with its question. The
// tab's sessionStorage keeps the answered ones across a reload of the page (F5).
const CHAT_STORE = 'leakwatch.copilot';

function storedChat(session) {
  try {
    const saved = JSON.parse(sessionStorage.getItem(CHAT_STORE) || 'null');
    return saved && saved.session === session && Array.isArray(saved.items) ? saved.items : [];
  } catch {
    return [];
  }
}

function useCopilot(inc, session) {
  const [items, setItems] = useState(() => storedChat(session));
  const last = useRef(session);
  useEffect(() => {
    if (last.current === session) return;  // a new leak or a restart: a new conversation
    last.current = session;
    setItems([]);
  }, [session]);
  useEffect(() => {
    try {
      sessionStorage.setItem(CHAT_STORE, JSON.stringify({ session, items: items.filter((q) => q.state !== 'asking') }));
    } catch { /* storage full or off: the conversation still lasts until the page is reloaded */ }
  }, [items, session]);
  const ask = async ({ question, direct = false }) => {
    const key = `q${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
    setItems((xs) => [...xs, { key, state: 'asking', question }]);
    const res = await call('POST', '/api/copilot/ask', { question, direct, alert_id: inc ? inc.alert_id : null });
    const done = res.ok ? { ...res, key, question } : { key, state: 'error', question, text: res.message };
    setItems((xs) => xs.map((q) => (q.key === key ? done : q)));
  };
  return { items, asking: items.some((q) => q.state === 'asking'), ask };
}

function Rail({ vm, net, size, canStart, why, busy, onStart, notice, revealed, onReveal, settings, session }) {
  const [unfolded, setUnfolded] = useState({});
  const copilot = useCopilot(vm.incident, session);
  const inc = vm.incident;
  const first = LeakStep({ leaks: vm.leaks, net, size, canStart, why, busy, onStart, inc,
    detector: settings ? settings.detector : null, days: settings ? settings.leak_visible_days : null });
  const bench = settings ? settings.localizer : null;
  const steps = [first, AlertStep({ inc }), LocatedStep({ inc, bench, net }), OrderStep({ inc }),
    RevealStep({ inc, revealed, onReveal, bench }), CopilotStep({ inc, items: copilot.items, asking: copilot.asking, ask: copilot.ask, settings, about: vm.ask_about })];
  const now = steps.reduce((last, s, i) => (s.on ? i : last), 0);
  // Finished steps fold to one line once the work order is on screen, so the open step fits.
  // The reveal stays open: it is the check of everything above it.
  const foldable = (i) => (steps[3].on && i < 3) || ((steps[4].on || steps[5].on) && i === 3);
  const id = (i) => `${inc ? inc.alert_id : ''}:${i}`;
  const folded = (i) => foldable(i) && !unfolded[id(i)];
  const toggle = (i) => setUnfolded({ ...unfolded, [id(i)]: !unfolded[id(i)] });
  return html`<aside class="rail" aria-label="Incident, step by step">
    <h2>Incident</h2>
    ${notice && html`<p class="problem" role="alert">${notice}</p>`}
    <ol class="steps">
      ${steps.map((s, i) => {
        const fold = folded(i);
        const head = html`${s.title}${fold && s.short && html`<span class="short">${s.short}</span>`}${s.when && html`<span class="when">${s.when}</span>`}`;
        return html`<li class=${`${s.on ? 'on' : ''} ${i === now && s.on ? 'now' : ''} ${fold ? 'fold' : ''}`}>
          <span class="n" aria-hidden="true">${i + 1}</span>
          <div><h3>${foldable(i) ? html`<button class="fold-toggle" aria-expanded=${fold ? 'false' : 'true'}
            onClick=${() => toggle(i)}>${head}</button>` : head}</h3>
            ${s.body && !fold && html`<div class="body">${s.body}</div>`}</div>
        </li>`;
      })}
    </ol>
  </aside>`;
}

/* ------------------------------------------------------------------- history, results */

function useOnce(url, active) {
  const [data, setData] = useState(null);
  useEffect(() => {
    if (!active || (data && data.ok)) return;
    call('GET', url).then(setData);
  }, [active]);
  return data;
}

function ResultsView({ active }) {
  const res = useOnce('/api/results', active);
  const det = useRef(null);
  const loc = useRef(null);
  useEffect(() => (res && res.ok && window.echarts ? drawResultsCharts(det.current, loc.current, res) : undefined), [res]);
  if (!res) return html`<main class="page" aria-busy="true"><p class="fine">Reading reports/metrics.</p></main>`;
  if (!res.ok) return html`<main class="page"><p class="problem" role="alert">${res.message}</p></main>`;
  const commit = (r) => (r.commit ? html`, code ${r.commit}` : '');
  return html`<main class="page" aria-labelledby="results-title">
    <h2 id="results-title">Results, as measured</h2>
    <p class="fine">The same rows as the README's results table, written by <code>make reports</code>. Misses are listed with their real numbers.</p>
    <div class="charts2">
      <figure><figcaption>2019 leaks detected with at most 2 false alarms per week, beside the chance level</figcaption>
        <div class="results-plot" ref=${det} aria-hidden="true"></div><p class="fine">${res.detection.file}${commit(res.detection)}</p></figure>
      <figure><figcaption>Localizers: medium and large simulated leaks with the true junction in the top 5</figcaption>
        <div class="results-plot" ref=${loc} aria-hidden="true"></div><p class="fine">${res.localizer.file}${commit(res.localizer)}</p></figure>
    </div>
    <table class="results">
      <thead><tr><th scope="col">Result</th><th scope="col">Measured</th><th scope="col">Source</th></tr></thead>
      <tbody>${res.rows.map((r) => html`<tr><th scope="row">${r.result}</th><td>${r.value}</td>
        <td class="src">${r.file}${r.commit ? html`<br />code ${r.commit}` : ''}</td></tr>`)}</tbody>
    </table>
  </main>`;
}

function HistoryView({ active }) {
  const res = useOnce('/api/history', active);
  const [year, setYear] = useState(null);
  if (!res) return html`<main class="page" aria-busy="true"><p class="fine">Reading the alerts of the main lake.</p></main>`;
  if (!res.ok) return html`<main class="page"><p class="problem" role="alert">${res.message}</p></main>`;
  const years = Object.keys(res.years);
  const shown = res.alerts.filter((a) => !year || a.year === year);
  return html`<main class="page" aria-labelledby="history-title">
    <h2 id="history-title">History: batch alerts of the main lake</h2>
    <p class="fine">BattLeDIM 2018 and 2019 scored by the live K-Means model (<code>make bdt-eval</code>), newest first. Times in IST.</p>
    <div class="row filters" role="group" aria-label="Year">
      <button class="btn ghost" aria-pressed=${year == null ? 'true' : 'false'} onClick=${() => setYear(null)}>All (${res.alerts.length})</button>
      ${years.map((y) => html`<button class="btn ghost" aria-pressed=${year === y ? 'true' : 'false'} onClick=${() => setYear(y)}>${y} (${res.years[y]})</button>`)}
    </div>
    ${shown.length === 0 ? html`<p>No batch alerts in the main lake.</p>` : html`<table class="results history">
      <thead><tr><th scope="col">Started</th><th scope="col">Ended</th><th scope="col">Lasted</th><th scope="col">Area</th>
        <th scope="col">Score × threshold</th><th scope="col">Loss, m³/h</th><th scope="col">Candidate 1</th><th scope="col">Work order</th></tr></thead>
      <tbody>${shown.map((a) => html`<tr><th scope="row">${a.started}</th><td>${a.ended}</td><td>${a.lasted || ''}</td><td>${a.area || ''}</td>
        <td>${a.ratio != null ? `${a.ratio}×` : ''}</td><td>${a.loss_m3h != null ? a.loss_m3h.toFixed(1) : ''}</td>
        <td>${a.candidate || ''}</td><td>${a.order || ''}</td></tr>`)}</tbody>
    </table>`}
  </main>`;
}

const VIEWS = [['live', 'Live'], ['history', 'History'], ['results', 'Results']];

function Tabs({ view, setView }) {
  return html`<nav class="tabs" aria-label="Views">
    ${VIEWS.map(([id, label]) => html`<button aria-current=${view === id ? 'page' : undefined} onClick=${() => setView(id)}>${label}</button>`)}
  </nav>`;
}

/* --------------------------------------------------------------------------------- app */

const SERVER_LOST = {
  runner: { code: 'error', label: 'Dashboard server not reachable', detail: 'Start it with `make demo`.' },
  clock: null,
  banner: { kind: 'unknown', title: 'Dashboard server not reachable', sub: 'This page stopped receiving data. Start the demo again with `make demo`.' },
  leaks: { pending: [], active: [], failed: [] },
  incident: null,
  drops: null,
  charts: null,
};

// The camera flies to the candidates once per alert, and back home when the alert is gone.
function useCamera(map, incident) {
  const [zoomed, setZoomed] = useState(false);
  const flown = useRef(null);
  const target = incident && incident.candidates.length ? incident.alert_id : null;
  useEffect(() => {
    if (!map) return;
    if (target && flown.current !== target) {
      flown.current = target;
      setZoomed(map.focus(incident.focus));
    } else if (!target && flown.current) {
      flown.current = null;
      map.home();
      setZoomed(false);
    }
  }, [map, target]);
  const frame = (nodes) => { if (map) setZoomed(map.focus(nodes, true)); };  // the reveal, framed close
  const toggle = () => {
    if (!map || !incident) return;
    if (zoomed) { map.home(); setZoomed(false); } else setZoomed(map.focus(incident.focus));
  };
  return { zoomed, toggle, frame };
}

function App() {
  const { vm, lost, settings } = useLive();
  const { net, error: netError } = useNetwork();
  const [pick, setPick] = useState(null);
  const [size, setSize] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [mapError, setMapError] = useState(null);
  const [map, setMap] = useState(null);
  const [revealed, setRevealed] = useState({});
  const [view, setView] = useState('live');
  const mapBox = useRef(null);
  const camera = useCamera(map, lost ? null : vm && vm.incident);
  // The copilot's conversation lasts until a new leak is started (the newest leak's id) or the
  // replay restarts; a moment without the server keeps it.
  const session = useRef('none');
  if (vm && !lost) {
    const { pending, active } = vm.leaks;
    const newestLeak = pending.length ? pending[pending.length - 1] : active[active.length - 1];
    session.current = vm.runner.code === 'starting' ? 'starting' : newestLeak ? newestLeak.leak_id : 'none';
  }
  useEffect(() => {
    if (!pick) return undefined;
    const close = (e) => { if (e.key === 'Escape') setPick(null); };
    addEventListener('keydown', close);
    return () => removeEventListener('keydown', close);
  }, [pick]);
  // The server's first snapshot can take a few seconds after `make demo` starts it.
  if (!vm && !lost) return html`<div class="app" aria-busy="true"><p class="boot">Reading the demo's state…</p></div>`;
  const shown = lost ? { ...(vm || {}), ...SERVER_LOST } : vm;
  const leakSize = size ?? (settings ? settings.leak_mm.default : 30);
  const canStart = shown.runner.code === 'running';
  const why = shown.runner.code === 'finished' || shown.runner.code === 'finishing'
    ? 'The replay has ended. Restart it to start a leak.'
    : `A leak can start once the replay says Running. Now: ${shown.runner.label}.`;
  const start = async (node) => {
    setBusy(true);
    const res = await call('POST', '/api/demo/inject', { node_id: node, diameter_mm: leakSize });
    setBusy(false);
    setNotice(res.ok ? null : res.message);
    if (res.ok) setPick(null);
  };
  const onPick = (node, at) => {
    const box = mapBox.current.getBoundingClientRect();
    setPick(node ? { node, x: at.x, y: at.y, maxX: box.width, maxY: box.height } : null);
  };
  const inc = shown.incident;
  const truth = inc && inc.truth && revealed[inc.alert_id] ? inc.truth : null;
  const reveal = () => {
    if (!inc || !inc.truth) return;
    setRevealed({ ...revealed, [inc.alert_id]: true });
    camera.frame([inc.truth.node_id, inc.candidates[0]]);  // the leak, candidate 1 and the line between them
  };
  const restart = async () => { const res = await call('POST', '/api/demo/restart'); setNotice(res.ok ? null : res.message); };
  return html`<div class="app">
    <${Header} runner=${shown.runner} clock=${shown.clock} analysed=${shown.analysed} tabs=${html`<${Tabs} view=${view} setView=${setView} />`}>
      <${Controls} settings=${settings} clock=${shown.clock} runner=${shown.runner} size=${leakSize} setSize=${setSize} notify=${setNotice} />
    <//>
    <div class=${view === 'live' ? 'live' : 'live off'}>
    <main class="map" ref=${mapBox}>
      ${net && !mapError && html`<${MapView} net=${net} selected=${pick ? pick.node : null} drops=${shown.drops}
        incident=${shown.incident} truth=${truth} label=${mapLabel(net, shown.incident, shown.drops, truth)}
        onReady=${setMap} onPick=${onPick}
        onError=${() => setMapError('The map needs WebGL 2, which this browser did not provide. The panels on the right still work.')} />`}
      ${(netError || mapError) && html`<p class="map-note" role="alert">${mapError || netError}</p>`}
      <${Banner} banner=${shown.banner} runner=${shown.runner} onRestart=${restart} />
      ${map && shown.incident && shown.incident.candidates.length > 0 && html`<div class="map-buttons">
        <button class="btn ghost" onClick=${camera.toggle}>${camera.zoomed ? 'Show the whole network' : 'Zoom to the candidates'}</button>
      </div>`}
      ${net && !mapError && html`<${Legend} net=${net} drops=${shown.drops} incident=${shown.incident} truth=${truth}
        floor=${shown.drops ? shown.drops.floor_m : settings ? settings.ring_floor_m : 0} />`}
      ${pick && net && html`<${PickCard} pick=${pick} net=${net} size=${leakSize} canStart=${canStart} why=${why}
        busy=${busy} onStart=${() => start(pick.node)} onClose=${() => setPick(null)} />`}
    </main>
    <${Instruments} charts=${shown.charts} />
    <${Rail} vm=${shown} net=${net} size=${leakSize} canStart=${canStart} why=${why} busy=${busy} onStart=${start} notice=${notice}
      revealed=${!!(inc && revealed[inc.alert_id])} onReveal=${reveal} settings=${settings} session=${session.current} />
    </div>
    ${view === 'history' && html`<${HistoryView} active=${true} />`}
    ${view === 'results' && html`<${ResultsView} active=${true} />`}
  </div>`;
}

render(html`<${App} />`, document.getElementById('app'));
