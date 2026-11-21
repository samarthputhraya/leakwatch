// Control-room screen (D-052). The server shapes everything that is shown (view.py); this file
// only lays it out. No build step: Preact + htm are one vendored module.
import { html, render, useEffect, useRef, useState } from './vendor/htm-preact.module.js';
import { createMap } from './map.js';

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
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
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
    call('GET', '/api/settings').then((s) => {
      if (s.ok) { pollMs = s.poll_ms; setSettings(s); }
      tick();
    });
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
  const down = runner.code === 'offline' || runner.code === 'stale' || runner.code === 'error';
  return html`<details class="controls">
    <summary>Demo controls</summary>
    <div class="controls-panel">
      <label for="leak-size">Leak size <output>${size} mm</output></label>
      <input id="leak-size" type="range" min=${settings.leak_mm.min} max=${settings.leak_mm.max} step="5"
        value=${size} onInput=${(e) => setSize(Number(e.target.value))} />
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
      <p class="fine">Restart clears the leaks and alerts of this run and replays the week from Monday.</p>
    </div>
  </details>`;
}

function Header({ runner, clock, children }) {
  const width = clock && clock.replay_fraction != null ? `${(clock.replay_fraction * 100).toFixed(1)}%` : '0';
  return html`<header class="header">
    <div class="brand"><b>LeakWatch</b><span>L-Town, the BattLeDIM benchmark network</span></div>
    <dl class="facts">
      ${clock && html`<div class="fact"><dt>Simulated time</dt><dd>${clock.sim_time}</dd></div>`}
      ${clock && clock.speed && html`<div class="fact"><dt>Replay speed</dt><dd>${clock.speed}</dd></div>`}
      ${clock && clock.replay_left && runner.code !== 'finished' && html`<div class="fact"><dt>Replay ends in</dt><dd>${clock.replay_left}</dd></div>`}
      <div class="status" data-code=${runner.code} title=${runner.detail}><span class="dot"></span>${runner.label}</div>
      ${children}
    </dl>
    <div class="replay" role="presentation"><i style=${{ width }}></i></div>
  </header>`;
}

/* --------------------------------------------------------------------------------- map */

function Banner({ banner, runner, onRestart }) {
  return html`<div class="banner" data-kind=${banner.kind} role="status" aria-live="polite">
    <${Icon} kind=${banner.kind} />
    <div><b>${banner.title}</b>${banner.sub && html`<small>${banner.sub}</small>`}</div>
    ${runner.code === 'finished' && html`<button class="btn" onClick=${onRestart}>Restart replay</button>`}
  </div>`;
}

function MapView({ net, selected, onPick, onError }) {
  const el = useRef(null);
  const map = useRef(null);
  const handlers = useRef({});
  handlers.current = { onPick, onError };
  useEffect(() => {
    map.current = createMap(el.current, net, {
      onPick: (node, at) => handlers.current.onPick(node, at),
      onError: (error) => handlers.current.onError(error),
    });
    return () => { if (map.current) map.current.destroy(); };
  }, [net]);
  useEffect(() => { if (map.current) map.current.set({ selected }); }, [selected]);
  return html`<div class="deck" ref=${el}></div>`;
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

/* -------------------------------------------------------------------------------- rail */

const STEPS = ['Start a leak', 'Alert', 'Located', 'Work order', 'Check against the true leak', 'Ask the copilot'];

function LeakStep({ leaks, net, size, canStart, why, busy, onStart }) {
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
    return { on: true, title: `Leak started at ${active.node_id}`, when: active.started, body: html`
      <p><b>${active.diameter_mm} mm</b> hole, simulated with EPANET${leaks.active.length > 1 ? `. ${leaks.active.length - 1} earlier leak${leaks.active.length > 2 ? 's' : ''} still running.` : ''}</p>` };
  }
  const known = net && net.junctions.includes(node);
  return { on: true, open: true, title: 'Start a leak', body: html`
    ${failed && html`<p class="problem" role="alert">The leak at ${failed.node_id} could not be simulated. Pick another junction.</p>`}
    <p>Click any junction on the map, or name one here. EPANET simulates the leak and adds it to the replayed readings.</p>
    <form class="row" onSubmit=${(e) => { e.preventDefault(); if (known) onStart(node); }}>
      <input list="junctions" value=${node} placeholder="Junction, for example n470" aria-label="Junction"
        onInput=${(e) => setNode(e.target.value.trim())} />
      <button class="btn" type="submit" disabled=${!known || !canStart || busy}>${busy ? 'Sending…' : `Start a ${size} mm leak`}</button>
    </form>
    ${net && html`<datalist id="junctions">${net.junctions.map((j) => html`<option value=${j} />`)}</datalist>`}
    ${!canStart && html`<p class="fine">${why}</p>`}` };
}

function Rail({ vm, net, size, canStart, why, busy, onStart, notice }) {
  const first = LeakStep({ leaks: vm.leaks, net, size, canStart, why, busy, onStart });
  const steps = [first, ...STEPS.slice(1).map((title) => ({ on: false, title }))];
  return html`<aside class="rail" aria-label="Incident, step by step">
    <h2>Incident</h2>
    ${notice && html`<p class="problem" role="alert">${notice}</p>`}
    <ol class="steps">
      ${steps.map((s, i) => html`<li class=${s.on ? 'on' : ''}>
        <span class="n">${i + 1}</span>
        <div><h3>${s.title}${s.when && html`<span class="when">${s.when}</span>`}</h3>
          ${s.body && html`<div class="body">${s.body}</div>`}</div>
      </li>`)}
    </ol>
  </aside>`;
}

/* --------------------------------------------------------------------------------- app */

const SERVER_LOST = {
  runner: { code: 'error', label: 'Dashboard server not reachable', detail: 'Start it with `make demo`.' },
  clock: null,
  banner: { kind: 'unknown', title: 'Dashboard server not reachable', sub: 'This page stopped receiving data. Start the demo again with `make demo`.' },
  leaks: { pending: [], active: [], failed: [] },
};

function App() {
  const { vm, lost, settings } = useLive();
  const { net, error: netError } = useNetwork();
  const [pick, setPick] = useState(null);
  const [size, setSize] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [mapError, setMapError] = useState(null);
  const mapBox = useRef(null);
  if (!vm && !lost) return html`<div class="app" aria-busy="true"></div>`;
  const shown = lost ? { ...(vm || {}), ...SERVER_LOST } : vm;
  const leakSize = size ?? (settings ? settings.leak_mm.default : 30);
  const canStart = shown.runner.code === 'running';
  const why = shown.runner.code === 'finished' ? 'The replay has finished. Restart it to start a leak.'
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
  const restart = async () => { const res = await call('POST', '/api/demo/restart'); setNotice(res.ok ? null : res.message); };
  return html`<div class="app">
    <${Header} runner=${shown.runner} clock=${shown.clock}>
      <${Controls} settings=${settings} clock=${shown.clock} runner=${shown.runner} size=${leakSize} setSize=${setSize} notify=${setNotice} />
    <//>
    <main class="map" ref=${mapBox}>
      ${net && !mapError && html`<${MapView} net=${net} selected=${pick ? pick.node : null} onPick=${onPick}
        onError=${() => setMapError('The map needs WebGL 2, which this browser did not provide. The panels on the right still work.')} />`}
      ${(netError || mapError) && html`<p class="map-note" role="alert">${mapError || netError}</p>`}
      <${Banner} banner=${shown.banner} runner=${shown.runner} onRestart=${restart} />
      ${pick && net && html`<${PickCard} pick=${pick} net=${net} size=${leakSize} canStart=${canStart} why=${why}
        busy=${busy} onStart=${() => start(pick.node)} onClose=${() => setPick(null)} />`}
    </main>
    <section class="strip" aria-label="Instruments"></section>
    <${Rail} vm=${shown} net=${net} size=${leakSize} canStart=${canStart} why=${why} busy=${busy} onStart=${start} notice=${notice} />
  </div>`;
}

render(html`<${App} />`, document.getElementById('app'));
