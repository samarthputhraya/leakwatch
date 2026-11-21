// Control-room screen (D-052). The server shapes everything that is shown (view.py); this file
// only lays it out. No build step: Preact + htm are one vendored module.
import { html, render, useEffect, useState } from './vendor/htm-preact.module.js';

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

// Poll the snapshot. `lost` is true while the dashboard server itself does not answer.
function useLive() {
  const [vm, setVm] = useState(null);
  const [lost, setLost] = useState(false);
  useEffect(() => {
    let stop = false;
    let timer = 0;
    let pollMs = 1000;
    const tick = async () => {
      try {
        const res = await fetch('/api/live', { cache: 'no-store' });
        if (!res.ok) throw new Error(String(res.status));
        setVm(await res.json());
        setLost(false);
      } catch {
        setLost(true);
      }
      if (!stop) timer = setTimeout(tick, pollMs);
    };
    fetch('/api/settings')
      .then((res) => res.json())
      .then((s) => { pollMs = s.poll_ms; })
      .catch(() => {})
      .finally(tick);
    return () => { stop = true; clearTimeout(timer); };
  }, []);
  return { vm, lost };
}

function Header({ runner, clock }) {
  const width = clock && clock.replay_fraction != null ? `${(clock.replay_fraction * 100).toFixed(1)}%` : '0';
  return html`<header class="header">
    <div class="brand"><b>LeakWatch</b><span>L-Town, the BattLeDIM benchmark network</span></div>
    <dl class="facts">
      ${clock && html`<div class="fact"><dt>Simulated time</dt><dd>${clock.sim_time}</dd></div>`}
      ${clock && clock.speed && html`<div class="fact"><dt>Replay speed</dt><dd>${clock.speed}</dd></div>`}
      ${clock && clock.replay_left && runner.code !== 'finished' && html`<div class="fact"><dt>Replay ends in</dt><dd>${clock.replay_left}</dd></div>`}
      <div class="status" data-code=${runner.code} title=${runner.detail}><span class="dot"></span>${runner.label}</div>
    </dl>
    <div class="replay" role="presentation"><i style=${{ width }}></i></div>
  </header>`;
}

function Banner({ banner }) {
  return html`<div class="banner" data-kind=${banner.kind} role="status" aria-live="polite">
    <${Icon} kind=${banner.kind} />
    <div><b>${banner.title}</b>${banner.sub && html`<small>${banner.sub}</small>`}</div>
  </div>`;
}

const SERVER_LOST = {
  runner: { code: 'error', label: 'Dashboard server not reachable', detail: 'Start it with `make demo`.' },
  clock: null,
  banner: { kind: 'unknown', title: 'Dashboard server not reachable', sub: 'This page stopped receiving data. Start the demo again with `make demo`.' },
};

function App() {
  const { vm, lost } = useLive();
  if (!vm && !lost) return html`<div class="app" aria-busy="true"></div>`;
  const shown = lost ? { ...(vm || {}), ...SERVER_LOST } : vm;
  return html`<div class="app">
    <${Header} runner=${shown.runner} clock=${shown.clock} />
    <main class="map"><${Banner} banner=${shown.banner} /></main>
    <section class="strip" aria-label="Instruments"></section>
    <aside class="rail" aria-label="Incident, step by step"></aside>
  </div>`;
}

render(html`<${App} />`, document.getElementById('app'));
