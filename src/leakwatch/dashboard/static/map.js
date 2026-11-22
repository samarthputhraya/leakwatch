// The L-Town map (D-052): deck.gl layers in the network's own metre coordinates, no basemap.
// `createMap` owns the WebGL canvas; the page calls `set()` with what changed.
// Everything drawn comes from /api/network and /api/live. The pipe tint is the only interpolated
// element: per-junction weights (geometry.py) times the sensors' measured drops (D-053).
const {
  DeckGL, OrthographicView, PathLayer, ScatterplotLayer, TextLayer, LineLayer, LinearInterpolator, PathStyleExtension,
} = window.deck;

const FONT = 'Bahnschrift, "DIN Alternate", Barlow, sans-serif';
// Additive blending for the amber glow under the pipes.
const ADD = {
  blendColorSrcFactor: 'src-alpha', blendColorDstFactor: 'one',
  blendAlphaSrcFactor: 'one', blendAlphaDstFactor: 'one', depthWriteEnabled: false,
};
// Smallest view around the candidates (metres), so the streets around them stay readable, and the
// margin added around them: [min width, min height, margin x, margin y].
const FOCUS_MIN = [760, 480, 240, 180];
// The reveal frames the true leak and candidate 1 closer, so a short distance stays readable.
const REVEAL_MIN = [300, 180, 160, 120];
// Work-order pipe numbers once a metre is this many pixels (closer than the candidates' view).
const NUMBERS_PX_PER_M = 1.4;
const MOVE_MS = 700; // rings and tint move between two real windows in this time

function token(name) {
  const hex = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const mix = (a, b, t) => [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t));

export function createMap(el, net, { onPick, onError }) {
  const C = {
    sheet: token('--sheet'), ink: token('--ink'), ink3: token('--ink-3'), rule: token('--rule'),
    pipe: token('--pipe'), amber: token('--amber'), coral: token('--coral'),
  };
  const xy = (n) => net.nodes[n];
  const [x0, y0, x1, y1] = net.bounds;
  const pipes = net.pipes.map((p) => ({ ...p, path: [xy(p.a), xy(p.b)], heat: 0 }));
  const pipeById = Object.fromEntries(pipes.map((p) => [p.id, p]));
  const grid = [];
  const gx0 = Math.floor(x0 / 500) * 500 - 500, gx1 = Math.ceil(x1 / 500) * 500 + 500;
  const gy0 = Math.floor(y0 / 500) * 500 - 500, gy1 = Math.ceil(y1 / 500) * 500 + 500;
  for (let x = gx0; x <= gx1; x += 100) grid.push({ a: [x, gy0], b: [x, gy1], major: x % 500 === 0 });
  for (let y = gy0; y <= gy1; y += 100) grid.push({ a: [gx0, y], b: [gx1, y], major: y % 500 === 0 });

  // `scale`: pixels per metre of the current view, in half-zoom steps (labels depend on it);
  // `moved`: the user has panned or zoomed since the view was last set.
  const st = { selected: null, drops: null, incident: null, truth: null, zoomed: false, key: '', ripple: 0, scale: null, moved: false };
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const move = (ms) => (reduced ? {} : ms);

  // The free band of the canvas, measured: the banner covers its top and the legend its bottom,
  // and both change height with the window and the incident.
  function pads() {
    const box = el.getBoundingClientRect();
    const rect = (sel) => { const n = el.parentElement.querySelector(sel); return n ? n.getBoundingClientRect() : null; };
    const banner = rect('.banner'), legend = rect('.legend');
    const top = banner && banner.height ? banner.bottom - box.top + 14 : 24;
    const bottom = legend && legend.height ? box.bottom - legend.top + 10 : 24;
    return [Math.max(24, top), Math.max(24, bottom)];
  }

  // View that fits a box [minx, miny, maxx, maxy] into the free band of the canvas.
  function fit(box, padX) {
    const [padTop, padBottom] = pads();
    const w = Math.max(200, el.clientWidth - padX), h = Math.max(140, el.clientHeight - padTop - padBottom);
    const zoom = Math.log2(Math.min(w / (box[2] - box[0]), h / (box[3] - box[1])));
    const lift = (padBottom - padTop) / 2 / 2 ** zoom; // world y grows upwards (flipY false)
    return { target: [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2 - lift, 0], zoom };
  }
  const home = () => fit(net.bounds, 64);

  // `tight` frames the reveal (the leak and candidate 1) closer than the candidates' view.
  function focusBox(nodes, tight = false) {
    const pts = nodes.map(xy).filter(Boolean);
    if (!pts.length) return null;
    const [minW, minH, marginW, marginH] = tight ? REVEAL_MIN : FOCUS_MIN;
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const w = Math.max(minW, Math.max(...xs) - Math.min(...xs) + marginW);
    const h = Math.max(minH, Math.max(...ys) - Math.min(...ys) + marginH);
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
  }

  // Pipe tint: each junction's weighted sum of its sensors' drops; a pipe takes the mean of its ends.
  function updateHeat() {
    const values = st.drops ? st.drops.values : {};
    const at = {};
    for (const [node, pairs] of Object.entries(net.tint)) {
      let v = 0;
      for (const [i, w] of pairs) v += w * (values[net.sensors[i]] || 0);
      at[node] = v;
    }
    for (const p of pipes) p.heat = Math.min(1, ((at[p.a] || 0) + (at[p.b] || 0)) / 2 / net.tint_full_m);
  }
  const drop = (s) => (st.drops ? st.drops.values[s] || 0 : 0);

  // The work order's pipes, numbered in inspection order (numbers once the view is close enough:
  // at the candidates' view they would cover the candidates).
  function orderLayers(inc, z, key) {
    const ids = inc && inc.order && inc.order.pipes ? inc.order.pipes : [];
    const data = ids.map((id, i) => ({ p: pipeById[id], n: String(i + 1) })).filter((d) => d.p);
    if (!data.length) return [];
    const out = [new PathLayer({ id: 'order-pipes', data, getPath: (d) => d.p.path, getColor: C.ink,
      getWidth: st.zoomed ? 5 : 3.5, widthUnits: 'pixels', capRounded: true, updateTriggers: { getWidth: z } })];
    if (st.scale != null && st.scale >= NUMBERS_PX_PER_M) {
      out.push(new TextLayer({ id: 'order-numbers', data, getText: (d) => d.n, getSize: 12.5, getColor: C.sheet,
        getPosition: (d) => [(d.p.path[0][0] + d.p.path[1][0]) / 2, (d.p.path[0][1] + d.p.path[1][1]) / 2],
        fontFamily: FONT, fontWeight: 700, background: true, getBackgroundColor: C.ink, backgroundPadding: [4, 1],
        updateTriggers: { getText: key } }));
    }
    return out;
  }

  // The true leak after the reveal: coral ring with a pulse, a line to candidate 1 and its distance.
  function truthLayers(z) {
    const t = st.truth;
    const at = t && xy(t.node_id);
    if (!at) return [];
    const first = st.incident && st.incident.candidates.length ? xy(st.incident.candidates[0]) : null;
    const out = [];
    if (first && t.rank !== 1) {
      out.push(new LineLayer({ id: 'truth-line', data: [0], getSourcePosition: () => at, getTargetPosition: () => first,
        getColor: [...C.coral, 230], getWidth: 1.6, widthUnits: 'pixels' }));
    }
    const r = st.ripple % 1;
    out.push(
      new ScatterplotLayer({ id: 'truth-pulse', data: [0], getPosition: () => at, radiusUnits: 'pixels',
        getRadius: (st.zoomed ? 18 : 12) + r * 30, filled: false, stroked: true, getLineColor: [...C.coral, Math.round(255 * (1 - r))],
        lineWidthUnits: 'pixels', getLineWidth: 2, updateTriggers: { getRadius: [st.ripple, z], getLineColor: st.ripple } }),
      new ScatterplotLayer({ id: 'truth', data: [0], getPosition: () => at, radiusUnits: 'pixels',
        getRadius: st.zoomed ? 18 : 12, filled: false, stroked: true, getLineColor: C.coral, lineWidthUnits: 'pixels',
        getLineWidth: 2.8, updateTriggers: { getRadius: z } }),
    );
    const below = [0, (st.zoomed ? 18 : 12) + 16];
    const labels = [{ p: at, text: `True leak ${t.node_id}`, o: below }];
    if (first && t.rank !== 1 && t.to_first_m != null) {
      const distance = `${t.to_first_m.toLocaleString('en')} m to candidate 1`;
      // On the line when it is long enough on screen to carry it; otherwise under the leak's name.
      const onScreen = Math.hypot(at[0] - first[0], at[1] - first[1]) * (st.scale || 0);
      if (onScreen >= 220) labels.push({ p: [(at[0] + first[0]) / 2, (at[1] + first[1]) / 2], text: distance, o: [0, 0] });
      else labels.push({ p: at, text: distance, o: [0, below[1] + 22] });
    }
    out.push(new TextLayer({ id: 'truth-labels', data: labels, getPosition: (d) => d.p, getText: (d) => d.text,
      getPixelOffset: (d) => d.o, getSize: 13.5, getColor: C.coral, fontFamily: FONT, fontWeight: 600, characterSet: 'auto',
      background: true, getBackgroundColor: [...C.sheet, 235], backgroundPadding: [5, 3],
      updateTriggers: { getText: [st.key, st.scale], getPixelOffset: [z, st.scale] } }));
    return out;
  }

  // Three pulses (about 4 s) when the true leak is revealed, then a still ring: moving content
  // stops within 5 s (WCAG 2.2.2). None under reduced motion.
  let pulse = 0;
  let pulsed = null;
  function animate() {
    const key = st.truth ? `${st.incident ? st.incident.alert_id : ''}:${st.truth.node_id}` : null;
    if (key && key !== pulsed && !reduced && !pulse) {
      pulsed = key;
      const t0 = performance.now();
      pulse = setInterval(() => {
        const done = performance.now() - t0 > 4200;
        st.ripple = done ? 0.999 : st.ripple + 0.035;
        draw();
        if (done) { clearInterval(pulse); pulse = 0; }
      }, 50);
    }
    if (!key && pulse) { clearInterval(pulse); pulse = 0; }
  }

  function layers() {
    const z = st.zoomed ? 1.4 : 1;
    const key = st.key;
    const inc = st.incident;
    const L = [
      new LineLayer({ id: 'grid', data: grid, getSourcePosition: (d) => d.a, getTargetPosition: (d) => d.b,
        getColor: (d) => [...C.rule, d.major ? 105 : 48], getWidth: 1, widthUnits: 'pixels' }),
      new PathLayer({ id: 'glow', data: pipes, getPath: (d) => d.path, getColor: (d) => [...C.amber, Math.round(d.heat * 90)],
        getWidth: 12, widthUnits: 'pixels', capRounded: true, jointRounded: true, parameters: ADD,
        updateTriggers: { getColor: key }, transitions: { getColor: move(MOVE_MS) } }),
    ];
    if (inc && net.outlines[inc.area]) {
      L.push(new PathLayer({ id: 'outline', data: net.outlines[inc.area].map((path) => ({ path })), getPath: (d) => d.path,
        getColor: [...C.amber, 215], getWidth: 1.6, widthUnits: 'pixels', jointRounded: true,
        extensions: [new PathStyleExtension({ dash: true })], getDashArray: [7, 5], dashJustified: true }));
    }
    L.push(
      new PathLayer({ id: 'pipes', data: pipes, getPath: (d) => d.path, getColor: (d) => mix(C.pipe, C.amber, d.heat),
        getWidth: (d) => (d.d >= 0.2 ? 3 : d.d >= 0.15 ? 2 : 1.15), widthUnits: 'pixels', capRounded: true,
        updateTriggers: { getColor: key }, transitions: { getColor: move(MOVE_MS) } }),
      ...orderLayers(inc, z, key),
      new ScatterplotLayer({ id: 'rings', data: net.sensors, getPosition: (d) => xy(d), radiusUnits: 'pixels',
        getRadius: (s) => (drop(s) > 0 ? (6 + Math.sqrt(drop(s)) * 12) * z : 0),
        getFillColor: [...C.amber, 56], stroked: true, getLineColor: C.amber, lineWidthUnits: 'pixels', getLineWidth: 1.5,
        updateTriggers: { getRadius: [key, z] }, transitions: { getRadius: move(MOVE_MS) } }),
      new ScatterplotLayer({ id: 'sensors', data: net.sensors, getPosition: (d) => xy(d), radiusUnits: 'pixels',
        getRadius: 4.2 * (st.zoomed ? 1.35 : 1), getFillColor: C.sheet, stroked: true, getLineColor: C.ink,
        lineWidthUnits: 'pixels', getLineWidth: 1.6, updateTriggers: { getRadius: z } }),
    );
    const labels = st.drops ? st.drops.labels.filter((l) => xy(l.sensor)) : [];
    if (labels.length) {
      const first = labels[0].sensor;
      L.push(new TextLayer({ id: 'sensor-labels', data: labels, getPosition: (d) => xy(d.sensor), getText: (d) => d.text,
        getSize: 13.5, getColor: C.ink, fontFamily: FONT, characterSet: 'auto',
        getPixelOffset: (d) => (d.sensor === first ? [0, -26 * z] : [16 * z, 0]),
        getTextAnchor: (d) => (d.sensor === first ? 'middle' : 'start'),
        background: true, getBackgroundColor: [...C.sheet, 225], backgroundPadding: [5, 3],
        updateTriggers: { getText: key, getPixelOffset: [key, z], getTextAnchor: key } }));
    }
    if (inc && inc.candidates.length) {
      const cands = inc.candidates.map((n, i) => ({ n, t: String(i + 1) })).filter((c) => xy(c.n)).reverse();
      L.push(
        new ScatterplotLayer({ id: 'candidates', data: cands, getPosition: (d) => xy(d.n), radiusUnits: 'pixels',
          getRadius: st.zoomed ? 12 : 9.5, getFillColor: C.amber, stroked: true, getLineColor: C.sheet,
          lineWidthUnits: 'pixels', getLineWidth: 2, updateTriggers: { getRadius: z } }),
        new TextLayer({ id: 'candidate-ranks', data: cands, getPosition: (d) => xy(d.n), getText: (d) => d.t,
          getSize: st.zoomed ? 14 : 12, getColor: C.sheet, fontFamily: FONT, fontWeight: 700, updateTriggers: { getSize: z } }),
      );
    }
    L.push(...truthLayers(z));
    const alertArea = inc ? inc.area : null;
    L.push(new TextLayer({ id: 'areas', data: Object.entries(net.area_labels), getPosition: (d) => d[1],
      getText: (d) => (d[0] === alertArea ? `Area ${d[0]}: alert` : `Area ${d[0]}`), getSize: 14,
      getColor: (d) => (d[0] === alertArea ? C.amber : C.ink3), fontFamily: FONT, characterSet: 'auto',
      background: true, getBackgroundColor: [...C.sheet, 200], backgroundPadding: [5, 2],
      updateTriggers: { getText: alertArea, getColor: alertArea } }));
    if (st.selected && xy(st.selected)) {
      L.push(new ScatterplotLayer({ id: 'selected', data: [st.selected], getPosition: (d) => xy(d), radiusUnits: 'pixels',
        getRadius: 11, filled: false, stroked: true, getLineColor: C.ink, lineWidthUnits: 'pixels', getLineWidth: 2 }));
      L.push(new TextLayer({ id: 'selected-label', data: [st.selected], getPosition: (d) => xy(d), getText: (d) => d,
        getSize: 13.5, getColor: C.sheet, getPixelOffset: [0, -24], fontFamily: FONT, fontWeight: 700,
        background: true, getBackgroundColor: C.ink, backgroundPadding: [6, 3] }));
    }
    // Invisible, generous hit areas: every junction can be hovered and clicked.
    L.push(new ScatterplotLayer({ id: 'junctions', data: net.junctions, getPosition: (d) => xy(d), radiusUnits: 'pixels',
      getRadius: 9, getFillColor: [0, 0, 0, 0], pickable: true, autoHighlight: true, highlightColor: [...C.ink, 150] }));
    return L;
  }

  const tip = document.createElement('div');
  tip.className = 'map-tip';
  tip.hidden = true;
  el.parentElement.appendChild(tip);

  // Whole-network inset: all pipes, and a box for the part the map shows.
  const inset = document.createElement('figure');
  inset.className = 'inset';
  inset.setAttribute('aria-hidden', 'true');
  const IW = 226, IH = 128;
  const k = Math.min((IW - 12) / (x1 - x0), (IH - 12) / (y1 - y0));
  const ox = (IW - (x1 - x0) * k) / 2, oy = (IH - (y1 - y0) * k) / 2;
  const P = (c) => [ox + (c[0] - x0) * k, oy + (y1 - c[1]) * k];
  const svgEl = (tag, attrs) => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
    return node;
  };
  const caption = document.createElement('figcaption');
  caption.textContent = 'Whole network, the box is the view';
  const svg = svgEl('svg', { width: IW, height: IH, viewBox: `0 0 ${IW} ${IH}` });
  const d = pipes.map((p) => { const a = P(p.path[0]), b = P(p.path[1]); return `M${a[0].toFixed(1)} ${a[1].toFixed(1)}L${b[0].toFixed(1)} ${b[1].toFixed(1)}`; }).join('');
  svg.append(svgEl('path', { d, fill: 'none', stroke: `rgb(${C.pipe})`, 'stroke-width': 0.8 }));
  const rect = svgEl('rect', { fill: `rgba(${C.amber},.22)`, stroke: `rgb(${C.amber})`, 'stroke-width': 1.5 });
  svg.append(rect);
  inset.append(caption, svg);
  el.parentElement.appendChild(inset);
  function updateInset(bounds) {
    const [a0, a1] = P([bounds[0], bounds[3]]), [b0, b1] = P([bounds[2], bounds[1]]);
    const cx0 = Math.max(1, Math.min(IW - 1, a0)), cy0 = Math.max(1, Math.min(IH - 1, a1));
    const cx1 = Math.max(1, Math.min(IW - 1, b0)), cy1 = Math.max(1, Math.min(IH - 1, b1));
    rect.setAttribute('x', cx0.toFixed(1)); rect.setAttribute('y', cy0.toFixed(1));
    rect.setAttribute('width', Math.max(4, cx1 - cx0).toFixed(1)); rect.setAttribute('height', Math.max(4, cy1 - cy0).toFixed(1));
  }

  let deckgl;
  let redraw = 0; // a pending redraw after the scale changed
  try {
    deckgl = new DeckGL({
      container: el,
      views: new OrthographicView({ flipY: false }),
      initialViewState: home(),
      controller: { inertia: !reduced, doubleClickZoom: false },
      getCursor: ({ isHovering, isDragging }) => (isDragging ? 'grabbing' : isHovering ? 'pointer' : 'grab'),
      onViewStateChange: ({ interactionState: i }) => {
        if (i && (i.isDragging || i.isPanning || i.isZooming)) st.moved = true;
      },
      onHover: ({ object, x, y }) => {
        tip.hidden = !object;
        if (object) {
          tip.textContent = `Junction ${object}, Area ${net.areas[object]}`;
          tip.style.left = `${x}px`;
          tip.style.top = `${y}px`;
        }
      },
      onAfterRender: () => {
        const vp = deckgl && deckgl.getViewports()[0];
        if (!vp) return;
        if (st.zoomed) updateInset(vp.getBounds());
        // Some labels depend on the scale: redraw (next frame) when it crosses a half-zoom step.
        const zoom = Array.isArray(vp.zoom) ? vp.zoom[0] : vp.zoom;
        const scale = 2 ** (Math.round(zoom * 2) / 2);
        if (Number.isFinite(scale) && scale !== st.scale && !redraw) {
          st.scale = scale;
          redraw = requestAnimationFrame(() => { redraw = 0; draw(); });
        }
      },
      onError: (error) => onError(error),
      layers: layers(),
    });
  } catch (error) {
    onError(error);
    inset.remove();
    tip.remove();
    return null;
  }

  const draw = () => deckgl.setProps({ layers: layers() });

  // A plain DOM click picks the nearest junction within reach (a drag to pan is not a click).
  let down = null;
  const onDown = (e) => { down = [e.clientX, e.clientY]; };
  const onClick = (e) => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    const box = el.getBoundingClientRect();
    const x = e.clientX - box.left, y = e.clientY - box.top;
    const hit = deckgl.pickObject({ x, y, radius: 14, layerIds: ['junctions'] });
    tip.hidden = true; // the card that opens names the junction; the hover tip would sit under it
    onPick(hit ? hit.object : null, { x, y });
  };
  el.addEventListener('pointerdown', onDown);
  el.addEventListener('click', onClick);
  function fly(view, ms = 1800) {
    deckgl.setProps({ initialViewState: { ...view, transitionDuration: reduced ? 0 : ms,
      transitionInterpolator: new LinearInterpolator(['target', 'zoom']), transitionEasing: easeInOut } });
  }
  // A window resized (or a page opened in a minimized window) refits the whole network; a view
  // the camera or the user moved is left as it is.
  let size = [el.clientWidth, el.clientHeight];
  const resized = new ResizeObserver(() => {
    const now = [el.clientWidth, el.clientHeight];
    if (now[0] === size[0] && now[1] === size[1]) return;
    size = now;
    if (!st.zoomed && !st.moved) deckgl.setProps({ initialViewState: home() });
  });
  resized.observe(el);

  return {
    set(patch) {
      Object.assign(st, patch);
      const order = st.incident && st.incident.order ? `${st.incident.order.state}:${(st.incident.order.pipes || []).join(',')}` : '';
      const key = `${st.drops ? st.drops.window_end : ''}|${st.incident ? st.incident.alert_id : ''}|${order}|${st.truth ? st.truth.node_id : ''}`;
      if (key !== st.key) { st.key = key; updateHeat(); }
      draw();
      animate();
    },
    // Fly to junctions (the candidates and the sensors that moved most; `tight` for the reveal);
    // false if none is on the map.
    focus(nodes, tight = false) {
      const box = focusBox(nodes, tight);
      if (!box) return false;
      st.zoomed = true;
      st.moved = false;
      inset.classList.add('on');
      fly(fit(box, 80));
      draw();
      return true;
    },
    home() {
      st.zoomed = false;
      st.moved = false;
      inset.classList.remove('on');
      fly(home(), 1200);
      draw();
    },
    destroy() {
      resized.disconnect();
      cancelAnimationFrame(redraw);
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('click', onClick);
      clearInterval(pulse);
      deckgl.finalize();
      tip.remove();
      inset.remove();
    },
  };
}
