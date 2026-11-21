// The L-Town map (D-052): deck.gl layers in the network's own metre coordinates, no basemap.
// `createMap` owns the WebGL canvas; the page calls `set()` with what changed.
const { DeckGL, OrthographicView, PathLayer, ScatterplotLayer, TextLayer, LineLayer, LinearInterpolator } = window.deck;

const FONT = 'Bahnschrift, "DIN Alternate", Barlow, sans-serif';

function token(name) {
  const hex = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export function createMap(el, net, { onPick, onError }) {
  const C = {
    sheet: token('--sheet'), ink: token('--ink'), ink3: token('--ink-3'), rule: token('--rule'),
    pipe: token('--pipe'), amber: token('--amber'), coral: token('--coral'),
  };
  const xy = (n) => net.nodes[n];
  const [x0, y0, x1, y1] = net.bounds;
  const pipes = net.pipes.map((p) => ({ ...p, path: [xy(p.a), xy(p.b)] }));
  const grid = [];
  const gx0 = Math.floor(x0 / 500) * 500 - 500, gx1 = Math.ceil(x1 / 500) * 500 + 500;
  const gy0 = Math.floor(y0 / 500) * 500 - 500, gy1 = Math.ceil(y1 / 500) * 500 + 500;
  for (let x = gx0; x <= gx1; x += 100) grid.push({ a: [x, gy0], b: [x, gy1], major: x % 500 === 0 });
  for (let y = gy0; y <= gy1; y += 100) grid.push({ a: [gx0, y], b: [gx1, y], major: y % 500 === 0 });

  const st = { selected: null, hovered: null };
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // View that fits a box [minx, miny, maxx, maxy] into the canvas, leaving room for overlays.
  function fit(box, padX, padY) {
    const w = Math.max(200, el.clientWidth - padX), h = Math.max(160, el.clientHeight - padY);
    const zoom = Math.log2(Math.min(w / (box[2] - box[0]), h / (box[3] - box[1])));
    return { target: [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 0], zoom };
  }
  const home = () => { const v = fit(net.bounds, 64, 104); v.target[1] -= 10 / 2 ** v.zoom; return v; };

  function layers() {
    const L = [
      new LineLayer({ id: 'grid', data: grid, getSourcePosition: (d) => d.a, getTargetPosition: (d) => d.b,
        getColor: (d) => [...C.rule, d.major ? 105 : 48], getWidth: 1, widthUnits: 'pixels' }),
      new PathLayer({ id: 'pipes', data: pipes, getPath: (d) => d.path, getColor: C.pipe,
        getWidth: (d) => (d.d >= 0.2 ? 3 : d.d >= 0.15 ? 2 : 1.15), widthUnits: 'pixels', capRounded: true }),
      new ScatterplotLayer({ id: 'sensors', data: net.sensors, getPosition: (d) => xy(d), radiusUnits: 'pixels',
        getRadius: 4.2, getFillColor: C.sheet, stroked: true, getLineColor: C.ink, lineWidthUnits: 'pixels', getLineWidth: 1.6 }),
      new TextLayer({ id: 'areas', data: Object.entries(net.area_labels), getPosition: (d) => d[1],
        getText: (d) => `Area ${d[0]}`, getSize: 14, getColor: C.ink3, fontFamily: FONT,
        background: true, getBackgroundColor: [...C.sheet, 200], backgroundPadding: [5, 2] }),
    ];
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

  let deckgl;
  try {
    deckgl = new DeckGL({
      container: el,
      views: new OrthographicView({ flipY: false }),
      initialViewState: home(),
      controller: { inertia: !reduced, doubleClickZoom: false },
      getCursor: ({ isHovering, isDragging }) => (isDragging ? 'grabbing' : isHovering ? 'pointer' : 'grab'),
      onHover: ({ object, x, y }) => {
        tip.hidden = !object;
        if (object) {
          tip.textContent = `Junction ${object}, Area ${net.areas[object]}`;
          tip.style.left = `${x}px`;
          tip.style.top = `${y}px`;
        }
      },
      onError: (error) => onError(error),
      layers: layers(),
    });
  } catch (error) {
    onError(error);
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
    onPick(hit ? hit.object : null, { x, y });
  };
  el.addEventListener('pointerdown', onDown);
  el.addEventListener('click', onClick);
  function fly(view, ms = 1600) {
    deckgl.setProps({ initialViewState: { ...view, transitionDuration: reduced ? 0 : ms,
      transitionInterpolator: new LinearInterpolator(['target', 'zoom']), transitionEasing: easeInOut } });
  }

  return {
    set(patch) { Object.assign(st, patch); draw(); },
    home() { fly(home(), 1200); },
    destroy() {
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('click', onClick);
      deckgl.finalize();
      tip.remove();
    },
  };
}
