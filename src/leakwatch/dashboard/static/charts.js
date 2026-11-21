// Score and pressure charts (D-052), Apache ECharts (vendored). The server shapes every value
// (view.charts_view); this file only draws them. The score's part above the threshold is a second
// series: a visualMap on a log axis throws in ECharts 6.1.0.
const MINUS = (v) => `${Number(v).toFixed(2)}`.replace('-', '−');

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function createCharts(scoreEl, pressureEl) {
  const { echarts } = window;
  const T = {
    ink: token('--ink'), ink2: token('--ink-2'), ink3: token('--ink-3'), rule: token('--rule'),
    amber: token('--amber'), sheet: token('--sheet'), font: token('--font'),
    series: [token('--series-1'), token('--series-2'), token('--series-3')],
  };
  // One line per sensor: colour, dash and width all differ (secondary encoding), plus end labels.
  const STYLES = [['solid', 2.3], ['dashed', 1.9], ['dotted', 2.2]];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const score = echarts.init(scoreEl);
  const pressure = echarts.init(pressureEl);
  const resize = new ResizeObserver(() => { score.resize(); pressure.resize(); });
  resize.observe(scoreEl);
  resize.observe(pressureEl);

  const text = { fontFamily: T.font, fontSize: 12, color: T.ink2 };
  const axis = {
    axisLine: { lineStyle: { color: T.rule } },
    axisTick: { show: false },
    axisLabel: { color: T.ink3, fontFamily: T.font, fontSize: 11.5 },
    splitLine: { lineStyle: { color: T.rule, opacity: 0.45 } },
  };
  const tip = {
    trigger: 'axis', backgroundColor: T.ink, borderWidth: 0, confine: true,
    textStyle: { color: T.sheet, fontFamily: T.font, fontSize: 13 },
    axisPointer: { lineStyle: { color: T.ink2 } },
  };
  const verticals = (marks) => marks.map((m) => ({
    xAxis: m.at,
    lineStyle: { color: m.kind === 'alert' ? T.amber : T.ink2, type: 'solid', width: 1.2 },
    // the leak's label left of its line, the alert's right of it: they are often an hour apart
    label: { ...text, formatter: m.text, position: 'end', distance: 3, align: m.kind === 'alert' ? 'left' : 'right' },
  }));

  function update(c) {
    if (!c) { score.clear(); pressure.clear(); return; }
    const clock = new Intl.DateTimeFormat('en-GB', { timeZone: c.tz, hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
    const time = (v) => clock.format(v);
    const xAxis = {
      type: 'time', min: c.start, max: c.end, ...axis, splitLine: { show: false },
      axisLabel: { ...axis.axisLabel, formatter: (v) => time(v), hideOverlap: true },
    };
    const anim = { animation: !reduced, animationDurationUpdate: 300 };

    const thr = c.score.threshold;
    const pts = c.score.points;
    score.setOption({
      ...anim,
      grid: { left: 46, right: 20, top: 24, bottom: 22 },
      tooltip: {
        ...tip,
        formatter: (ps) => {
          const p = ps.find((x) => x.seriesId === 'score');
          if (!p) return '';
          const real = p.value[2];
          const side = thr == null ? '' : `, ${real > thr ? 'above' : 'below'} the threshold ${thr.toFixed(2)}`;
          return `${time(p.value[0])} ${c.tz_name}<br><b>Score ${real.toFixed(2)}</b>${side}`;
        },
      },
      xAxis,
      yAxis: {
        type: 'log', min: c.score.axis_min, ...axis,
        axisLabel: { ...axis.axisLabel, formatter: (v) => String(Number(v.toPrecision(2))) },
      },
      series: [
        {
          id: 'score', name: 'Score', type: 'line', showSymbol: false, data: pts, encode: { x: 0, y: 1 },
          lineStyle: { width: 2, color: T.ink }, itemStyle: { color: T.ink },
          markLine: {
            symbol: 'none', silent: true, animation: false,
            data: [
              ...(thr == null ? [] : [{
                yAxis: thr, lineStyle: { color: T.amber, type: 'dashed', width: 1.4 },
                label: { ...text, formatter: c.score.threshold_text, position: 'insideStartTop' },
              }]),
              ...verticals(c.marks),
            ],
          },
        },
        {
          id: 'above', name: 'Above the threshold', type: 'line', showSymbol: false, silent: true, z: 3,
          connectNulls: false, encode: { x: 0, y: 1 },
          data: thr == null ? [] : pts.map((p) => [p[0], p[2] > thr ? p[1] : null, p[2]]),
          lineStyle: { width: 2.8, color: T.amber }, itemStyle: { color: T.amber },
        },
      ],
    }, { replaceMerge: ['series'] });

    const series = c.pressure.series;
    pressure.setOption({
      ...anim,
      grid: { left: 40, right: 104, top: 26, bottom: 22 },
      legend: {
        top: 0, right: 6, itemWidth: 24, itemHeight: 9, textStyle: text, data: series.map((s) => s.sensor),
      },
      tooltip: {
        ...tip,
        formatter: (ps) => `${time(ps[0].value[0])} ${c.tz_name}<br>${ps.map((p) => `${p.marker}${p.seriesName} <b>${p.value[1] == null ? 'no reading' : `${MINUS(p.value[1])} m`}</b>`).join('<br>')}`,
      },
      xAxis,
      yAxis: {
        type: 'value', ...axis,
        min: (v) => Math.min(-1, Math.floor(v.min * 2) / 2),
        max: (v) => Math.max(0.5, Math.ceil(v.max * 2) / 2),
        axisLabel: { ...axis.axisLabel, formatter: (v) => `${v}`.replace('-', '−') },
      },
      series: series.map((s, i) => ({
        id: s.sensor, name: s.sensor, type: 'line', showSymbol: false, data: s.points, connectNulls: false,
        lineStyle: { color: T.series[i], type: STYLES[i][0], width: STYLES[i][1] }, itemStyle: { color: T.series[i] },
        endLabel: { show: true, ...text, formatter: (p) => (p.value[1] == null ? s.sensor : `${s.sensor}  ${MINUS(p.value[1])} m`) },
        labelLayout: { moveOverlap: 'shiftY' },
        markLine: i > 0 ? undefined : {
          symbol: 'none', silent: true, animation: false,
          data: [
            { yAxis: 0, lineStyle: { color: T.ink3, type: 'solid', width: 1 }, label: { ...text, color: T.ink3, formatter: 'Usual level', position: 'insideStartTop' } },
            ...verticals(c.marks),
          ],
        },
      })),
    }, { replaceMerge: ['series'] });
  }

  return {
    update,
    destroy() { resize.disconnect(); score.dispose(); pressure.dispose(); },
  };
}

// A chart returned by a copilot tool (one or more lines over time), drawn small in the rail.
export function drawAnswerChart(el, chart, tz, tzName) {
  const { echarts } = window;
  const T = {
    ink2: token('--ink-2'), ink3: token('--ink-3'), rule: token('--rule'), ink: token('--ink'), sheet: token('--sheet'),
    font: token('--font'), series: [token('--series-1'), token('--series-2'), token('--series-3')],
  };
  const STYLES = ['solid', 'dashed', 'dotted'];
  const clock = new Intl.DateTimeFormat('en-GB', { timeZone: tz, day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
  const text = { fontFamily: T.font, fontSize: 11.5, color: T.ink3 };
  const c = echarts.init(el);
  c.setOption({
    animation: false,
    grid: { left: 44, right: 12, top: chart.series.length > 1 || chart.markers.length ? 24 : 10, bottom: 22 },
    legend: chart.series.length > 1 ? { top: 0, right: 0, textStyle: { ...text, color: T.ink2 }, itemWidth: 20, itemHeight: 8 } : undefined,
    tooltip: {
      trigger: 'axis', backgroundColor: T.ink, borderWidth: 0, confine: true,
      textStyle: { color: T.sheet, fontFamily: T.font, fontSize: 12.5 },
      formatter: (ps) => `${clock.format(ps[0].value[0])} ${tzName}<br>${ps.map((p) => `${p.marker}${echarts.format.encodeHTML(p.seriesName)} <b>${p.value[1] == null ? 'no value' : `${Number(p.value[1]).toFixed(2)} ${chart.unit}`}</b>`).join('<br>')}`,
    },
    xAxis: { type: 'time', axisLine: { lineStyle: { color: T.rule } }, axisTick: { show: false }, splitLine: { show: false },
      axisLabel: { ...text, hideOverlap: true, formatter: (v) => clock.format(v) } },
    yAxis: { type: 'value', scale: true, axisLabel: text, splitLine: { lineStyle: { color: T.rule, opacity: 0.45 } } },
    series: chart.series.map((s, i) => ({
      name: s.name, type: 'line', showSymbol: false, data: s.points, connectNulls: false,
      lineStyle: { color: T.series[i % 3], type: STYLES[i % 3], width: 2 }, itemStyle: { color: T.series[i % 3] },
      markLine: i > 0 || !chart.markers.length ? undefined : {
        symbol: 'none', silent: true, animation: false,
        data: chart.markers.map((m) => ({ xAxis: m.at, lineStyle: { color: T.ink2, type: 'solid', width: 1 },
          label: { ...text, color: T.ink2, formatter: m.text, position: 'end' } })),
      },
    })),
  });
  const resize = new ResizeObserver(() => c.resize());
  resize.observe(el);
  return () => { resize.disconnect(); c.dispose(); };
}
