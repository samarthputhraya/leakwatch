# Vendored front-end libraries

The dashboard is served without a build step and without a CDN, so the demo needs no network for
its own page. These files are unmodified copies; `manifest.json` records where each came from, its
size and its SHA-256, and `tests/unit/test_dashboard_vendor.py` checks them.

| File | Package | Version | Licence |
|---|---|---|---|
| `deck.gl.min.js` | deck.gl (WebGL map layers) | 9.4.1 | MIT, `LICENSE-deck.gl.txt` |
| `echarts.min.js` | Apache ECharts (charts) | 6.1.0 | Apache-2.0, `LICENSE-echarts.txt`, `NOTICE-echarts.txt` |
| `htm-preact.module.js` | htm with Preact bundled (components) | 3.1.1 | Apache-2.0 (htm), MIT (Preact): `LICENSE-htm.txt`, `LICENSE-preact.txt` |

To update a library: download the new file from the `source` URL pattern in `manifest.json`,
replace the file, and update `version`, `source`, `bytes` and `sha256` there.
