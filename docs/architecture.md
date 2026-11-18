# Architecture

What exists and where it lives. Section numbers like "PRD §7.2" in code comments refer to the
project brief; `D-0xx` refers to [DECISIONS.md](DECISIONS.md). Numbers are not repeated here:
they live in `reports/metrics/*.json` (summarised in the README and the report drafts).

## Data flow

```
 Zenodo 4017659 ──make data──► data/raw/battledim/  (CSV, INP, YAML; MD5-checked)
                                 ├─► parquet/scada_{2018,2019}.parquet   wide, UTC, 119 sensors
                                 ├─► parquet/leak_flows_{year}.parquet   ground-truth outflows
                                 └─► parquet/ground_truth.parquet        33 leaks, cross-checked

 epyt-flow scenarios (make sim-scale: Spark mapPartitions, 7-day runs)  ──► lake/sim/scenarios
 EPANET emitter scenarios (make gnn-data: process pool)                 ──► data/sim/gnn/*.npz

 replayer / sim.replay ──► Kafka `scada.readings` ──► Spark Structured Streaming ──► lake/raw
     (parse + validate → watermark dedup → range flag → exactly-once commit; rejects → lake/_rejects)
                                                │
   ├─ Spark batch  (make features)        lake/clean → lake/baselines/hour_of_week → lake/features
   └─ Spark stream (make stream-features) Kafka → same window_features() → lake/features_stream
        1 h windows every 15 min · 38 series × 6 stats = 228 features · batch = stream (parity test)
                                                │
   K-Means (Spark MLlib: scaler → PCA → KMeans, exported to a numpy scorer)
   ├─ batch  (make bdt-eval)   lake/scores/origin=batch,  lake/alerts/origin=batch
   └─ live   (feature stream foreachBatch) lake/scores/origin=stream, lake/alerts/origin=stream
                                           + Kafka `alerts` (one message per alert version)
                                                │
   GNN / MLP localizer ──► alert version with gnn_top5     (live: on new alerts; batch: make gnn)
   LSTM autoencoder    ──► offline comparison with K-Means (make lstm-ae)
                                                │
   copilot (OpenAI-compatible LLM + tools; DuckDB read-only; RAG over the manuals)
   ├─ dispatch: one work order per new alert ──► lake/work_orders (template fallback without LLM)
   └─ chat: data questions, procedures, complaints
                                                │
   Streamlit dashboard ◄── DuckDB views ── lake (main or demo)      demo.commands ──► demo runner
```

Rules (PRD §5): Kafka is the only entry point for readings into the lake; Spark writes the lake;
DuckDB only reads it. Two documented exceptions write small files with pyarrow instead of a Spark
job: live score/alert rows inside the stream's own commit (D-022) and work orders (D-021).
`data/raw/` holds downloaded sources and their Parquet conversions; it is the replayer's input,
not part of the lake.

## Contracts

All in [`src/leakwatch/common/schemas.py`](../src/leakwatch/common/schemas.py):

| Contract | pydantic | pandera / columns | Spark |
|---|---|---|---|
| `scada.readings` message | `Reading` | `READINGS_SCHEMA` | `readings_payload_spark_schema()` |
| raw lake rows | – | `RAW_LAKE_SCHEMA` | written by `stream.raw_ingest` |
| cleaned rows | – | `CLEAN_LAKE_SCHEMA` | written by `features.build` |
| hour-of-week baselines | – | `BASELINES_SCHEMA` | written by `features.build` |
| feature rows (batch + live) | – | `feature_table_schema(spec.feature_columns())` | `features.windows.window_features` |
| window scores | – | `SCORES_SCHEMA` (`SCORE_COLUMNS`) | `bdt.data.write_scores` |
| alerts table / `alerts` topic | `Alert` (+ `TopSensor`, `GnnCandidate`) | `ALERT_LAKE_COLUMNS` | `alerts_spark_schema()` |
| work order | `WorkOrder` (+ `ManualCitation`) | `WORK_ORDER_COLUMNS` | – (pyarrow) |
| `demo.commands` message | `DemoCommand` | – | – |
| leak ground truth | – | `GROUND_TRUTH_SCHEMA` | – |

DuckDB views (`leakwatch.common.lake.connect()`): `raw_readings`, `clean_readings`, `baselines`,
`features`, `features_stream`, `scores`, `alert_events` (every alert version), `work_orders`, and
`alerts` = the latest version of each alert joined with its latest work order.

Sensor naming: `sensor_id` is the L-Town element ID (`n54`, `p227`, `PUMP_1`, `T1`). In wide tables
columns are `<sensor_type>__<sensor_id>` (`pressure__n54`, `amr__n1`) because nodes n1, n4 and n31
carry both a pressure sensor and an AMR.

## Streaming guarantees

- **No loss:** every message is either written to `lake/raw`, kept in `lake/_rejects` (breaks the
  contract), or counted as a duplicate / late row in the run's metrics.
- **No duplicates:** idempotent producer; watermark dedup on (source, scenario, sensor, ts);
  `foreachBatch` commit protocol so a retried micro-batch is never written twice (D-013). The live
  feature, score and alert files of one micro-batch are committed together
  (`lake_writer.commit_batch_multi`).
- **Order:** event time must move forward; years are streamed oldest first; simulated scenarios
  follow BattLeDIM in event time (D-023).

## Features (Phase 2)

- **One function, two engines:** `features.windows.window_features(readings, baselines, spec)`
  (cleaning → series → residual → 1 h / 15 min aggregates → one wide row per window) is called by
  the batch job, the streaming job (every micro-batch plus carried history) and the scaling
  benchmark (D-017, D-018).
- **Series:** 33 pressure + 3 flow + 1 level sensors and the AMR sum of area C = 38 series; stats
  mean, std, min, max, slope (per hour), residual (vs the hour-of-week baseline).
- **Normal (D-015):** background leaks count as normal after their first 7 days; every other leak
  is excluded while active. Baselines and K-Means training use normal 2018 only.

## Detection (Phase 3, BDT)

- `bdt.train`: Spark MLlib pipeline per variant, k at the SSE elbow (silhouette reported),
  τ = p99 of held-out threshold days, persistence m tuned on 2018 validation days (D-020).
- `bdt.scorer`: the fitted scaler/PCA/centroids as numpy; tested equal to Spark's `transform`.
  The same scorer runs in `bdt.evaluate` (batch), in the live stream (`bdt.live.LiveScoring`) and
  inside Spark tasks in the scaling benchmark (`bdt.scaling`, `mapInPandas`).
- `bdt.alerts`: alert rule (m windows above τ; closes after 4 below), per-series contribution,
  area of the top series, estimated loss from excess inflow. Alert IDs are deterministic (uuid5).
- `eval.detection`: event-based labels (tp / pre-existing / fp), detection by leak type, chance
  level and onset contrast.

## Localization and the deep-learning comparison (Phase 5, DL)

- `dl.gnn_data`: single-leak EPANET scenarios with demand and sensor noise, baselines from noisy
  leak-free runs (train/serve consistency, D-024).
- `dl.gnn`: nearest-sensor heuristic, MLP, GCN and GAT on the L-Town graph (node features:
  elevation, base demand, sensor flag, two hourly residuals); random and unseen-size splits;
  2019 benchmark on detected leaks; `make_localizer()` for the live path.
- `dl.lstm_ae`: LSTM autoencoder on 72-step residual sequences; same threshold and persistence
  procedure as K-Means; comparison at matched false-alarm rates.

## Copilot (Phase 6, PE)

`copilot/`: OpenAI-compatible client (`llm.py`), tools with JSON schemas (`tools.py`: schema,
read-only SQL with a sqlglot guard, alerts, sensor series, localizer candidates, loss estimate,
manual search, complaint intake), prompt versions v1–v3 (`prompts/`), guardrails (number check,
citations, refusals), RAG index (`rag/`), dispatch of work orders, and the evaluation harness over
`evals/copilot_questions.jsonl` (D-026).

## Demo (Phase 7)

`demo.runner` replays a quiet BattLeDIM week through `demo.readings` at `demo.speed`, runs the raw
ingest and the scored feature stream into `data/demo/lake`, localizes and dispatches new alerts,
and applies `inject_leak` commands as EPANET leak signatures (D-025). `dashboard/app.py` shows the
map, alerts, details, work orders, charts and the copilot. `demo.check` rehearses the demo without
a person.

## Network areas

Areas A/B/C are computed, not hard-coded: cut L-Town at PUMP_1 and PRV-1/2/3, take the connected
zones, and name each by an anchor node in `configs/default.yaml` (D-008).

## Where things run

Windows 11 host, everything executed inside WSL2 Ubuntu (D-001). Source in the workspace folder;
venv (`~/.venvs/leakwatch`) and data (`~/leakwatch-data`) on the WSL disk (D-002). Spark runs in
local mode; Kafka is a single KRaft broker in Docker.
