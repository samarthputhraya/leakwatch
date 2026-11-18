# Architecture

This page tracks what exists and where it lives (section numbers like "PRD §7.2" in code
comments refer to the project brief).
Status markers: **built** (code + tests in the repo), **planned** (phase number).

## Data flow

```
 Zenodo 4017659 ──make data──► data/raw/battledim/ (CSV, INP, YAML; MD5-checked)          [built]
                                    │
                                    ├─► parquet/scada_{2018,2019}.parquet  (wide, UTC, 119 sensors)  [built]
                                    ├─► parquet/leak_flows_{year}.parquet  (ground-truth outflows)    [built]
                                    └─► parquet/ground_truth.parquet       (33 leaks, cross-checked)  [built]

 epyt-flow (bulk) / WNTR (GNN scenarios) ──► simulated readings                    [benchmarked; P4/P5]
                                    │
 replayer ──► Kafka `scada.readings` ──► Spark Structured Streaming ──► data/lake/raw/...  [built]
              (parse + validate → watermark dedup → range flag → exactly-once commit; rejects → lake/_rejects)
                                                  │
      ├─ Spark batch (make features): lake/clean → lake/baselines/hour_of_week → lake/features  [built]
      └─ Spark stream (make stream-features): Kafka → same window_features() → lake/features_stream [built]
         (1 h windows every 15 min, 38 series × 6 stats = 228 features; parity checked)
                                                  │
                     K-Means (MLlib) score ──► alerts table + Kafka `alerts`                  [P3]
                                                  │
                      LSTM-AE score, GNN top-5 ──► alerts.lstm_score_max, alerts.gnn_top5     [P5]
                                                  │
                  copilot (tools + RAG, DuckDB read-only) ──► alerts.work_order_json          [P6]
                                                  │
                        Streamlit dashboard + `demo.commands`                                 [P7]
```

Rules (PRD §5): Kafka is the only entry point for readings into the lake; Spark writes the lake;
DuckDB only reads it. `data/raw/` holds downloaded sources and their Parquet conversions — it is
the replayer's input, not part of the lake.

## Contracts

All in [`src/leakwatch/common/schemas.py`](../src/leakwatch/common/schemas.py):

| Contract | pydantic | pandera | Spark |
|---|---|---|---|
| `scada.readings` message | `Reading` | `READINGS_SCHEMA` | `readings_payload_spark_schema()` |
| raw lake rows | – | `RAW_LAKE_SCHEMA` (`RAW_LAKE_COLUMNS`) | written by `stream.raw_ingest` |
| cleaned rows | – | `CLEAN_LAKE_SCHEMA` (`CLEAN_LAKE_COLUMNS`) | written by `features.build` |
| hour-of-week baselines | – | `BASELINES_SCHEMA` | written by `features.build` |
| feature rows (batch + live) | – | `feature_table_schema(spec.feature_columns())` | `features.windows.window_features` |
| `demo.commands` message | `DemoCommand` | – | – |
| alerts table / `alerts` topic | `Alert` (+ `TopSensor`, `GnnCandidate`) | Phase 3 | Phase 3 |
| work order | `WorkOrder` (+ `ManualCitation`) | – | – |
| leak ground truth | – | `GROUND_TRUTH_SCHEMA` | – |
| wide SCADA table | – | `wide_scada_schema()` | – |

Sensor naming: `sensor_id` is the L-Town element ID (`n54`, `p227`, `PUMP_1`, `T1`). In wide tables
columns are `<sensor_type>__<sensor_id>` (`pressure__n54`, `amr__n1`) because nodes n1, n4 and n31
carry both a pressure sensor and an AMR.

## Streaming guarantees (Phase 1)

- **No loss:** every message is either written to `lake/raw`, kept in `lake/_rejects` (breaks the
  contract), or counted as a duplicate / late row in the run's metrics.
- **No duplicates:** idempotent producer; watermark dedup on (source, scenario, sensor, ts);
  `foreachBatch` commit protocol so a retried micro-batch is never written twice (D-013).
- **Order:** event time must move forward; years are streamed oldest first.
- **Reading the lake:** `leakwatch.common.lake.connect()` gives a DuckDB view `raw_readings` with
  typed partition columns.

## Features (Phase 2)

- **One function, two engines:** `features.windows.window_features(readings, baselines, spec)`
  (cleaning → series → residual → 1 h / 15 min aggregates → one wide row per window) is called by
  the batch job on raw lake rows and by the streaming job on every micro-batch plus carried
  history (D-017, D-018). `features.spec.FeatureSpec` fixes series, stats and column order.
- **Series:** 33 pressure + 3 flow + 1 level sensors and the AMR sum of area C (all 82 AMRs are
  in C) = 38 series; stats mean, std, min, max, slope (per hour), residual (vs baseline).
- **Normal (D-015):** background leaks (incipient, peak ≤ 10 m³/h) count as normal after their
  first 7 days; every other leak is excluded while active. Baselines use normal 2018 only.
- **Live path guarantees:** each window once, in order, emitted when data 15 min past its last
  step has arrived; state (tail of readings) committed with the batch's files (D-013 protocol).
- **Parity:** `tests/contract/test_feature_parity.py` (file stream, restart mid-way),
  `tests/integration/test_kafka_features.py` (Kafka), and `make features-parity` (full years).

## Network areas

Areas A/B/C are computed, not hard-coded: cut L-Town at PUMP_1 and PRV-1/2/3, take the connected
zones, and name each by an anchor node in `configs/default.yaml` (D-008). Measured sizes are in
`reports/metrics/network_stats.json`.

## Where things run

Windows 11 host, everything executed inside WSL2 Ubuntu (D-001). Source in the OneDrive workspace;
venv (`~/.venvs/leakwatch`) and data (`~/leakwatch-data`) on the WSL disk (D-002).
