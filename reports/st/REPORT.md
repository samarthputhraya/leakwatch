# LeakWatch ST module: how we know it works

Draft generated from the metrics files.

- `make test-report`: 465 passed, 0 failed, 0 skipped in 463 s; line coverage of `src/leakwatch` 70.9% (`reports/metrics/test_summary.json` (code c5c020c)).

| test folder | tests |
|---|---|
| contract | 39 |
| integration | 2 |
| metamorphic | 5 |
| unit | 419 |

Tests run: `slow or not slow`. Modules with the most uncovered lines (most are job entry points exercised by `make` targets rather than by unit tests):

| module | statements | covered |
|---|---|---|
| `demo/runner.py` | 217 | 18.0% |
| `dl/gnn.py` | 325 | 50.2% |
| `dl/gnn_data.py` | 127 | 0.0% |
| `features/build.py` | 122 | 0.0% |
| `bdt/train.py` | 144 | 25.0% |
| `stream/live_replay.py` | 103 | 0.0% |
| `bdt/evaluate.py` | 129 | 24.0% |
| `dl/lstm_ae.py` | 202 | 55.0% |
| `sim/bench.py` | 89 | 0.0% |
| `stream/features_stream.py` | 212 | 59.0% |

## Test kinds (PRD §11)

- **Unit:** loaders, ground truth, cleaning (Spark vs pandas reference), features (hand-computed values), scorer (numpy vs Spark model), alert rule, live scoring (the same alerts however the windows are split into micro-batches), evaluation labels, scenario draws, a one-day epyt-flow simulation with and without a leak, copilot SQL guard, guardrails and chat CLI (scripted model).
- **Data contracts (pandera):** raw, cleaned, baseline and feature tables on the real lake.
- **Batch/stream parity:** file stream with a restart, Kafka end to end, full two years (`features_parity_battledim.json`).
- **Metamorphic** (`tests/metamorphic/`): shifting a day by whole weeks leaves its features unchanged; a bigger simulated leak never lowers the score; appending later data does not change alerts that already started; the quiet day without a leak raises no alert.
- **Simulator as oracle:** a large simulated leak on real background raises an alert and the localizer ranks nearby junctions.
- **Integration:** replayer → Kafka → Spark → lake, and the Kafka feature stream vs batch.
