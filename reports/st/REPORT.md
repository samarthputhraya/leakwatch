# LeakWatch ST module: how we know it works

Draft generated from the metrics files.

- `make test-report`: 610 passed, 0 failed, 0 skipped in 628 s; line coverage of `src/leakwatch` 71.4% (`reports/metrics/test_summary.json` (code 4bfbde7)).

| test folder | tests |
|---|---|
| contract | 39 |
| integration | 2 |
| metamorphic | 5 |
| unit | 564 |

Tests run: `slow or not slow`. Modules with the most uncovered lines (most are job entry points exercised by `make` targets rather than by unit tests):

| module | statements | covered |
|---|---|---|
| `dl/gnn.py` | 405 | 51.9% |
| `demo/runner.py` | 226 | 27.0% |
| `dl/gnn_data.py` | 170 | 26.5% |
| `features/build.py` | 122 | 0.0% |
| `bdt/train.py` | 166 | 28.9% |
| `bdt/evaluate.py` | 141 | 23.4% |
| `dl/lstm_ae.py` | 219 | 50.7% |
| `stream/live_replay.py` | 103 | 0.0% |
| `bdt/scaling.py` | 146 | 32.9% |
| `eval/battledim_score.py` | 246 | 61.8% |

## Test kinds (PRD §11)

- **Unit:** loaders, ground truth, cleaning (Spark vs pandas reference), features (hand-computed values), scorer (numpy vs Spark model), alert rule, live scoring (the same alerts however the windows are split into micro-batches), evaluation labels, scenario draws, a one-day epyt-flow simulation with and without a leak, copilot SQL guard, guardrails and chat CLI (scripted model).
- **Data contracts (pandera):** raw, cleaned, baseline and feature tables on the real lake.
- **Batch/stream parity:** file stream with a restart, Kafka end to end, full two years (`features_parity_battledim.json`).
- **Metamorphic** (`tests/metamorphic/`): shifting a day by whole weeks leaves its features unchanged; a bigger simulated leak never lowers the score; appending later data does not change alerts that already started; the quiet day without a leak raises no alert.
- **Simulator as oracle:** a large simulated leak on real background raises an alert and the localizer ranks nearby junctions.
- **Integration:** replayer → Kafka → Spark → lake, and the Kafka feature stream vs batch.
- **Property-based** (`tests/unit/test_properties.py`, hypothesis, D-040): the official score does not depend on submission order and its bookkeeping adds up; pipe distances are symmetric; re-baselining ignores a constant offset; the SQL guard never passes a write; the number guardrail accepts every value it was given.

## Mutation testing (D-040)

mutmut 3.8.1 changed the code one small edit at a time and reran the unit tests: 2,735 mutants, 2,181 killed, 296 survived, 0 timeouts, 258 in code no unit test reaches (mostly CLI entry points). Mutation score killed / (killed + survived) = **88.0%** (`reports/metrics/st_mutation.json` (code fa0a731)). Every survivor is listed in the metrics file; many are equivalent (same behaviour).

| module | killed | survived | timeout | no tests | score |
|---|---|---|---|---|---|
| `src/leakwatch/bdt/alerts.py` | 481 | 42 | 0 | 0 | 92.0% |
| `src/leakwatch/copilot/guardrails.py` | 545 | 78 | 0 | 0 | 87.5% |
| `src/leakwatch/copilot/sql_guard.py` | 113 | 23 | 0 | 0 | 83.1% |
| `src/leakwatch/eval/battledim_score.py` | 461 | 52 | 0 | 258 | 89.9% |
| `src/leakwatch/eval/detection.py` | 488 | 91 | 0 | 0 | 84.3% |
| `src/leakwatch/features/rebaseline.py` | 93 | 10 | 0 | 0 | 90.3% |
