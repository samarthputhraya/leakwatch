# LeakWatch ST module: how we know it works

Draft generated from the metrics files.

- `make test-report`: 423 passed, 0 failed, 0 skipped in 1,111 s; line coverage of `src/leakwatch` 62.4% (`reports/metrics/test_summary.json` (code fe9da9b)).

| test folder | tests |
|---|---|
| contract | 39 |
| unit | 384 |

## Test kinds (PRD §11)

- **Unit:** loaders, ground truth, cleaning (Spark vs pandas reference), features (hand-computed values), scorer (numpy vs Spark model), alert rule, evaluation labels, copilot SQL guard and guardrails (mocked LLM).
- **Data contracts (pandera):** raw, cleaned, baseline and feature tables on the real lake.
- **Batch/stream parity:** file stream with a restart, Kafka end to end, full two years (`features_parity_battledim.json`).
- **Metamorphic** (`tests/metamorphic/`): shifting a day by whole weeks leaves its features unchanged; a bigger simulated leak never lowers the score; appending later data does not change alerts that already started; the quiet day without a leak raises no alert.
- **Simulator as oracle:** a large simulated leak on real background raises an alert and the localizer ranks nearby junctions.
- **Integration:** replayer → Kafka → Spark → lake, and the Kafka feature stream vs batch.
