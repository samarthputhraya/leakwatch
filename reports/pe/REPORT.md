# LeakWatch PE module: operator copilot

Draft generated from the metrics files.

## Design (PRD §9)

- Tools: describe_schema, run_sql (SQL guard), list_alerts, get_alert, get_sensor_series, get_gnn_candidates, estimate_loss, search_manual (RAG), log_complaint.
- Prompt versions: v1 zero-shot, v2 few-shot, v3 few-shot + schema + self-correction + guardrails.
- Guardrails: no number without a tool result (one regeneration, then tool output), SQL safety, refusal for out-of-scope questions, citations required for manual answers.
- Work orders: structured output validated with pydantic; template fallback when no model is available.

## Evaluation

Model: nvidia/nemotron-3-ultra-550b-a55b @ integrate.api.nvidia.com. Questions: `evals/copilot_questions.jsonl`; expected answers are computed by reference SQL on the lake at evaluation time. Rates are shown as value (number of items the metric applies to).

| version | SQL exec acc. | answer correct | tool calls | halluc. numbers before guardrail | after guardrail | refusal acc. | citations | complaint match | latency p50 (s) |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 43.8% (32) | 65.3% (72) | 93.1% (72) | 25.0% (72) | 25.0% (72) | 86.1% (72) | 27.8% (18) | 100.0% (9) | 11.1 |
| v2 | 81.2% (32) | 90.3% (72) | 97.2% (72) | 6.5% (62) | 6.5% (62) | 100.0% (72) | 72.2% (18) | 100.0% (9) | 7.9 |
| v3 | 90.6% (32) | 97.2% (72) | 100.0% (72) | 8.1% (62) | 0.0% (62) | 100.0% (72) | 94.4% (18) | 100.0% (9) | 13.6 |

Answer correctness by category:

| version | data | procedure | out of scope | complaint |
|---|---|---|---|---|
| v1 | 97.1% (35) | 27.8% (18) | 0.0% (10) | 88.9% (9) |
| v2 | 97.1% (35) | 72.2% (18) | 100.0% (10) | 88.9% (9) |
| v3 | 100.0% (35) | 94.4% (18) | 100.0% (10) | 88.9% (9) |

Latency of data questions (PRD §10 aim: at most 10 s with a hosted model). Time spent pacing requests and waiting out the free tier's rate limits is measured separately and not counted:

| version | data questions | median (s) | p95 (s) | within 10 s | aim | rate-limit waits, whole run (s) |
|---|---|---|---|---|---|---|
| v1 | 35 | 11.1 | 20.1 | 14 | missed | 6 |
| v2 | 35 | 7.4 | 17.2 | 23 | met | 597 |
| v3 | 35 | 13.2 | 38.0 | 9 | missed | 1,037 |

v1 items with a failed check (50): alerts-01 (SQL result); alerts-02 (SQL result); alerts-03 (SQL result); alerts-05 (SQL result); alerts-06 (SQL result); alerts-07 (SQL result, unsupported number); alerts-10 (SQL result); alerts-11 (SQL result); alerts-13 (SQL result); alerts-16 (unsupported number); alerts-17 (SQL result, tools, unsupported number); readings-01 (SQL result, unsupported number); readings-02 (SQL result, tools, unsupported number); readings-03 (unsupported number); readings-07 (SQL result, unsupported number); readings-08 (SQL result, unsupported number); readings-09 (SQL result, tools, unsupported number); readings-10 (SQL result, tools, unsupported number); readings-11 (SQL result, unsupported number); features-01 (answer, SQL result, tools); chart-01 (unsupported number); chart-02 (unsupported number); proc-01 (answer, citation); proc-02 (answer, citation); proc-03 (answer, citation); proc-04 (answer, citation); proc-05 (answer, citation); proc-06 (answer, citation, unsupported number); proc-07 (answer, citation); proc-08 (answer, citation); proc-09 (answer, citation); proc-11 (answer, citation); proc-12 (answer, citation); proc-14 (answer, citation); proc-16 (answer, citation); proc-17 (unsupported number); proc-18 (unsupported number); oos-01 (answer, refusal); oos-02 (answer, refusal); oos-03 (answer, refusal); oos-04 (answer, refusal); oos-05 (answer, refusal); oos-06 (answer, refusal); oos-07 (answer, refusal); oos-08 (answer, refusal); oos-09 (answer, refusal); oos-10 (answer, refusal); complaint-01 (answer, complaint fields); complaint-04 (unsupported number); complaint-09 (unsupported number).

v2 items with a failed check (16): alerts-03 (SQL result); alerts-07 (SQL result); alerts-10 (SQL result); alerts-13 (SQL result); alerts-17 (SQL result); readings-01 (unsupported number); readings-10 (unsupported number); features-01 (answer, SQL result, tools); proc-08 (answer, citation); proc-09 (answer, tools, citation); proc-11 (answer, citation); proc-13 (answer, citation); proc-16 (answer, citation); proc-18 (unsupported number); complaint-01 (answer, complaint fields); complaint-04 (unsupported number).

v3 items with a failed check (5): alerts-10 (SQL result); alerts-13 (SQL result); alerts-17 (SQL result); proc-16 (answer, citation); complaint-01 (answer, complaint fields).

Sources: `reports/metrics/pe_eval_v1.json` (code f2a35c9), `reports/metrics/pe_eval_v2.json` (code f2a35c9), `reports/metrics/pe_eval_v3.json` (code f2a35c9).

PRD §10 targets: `{"sql_execution_accuracy": 0.85, "hallucinated_number_rate_after": 0.0, "refusal_accuracy": 0.9}`.

RAG index: 3,275 passages (sentence-transformers/all-MiniLM-L6-v2): CPHEEO O&M Manual 2005 1,378, CPHEEO Water Supply Manual 1999 1,868 (OCR of a scan), knowledge base 29 (`reports/metrics/rag_index.json` (code ae45577)).
