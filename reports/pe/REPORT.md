# LeakWatch PE module: operator copilot

Draft generated from the metrics files.

## Design (PRD §9)

- Tools: describe_schema, run_sql (SQL guard), list_alerts, get_alert, get_sensor_series, get_gnn_candidates, estimate_loss, search_manual (RAG), log_complaint.
- Prompt versions: v1 zero-shot, v2 few-shot, v3 few-shot + schema + self-correction + guardrails.
- Guardrails: no number without a tool result (one regeneration, then tool output), SQL safety, refusal for out-of-scope questions, citations required for manual answers.
- Work orders: structured output validated with pydantic; template fallback when no model is available.

## Evaluation

_Not run yet._ The evaluation needs an LLM endpoint (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in `.env`: a free endpoint from `.env.example`, D-035, or the local CPU model from `make llm-local`, D-027); none was configured on the build machine. `make llm-check` tests an endpoint in a minute; `make pe-eval` writes `reports/metrics/pe_eval_v1|v2|v3.json` (a run stopped by a free tier's quota continues when rerun) and `make reports` regenerates this section. The harness itself is tested with a scripted model (`make pe-eval` refuses to produce numbers without a real one).

RAG index: 3,275 passages (sentence-transformers/all-MiniLM-L6-v2): CPHEEO O&M Manual 2005 1,378, CPHEEO Water Supply Manual 1999 1,868 (OCR of a scan), knowledge base 29 (`reports/metrics/rag_index.json` (code ae45577)).
