"""Report drafts and the README results table come only from metrics files."""

from __future__ import annotations

import json

import pytest

from leakwatch.eval import reports


@pytest.fixture
def metrics_dir(tmp_path, monkeypatch):
    folder = tmp_path / "metrics"
    folder.mkdir()
    monkeypatch.setattr(reports, "METRICS_DIR", folder)
    return folder


def write(folder, name, payload):
    payload = {"meta": {"git_commit": "abc1234"}, **payload}
    (folder / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_every_report_renders_with_no_metrics(metrics_dir):
    for fn in reports.REPORTS.values():
        text = fn()
        assert text.startswith("# ")
    assert "make bdt-eval" in reports.bdt_report()
    assert "Not run yet" in reports.pe_report()


def test_readme_table_cites_the_metrics_file_of_each_number(metrics_dir):
    write(metrics_dir, "replay_battledim_2018", {"readings_per_s": 123851.2, "delivery_errors": 0})
    write(metrics_dir, "test_summary", {"passed": 10, "failed": 0, "coverage_percent": 71.26})
    table = reports.readme_results()
    assert "123,851 readings/s" in table
    assert "`replay_battledim_2018.json`" in table
    assert "71.3%" in table
    assert "Localizer" not in table  # no dl_gnn.json -> no row, never a placeholder number


def test_update_readme_replaces_only_the_marked_block(metrics_dir, tmp_path):
    write(metrics_dir, "test_summary", {"passed": 3, "failed": 1, "coverage_percent": 50.0})
    readme = tmp_path / "README.md"
    readme.write_text(
        f"intro\n{reports.README_START}\nold\n{reports.README_END}\noutro\n", encoding="utf-8"
    )
    assert reports.update_readme(readme)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n") and text.endswith("outro\n")
    assert "old" not in text and "3 passed, 1 failed" in text
    plain = tmp_path / "PLAIN.md"
    plain.write_text("no markers", encoding="utf-8")
    assert not reports.update_readme(plain)


def test_pe_report_reads_nested_rates(metrics_dir):
    rate = {"value": 0.875, "n": 8}
    write(
        metrics_dir,
        "pe_eval_v3",
        {
            "model": {"model": "m", "endpoint_host": "127.0.0.1:8080"},
            "sql_execution_accuracy": rate,
            "answer_correctness": {"overall": rate, "by_category": {"data": rate}},
            "hallucinated_number_rate": {"before_guardrail": rate, "after_guardrail": rate},
            "tool_call_correctness": rate,
            "refusal_accuracy": rate,
            "citation_correctness": rate,
            "complaint_match_accuracy": rate,
            "latency_s": {"p50": 12.34},
        },
    )
    text = reports.pe_report()
    assert "87.5% (8)" in text
    assert "m @ 127.0.0.1:8080" in text
    assert "12.3" in text
