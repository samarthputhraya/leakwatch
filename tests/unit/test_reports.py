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


def _model(top5, by_size):
    test = {"top1": 0.1, "top5": top5, "mean_distance_top1_m": 200.0, "within_m_best_of_top5": 0.9}
    return {"test": test, "unseen_sizes": test, "test_by_size": by_size}


def test_localizer_target_uses_medium_and_large_leaks_weighted_by_count(metrics_dir):
    gat = _model(
        0.5,
        {
            "small": {"n": 10, "top5": 0.0, "within_m_best_of_top5": 0.5},
            "medium": {"n": 30, "top5": 0.5, "within_m_best_of_top5": 0.9},
            "large": {"n": 10, "top5": 0.9, "within_m_best_of_top5": 1.0},
        },
    )
    assert reports.medium_large_top5(gat) == pytest.approx((30 * 0.5 + 10 * 0.9) / 40)
    assert reports.medium_large_top5({}) is None
    mlp = _model(0.3, {"medium": {"n": 10, "top5": 0.3}, "large": {"n": 10, "top5": 0.3}})
    bench = {
        "leaks_localized": 1,
        "model": "gat",
        "median_distance_top1_m": 900.0,
        "median_distance_best_of_top5_m": 800.0,
        "nearest_sensor_median_distance_top1_m": 400.0,
        "within_m_best_of_top5": 0.0,
        "per_leak": [],
    }
    write(
        metrics_dir,
        "dl_gnn",
        {
            "dataset": {"test": 50, "unseen_sizes": 5, "unseen_diameter_m": [0.014, 0.017]},
            "variants": {"mlp": {"kind": "mlp"}, "gat": {"kind": "gat"}},
            "models": {"mlp": mlp, "gat": gat},
            "chosen_model": "gat",
            "gnn_beats_mlp_top5": True,
            "battledim_2019": bench,
        },
    )
    text = reports.dl_report()
    assert "Measured: 60.0% (gat) vs 30.0% (MLP): met on the 60% level, better than" in text
    assert "worse than the nearest-sensor heuristic" in text
    assert "GAT, residuals at sensor nodes + closeness to each sensor + global sensor" in text
    row = reports.readme_results()
    assert "medium + large leaks 60.0% (target 60%, MLP 30.0%)" in row
    assert "900 m from the top-1 junction" in row and "400 m for the nearest-sensor" in row


def test_kmeans_target_names_the_variants_that_meet_it():
    def variant(abrupt, fa, live=False):
        return {
            "model_dir": "/m/kmeans_live" if live else "/m/kmeans_other",
            "eval_2019": {
                "detection": {"abrupt": {"detected": abrupt, "leaks": 8}},
                "false_alarms_per_week": fa,
            },
        }

    ev = {
        "live_model": "kmeans_live",
        "models": {"a": variant(4, 1.9, live=True), "b": variant(8, 2.5), "c": variant(3, 1.0)},
    }
    text = reports.kmeans_target_line(ev)
    assert "a 4/8 at 1.90/week" in text
    assert "met by a" in text and "the live model (a) meets it" in text
    ev["models"]["a"] = variant(3, 1.9, live=True)
    assert "not met by any variant" in reports.kmeans_target_line(ev)
