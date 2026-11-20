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


def _item(item_id, category, latency, **checks):
    return {"id": item_id, "category": category, "skipped": None, "latency_s": latency, **checks}


def test_pe_report_data_latency_aim_and_misses(metrics_dir):
    """Data-question latency comes from the per-item rows; every failed check is named."""
    write(
        metrics_dir,
        "pe_eval_v3",
        {
            "model": {"model": "m", "endpoint_host": "h"},
            "rate_limit_wait_s": 42.0,
            "per_item": [
                _item("d1", "data", 8.0, correct=True, sql_exec_match=True),
                _item("d2", "data", 12.0, correct=True, sql_exec_match=False),
                _item("d3", "data", 14.0, correct=True, hallucinated_after=False),
                _item("p1", "procedure", 30.0, correct=False, citation_ok=False),
                _item("k1", "complaint", 20.0, correct=True, hallucinated_after=True),
                {**_item("d4", "data", 1.0, correct=False), "skipped": "lake has no scores"},
            ],
        },
    )
    assert reports.category_latency(reports.load("pe_eval_v3"), "data")["p50"] == 12.0
    text = reports.pe_report()
    assert "| v3 | 3 | 12.0 | 14.0 | 1 | missed | 42 |" in text
    assert (
        "v3 items with a failed check (3): d2 (SQL result); p1 (answer, citation); "
        "k1 (unsupported number)." in text
    )
    row = next(r for r in reports.readme_results().splitlines() if "Copilot v3" in r)
    assert "data answers median 12.0 s (aim ≤ 10 s" in row


def test_pe_report_prefers_stored_category_latency(metrics_dir):
    stored = {"n": 2, "mean": 5.0, "p50": 4.0, "p95": 6.0, "max": 6.0}
    write(
        metrics_dir,
        "pe_eval_v3",
        {
            "model": {"model": "m", "endpoint_host": "h"},
            "latency_by_category_s": {"data": stored},
            "per_item": [_item("d1", "data", 99.0, correct=True)],
        },
    )
    assert reports.category_latency(reports.load("pe_eval_v3"), "data") == stored
    assert "| v3 | 2 | 4.0 | 6.0 | 0 | met |" in reports.pe_report()


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


def test_reports_render_from_the_committed_metrics():
    """Regression: every draft and the README table build from the metrics files in the repo."""
    if not any(reports.METRICS_DIR.glob("*.json")):
        pytest.skip("no metrics files")
    for name, fn in reports.REPORTS.items():
        text = fn()
        assert text.startswith("# "), name
        assert "(code " in text, name  # numbers carry their source
    table = reports.readme_results()
    assert table.count("\n") >= 5 and "`data_summary.json`" in table


def test_demo_distances_show_refined_and_first_localization():
    runs = [
        {
            "node": "n1",
            "distance": {"best_of_top5_m": 44.0},
            "distance_at_alert": {"best_of_top5_m": 900.0},
        },
        {"node": "n2", "distance": None, "distance_at_alert": None},
    ]
    text = reports.demo_distances({"runs": runs})
    assert (
        "n1 44 m / n2 n/a m" in text and "(at the alert, before refinement: 900 m / n/a m)" in text
    )
    assert reports.demo_distances({"runs": [{"node": "n1"}]}) == ""


def test_official_score_row_and_report_section(metrics_dir):
    year = {
        "role": "test (official)",
        "total_score_eur": 230_000.0,
        "true_positives": 10,
        "leaks": 23,
        "false_positives": 4,
        "alerts": 30,
        "rank_among_top6": 3,
        "sixth_place_eur": 127_626,
    }
    write(
        metrics_dir,
        "battledim_score",
        {
            "detector": "kmeans_rebaselined",
            "years": {"2019": year},
            "leaderboard_top6": [{"team": "Tongji-Team", "score_eur": 264_873}],
        },
    )
    row = next(r for r in reports.readme_results().splitlines() if "official score" in r)
    assert "€230,000: 10 of 23 leaks, 4 false positives; would rank 3" in row
    text = reports.bdt_report()
    assert "## Official BattLeDIM 2020 score" in text and "rank **3**" in text


def test_real_leak_localization_table(metrics_dir):
    loc = {"median_distance_top1_m": 250.0, "within_m_top1": 0.5, "within_m_best_of_top5": 0.75}
    write(
        metrics_dir,
        "dl_gnn",
        {
            "models": {
                "gcn_sens": {
                    "test": {
                        "top1": 0.1,
                        "top5": 0.3,
                        "mean_distance_top1_m": 400,
                        "within_m_best_of_top5": 0.8,
                    }
                }
            },
            "chosen_model": "gcn_sens",
            "gnn_beats_mlp_top5": True,
            "dataset": {"test": 10, "unseen_sizes": 0, "unseen_diameter_m": [0.014, 0.017]},
            "real_leaks": {
                y: {"leaks": 14, "localizers": {"nearest_sensor": loc, "gcn_sens": loc}}
                for y in ("2018", "2019")
            },
        },
    )
    text = reports.dl_report()
    assert "Recorded BattLeDIM leaks (D-038)" in text
    assert "| gcn_sens | 250 | 50.0% | 75.0% | 250 | 50.0% | 75.0% |" in text
    row = next(r for r in reports.readme_results().splitlines() if "2019 BattLeDIM leaks" in r)
    assert "median 250 m" in row
