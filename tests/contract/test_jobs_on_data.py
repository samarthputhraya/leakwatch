"""Core jobs end to end on the real lake, results to a temp dir (D-049).

The official scorer and the localizer choice are the code behind the headline numbers; their
``main`` functions only run in ``make`` targets otherwise. These tests run them on the built lake
(alerts, features, the trained localizer) with fewer random draws, and check invariants that do
not depend on the current alerts: the perfect submission scores the paper's €523,154, the
bookkeeping adds up, the chosen localizer is one of the candidates and 2019 never decides it.
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.needs_features, pytest.mark.needs_data]


@pytest.fixture
def metrics_dir(tmp_path, monkeypatch):
    from leakwatch.common import metrics

    monkeypatch.setattr(metrics, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(metrics, "git_commit", lambda: "test")  # the stamp expects the repo dir
    return tmp_path


def _alerts_exist() -> bool:
    from leakwatch.common import lake, paths

    return lake.has_parquet(paths.lake_alerts_dir())


def test_official_score_job(metrics_dir, monkeypatch):
    if not _alerts_exist():
        pytest.skip("no batch alerts (run `make bdt-eval`)")
    monkeypatch.setenv("LEAKWATCH__BATTLEDIM_SCORE__RANDOM_BASELINE_DRAWS", "5")
    from leakwatch.eval import battledim_score

    assert battledim_score.main([]) == 0
    out = json.loads((metrics_dir / "battledim_score.json").read_text(encoding="utf-8"))
    assert out["perfect_submission_2019"]["total_score_eur"] == pytest.approx(523_153.82, abs=1)
    for year in ("2018", "2019"):
        y = out["years"][year]
        assert y["true_positives"] + y["false_negatives"] == y["leaks"]
        labels = [r["label"] for r in y["per_detection"]]
        assert labels.count("tp") == y["true_positives"]
        assert labels.count("fp") == y["false_positives"]
        assert len(labels) == y["detections"]
        assert out["random_baseline"][year]["draws"] == 5
        t = out["timing_tests"][year]
        assert t["oracle_placement_eur"]["draws"] == 5
        assert 0 < t["oracle_placement_eur"]["p_value"] <= 1
    assert out["placement"] in ("localizer", "sensor_drop")
    assert out["revisit_rule"]["chosen_days"] in [0, 3, 7, 14, 30]


def test_localizer_choice_job_never_uses_2019(metrics_dir, monkeypatch):
    from leakwatch.common import paths

    if not (paths.models_dir() / "gnn" / "meta.json").exists():
        pytest.skip("no trained localizer (run `make gnn`)")
    from leakwatch.dl import gnn, localizer

    called = []
    real = gnn.real_leak_benchmark

    def spy(year, *a, **k):
        called.append(year)
        return real(year, *a, **k)

    monkeypatch.setattr(gnn, "real_leak_benchmark", spy)
    assert localizer.main(["--no-write"]) == 0
    assert called == [2018]  # the dry run chooses without computing 2019 at all
    assert not (metrics_dir / "dl_localizer.json").exists()
