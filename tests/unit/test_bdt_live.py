"""Live scoring (D-022): alerts do not depend on how windows are split into micro-batches."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from leakwatch.bdt.live import LiveScoring, alert_message, write_alerts_staged, write_scores_staged
from leakwatch.bdt.scorer import KMeansParams

COLUMNS = ("flow__p1__residual", "pressure__n1__residual", "pressure__n2__residual")
AREAS = {"flow__p1": "A", "pressure__n1": "A", "pressure__n2": "B"}
CFG = {
    "bdt": {
        "close_after_windows": 2,
        "top_sensors": 2,
        "model": "kmeans",
        "inflow_series": ["flow__p1"],
    }
}


@pytest.fixture
def scoring(tmp_path) -> LiveScoring:
    KMeansParams(
        feature_columns=COLUMNS,
        mean=np.zeros(3),
        std=np.ones(3),
        pc=np.eye(3),
        centers=np.zeros((1, 3)),
        threshold=1.0,
        persistence_windows=2,
    ).save(tmp_path / "model")
    return LiveScoring.from_model_dir(tmp_path / "model", CFG, AREAS, origin="stream")


def windows(values: list[tuple[float, float, float]], start="2019-03-01 00:00") -> pd.DataFrame:
    t0 = pd.Timestamp(start, tz="UTC")
    starts = [t0 + pd.Timedelta(minutes=15 * i) for i in range(len(values))]
    df = pd.DataFrame(values, columns=list(COLUMNS))
    df.insert(0, "source", "battledim")
    df.insert(1, "scenario_id", None)
    df.insert(2, "window_start", starts)
    df.insert(3, "window_end", [s + pd.Timedelta(hours=1) for s in starts])
    return df


QUIET, LEAK = (0.1, 0.0, 0.0), (5.0, 2.0, 0.0)
SERIES = [QUIET] * 3 + [LEAK] * 4 + [QUIET] * 4


def final_versions(rows):
    latest = {}
    for r in rows:
        if r["alert_id"] not in latest or r["version"] >= latest[r["alert_id"]]["version"]:
            latest[r["alert_id"]] = r
    return latest


def test_one_alert_opens_after_m_windows_and_closes(scoring):
    scored, rows, state = scoring.process(windows(SERIES), None)
    assert scored["above"].tolist() == [False] * 3 + [True] * 4 + [False] * 4
    alerts = final_versions(rows)
    assert len(alerts) == 1
    (alert,) = alerts.values()
    assert alert["status"] == "closed"
    assert alert["area"] == "A"  # flow__p1 and pressure__n1 explain the score
    assert alert["origin"] == "stream"
    assert "battledim|" in state
    assert scoring.stats == {"windows": len(SERIES), "alerts": len(rows)}


@pytest.mark.parametrize("cuts", [[1], [4], [5, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
def test_alerts_do_not_depend_on_micro_batch_boundaries(scoring, cuts):
    whole = final_versions(scoring.process(windows(SERIES), None)[1])
    df, state, rows = windows(SERIES), None, []
    for lo, hi in zip([0, *cuts], [*cuts, len(SERIES)], strict=True):
        _, changed, state = scoring.process(df.iloc[lo:hi], state)
        rows += changed
    split = final_versions(rows)
    keep = ("alert_id", "start_ts", "end_ts", "status", "area", "kmeans_score_max")
    assert [{k: a[k] for k in keep} for a in split.values()] == [
        {k: a[k] for k in keep} for a in whole.values()
    ]


def test_replayed_windows_are_ignored(scoring):
    df = windows(SERIES)
    _, rows, state = scoring.process(df, None)
    _, again, _ = scoring.process(df, state)
    assert rows and again == []


def test_localizer_fills_the_candidates(scoring):
    seen = []

    def localize(row, part):
        seen.append(len(part))
        return [{"node_id": "n1", "prob": 0.9}]

    scoring.localize = localize
    _, rows, _ = scoring.process(windows(SERIES), None)
    assert seen and all(r["gnn_top5"] == [{"node_id": "n1", "prob": 0.9}] for r in rows)


def test_empty_batch_keeps_the_state(scoring):
    scored, rows, state = scoring.process(windows([]), {"x": 1})
    assert scored.empty and rows == [] and state == {"x": 1}


def test_staged_scores_and_alerts_are_readable_parquet(scoring, tmp_path):
    scored, rows, _ = scoring.process(windows(SERIES), None)
    assert write_scores_staged(scored, tmp_path / "scores", "kmeans", "stream") == len(SERIES)
    part = tmp_path / "scores" / "model=kmeans" / "origin=stream" / "date=2019-03-01"
    table = pq.read_table(part / "part-00000.parquet")
    assert table.num_rows == len(SERIES)
    assert write_alerts_staged(rows, tmp_path / "alerts", "stream") == len(rows)
    alerts = pq.read_table(tmp_path / "alerts" / "origin=stream" / "part-00000.parquet")
    assert alerts.num_rows == len(rows)
    assert write_scores_staged(scored.iloc[:0], tmp_path / "none", "kmeans", "stream") == 0
    assert write_alerts_staged([], tmp_path / "none", "stream") == 0


def test_alert_message_is_json_safe(scoring):
    _, rows, _ = scoring.process(windows(SERIES), None)
    msg = alert_message(rows[0])
    assert isinstance(msg["start_ts"], str) and msg["start_ts"].startswith("2019-03-01T")


def test_model_without_threshold_is_refused(tmp_path):
    KMeansParams(
        feature_columns=COLUMNS,
        mean=np.zeros(3),
        std=np.ones(3),
        pc=np.eye(3),
        centers=np.zeros((1, 3)),
    ).save(tmp_path / "untrained")
    with pytest.raises(ValueError, match="no threshold"):
        LiveScoring.from_model_dir(tmp_path / "untrained", CFG, AREAS)
