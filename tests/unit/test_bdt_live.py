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


def test_later_versions_keep_the_localizer_candidates(scoring):
    """The live localizer only has its input in the opening batch; updates must carry its result."""
    import json

    calls = []

    def localize(row, part):
        calls.append(row["version"])
        return [{"node_id": "n1", "prob": np.float32(0.5)}] if len(calls) == 1 else None

    scoring.localize = localize
    df, state, rows = windows(SERIES), None, []
    for i in range(len(SERIES)):  # one window per micro-batch, as in the live demo
        _, changed, state = scoring.process(df.iloc[i : i + 1], state)
        rows += changed
    assert calls == [1]  # localized once, when the alert opened
    assert len(rows) > 1 and all(r["gnn_top5"] == [{"node_id": "n1", "prob": 0.5}] for r in rows)
    json.dumps(state)  # the carried state stays JSON (it is stored in the commit marker)


def test_open_alert_is_refined_during_its_first_hours_then_kept(tmp_path):
    """D-033: a fast alert's first windows are mostly pre-leak; later windows refine the top 5."""
    KMeansParams(
        feature_columns=COLUMNS,
        mean=np.zeros(3),
        std=np.ones(3),
        pc=np.eye(3),
        centers=np.zeros((1, 3)),
        threshold=1.0,
        persistence_windows=2,
    ).save(tmp_path / "model")
    cfg = {**CFG, "dl": {"gnn": {"live_refine_hours": 1.0}}}
    live = LiveScoring.from_model_dir(tmp_path / "model", cfg, AREAS)
    assert live.refine_hours == 1.0
    seen = []

    def localize(row, recent):
        seen.append((row["version"], len(recent)))
        return [{"node_id": f"n{len(seen)}", "prob": 0.5}]

    live.localize = localize
    series = [QUIET] * 3 + [LEAK] * 12
    df, state, rows = windows(series), None, []
    for i in range(len(series)):
        _, changed, state = live.process(df.iloc[i : i + 1], state)
        rows += changed
    # opened at the 5th window (m = 2); refined while its last window is <= 1 h after the start
    versions = [v for v, _ in seen]
    assert versions == list(range(1, len(versions) + 1)) and len(versions) == 5
    assert [n for _, n in seen][-1] == 9  # the localizer sees the stream's recent windows
    assert rows[-1]["gnn_top5"] == [{"node_id": "n5", "prob": 0.5}]  # last refinement is kept


def test_live_rebaseline_seeds_from_history_and_keeps_its_own():
    """D-038: a step in a live residual shows as a change against the seeded reference."""
    from leakwatch.bdt.live import LiveScoring
    from leakwatch.features.rebaseline import Rebaseline

    cols = ["pressure__n1__residual"]
    calls = []

    def seed(first, back):
        calls.append((first, back))
        ts = pd.date_range(first - back, first, freq="15min", tz="UTC", inclusive="left")
        return pd.DataFrame({"window_start": ts, cols[0]: 1.0})

    live = LiveScoring(None, None, [])  # type: ignore[arg-type]
    live.rebaseline, live.rebaseline_columns, live.seed = Rebaseline(2, 1), cols, seed
    start = pd.Timestamp("2018-11-19 00:00", tz="UTC")
    part = pd.DataFrame(
        {"window_start": pd.date_range(start, periods=4, freq="15min", tz="UTC"), cols[0]: -2.0}
    )
    out = live._rebaselined("battledim|", part)
    assert np.allclose(out[cols[0]], -3.0)  # -2 against a reference of +1
    later = part.assign(window_start=part["window_start"] + pd.Timedelta(hours=1))
    live._rebaselined("battledim|", later)
    assert len(calls) == 1  # the stream keeps its own history after the first batch
    assert live.reference["battledim|"]["window_start"].max() == later["window_start"].max()
