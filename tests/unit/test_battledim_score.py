"""Official BattLeDIM scoring port (D-037): ``Scoring_Algorithm.m`` rules on a toy network."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from leakwatch.eval import battledim_score as bs

PARAMS = bs.ScoringParams()
YEAR_STEPS = 105_120


class ToyNetwork:
    """A line of junctions a - b - c - d - e (100 m pipes) plus a valve d - f."""

    node_name_list = ("a", "b", "c", "d", "e", "f")

    def links(self):
        pipes = [("p1", "a", "b"), ("p2", "b", "c"), ("p3", "c", "d"), ("p4", "d", "e")]
        for name, s, e in pipes:
            yield (
                name,
                SimpleNamespace(start_node_name=s, end_node_name=e, link_type="Pipe", length=100.0),
            )
        yield "v1", SimpleNamespace(start_node_name="d", end_node_name="f", link_type="Valve")


def leak(link: str, start: int, end: int, flow: float = 12.0) -> bs.OfficialLeak:
    series = np.zeros(YEAR_STEPS)
    series[max(start, 0) : end + 1] = flow
    return bs.OfficialLeak(link, start, end, series)


def at(step: int) -> pd.Timestamp:
    return bs.YEAR_START + bs.STEP * step


@pytest.fixture
def dist() -> bs.LinkDistances:
    return bs.LinkDistances(ToyNetwork())


def test_link_distance_is_pipe_path_between_centres(dist):
    assert dist.between("p1", "p1") == 0.0
    assert dist.between("p1", "p2") == pytest.approx(100.0)  # 0 between ends + 50 + 50
    assert dist.between("p1", "p4") == pytest.approx(300.0)
    # valves are not part of the distance graph: f is unreachable
    assert np.isinf(dist.node_distance[dist.index["a"], dist.index["f"]])
    assert dist.link_at_node("c") in {"p2", "p3"} and dist.link_at_node("f") is None


def test_to_step_matches_time2steps():
    assert bs.to_step("2019-01-01 00:05") == 1
    assert bs.to_step(pd.Timestamp("2018-12-31 23:55", tz="UTC")) == -1


def test_true_positive_scores_water_saved_minus_crew_cost(dist):
    leaks = [leak("p1", 100, 1_000)]
    det = pd.DataFrame({"link_id": ["p2"], "ts": [at(400)]})
    r = bs.score_detections(det, leaks, dist, PARAMS)
    water = 12.0 * (1_000 - 400 + 2) / 12  # flow[td-1 : end], 5-minute steps
    assert r["true_positives"] == 1 and r["false_positives"] == 0
    assert r["total_score_eur"] == pytest.approx(water * 0.8 - 100 / 300 * 500, abs=0.01)
    assert r["per_detection"][0]["delay_h"] == pytest.approx(300 * 5 / 60)


def test_too_far_too_early_and_2018_detections(dist):
    leaks = [leak("p1", 100, 1_000)]
    det = pd.DataFrame(
        {
            "link_id": ["p4", "p1", "p1"],  # 300 m away is still in; then before the start; 2018
            "ts": [at(200), at(50), bs.YEAR_START - pd.Timedelta(hours=1)],
        }
    )
    r = bs.score_detections(det, leaks, dist, PARAMS)
    assert r["detections"] == 2  # the 2018 entry is dropped
    labels = [d["label"] for d in r["per_detection"]]
    assert labels == ["fp", "tp"]  # sorted by time: step 50 (too early), then step 200
    far = bs.score_detections(
        pd.DataFrame({"link_id": ["p4"], "ts": [at(200)]}),
        [leak("p1", 100, 1_000)],
        dist,
        bs.ScoringParams(max_distance_m=299.0),
    )
    assert far["false_positives"] == 1 and far["total_score_eur"] == -500.0


def test_repeat_scores_zero_and_closest_new_leak_is_claimed(dist):
    leaks = [leak("p1", 0, 2_000), leak("p3", 0, 2_000)]
    det = pd.DataFrame({"link_id": ["p2", "p2", "p2"], "ts": [at(10), at(20), at(30)]})
    r = bs.score_detections(det, leaks, dist, PARAMS)
    # p2 is 100 m from both: first claims p1 (tie -> first leak), second claims p3, third repeats
    assert [d["leak"] for d in r["per_detection"]] == ["p1", "p3", None]
    assert [d["label"] for d in r["per_detection"]] == ["tp", "tp", "repeat"]
    assert r["true_positives"] == 2 and r["false_positives"] == 0


def test_perfect_submission_and_rank(dist):
    leaks = [leak("p1", -500, 3_000), leak("p3", 10, 3_000)]
    perfect = bs.score_detections(bs.perfect_detections(leaks), leaks, dist, PARAMS)
    assert perfect["true_positives"] == 2 and perfect["false_positives"] == 0
    assert bs.rank(300_000)["rank_among_top6"] == 1
    assert bs.rank(200_000)["rank_among_top6"] == 4
    assert bs.rank(1_000)["below_tabulated"]


def test_detections_from_alerts_use_localizer_then_top_sensor(dist):
    alerts = pd.DataFrame(
        {
            "alert_id": ["x", "y", "z"],
            "start_ts": [at(1), at(2), at(3)],
            "gnn_top5": [[{"node_id": "b", "prob": 0.4}], None, None],
            "top_sensors": [[], [{"sensor_id": "e", "contribution": 0.9}], []],
        }
    )
    det = bs.detections_from_alerts(alerts, dist)
    assert list(det["alert_id"]) == ["x", "y"]
    assert list(det["how"]) == ["localizer", "top_sensor"]
    assert det.loc[1, "link_id"] == "p4"


@pytest.mark.needs_data
def test_perfect_score_matches_the_paper():
    from leakwatch.common import paths
    from leakwatch.data import network
    from leakwatch.data.ground_truth import load_ground_truth

    flows = pd.read_parquet(paths.battledim_parquet_dir() / "leak_flows_2019.parquet")
    leaks = bs.official_leaks(load_ground_truth(), flows)
    dist = bs.LinkDistances(network.load_network())
    r = bs.score_detections(bs.perfect_detections(leaks), leaks, dist, PARAMS)
    assert len(leaks) == 23
    assert round(r["total_score_eur"]) == bs.PERFECT_SCORE_EUR


def test_official_leaks_per_year_and_steps():
    ts = pd.Timestamp
    gt = pd.DataFrame(
        {
            "pipe_id": ["p1", "p2", "p3"],
            "start_ts": [
                ts("2018-03-01", tz="UTC"),
                ts("2018-06-01", tz="UTC"),
                ts("2019-02-01", tz="UTC"),
            ],
            "end_ts": [pd.NaT, ts("2018-07-01", tz="UTC"), ts("2019-03-01", tz="UTC")],
        }
    )
    flows = pd.DataFrame({p: np.zeros(10) for p in ("p1", "p2", "p3")})
    l18 = bs.official_leaks(gt, flows, 2018)
    l19 = bs.official_leaks(gt, flows, 2019)
    assert [x.link_id for x in l18] == ["p1", "p2"]
    assert [x.link_id for x in l19] == ["p1", "p3"]  # p1 never repaired: carried into 2019
    assert l18[0].end_step == bs.to_step("2018-12-31 23:55", 2018)
    assert bs.to_step("2018-01-01 00:05", 2018) == 1 and l19[0].start_step < 0


def test_detections_after_the_scored_year_are_left_out(dist):
    leaks = [leak("p1", 100, 1_000)]
    det = pd.DataFrame(
        {"link_id": ["p1", "p1"], "ts": [at(200), pd.Timestamp("2020-01-02", tz="UTC")]}
    )
    r = bs.score_detections(det, leaks, dist, PARAMS)
    assert r["detections"] == 1 and r["false_positives"] == 0
