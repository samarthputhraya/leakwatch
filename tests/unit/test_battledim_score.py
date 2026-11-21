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


def test_placement_is_chosen_on_2018_with_the_localizer_on_ties():
    assert bs.choose_placement({"localizer": 10.0, "sensor_drop": 20.0}) == "sensor_drop"
    assert bs.choose_placement({"localizer": 20.0, "sensor_drop": 10.0}) == "localizer"
    assert bs.choose_placement({"localizer": 5.0, "sensor_drop": 5.0}) == "localizer"
    assert bs.PLACEMENTS == ("localizer", "sensor_drop")


# Gaps found by mutation testing (D-040).


def test_rank_at_ties_and_edges():
    top = max(t["score_eur"] for t in bs.LEADERBOARD)
    last = min(t["score_eur"] for t in bs.LEADERBOARD)
    tie = bs.rank(260_562)  # equal to the 2nd team: placed beside it, not below
    assert tie["rank_among_top6"] == 2 and not tie["above_all_tabulated"]
    assert bs.rank(top)["above_all_tabulated"] and bs.rank(top)["rank_among_top6"] == 1
    at_last = bs.rank(last)
    assert at_last["rank_among_top6"] == 6 and not at_last["below_tabulated"]
    below = bs.rank(last - 1)
    assert below["rank_among_top6"] is None and below["below_tabulated"]
    assert below["sixth_place_eur"] == last and below["teams_in_competition"] == 18


def test_bounds_of_the_year_and_of_a_leak_are_inclusive(dist):
    leaks = [leak("p1", 100, 1_000)]
    on_end = bs.score_detections(
        pd.DataFrame({"link_id": ["p1"], "ts": [at(1_000)]}), leaks, dist, PARAMS
    )
    assert on_end["true_positives"] == 1  # still running on its end step
    after = bs.score_detections(
        pd.DataFrame({"link_id": ["p1"], "ts": [at(1_001)]}), leaks, dist, PARAMS
    )
    assert after["false_positives"] == 1
    last_step = bs.to_step(bs.year_bounds(2019)[1])
    late = [leak("p1", 100, last_step)]
    r = bs.score_detections(
        pd.DataFrame({"link_id": ["p1"], "ts": [at(last_step)]}), late, dist, PARAMS
    )
    assert r["detections"] == 1 and r["true_positives"] == 1


def test_per_detection_rows(dist):
    leaks = [leak("p1", 100, 2_000)]
    det = pd.DataFrame({"link_id": ["p1", "p1", "p4"], "ts": [at(110), at(120), at(5)]})
    r = bs.score_detections(det, leaks, dist, PARAMS)
    fp, tp, rep = r["per_detection"]  # in time order
    assert fp == {
        "link_id": "p4",
        "ts": at(5).isoformat(),
        "leak": None,
        "distance_m": None,
        "score_eur": -500.0,
        "label": "fp",
    }  # before the leak started
    assert tp["leak"] == "p1" and tp["distance_m"] == 0.0 and tp["label"] == "tp"
    assert tp["delay_h"] == pytest.approx(10 * 5 / 60, abs=0.01)
    assert rep == {
        "link_id": "p1",
        "ts": at(120).isoformat(),
        "leak": None,
        "distance_m": None,
        "score_eur": 0.0,
        "label": "repeat",
    }
    assert r["repeated_detections"] == 1 and r["missed_leaks"] == []
    assert r["detected_leaks"] == ["p1"] and r["tp_rate"] == 1.0


def test_official_leaks_of_a_year(tmp_path):
    first, last = bs.year_bounds(2019)
    gt = pd.DataFrame(
        {
            "leak_id": ["ended_at_start", "ended_before", "starts_last", "open", "next_year"],
            "pipe_id": ["p1", "p2", "p3", "p4", "p2"],
            "start_ts": [
                first - pd.Timedelta(days=30),
                first - pd.Timedelta(days=30),
                last,
                first + pd.Timedelta(days=3),
                last + bs.STEP,
            ],
            "end_ts": [first, first - bs.STEP, pd.NaT, pd.NaT, pd.NaT],
        }
    )
    gt["end_ts"] = pd.to_datetime(gt["end_ts"], utc=True)
    flows = pd.DataFrame(
        {p: np.full(YEAR_STEPS, i + 1.0) for i, p in enumerate(["p1", "p2", "p3", "p4"])}
    )
    leaks = bs.official_leaks(gt, flows)  # 2019 by default
    assert [lk.link_id for lk in leaks] == ["p1", "p4", "p3"]  # start order; two left out
    assert leaks[0].end_step == 0 and leaks[0].start_step < 0
    assert leaks[1].end_step == bs.to_step(last) and leaks[1].start_step == 3 * 288
    assert leaks[2].start_step == bs.to_step(last)
    assert leaks[1].flow_m3h.dtype == np.float64 and leaks[1].flow_m3h[0] == 4.0


def test_perfect_detections_put_old_leaks_at_the_first_step():
    leaks = [leak("p1", -500, 3_000), leak("p3", 0, 3_000), leak("p4", 7, 3_000)]
    det = bs.perfect_detections(leaks)
    assert list(det["ts"]) == [at(1), at(1), at(7)]


class _Net:
    """b - a - c with a 0.5 m pipe a-c and a 10 m pipe a-b."""

    node_name_list = ("a", "b", "c")

    def links(self):
        for name, s, e, length in (("zz_short", "a", "c", 0.5), ("aa_long", "a", "b", 10.0)):
            yield (
                name,
                SimpleNamespace(
                    start_node_name=s, end_node_name=e, link_type="Pipe", length=length
                ),
            )


def test_short_pipes_are_edges_and_the_shortest_incident_pipe_wins():
    d = bs.LinkDistances(_Net())
    assert d.link_at_node("a") == "zz_short"  # shortest, though not first by name
    assert d.link_at_node("c") == "zz_short" and d.link_at_node("b") == "aa_long"
    assert d.node_distance[d.index["b"], d.index["c"]] == pytest.approx(10.5)
    assert d.node_distance[d.index["c"], d.index["b"]] == pytest.approx(10.5)
    assert d.nodes == ["a", "b", "c"]


def test_localizer_candidates_as_plain_ids_or_objects(dist):
    alerts = pd.DataFrame(
        {
            "alert_id": ["x", "y"],
            "start_ts": [at(1), at(2)],
            "gnn_top5": [["b"], [SimpleNamespace(node_id="e", prob=0.3)]],
            "top_sensors": [[], []],
        }
    )
    det = bs.detections_from_alerts(alerts, dist)
    assert list(det.columns) == ["alert_id", "ts", "link_id", "how"]
    assert list(det["link_id"]) == ["p1", "p4"] and list(det["ts"]) == [at(1), at(2)]
    empty = bs.detections_from_alerts(alerts.iloc[:0], dist)
    assert list(empty.columns) == ["alert_id", "ts", "link_id", "how"] and empty.empty


def test_revisit_window_is_chosen_on_2018_with_the_shorter_one_on_ties():
    assert bs.choose_revisit_days({0.0: 10.0, 7.0: 30.0, 14.0: 20.0}) == 7.0
    assert bs.choose_revisit_days({0.0: 30.0, 7.0: 30.0, 14.0: 30.0}) == 0.0


def test_random_alert_times_are_distinct_on_the_grid_and_inside_the_year():
    a = bs.random_alert_times(2019, 50, np.random.default_rng(0))
    b = bs.random_alert_times(2019, 50, np.random.default_rng(0))
    assert a.equals(b) and len(a) == 50 and a["start_ts"].is_unique
    first, last = bs.year_bounds(2019)
    assert (a["start_ts"] > first).all() and (a["start_ts"] <= last).all()
    assert (a["start_ts"].dt.minute % 15 == 0).all()
    assert a["start_ts"].is_monotonic_increasing


def test_empirical_p_value_counts_draws_at_least_as_good():
    assert bs.empirical_p_value(10.0, [1.0, 2.0, 3.0]) == pytest.approx(0.25)
    assert bs.empirical_p_value(2.0, [1.0, 2.0, 3.0]) == pytest.approx(0.75)
    assert bs.empirical_p_value(5.0, []) == 1.0


# ---- timing tests (D-047)


def test_far_link_is_away_from_every_leak(dist):
    assert bs.far_link([leak("p1", 0, 10)], dist) == "p4"
    assert bs.far_link([leak("p4", 0, 10)], dist) == "p1"


def test_oracle_reports_go_to_the_newest_unreported_running_leak(dist):
    leaks = [leak("p1", 100, 1_000), leak("p4", 500, 2_000)]
    times = [at(3_000), at(200), at(600), at(700)]  # unsorted on purpose
    det = bs.oracle_detections(times, leaks, 2019, far="p2")
    assert list(det["link_id"]) == ["p1", "p4", "p4", "p2"]  # A, then B, B again, none running
    r = bs.score_detections(det, leaks, dist, PARAMS)
    assert r["true_positives"] == 2 and r["repeated_detections"] == 1
    assert r["false_positives"] == 1


def test_first_alert_delays_skip_old_and_unalerted_leaks():
    leaks = [leak("p1", 100, 1_000), leak("p2", 500, 2_000), leak("p3", -50, 90), leak("p4", 5, 50)]
    delays = bs.first_alert_delays_h([at(600), at(200)], leaks, 2019)
    assert delays == [pytest.approx(100 * 5 / 60), pytest.approx(100 * 5 / 60)]


def test_compare_to_random_both_directions():
    up = bs.compare_to_random(10.0, [1.0, 2.0, 3.0, 11.0])
    assert up["p_value"] == pytest.approx(2 / 5) and up["random_mean"] == pytest.approx(4.25)
    down = bs.compare_to_random(2.0, [1.0, 5.0, 6.0, 7.0], smaller_is_better=True)
    assert down["p_value"] == pytest.approx(2 / 5)
    assert bs.compare_to_random(1.0, [float("nan")]) == {"observed": 1.0, "draws": 0}


def test_timing_tests_report_every_measure(dist):
    leaks = [leak("p1", 100, 1_000), leak("p4", 500, 2_000)]
    scorer = SimpleNamespace(
        dist=dist, score=lambda d, year: bs.score_detections(d, leaks, dist, PARAMS, year)
    )
    system_times = [at(110), at(510)]  # right after each onset
    system = scorer.score(pd.DataFrame({"link_id": ["p1", "p4"], "ts": system_times}), 2019)
    rng = np.random.default_rng(0)
    sets = [bs.random_alert_times(2019, 2, rng) for _ in range(20)]
    rand = [
        scorer.score(bs.oracle_detections(list(s["start_ts"]), leaks, 2019, "p2"), 2019)
        for s in sets
    ]
    out = bs.timing_tests(system_times, sets, system, rand, leaks, scorer, 2019)
    assert out["oracle_placement_true_positives"]["observed"] == 2
    assert out["median_delay_to_first_alert_h"]["observed"] == pytest.approx(10 * 5 / 60, abs=0.01)
    assert out["median_delay_to_first_alert_h"]["leaks_starting_in_year"] == 2
    assert out["median_delay_to_first_correct_report_h"]["leaks_found"] == 2
    assert out["oracle_placement_eur"]["p_value"] <= 0.2  # early reports beat random ones
