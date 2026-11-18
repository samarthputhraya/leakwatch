"""Ground truth (PRD §4.1 step 4). Runs in CI on the committed official config fixture."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from leakwatch.data import ground_truth as gtm


@pytest.fixture(scope="module")
def gt(official_config: Path) -> pd.DataFrame:
    return gtm.build_ground_truth(official_config)


def test_23_leaks_active_in_2019(gt):
    """Phase 0 Definition of Done: 23 leaks are active in 2019."""
    assert len(gtm.active_in_year(gt, 2019)) == 23


def test_leak_counts_match_benchmark_description(gt):
    assert len(gt) == 33
    assert (gt["year_started"] == 2018).sum() == 14
    assert (gt["year_started"] == 2019).sum() == 19
    active18 = gtm.active_in_year(gt, 2018)
    assert len(active18) == 14
    # 10 of the 2018 leaks were found and fixed in 2018; 4 carry over into 2019
    assert active18["end_ts"].dt.year.eq(2018).sum() == 10
    active19 = gtm.active_in_year(gt, 2019)
    assert (active19["year_started"] == 2018).sum() == 4


def test_types_and_repairs(gt):
    assert set(gt["type"]) == {"abrupt", "incipient"}
    # unrepaired leaks have no end time; repaired ones end after they start
    assert gt.loc[~gt["repaired_in_data"], "end_ts"].isna().all()
    repaired = gt[gt["repaired_in_data"]]
    assert (repaired["end_ts"] > repaired["start_ts"]).all()
    # abrupt leaks peak at their start
    abrupt = gt[gt["type"] == "abrupt"]
    assert (abrupt["peak_ts"] == abrupt["start_ts"]).all()


def test_known_rows(gt):
    row = gt.set_index("pipe_id").loc["p257"]
    assert row["start_ts"] == pd.Timestamp("2018-01-08 13:30", tz="UTC")
    assert pd.isna(row["end_ts"])
    assert row["type"] == "incipient"
    assert row["diameter_m"] == pytest.approx(0.011843)
    row = gt.set_index("pipe_id").loc["p523"]
    assert row["end_ts"] == pd.Timestamp("2019-02-01 09:50", tz="UTC")
    assert row["leak_id"] == "L2019_p523"


def test_columns_follow_prd(gt):
    for col in ["pipe_id", "start_ts", "end_ts", "type", "peak_outflow_m3h", "year_started"]:
        assert col in gt.columns
    assert gt["start_ts"].is_monotonic_increasing


def test_cross_check_agrees_with_fixed_report_and_epyt_flow(gt, fixed_report: Path):
    assert gtm.cross_check(gt, fixed_report=fixed_report) == []


def test_cross_check_catches_a_wrong_repair_time(gt, fixed_report: Path):
    tampered = gt.copy()
    idx = tampered.index[tampered["pipe_id"] == "p461"][0]
    tampered.loc[idx, "end_ts"] = tampered.loc[idx, "end_ts"] + pd.Timedelta(hours=1)
    problems = gtm.cross_check(tampered, fixed_report=fixed_report)
    assert any("p461" in p for p in problems)


def test_cross_check_catches_a_missing_leak(gt):
    problems = gtm.cross_check(gt[gt["pipe_id"] != "p879"].reset_index(drop=True))
    assert any("p879" in p for p in problems)


def test_active_between_is_inclusive(gt):
    start = pd.Timestamp("2019-02-01 09:50", tz="UTC")  # p523's last leaking step
    assert "p523" in set(gtm.active_between(gt, start, start)["pipe_id"])
    after = start + pd.Timedelta(minutes=5)
    assert "p523" not in set(gtm.active_between(gt, after, after)["pipe_id"])


def test_peak_outflow_from_leak_flows(official_config: Path):
    idx = pd.date_range("2019-01-01", periods=3, freq="5min", tz="UTC")
    flows = {2019: pd.DataFrame({"p523": [0.0, 3.5, 2.0], "p257": [6.8, 6.9, 7.0]}, index=idx)}
    gt = gtm.build_ground_truth(official_config, flows)
    by_pipe = gt.set_index("pipe_id")["peak_outflow_m3h"]
    assert by_pipe["p523"] == pytest.approx(3.5)
    assert by_pipe["p257"] == pytest.approx(7.0)
    assert pd.isna(by_pipe["p879"])  # no flow data given for it


@pytest.mark.parametrize(
    "line",
    [
        "p1, 2018-01-01 00:00, 2018-01-02 00:00, 0.01, sudden, 2018-01-01 00:00",
        "p1, 2018-01-01 00:00, 0.01, abrupt, 2018-01-01 00:00",
    ],
)
def test_parse_leak_line_rejects_bad_input(line):
    with pytest.raises(ValueError):
        gtm.parse_leak_line(line)


def test_parse_fixed_report(fixed_report: Path):
    report = gtm.parse_fixed_report(fixed_report)
    assert len(report) == 10
    assert report.set_index("pipe_id").loc["p31", "end_ts"] == pd.Timestamp(
        "2018-08-12 17:30", tz="UTC"
    )
