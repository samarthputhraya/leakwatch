"""Alert versions read back from the lake can be written again (how ``make gnn`` adds gnn_top5)."""

from __future__ import annotations

import numpy as np
import pytest

from leakwatch.bdt.data import _plain, write_alerts
from leakwatch.common import lake, paths

pytestmark = pytest.mark.spark


def test_plain_turns_arrays_of_structs_into_lists():
    value = np.array([{"sensor_id": "n1", "contribution": np.float64(0.5)}], dtype=object)
    assert _plain(value) == [{"sensor_id": "n1", "contribution": 0.5}]
    assert _plain(None) is None


def test_alert_versions_round_trip_through_the_lake(spark, tmp_path, monkeypatch):
    from leakwatch.copilot import testing

    monkeypatch.setenv("LEAKWATCH_DATA_DIR", str(tmp_path))
    testing.write_alerts(paths.lake_dir(), testing.alert_rows())
    sql = (
        "SELECT * EXCLUDE (work_order_json, work_order_generator) FROM alerts "
        "WHERE origin = 'batch' ORDER BY alert_id"
    )
    before = lake.query_df(lake.connect(), sql)
    updated = before.copy()
    updated["version"] = updated["version"] + 1
    updated["gnn_top5"] = [[{"node_id": "n7", "prob": 0.25}] for _ in range(len(updated))]
    assert write_alerts(spark, updated, "batch", mode="append") == len(updated)
    after = lake.query_df(lake.connect(), sql)
    assert list(after["version"]) == list(before["version"] + 1)
    assert [list(t) for t in after["top_sensors"]] == [list(t) for t in before["top_sensors"]]
    assert all(g[0]["node_id"] == "n7" for g in after["gnn_top5"])
