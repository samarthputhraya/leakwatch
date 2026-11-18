from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pandera.errors
import pytest
from pydantic import ValidationError

from leakwatch.common.schemas import (
    READINGS_SCHEMA,
    Alert,
    DemoCommand,
    Reading,
    WorkOrder,
    sensor_column,
    split_sensor_column,
)

T0 = datetime(2019, 3, 1, 12, 0, tzinfo=UTC)


def reading(**overrides):
    base = {
        "ts": T0,
        "sensor_id": "n54",
        "sensor_type": "pressure",
        "value": 41.2,
        "unit": "m",
        "source": "battledim",
        "scenario_id": None,
        "year": 2019,
    }
    return {**base, **overrides}


def alert(**overrides):
    base = {
        "alert_id": uuid.uuid4(),
        "source": "battledim",
        "start_ts": T0,
        "status": "open",
        "kmeans_score_max": 7.5,
        "threshold": 4.2,
        "top_sensors": [{"sensor_id": "n54", "contribution": 0.6}],
        "area": "A",
        "created_at": T0,
        "updated_at": T0,
    }
    return {**base, **overrides}


class TestReading:
    def test_valid_reading_round_trips_through_json(self):
        r = Reading(**reading())
        again = Reading.model_validate_json(r.model_dump_json())
        assert again == r
        assert again.ts.tzinfo is not None

    def test_non_utc_timestamp_is_normalised_to_utc(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        r = Reading(**reading(ts=datetime(2019, 3, 1, 23, 0, tzinfo=ist)))
        assert r.ts == datetime(2019, 3, 1, 17, 30, tzinfo=UTC)
        assert r.ts.utcoffset() == timedelta(0)

    @pytest.mark.parametrize(
        "bad",
        [
            {"ts": datetime(2019, 3, 1, 12, 0)},  # naive
            {"unit": "m3/h"},  # wrong unit for pressure
            {"value": float("nan")},
            {"value": float("inf")},
            {"sensor_type": "temperature"},
            {"source": "sim", "scenario_id": None},  # sim needs a scenario
            {"extra_field": 1},
        ],
    )
    def test_invalid_readings_rejected(self, bad):
        with pytest.raises(ValidationError):
            Reading(**reading(**bad))

    def test_sim_reading_with_scenario(self):
        r = Reading(**reading(source="sim", scenario_id="s-001"))
        assert r.scenario_id == "s-001"


class TestAlertAndWorkOrder:
    def test_valid_alert(self):
        a = Alert(**alert(gnn_top5=[{"node_id": f"n{i}", "prob": 0.1} for i in range(5)]))
        assert a.lstm_score_max is None
        assert len(a.gnn_top5 or []) == 5

    @pytest.mark.parametrize(
        "bad",
        [
            {"top_sensors": [{"sensor_id": f"n{i}", "contribution": 0.1} for i in range(4)]},
            {"gnn_top5": [{"node_id": f"n{i}", "prob": 0.1} for i in range(6)]},
            {"gnn_top5": [{"node_id": "n1", "prob": 1.5}]},
            {"area": "D"},
            {"status": "unknown"},
            {"end_ts": T0 - timedelta(minutes=5)},
            {"updated_at": T0 - timedelta(seconds=1)},
        ],
    )
    def test_invalid_alerts_rejected(self, bad):
        with pytest.raises(ValidationError):
            Alert(**alert(**bad))

    def test_work_order_json_round_trip(self):
        wo = WorkOrder(
            alert_id=uuid.uuid4(),
            priority="P1",
            summary="Pressure drop in area A",
            area="A",
            pipes_to_inspect=["p461", "p460"],
            estimated_loss_m3_per_hour=12.5,
            evidence=["get_alert: score 7.5 > threshold 4.2"],
            recommended_actions=["Step test around n54"],
            manual_citations=[{"doc": "CPHEEO O&M", "chapter": "Leakage Control", "page": 112}],
            generated_at=T0,
        )
        payload = json.loads(wo.model_dump_json())
        assert WorkOrder.model_validate(payload) == wo

    def test_demo_command(self):
        cmd = DemoCommand(command="inject_leak", params={"node_id": "n54"})
        assert cmd.params["node_id"] == "n54"
        with pytest.raises(ValidationError):
            DemoCommand(command="explode")


def test_sensor_column_round_trip():
    assert sensor_column("amr", "n1") == "amr__n1"
    assert split_sensor_column("pressure__n54") == ("pressure", "n54")
    with pytest.raises(ValueError):
        split_sensor_column("n54")
    with pytest.raises(ValueError):
        sensor_column("temperature", "n1")


def test_readings_schema_checks_unit_against_type():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2019-01-01 00:00", "2019-01-01 00:00"], utc=True),
            "sensor_id": ["n54", "p227"],
            "sensor_type": ["pressure", "flow"],
            "value": [40.0, 80.0],
            "unit": ["m", "m3/h"],
            "source": ["battledim", "battledim"],
            "scenario_id": [None, None],
            "year": [2019, 2019],
        }
    )
    READINGS_SCHEMA.validate(df)
    df.loc[1, "unit"] = "m"
    with pytest.raises(pandera.errors.SchemaError):
        READINGS_SCHEMA.validate(df)
