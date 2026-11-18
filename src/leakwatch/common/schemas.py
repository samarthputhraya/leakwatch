"""Shared contracts (PRD §6): the only interfaces between modules.

- pydantic models validate single records at runtime (Kafka messages, alerts, work orders).
- pandera schemas validate whole tables in tests and at module boundaries.
- Spark StructTypes are added here in Phase 1, built from the same field lists.

Changing anything in this file changes a contract between subjects: agree it with the team first
(PRD §15) and log it in docs/DECISIONS.md.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import pandas as pd
import pandera.pandas as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------------

SensorType = Literal["pressure", "flow", "level", "amr"]
Source = Literal["battledim", "sim"]
Area = Literal["A", "B", "C"]
AlertStatus = Literal["open", "closed", "confirmed", "false_alarm"]
LeakType = Literal["abrupt", "incipient"]
Priority = Literal["P1", "P2", "P3"]

SENSOR_TYPES: tuple[str, ...] = ("pressure", "flow", "level", "amr")
SOURCES: tuple[str, ...] = ("battledim", "sim")
AREAS: tuple[str, ...] = ("A", "B", "C")
LEAK_TYPES: tuple[str, ...] = ("abrupt", "incipient")

#: Unit of each sensor type, as published by BattLeDIM.
UNITS: dict[str, str] = {"pressure": "m", "flow": "m3/h", "level": "m", "amr": "L/h"}

#: Separator between sensor type and sensor id in wide-table column names (``pressure__n54``).
#: Needed because some node ids carry both a pressure sensor and an AMR (n1, n4, n31).
COLUMN_SEP = "__"


def sensor_column(sensor_type: str, sensor_id: str) -> str:
    """Return the wide-table column name for a sensor, e.g. ``pressure__n54``."""
    if sensor_type not in SENSOR_TYPES:
        raise ValueError(f"unknown sensor_type {sensor_type!r}")
    return f"{sensor_type}{COLUMN_SEP}{sensor_id}"


def split_sensor_column(column: str) -> tuple[str, str]:
    """Inverse of :func:`sensor_column`: ``pressure__n54`` -> ``("pressure", "n54")``."""
    sensor_type, sep, sensor_id = column.partition(COLUMN_SEP)
    if not sep or sensor_type not in SENSOR_TYPES or not sensor_id:
        raise ValueError(f"not a sensor column: {column!r}")
    return sensor_type, sensor_id


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware (UTC)")
    return value.astimezone(UTC)


# --------------------------------------------------------------------------------------------
# Kafka: scada.readings (PRD §6.1)
# --------------------------------------------------------------------------------------------


class Reading(BaseModel):
    """One sensor reading on topic ``scada.readings``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ts: datetime
    sensor_id: str = Field(min_length=1)
    sensor_type: SensorType
    value: float
    unit: str
    source: Source
    scenario_id: str | None = None
    year: int = Field(ge=1900, le=2200)

    @field_validator("ts")
    @classmethod
    def _ts_utc(cls, value: datetime) -> datetime:
        return _to_utc(value)

    @field_validator("value")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite (missing readings are not published)")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Reading:
        if self.unit != UNITS[self.sensor_type]:
            raise ValueError(f"unit {self.unit!r} does not match sensor_type {self.sensor_type!r}")
        if self.source == "sim" and not self.scenario_id:
            raise ValueError("simulated readings need a scenario_id")
        return self


# --------------------------------------------------------------------------------------------
# Kafka: demo.commands (PRD §6.1)
# --------------------------------------------------------------------------------------------


class DemoCommand(BaseModel):
    """Dashboard -> simulator/replayer command on topic ``demo.commands``."""

    model_config = ConfigDict(extra="forbid")

    command: Literal["inject_leak", "reset", "set_speed"]
    params: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Alerts table + Kafka topic alerts (PRD §6.3)
# --------------------------------------------------------------------------------------------


class TopSensor(BaseModel):
    """A sensor's share of an anomaly score."""

    model_config = ConfigDict(extra="forbid")

    sensor_id: str
    contribution: float


class GnnCandidate(BaseModel):
    """One candidate leak location from the localizer."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    prob: float = Field(ge=0.0, le=1.0)


class Alert(BaseModel):
    """One row of the alerts table; also the payload on topic ``alerts``."""

    model_config = ConfigDict(extra="forbid")

    alert_id: UUID
    source: Source
    scenario_id: str | None = None
    start_ts: datetime
    end_ts: datetime | None = None
    status: AlertStatus
    kmeans_score_max: float
    threshold: float
    top_sensors: list[TopSensor] = Field(max_length=3)
    area: Area
    lstm_score_max: float | None = None
    gnn_top5: list[GnnCandidate] | None = Field(default=None, max_length=5)
    est_loss_m3h: float | None = None
    work_order_json: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("start_ts", "end_ts", "created_at", "updated_at")
    @classmethod
    def _ts_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _to_utc(value)

    @model_validator(mode="after")
    def _ordered(self) -> Alert:
        if self.end_ts is not None and self.end_ts < self.start_ts:
            raise ValueError("end_ts is before start_ts")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at is before created_at")
        return self


# --------------------------------------------------------------------------------------------
# Work order (PRD §9.3), stored as JSON in alerts.work_order_json
# --------------------------------------------------------------------------------------------


class ManualCitation(BaseModel):
    """A passage of a CPHEEO manual the copilot relied on."""

    model_config = ConfigDict(extra="forbid")

    doc: str
    chapter: str
    page: int | str


class WorkOrder(BaseModel):
    """Crew work order generated for an alert."""

    model_config = ConfigDict(extra="forbid")

    alert_id: UUID
    priority: Priority
    summary: str
    area: Area
    pipes_to_inspect: list[str]
    estimated_loss_m3_per_hour: float | None
    evidence: list[str]
    recommended_actions: list[str]
    manual_citations: list[ManualCitation]
    generated_at: datetime


# --------------------------------------------------------------------------------------------
# Table schemas (pandera)
# --------------------------------------------------------------------------------------------

_UTC = pd.DatetimeTZDtype("ns", "UTC")

#: Long-format readings: the Kafka message fields as a table (raw lake layer from Phase 1).
READINGS_SCHEMA = pa.DataFrameSchema(
    {
        "ts": pa.Column(_UTC),
        "sensor_id": pa.Column(str),
        "sensor_type": pa.Column(str, pa.Check.isin(SENSOR_TYPES)),
        "value": pa.Column(float, pa.Check(lambda s: s.map(math.isfinite), element_wise=False)),
        "unit": pa.Column(str, pa.Check.isin(sorted(set(UNITS.values())))),
        "source": pa.Column(str, pa.Check.isin(SOURCES)),
        "scenario_id": pa.Column(str, nullable=True),
        "year": pa.Column(int, pa.Check.in_range(1900, 2200)),
    },
    checks=[
        pa.Check(
            lambda df: df["unit"] == df["sensor_type"].map(UNITS),
            error="unit does not match sensor_type",
        ),
    ],
    strict=True,
    name="readings",
)

#: Tidy table of benchmark leaks (built by ``leakwatch.data.ground_truth``).
GROUND_TRUTH_SCHEMA = pa.DataFrameSchema(
    {
        "leak_id": pa.Column(str, unique=True),
        "pipe_id": pa.Column(str, pa.Check.str_matches(r"^p\d+$")),
        "start_ts": pa.Column(_UTC),
        "end_ts": pa.Column(_UTC, nullable=True),
        "peak_ts": pa.Column(_UTC),
        "type": pa.Column(str, pa.Check.isin(LEAK_TYPES)),
        "diameter_m": pa.Column(float, pa.Check.in_range(0.0, 0.5, include_min=False)),
        "peak_outflow_m3h": pa.Column(float, pa.Check.ge(0.0), nullable=True),
        "year_started": pa.Column(int),
        "repaired_in_data": pa.Column(bool),
    },
    checks=[
        pa.Check(
            lambda df: df["end_ts"].isna() | (df["end_ts"] > df["start_ts"]),
            error="end_ts must be after start_ts",
        ),
        pa.Check(
            lambda df: df["peak_ts"] >= df["start_ts"],
            error="peak_ts must not precede start_ts",
        ),
        pa.Check(
            lambda df: df["repaired_in_data"] == df["end_ts"].notna(),
            error="repaired_in_data must equal end_ts being set",
        ),
    ],
    strict=True,
    ordered=True,
    name="ground_truth",
)


def wide_scada_schema(expected_rows: int | None = None) -> pa.DataFrameSchema:
    """Schema for a wide SCADA table: UTC index, regular 5-minute steps, sensor columns."""
    checks = [
        pa.Check(lambda df: df.index.is_monotonic_increasing, error="index not sorted"),
        pa.Check(lambda df: df.index.is_unique, error="duplicate timestamps"),
    ]
    if expected_rows is not None:
        checks.append(pa.Check(lambda df: len(df) == expected_rows, error="unexpected row count"))
    return pa.DataFrameSchema(
        columns={
            r"^(pressure|flow|level|amr)__.+$": pa.Column(float, regex=True, nullable=True),
        },
        index=pa.Index(_UTC, name="ts"),
        checks=checks,
        strict=True,
        name="scada_wide",
    )
