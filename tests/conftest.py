from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leakwatch.common import paths

FIXTURES = paths.TESTS_FIXTURES_DIR / "battledim"


def _have_data() -> bool:
    return (
        paths.ground_truth_path().exists()
        and all(paths.scada_parquet_path(y).exists() for y in (2018, 2019))
        and paths.network_inp_path().exists()
    )


def _have_raw_lake_2018() -> bool:
    return (paths.lake_raw_dir() / "source=battledim" / "year=2018").exists()


def _have_features() -> bool:
    return (paths.lake_features_dir() / "source=battledim").exists() and any(
        paths.lake_baselines_dir().glob("*.parquet")
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests whose inputs are missing on this machine (e.g. in CI)."""
    have_data, have_lake, have_features = _have_data(), _have_raw_lake_2018(), _have_features()
    skip_data = pytest.mark.skip(reason="needs `make data` (BattLeDIM files not on this machine)")
    skip_lake = pytest.mark.skip(reason="needs the 2018 raw lake (`make ingest-year YEAR=2018`)")
    skip_features = pytest.mark.skip(reason="needs the feature tables (`make features`)")
    for item in items:
        if "needs_data" in item.keywords and not have_data:
            item.add_marker(skip_data)
        if "needs_lake" in item.keywords and not have_lake:
            item.add_marker(skip_lake)
        if "needs_features" in item.keywords and not have_features:
            item.add_marker(skip_features)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def official_config() -> Path:
    return FIXTURES / "dataset_configuration.yaml"


@pytest.fixture(scope="session")
def fixed_report() -> Path:
    return FIXTURES / "2018_Fixed_Leakages_Report.txt"


@pytest.fixture(scope="session")
def spark():
    """Small local SparkSession shared by all Spark tests (no Kafka connector)."""
    from leakwatch.common.config import load_config
    from leakwatch.stream.spark import get_spark

    cfg = load_config(environ={})["spark"]
    session = get_spark(
        {**cfg, "shuffle_partitions": 2, "ui_enabled": False, "driver_memory": "1g"},
        "leakwatch-tests",
        master="local[2]",
    )
    yield session
    session.stop()


# --------------------------------------------------------------------------------------------
# Feature tests: a small network and helpers that turn pandas frames into Spark readings
# --------------------------------------------------------------------------------------------

#: Spark schema of the reading rows the feature functions consume (clean.READING_COLUMNS).
READINGS_DDL = (
    "source string, scenario_id string, sensor_type string, sensor_id string, "
    "t long, value double, out_of_range boolean"
)


def spark_readings(spark, df: pd.DataFrame):
    """Long pandas readings (``sensor_type, sensor_id, ts, value`` [+ ``out_of_range``,
    ``source``, ``scenario_id``]) -> Spark DataFrame with ``clean.READING_COLUMNS``."""
    from pyspark.sql import functions as F

    rows = [
        (
            r.get("source", "battledim"),
            r.get("scenario_id"),
            r["sensor_type"],
            r["sensor_id"],
            int(pd.Timestamp(r["ts"]).timestamp()),
            float(r["value"]),
            bool(r.get("out_of_range", False)),
        )
        for r in df.to_dict("records")
    ]
    return (
        spark.createDataFrame(rows, READINGS_DDL)
        .withColumn("ts", F.timestamp_seconds("t"))
        .drop("t")
    )


@pytest.fixture(scope="session")
def small_spec():
    """FeatureSpec for a toy network: 2 pressure, 1 flow, 1 level sensor, AMRs in areas A and C."""
    from leakwatch.features.spec import FeatureSpec

    return FeatureSpec(
        step_s=300,
        window_s=3600,
        slide_s=900,
        stats=("mean", "std", "min", "max", "slope", "residual"),
        sensor_series=("pressure__n1", "pressure__n2", "flow__p1", "level__T1"),
        amr_areas={"a1": "C", "a2": "C", "a3": "A"},
        max_ffill_gap_steps=2,
        stuck_min_steps=12,
        stuck_exempt_types=("flow", "level", "amr"),
        exclude_out_of_range=True,
        lateness_s=900,
    )


def toy_readings(hours: int = 12, seed: int = 7) -> pd.DataFrame:
    """Toy readings for ``small_spec`` from 2018-01-01 00:00 UTC (a Monday), 5-minute steps.

    Includes what the cleaning rules act on: a 2-step gap (filled) and a 4-step gap (not filled)
    in pressure n1, a 20-step constant run in n1 (stuck), constant runs in the exempt flow sensor,
    a 1-step AMR gap, an out-of-range AMR value, and a sensor (n9) that is not a feature series.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=hours * 12, freq="5min", tz="UTC")
    i = np.arange(len(idx))
    series = {
        ("pressure", "n1"): 40 + np.sin(i / 9) + rng.normal(0, 0.05, len(i)),
        ("pressure", "n2"): 35 + 0.01 * i + rng.normal(0, 0.02, len(i)),
        ("pressure", "n9"): 30 + rng.normal(0, 0.1, len(i)),
        ("flow", "p1"): np.where((i // 30) % 2 == 0, 0.0, 44.07),
        ("level", "T1"): 3 + 0.5 * np.sin(i / 40),
        ("amr", "a1"): 100 + rng.normal(0, 5, len(i)),
        ("amr", "a2"): 200 + rng.normal(0, 5, len(i)),
        ("amr", "a3"): 50 + rng.normal(0, 2, len(i)),
    }
    series[("pressure", "n1")][80:100] = 39.5  # stuck run of 20 steps
    frames = []
    for (sensor_type, sensor_id), values in series.items():
        df = pd.DataFrame(
            {"sensor_type": sensor_type, "sensor_id": sensor_id, "ts": idx, "value": values}
        )
        df["out_of_range"] = False
        if sensor_id == "n1":
            df = df.drop(index=[30, 31, 60, 61, 62, 63])  # gap of 2 (filled), gap of 4 (not)
        if sensor_id == "a2":
            df = df.drop(index=[50])
        if sensor_id == "a1":
            df.loc[70, ["value", "out_of_range"]] = [1e9, True]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def toy_baselines(spark, spec):
    """A baseline per series and hour of week that varies with the hour (non-trivial residuals)."""
    rows = [
        (s, h, 10.0 + (h % 24) + 0.5 * k, 100)
        for k, s in enumerate(spec.series)
        for h in range(168)
    ]
    return spark.createDataFrame(
        rows, "series_id string, hour_of_week int, baseline double, n_readings long"
    )


@pytest.fixture(scope="session")
def to_spark(spark):
    """Function: long pandas readings -> Spark readings DataFrame."""
    return lambda df: spark_readings(spark, df)


@pytest.fixture(scope="session")
def toy_pdf() -> pd.DataFrame:
    """12 hours of toy readings (see ``toy_readings``)."""
    return toy_readings()


@pytest.fixture(scope="session")
def toy_base(spark, small_spec):
    """Toy hour-of-week baselines for ``small_spec``."""
    return toy_baselines(spark, small_spec)
