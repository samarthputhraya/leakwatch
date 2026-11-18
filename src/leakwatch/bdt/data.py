"""Data plumbing for the BDT detector: feature tables in, scores and alerts out.

- feature tables are read with DuckDB (``lake.features``) into pandas: 35k windows × 233 columns
  per year fit easily in memory, and the numpy scorer works on them directly;
- normal 2018 windows (D-015) are split into time blocks of whole UTC days, cycling through
  train / threshold / validation days (``bdt.split_cycle_days``): overlapping windows must never be
  split at random;
- scores and alert versions are written to the lake with Spark (PRD §5: Spark writes the lake).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from leakwatch.common import lake, paths
from leakwatch.common.schemas import ALERT_LAKE_COLUMNS, SCORE_COLUMNS, alerts_spark_schema
from leakwatch.features.normal import intervals_from_config, window_normal_mask
from leakwatch.features.spec import FeatureSpec


def load_features(year: int, source: str = "battledim", table: str = "features") -> pd.DataFrame:
    """Batch feature rows of one year (``window_start`` in that year), in time order."""
    con = lake.connect()
    df = lake.query_df(
        con,
        f"""SELECT * EXCLUDE ("window", date) FROM {table}
            WHERE source = '{source}' AND year(window_start) = {int(year)}
            ORDER BY window_start""",
    )
    if df.empty:
        raise RuntimeError(f"no {table} rows for {source} {year}: run `make features`")
    return df


def split_roles(
    window_start: pd.Series, cfg: Mapping[str, Any], origin: pd.Timestamp | None = None
) -> np.ndarray:
    """``train`` / ``threshold`` / ``validation`` per window by UTC day (see module docstring)."""
    origin = origin or pd.Timestamp("2018-01-01", tz="UTC")
    day = ((pd.DatetimeIndex(window_start) - origin) // pd.Timedelta(days=1)).to_numpy()
    cyc = day % int(cfg["split_cycle_days"])
    roles = np.full(len(day), "train", dtype=object)
    roles[cyc == int(cfg["threshold_day"])] = "threshold"
    roles[cyc == int(cfg["validation_day"])] = "validation"
    return roles


def normal_mask(
    features: pd.DataFrame, spec: FeatureSpec, gt: pd.DataFrame, cfg: Mapping[str, Any], rule: str
) -> np.ndarray:
    """D-015 (``rule="main"``) or strict ablation mask over feature windows."""
    intervals = intervals_from_config(gt, cfg["normal"], rule=rule)  # type: ignore[arg-type]
    return window_normal_mask(features["window_start"], spec.window_s, spec.step_s, intervals)


def series_area_map(spec: FeatureSpec) -> dict[str, str]:
    """Area of every feature series (for alert areas)."""
    from leakwatch.common.config import load_config
    from leakwatch.data import network

    wn = network.load_network()
    areas = network.node_areas(wn, load_config()["network"]["area_anchors"])
    return network.series_areas(wn, spec.series, areas)


def write_scores(
    spark: SparkSession, scored: pd.DataFrame, model: str, origin: str, overwrite: bool = True
) -> int:
    """Write score rows to ``lake/scores/model=…/origin=…/date=…`` (partitions replaced)."""
    out = scored[list(SCORE_COLUMNS)].copy()
    out["above"] = out["above"].astype(bool)
    out["cluster"] = out["cluster"].astype("int32")
    out["model"] = model
    out["origin"] = origin
    out["date"] = out["window_start"].dt.strftime("%Y-%m-%d")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    sdf = spark.createDataFrame(out)
    (
        sdf.repartition("date")
        .write.mode("overwrite" if overwrite else "append")
        .partitionBy("model", "origin", "date")
        .parquet(str(paths.lake_scores_dir()))
    )
    return len(out)


def alerts_to_spark(spark: SparkSession, alerts: pd.DataFrame):  # -> DataFrame
    """Alert rows (``ALERT_LAKE_COLUMNS``) as a Spark DataFrame with the lake schema."""
    rows = []
    for rec in alerts[list(ALERT_LAKE_COLUMNS)].to_dict("records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, pd.Timestamp):
                clean[k] = v.to_pydatetime()
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                clean[k] = None
            elif k in ("version", "windows_above"):
                clean[k] = int(v)
            else:
                clean[k] = v
        rows.append(clean)
    return spark.createDataFrame(rows, schema=alerts_spark_schema())


def write_alerts(spark: SparkSession, alerts: pd.DataFrame, origin: str, mode: str) -> int:
    """Write alert versions to ``lake/alerts/origin=…/`` (``mode``: overwrite or append)."""
    if alerts.empty:
        return 0
    sdf = alerts_to_spark(spark, alerts).drop("origin").withColumn("origin", F.lit(origin))
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    sdf.coalesce(1).write.mode(mode).partitionBy("origin").parquet(str(paths.lake_alerts_dir()))
    return len(alerts)
