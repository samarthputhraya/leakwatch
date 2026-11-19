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


WEEK = pd.Timedelta(days=7)


def weekly_change(
    features: pd.DataFrame, columns: list[str], lag: pd.Timedelta = WEEK
) -> pd.DataFrame:
    """Week-over-week change features: each window minus the same window ``lag`` earlier.

    Rows without a reference window (the first week, gaps) are dropped. Key columns are kept and
    the feature columns keep their names (values are differences), so the scorer and the
    explanations work unchanged; the inflow residual columns become the change in excess inflow.
    """
    keys = ["source", "scenario_id", "window_start", "window_end"]
    base = features[keys + list(columns)].copy()
    ref = base[["source", "scenario_id", "window_start", *columns]].copy()
    ref["window_start"] = ref["window_start"] + lag
    ref["scenario_id"] = ref["scenario_id"].fillna("")
    base["_sc"] = base["scenario_id"].fillna("")
    merged = base.merge(
        ref.rename(columns={"scenario_id": "_sc"}),
        on=["source", "_sc", "window_start"],
        how="inner",
        suffixes=("", "__ref"),
    )
    cols = list(columns)
    diff = pd.DataFrame(
        merged[cols].to_numpy() - merged[[f"{c}__ref" for c in cols]].to_numpy(),
        columns=cols,
        index=merged.index,
    )
    out = pd.concat([merged[keys], diff], axis=1)
    return out.sort_values(["source", "scenario_id", "window_start"]).reset_index(drop=True)


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
    # An all-null scenario_id would be inferred as VOID, which Parquet cannot store.
    sdf = spark.createDataFrame(out).withColumn("scenario_id", F.col("scenario_id").cast("string"))
    (
        sdf.repartition("date")
        .write.mode("overwrite" if overwrite else "append")
        .partitionBy("model", "origin", "date")
        .parquet(str(paths.lake_scores_dir()))
    )
    return len(out)


def _plain(value: Any) -> Any:
    """Arrays and nested structures as read back from Parquet/DuckDB -> plain lists/dicts."""
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def alerts_to_spark(spark: SparkSession, alerts: pd.DataFrame):  # -> DataFrame
    """Alert rows (``ALERT_LAKE_COLUMNS``) as a Spark DataFrame with the lake schema.

    Accepts rows built in Python and rows read back from the lake, where list columns
    (``top_sensors``, ``gnn_top5``) arrive as numpy arrays.
    """
    rows = []
    for rec in alerts[list(ALERT_LAKE_COLUMNS)].to_dict("records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, list | tuple | dict | np.ndarray):
                clean[k] = _plain(v)
            elif v is None or v is pd.NaT or (np.isscalar(v) and pd.isna(v)):
                clean[k] = None
            elif isinstance(v, pd.Timestamp):
                clean[k] = v.to_pydatetime()
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
