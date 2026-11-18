"""Batch feature pipeline (``make features``): cleaned layer, baselines, feature tables.

Steps (``--steps``, default all, in this order):

1. ``clean``: raw lake of year Y (plus the history the rules need from Y-1) →
   ``clean_readings`` → ``lake/clean`` (partitions of year Y replaced);
2. ``baselines``: ``lake/clean`` of the baseline year, normal timestamps only (D-015) →
   ``lake/baselines/hour_of_week``;
3. ``features``: raw lake from Y minus the cleaning history to the first hour of Y+1 →
   ``window_features`` (the same function the streaming job calls) → windows that start in Y and
   are fully covered by data → ``lake/features`` (partitions of year Y replaced).

Every step writes ``reports/metrics/*.json``. Spark writes the lake; reruns are idempotent.

Usage:
    python -m leakwatch.features.build                          # all steps, 2018 and 2019
    python -m leakwatch.features.build --steps features --years 2019
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.common.schemas import CLEAN_LAKE_COLUMNS, UNITS
from leakwatch.data.ground_truth import load_ground_truth
from leakwatch.features import normal
from leakwatch.features.baselines import compute_baselines, read_baselines
from leakwatch.features.clean import clean_readings
from leakwatch.features.spec import FeatureSpec, load_feature_spec
from leakwatch.features.windows import (
    complete_window_range,
    select_windows,
    window_features,
    with_partitions,
)
from leakwatch.stream.spark import get_spark

log = get_logger("leakwatch.features.build")

STEPS = ("clean", "baselines", "features")
CLEAN_PARTITIONS = ("source", "year", "date", "sensor_type")
FEATURE_PARTITIONS = ("source", "window", "date")


def year_start_s(year: int) -> int:
    """Epoch seconds of ``year``-01-01 00:00 UTC."""
    return int(datetime(year, 1, 1, tzinfo=UTC).timestamp())


def _utc(epoch_s: int) -> datetime:
    return datetime.fromtimestamp(epoch_s, tz=UTC)


def _iso(epoch_s: int | None) -> str | None:
    return None if epoch_s is None else _utc(epoch_s).isoformat()


def read_layer_range(
    spark: SparkSession, layer_dir: Path, source: str, start_s: int, end_s: int
) -> DataFrame:
    """Rows of a ``source=/year=/date=/sensor_type=`` layer with ``start_s <= ts < end_s``."""
    first, last = _utc(start_s), _utc(end_s - 1)
    days: tuple[date, date] = (first.date(), last.date())
    df = spark.read.parquet(str(layer_dir))
    return df.where(
        (F.col("source") == source)
        & F.col("year").between(first.year, last.year)
        & F.col("date").between(F.lit(days[0]), F.lit(days[1]))
        & (F.col("ts") >= F.timestamp_seconds(F.lit(start_s)))
        & (F.col("ts") < F.timestamp_seconds(F.lit(end_s)))
    )


# --------------------------------------------------------------------------------------------
# Step 1: cleaned layer
# --------------------------------------------------------------------------------------------


def build_clean(spark: SparkSession, spec: FeatureSpec, year: int, source: str) -> dict[str, Any]:
    """Write ``lake/clean`` for one year and return its metrics."""
    t0 = time.perf_counter()
    y0, y1 = year_start_s(year), year_start_s(year + 1)
    raw = read_layer_range(spark, paths.lake_raw_dir(), source, y0 - spec.history_s, y1)
    units = F.create_map(*[F.lit(x) for kv in UNITS.items() for x in kv])
    clean = (
        clean_readings(raw, spec)
        .where(
            (F.col("ts") >= F.timestamp_seconds(F.lit(y0)))
            & (F.col("ts") < F.timestamp_seconds(F.lit(y1)))
        )
        .withColumn("unit", units[F.col("sensor_type")])
        .withColumn("year", F.year("ts"))
        .withColumn("date", F.date_format("ts", "yyyy-MM-dd"))
        .select(*CLEAN_LAKE_COLUMNS)
    )
    (
        clean.repartition("date", "sensor_type")
        .sortWithinPartitions("sensor_id", "ts")
        .write.mode("overwrite")
        .partitionBy(*CLEAN_PARTITIONS)
        .parquet(str(paths.lake_clean_dir()))
    )
    written = spark.read.parquet(str(paths.lake_clean_dir())).where(
        (F.col("source") == source) & (F.col("year") == year)
    )
    by_type = {
        r["sensor_type"]: {
            "sensors": r["sensors"],
            "rows": r["rows"],
            "filled": r["filled"],
            "stuck": r["stuck"],
            "out_of_range": r["oor"],
        }
        for r in written.groupBy("sensor_type")
        .agg(
            F.countDistinct("sensor_id").alias("sensors"),
            F.count(F.lit(1)).alias("rows"),
            F.sum(F.col("filled").cast("long")).alias("filled"),
            F.sum(F.col("stuck").cast("long")).alias("stuck"),
            F.sum(F.col("out_of_range").cast("long")).alias("oor"),
        )
        .collect()
    }
    raw_rows = (
        spark.read.parquet(str(paths.lake_raw_dir()))
        .where((F.col("source") == source) & (F.col("year") == year))
        .count()
    )
    total = {
        k: sum(v[k] for v in by_type.values()) for k in ("rows", "filled", "stuck", "out_of_range")
    }
    stuck_sensors = (
        written.where("stuck")
        .groupBy("sensor_type", "sensor_id")
        .count()
        .orderBy(F.desc("count"))
        .collect()
    )
    return {
        "source": source,
        "year": year,
        "raw_rows": raw_rows,
        "clean_rows": total["rows"],
        "filled_rows": total["filled"],
        "stuck_rows": total["stuck"],
        "out_of_range_rows": total["out_of_range"],
        "by_sensor_type": by_type,
        "stuck_by_sensor": {
            f"{r['sensor_type']}__{r['sensor_id']}": r["count"] for r in stuck_sensors
        },
        "rules": {
            "max_ffill_gap_steps": spec.max_ffill_gap_steps,
            "stuck_min_steps": spec.stuck_min_steps,
            "stuck_exempt_types": list(spec.stuck_exempt_types),
        },
        "wall_s": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------------------------
# Step 2: baselines
# --------------------------------------------------------------------------------------------


def build_baselines(
    spark: SparkSession, spec: FeatureSpec, cfg: Mapping[str, Any], source: str
) -> dict[str, Any]:
    """Write ``lake/baselines/hour_of_week`` from normal timestamps of the baseline year."""
    t0 = time.perf_counter()
    normal_cfg = cfg["normal"]
    year = int(normal_cfg["baseline_year"])
    gt = load_ground_truth()
    intervals = normal.intervals_from_config(gt, normal_cfg, rule="main")
    strict = normal.intervals_from_config(gt, normal_cfg, rule="strict")
    y0, y1 = year_start_s(year), year_start_s(year + 1)
    clean = read_layer_range(spark, paths.lake_clean_dir(), source, y0, y1)
    baselines = compute_baselines(clean, spec, normal.normal_condition("ts", intervals))
    target = paths.lake_baselines_dir()
    baselines.coalesce(1).write.mode("overwrite").parquet(str(target))

    table = read_baselines(spark, target).toPandas()
    index = pd.date_range(_utc(y0), _utc(y1 - spec.step_s), freq=f"{spec.step_s}s", name="ts")
    per_series = table.groupby("series_id")["hour_of_week"].nunique()
    expected = set(spec.series)
    return {
        "source": source,
        "baseline_year": year,
        "rule": "D-015 (docs/DECISIONS.md)",
        "background_max_peak_m3h": float(normal_cfg["background_max_peak_m3h"]),
        "onset_exclusion_days": float(normal_cfg["onset_exclusion_days"]),
        "excluded_intervals": [
            {
                "leak_id": r.leak_id,
                "type": r.type,
                "peak_outflow_m3h": r.peak_outflow_m3h,
                "reason": r.reason,
                "start": r.start_ts.isoformat(),
                "end": r.end_ts.isoformat(),
            }
            for r in intervals.itertuples(index=False)
            if r.start_ts < _utc(y1) and r.end_ts >= _utc(y0)
        ],
        "normal": normal.normal_summary(index, intervals, spec.step_s),
        "normal_strict_ablation": {
            k: v
            for k, v in normal.normal_summary(index, strict, spec.step_s).items()
            if k != "normal_stretches"
        },
        "table": {
            "rows": len(table),
            "series": int(table["series_id"].nunique()),
            "series_expected": len(expected),
            "series_missing": sorted(expected - set(table["series_id"])),
            "series_with_all_168_hours": int((per_series == 168).sum()),
            "readings_per_bin": {
                "min": int(table["n_readings"].min()),
                "median": float(table["n_readings"].median()),
                "max": int(table["n_readings"].max()),
            },
        },
        "wall_s": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------------------------
# Step 3: feature tables
# --------------------------------------------------------------------------------------------


def build_features(
    spark: SparkSession, spec: FeatureSpec, year: int, source: str
) -> dict[str, Any]:
    """Write ``lake/features`` for the windows that start in ``year``; return metrics."""
    t0 = time.perf_counter()
    y0, y1 = year_start_s(year), year_start_s(year + 1)
    raw = read_layer_range(
        spark,
        paths.lake_raw_dir(),
        source,
        y0 - spec.history_s,
        y1 + spec.window_s + spec.history_s,
    )
    raw.persist()
    span = raw.agg(
        F.min(F.col("ts").cast("long")).alias("lo"), F.max(F.col("ts").cast("long")).alias("hi")
    ).first()
    if span is None or span["lo"] is None:
        raise RuntimeError(f"no raw readings for {source} {year}: run `make ingest-year`")
    first, last = complete_window_range(int(span["lo"]), int(span["hi"]), spec)
    lo, hi = max(first, y0), min(last, y1 - spec.slide_s)
    baselines = read_baselines(spark, paths.lake_baselines_dir())
    features = with_partitions(select_windows(window_features(raw, baselines, spec), lo, hi), spec)
    (
        features.repartition("date")
        .sortWithinPartitions("window_start")
        .write.mode("overwrite")
        .partitionBy(*FEATURE_PARTITIONS)
        .parquet(str(paths.lake_features_dir()))
    )
    raw.unpersist()

    written = spark.read.parquet(str(paths.lake_features_dir())).where(
        (F.col("source") == source)
        & (F.col("window") == spec.window_label)
        & F.col("window_start").cast("long").between(lo, hi)
    )
    cols = spec.feature_columns()
    full = spec.steps_per_window * len(spec.series)
    row = written.agg(
        F.count(F.lit(1)).alias("windows"),
        F.min(F.col("window_start").cast("long")).alias("first"),
        F.max(F.col("window_start").cast("long")).alias("last"),
        F.sum(
            ((F.col("n_readings") == full) & (F.col("n_series_missing") == 0)).cast("long")
        ).alias("complete"),
        F.sum("n_filled").alias("filled"),
        F.sum("n_stuck").alias("stuck"),
        F.sum("n_out_of_range").alias("oor"),
        *[F.sum(F.col(c).isNull().cast("long")).alias(c) for c in cols],
    ).first()
    nulls = {c: int(row[c]) for c in cols if row[c]}
    expected = (hi - lo) // spec.slide_s + 1
    return {
        "source": source,
        "year": year,
        "window": spec.window_label,
        "slide_minutes": spec.slide_s // 60,
        "windows": int(row["windows"]),
        "windows_expected": expected,
        "first_window_start": _iso(row["first"]),
        "last_window_start": _iso(row["last"]),
        "complete_windows": int(row["complete"]),
        "readings_per_complete_window": full,
        "series": len(spec.series),
        "series_ids": list(spec.series),
        "stats": list(spec.stats),
        "feature_columns": len(cols),
        "null_feature_cells": sum(nulls.values()),
        "null_feature_cells_by_column": nulls,
        "filled_points": int(row["filled"] or 0),
        "stuck_points": int(row["stuck"] or 0),
        "out_of_range_points": int(row["oor"] or 0),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def run(
    cfg: Mapping[str, Any],
    steps: Sequence[str],
    years: Sequence[int],
    source: str = "battledim",
    spark: SparkSession | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the selected steps; return ``{metrics name: payload}`` (also written to disk)."""
    spec = load_feature_spec(cfg)
    spark = spark or get_spark(cfg["spark"], "leakwatch-features")
    # Overwrite only the partitions a run writes (one year), not the whole layer.
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    results: dict[str, dict[str, Any]] = {}

    def save(name: str, payload: dict[str, Any]) -> None:
        results[name] = payload
        log.info("wrote %s", write_metrics(name, payload))

    if "clean" in steps:
        for year in years:
            log.info("clean %s %d", source, year)
            save(f"clean_{source}_{year}", build_clean(spark, spec, year, source))
    if "baselines" in steps:
        log.info("baselines from %s %s", source, cfg["normal"]["baseline_year"])
        save("baselines_hour_of_week", build_baselines(spark, spec, cfg, source))
    if "features" in steps:
        for year in years:
            log.info("features %s %d", source, year)
            save(f"features_{source}_{year}", build_features(spark, spec, year, source))
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cleaned layer, baselines and feature tables.")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--source", default="battledim")
    args = parser.parse_args(argv)
    cfg = load_config()
    years = args.years or [int(y) for y in cfg["battledim"]["years"]]
    steps = [s for s in STEPS if s in args.steps]  # always in pipeline order
    run(cfg, steps, years, source=args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
