"""Hour-of-week baselines (PRD §6.2, §7.2): the typical value of each series at each hour of week.

For every feature series (pressure/flow/level sensors and AMR area sums) and every hour of the
week (0 = Monday 00:00 UTC … 167 = Sunday 23:00), the baseline is the exact median of the cleaned
values in that hour over the *normal* timestamps of the baseline year (2018; rule D-015 in
``features.normal``). Residual features are value minus this baseline.

Stored in ``lake/baselines/hour_of_week/`` as one small Parquet table
(``series_id, hour_of_week, baseline, n_readings``).
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from leakwatch.features.spec import FeatureSpec
from leakwatch.features.windows import feature_series, hour_of_week

BASELINE_COLUMNS: tuple[str, ...] = ("series_id", "hour_of_week", "baseline", "n_readings")
HOURS_PER_WEEK = 168


def compute_baselines(clean: DataFrame, spec: FeatureSpec, normal: Column | None) -> DataFrame:
    """Median of each series per hour of week over the rows where ``normal`` holds.

    Args:
        clean: cleaned readings (``clean.CLEAN_COLUMNS``) of the baseline period.
        spec: feature specification (which series exist).
        normal: Spark condition on ``ts`` selecting normal timestamps (``normal_condition``);
            ``None`` uses every row.

    """
    series = feature_series(clean, spec).where(F.col("value").isNotNull())
    if normal is not None:
        series = series.where(normal)
    return (
        series.groupBy("series_id", hour_of_week(F.col("ts")).alias("hour_of_week"))
        .agg(F.median("value").alias("baseline"), F.count("value").alias("n_readings"))
        .select(*BASELINE_COLUMNS)
    )


def read_baselines(spark: SparkSession, path: Path) -> DataFrame:
    """Read the baseline table written by the batch job (``make features``)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `make features` (baselines step)")
    return spark.read.parquet(str(path)).select(*BASELINE_COLUMNS)
