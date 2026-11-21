"""Windowed features (PRD §7.2): the one function shared by Spark batch and Spark streaming.

:func:`window_features` turns readings into one row per (source, scenario, window):

1. :func:`~leakwatch.features.clean.clean_readings` (forward fill, stuck flags);
2. :func:`feature_series`: one series per pressure/flow/level sensor, plus the summed AMR
   consumption of each area; out-of-range values are left out (``exclude_out_of_range``);
3. :func:`with_residual`: value minus the hour-of-week baseline of the series;
4. :func:`window_aggregates`: per series and window (``window_s`` long, every ``slide_s``,
   aligned to the epoch): mean, population std, min, max, least-squares slope per hour, mean
   residual, and counts;
5. :func:`pivot_windows`: one row per window with ``<series>__<stat>`` columns in the order of
   ``FeatureSpec.feature_columns()`` plus data-quality counts.

The function returns every window the input touches, including partial ones at its edges; callers
keep the complete windows with :func:`select_windows` (the batch job by date range, the streaming
job by watermark).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from leakwatch.common.schemas import COLUMN_SEP
from leakwatch.features.clean import clean_readings
from leakwatch.features.spec import AMR_SUM_PREFIX, QUALITY_COLUMNS, FeatureSpec, feature_column

GROUP_KEY: tuple[str, ...] = ("source", "scenario_id")
SERIES_COLUMNS: tuple[str, ...] = (
    *GROUP_KEY,
    "series_id",
    "ts",
    "value",
    "filled",
    "stuck",
    "out_of_range",
)
WINDOW_KEY: tuple[str, ...] = (*GROUP_KEY, "window_start", "window_end")


def hour_of_week(ts: Column) -> Column:
    """0 = Monday 00:00–00:59 … 167 = Sunday 23:00–23:59, in the session time zone (UTC)."""
    return ((F.dayofweek(ts) + 5) % 7) * 24 + F.hour(ts)


def _lookup(mapping: dict[str, str], key: Column) -> Column:
    pairs = [F.lit(x) for kv in sorted(mapping.items()) for x in kv]
    return F.create_map(*pairs)[key]


def feature_series(clean: DataFrame, spec: FeatureSpec) -> DataFrame:
    """Clean readings → the series features are computed on (columns ``SERIES_COLUMNS``).

    AMR readings are summed per area and timestamp; the sum covers the AMRs that reported (and
    were in range) at that step, and its flags are set if any contributing reading had them.
    """
    value = F.col("value")
    if spec.exclude_out_of_range:
        value = F.when(F.col("out_of_range"), F.lit(None).cast("double")).otherwise(value)
    rows = clean.withColumn("value", value)

    series_id = F.concat_ws(COLUMN_SEP, F.col("sensor_type"), F.col("sensor_id"))
    sensors = rows.withColumn("series_id", series_id).where(
        F.col("series_id").isin(*spec.sensor_series)
    )
    parts = [sensors.select(*SERIES_COLUMNS)]
    if spec.amr_areas:
        area = _lookup(dict(spec.amr_areas), F.col("sensor_id"))
        amr = (
            rows.where(F.col("sensor_type") == "amr")
            .withColumn("_area", area)
            .where(F.col("_area").isNotNull())
            .groupBy(*GROUP_KEY, "_area", "ts")
            .agg(
                F.sum("value").alias("value"),
                F.max("filled").alias("filled"),
                F.max("stuck").alias("stuck"),
                F.max("out_of_range").alias("out_of_range"),
            )
            .withColumn("series_id", F.concat_ws(COLUMN_SEP, F.lit(AMR_SUM_PREFIX), "_area"))
        )
        parts.append(amr.select(*SERIES_COLUMNS))
    out = parts[0]
    for part in parts[1:]:
        out = out.unionByName(part)
    return out


def with_residual(series: DataFrame, baselines: DataFrame) -> DataFrame:
    """Add ``residual`` = value − hour-of-week baseline of the series (null without a baseline)."""
    base = F.broadcast(baselines.select("series_id", "hour_of_week", "baseline"))
    return (
        series.withColumn("hour_of_week", hour_of_week(F.col("ts")))
        .join(base, on=["series_id", "hour_of_week"], how="left")
        .withColumn("residual", F.col("value") - F.col("baseline"))
        .drop("hour_of_week", "baseline")
    )


def window_aggregates(series: DataFrame, spec: FeatureSpec) -> DataFrame:
    """Per (source, scenario, window, series): the statistics and counts, in long format.

    Time for the slope is measured in hours from the window's own start (0 … 11/12 h), not from
    the epoch: with epoch hours (~4e5) the covariance loses ~1e-10 of each time step, which large
    series such as the AMR area sum (~3e4 L/h) turn into slope noise of ~1e-6 that depends on the
    order Spark aggregates rows in (docs/DECISIONS.md D-018, first full-year parity run).
    """
    win = F.window("ts", f"{spec.window_s} seconds", f"{spec.slide_s} seconds")
    # A sliding window() in a projection expands each row into every window that contains it.
    windowed = series.withColumn("_w", win)
    offset_s = F.col("ts").cast("long") - F.col("_w.start").cast("long")
    hours = F.when(F.col("value").isNotNull(), offset_s / 3600.0)
    agg = (
        windowed.withColumn("_x", hours)
        .groupBy(*GROUP_KEY, "_w", "series_id")
        .agg(
            F.avg("value").alias("mean"),
            F.stddev_pop("value").alias("std"),
            F.min("value").alias("min"),
            F.max("value").alias("max"),
            F.covar_pop("_x", "value").alias("_cov"),
            F.var_pop("_x").alias("_var"),
            F.avg("residual").alias("residual"),
            F.count("value").alias("n"),
            F.sum(F.col("filled").cast("long")).alias("n_filled"),
            F.sum(F.col("stuck").cast("long")).alias("n_stuck"),
            F.sum(F.col("out_of_range").cast("long")).alias("n_out_of_range"),
        )
    )
    # Least-squares slope of value against time, in value units per hour.
    slope = F.when(F.col("_var") > 0, F.col("_cov") / F.col("_var"))
    return agg.select(
        *GROUP_KEY,
        F.col("_w.start").alias("window_start"),
        F.col("_w.end").alias("window_end"),
        "series_id",
        "mean",
        "std",
        "min",
        "max",
        slope.alias("slope"),
        "residual",
        "n",
        "n_filled",
        "n_stuck",
        "n_out_of_range",
    )


def pivot_windows(agg: DataFrame, spec: FeatureSpec) -> DataFrame:
    """Long aggregates → one row per window: ``WINDOW_KEY``, quality counts, feature columns."""
    quality = [
        F.sum("n").alias("n_readings"),
        F.sum("n_filled").alias("n_filled"),
        F.sum("n_stuck").alias("n_stuck"),
        F.sum("n_out_of_range").alias("n_out_of_range"),
        (F.lit(len(spec.series)) - F.sum(F.when(F.col("n") > 0, 1).otherwise(0))).alias(
            "n_series_missing"
        ),
    ]
    features = [
        F.max(F.when(F.col("series_id") == s, F.col(stat))).alias(feature_column(s, stat))
        for s in spec.series
        for stat in spec.stats
    ]
    return agg.groupBy(*WINDOW_KEY).agg(*quality, *features)


_NO_KEY = "<no scenario>"


def pivot_windows_pandas(agg: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """:func:`pivot_windows` on collected long aggregates (pandas), value for value (D-047).

    The Spark pivot is a 233-expression aggregation whose planning and code generation cost ~3 s
    per streaming micro-batch for a few hundred long rows. The pivot only *selects* values (one
    row per window and series; ``max`` of that single value) and sums integer counts, so doing it
    after collecting the long rows gives the same numbers. Column order and types follow
    :func:`window_features`: ``WINDOW_KEY``, ``QUALITY_COLUMNS`` (int64), feature columns
    (float64, NaN where Spark has NULL).
    """
    feature_cols = list(spec.feature_columns())
    key = list(WINDOW_KEY)
    if agg.empty:
        empty = pd.DataFrame(columns=[*key, *QUALITY_COLUMNS, *feature_cols])
        return empty.astype({c: "float64" for c in feature_cols})
    df = agg.copy()
    # NULL keys (BattLeDIM rows have no scenario) do not align reliably in pandas MultiIndexes
    df["scenario_id"] = df["scenario_id"].astype(object).where(df["scenario_id"].notna(), _NO_KEY)
    df["_present"] = (df["n"] > 0).astype("int64")
    by = df.groupby(key, dropna=False, sort=True)
    quality = by.agg(
        n_readings=("n", "sum"),
        n_filled=("n_filled", "sum"),
        n_stuck=("n_stuck", "sum"),
        n_out_of_range=("n_out_of_range", "sum"),
        _present=("_present", "sum"),
    )
    quality["n_series_missing"] = len(spec.series) - quality.pop("_present")
    wide = df.pivot_table(
        index=key, columns="series_id", values=list(spec.stats), aggfunc="max", dropna=False
    )
    cols = {
        feature_column(series, stat): wide[(stat, series)]
        if (stat, series) in wide.columns
        else pd.Series(np.nan, index=wide.index)
        for series in spec.series
        for stat in spec.stats
    }
    features = pd.DataFrame(cols, index=wide.index).reindex(quality.index)
    out = pd.concat([quality.astype("int64"), features.astype("float64")], axis=1).reset_index()
    out["scenario_id"] = out["scenario_id"].astype(object)
    out.loc[out["scenario_id"] == _NO_KEY, "scenario_id"] = None
    return out[[*key, *QUALITY_COLUMNS, *feature_cols]]


def window_aggregates_of(readings: DataFrame, baselines: DataFrame, spec: FeatureSpec) -> DataFrame:
    """The long aggregates :func:`window_features` pivots (cleaning, series, residual, stats)."""
    series = with_residual(feature_series(clean_readings(readings, spec), spec), baselines)
    return window_aggregates(series, spec)


def window_features(readings: DataFrame, baselines: DataFrame, spec: FeatureSpec) -> DataFrame:
    """Readings → one feature row per window (see module docstring). Shared by batch and stream.

    Args:
        readings: static DataFrame with ``clean.READING_COLUMNS`` (raw lake rows or parsed Kafka
            messages).
        baselines: hour-of-week baselines (``series_id, hour_of_week, baseline``).
        spec: the feature specification.

    Returns:
        Columns ``WINDOW_KEY``, ``QUALITY_COLUMNS``, then ``spec.feature_columns()``.

    """
    wide = pivot_windows(window_aggregates_of(readings, baselines, spec), spec)
    return wide.select(*WINDOW_KEY, *QUALITY_COLUMNS, *spec.feature_columns())


def select_windows(features: DataFrame, first_start_s: int, last_start_s: int) -> DataFrame:
    """Keep windows whose start lies in ``[first_start_s, last_start_s]`` (epoch seconds)."""
    start = F.col("window_start").cast("long")
    return features.where(start.between(first_start_s, last_start_s))


def with_partitions(features: DataFrame, spec: FeatureSpec) -> DataFrame:
    """Add the lake partition columns ``window`` (e.g. ``1h``) and ``date`` (UTC, of the start)."""
    return features.withColumn("window", F.lit(spec.window_label)).withColumn(
        "date", F.date_format("window_start", "yyyy-MM-dd")
    )


def ceil_to(epoch_s: int, every_s: int) -> int:
    """Smallest multiple of ``every_s`` at or after ``epoch_s``."""
    return -(-epoch_s // every_s) * every_s


def floor_to(epoch_s: int, every_s: int) -> int:
    """Largest multiple of ``every_s`` at or before ``epoch_s``."""
    return (epoch_s // every_s) * every_s


def complete_window_range(
    first_ts_s: int, last_ts_s: int, spec: FeatureSpec, lateness_s: int = 0
) -> tuple[int, int]:
    """First and last window start fully covered by data from ``first_ts_s`` to ``last_ts_s``.

    A window ``[s, s + window)`` is complete when its last step ``s + window - step`` is at most
    ``last_ts_s - lateness_s``. The streaming job passes its lateness so the reading that closes a
    short gap after the window has arrived; the batch job sees all data and passes 0. If no window
    is complete the returned last start is below the first.
    """
    first = ceil_to(first_ts_s, spec.slide_s)
    last = floor_to(last_ts_s - lateness_s + spec.step_s - spec.window_s, spec.slide_s)
    return first, last
