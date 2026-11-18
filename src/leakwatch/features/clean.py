"""Cleaned readings (PRD §7.2; docs/DECISIONS.md D-011, D-016).

:func:`clean_readings` is the one Spark implementation of the cleaning rules. The batch job uses it
to build ``lake/clean`` and inside the feature tables; the streaming job applies it to each
micro-batch plus the history it carries. It works on any static DataFrame of readings.

Rules, per sensor (key: source, scenario, sensor type, sensor id):

- **forward fill:** an *interior* gap of 1 to ``max_ffill_gap_steps`` missing steps is filled with
  the last value before it (``filled = True``). A gap is only known once the reading after it has
  arrived, so a gap at the very end of the data is never filled (the pandas reference
  ``data.quality.forward_fill_short_gaps`` also fills a short trailing gap; that is the only
  difference). Longer gaps stay missing.
- **stuck:** an observed reading is flagged once it and the ``stuck_min_steps - 1`` steps before it
  were all observed with the same value (causal rule, reference ``data.quality.stuck_flags``).
  A missing step breaks a run; filled readings are never flagged. Sensor types listed in
  ``stuck_exempt_types`` are never flagged.
- **out of range:** carried over from the raw layer (flagged in Phase 1, never dropped).
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from leakwatch.features.spec import FeatureSpec

#: Identifies one sensor's series.
READING_KEY: tuple[str, ...] = ("source", "scenario_id", "sensor_type", "sensor_id")
#: Columns the cleaning step reads.
READING_COLUMNS: tuple[str, ...] = (*READING_KEY, "ts", "value", "out_of_range")
#: Columns it returns.
CLEAN_COLUMNS: tuple[str, ...] = (*READING_COLUMNS, "filled", "stuck")


def clean_readings(readings: DataFrame, spec: FeatureSpec) -> DataFrame:
    """Apply the cleaning rules (module docstring) to a static DataFrame of readings.

    Args:
        readings: at least the columns in ``READING_COLUMNS``; one row per sensor and timestamp.
        spec: step length, gap and stuck thresholds, exempt sensor types.

    Returns:
        Columns ``CLEAN_COLUMNS``: the observed readings plus one row per filled step.

    """
    step = spec.step_s
    k = spec.stuck_min_steps
    base = readings.select(*READING_COLUMNS).withColumn("_t", F.col("ts").cast("long"))
    by_sensor = Window.partitionBy(*READING_KEY).orderBy("_t")
    last_k = by_sensor.rangeBetween(-(k - 1) * step, 0)

    stuck = (F.count("value").over(last_k) == k) & (
        F.min("value").over(last_k) == F.max("value").over(last_k)
    )
    if spec.stuck_exempt_types:
        stuck = stuck & ~F.col("sensor_type").isin(*spec.stuck_exempt_types)

    # Missing steps between this reading and the next one of the same sensor.
    missing = ((F.lead("_t").over(by_sensor) - F.col("_t")) / step).cast("long") - 1
    fill_n = F.when(missing.between(1, spec.max_ffill_gap_steps), missing).otherwise(F.lit(0))

    observed = base.select(
        *READING_COLUMNS,
        "_t",
        F.lit(False).alias("filled"),
        F.coalesce(stuck, F.lit(False)).alias("stuck"),
        fill_n.alias("_fill_n"),
    )
    fills = (
        observed.where(F.col("_fill_n") > 0)
        .withColumn("_i", F.explode(F.sequence(F.lit(1), F.col("_fill_n").cast("int"))))
        .withColumn("ts", F.timestamp_seconds(F.col("_t") + F.col("_i") * step))
        .withColumn("filled", F.lit(True))
        .withColumn("stuck", F.lit(False))
    )
    return observed.select(*CLEAN_COLUMNS).unionByName(fills.select(*CLEAN_COLUMNS))
