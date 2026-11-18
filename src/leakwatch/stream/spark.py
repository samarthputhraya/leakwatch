"""SparkSession factory: one place for the laptop-safe Spark settings (configs/*.yaml → spark)."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

from pyspark.sql import SparkSession


def get_spark(
    spark_cfg: Mapping[str, Any],
    app_name: str,
    with_kafka: bool = False,
    master: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> SparkSession:
    """Create (or reuse) a local SparkSession.

    Args:
        spark_cfg: the ``spark`` section of the config.
        app_name: shown in the Spark UI.
        with_kafka: add the ``spark-sql-kafka`` connector (downloaded once into ``~/.ivy2``).
        master: override ``spark_cfg["master"]`` (tests use ``local[2]``).
    """
    # Python workers must use this venv's interpreter, not whatever `python3` is on PATH.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    package = str(spark_cfg["kafka_package"])
    active = SparkSession.getActiveSession()
    if (
        with_kafka
        and active is not None
        and package not in active.conf.get("spark.jars.packages", "")
    ):
        # Packages are fixed when the JVM starts: restart without leaking the old session.
        active.stop()
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master or str(spark_cfg["master"]))
        .config("spark.driver.memory", str(spark_cfg["driver_memory"]))
        .config("spark.sql.shuffle.partitions", str(spark_cfg["shuffle_partitions"]))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        .config("spark.sql.parquet.compression.codec", "zstd")
        # UTC-adjusted timestamps instead of legacy INT96, so DuckDB/pyarrow read them right.
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .config("spark.ui.enabled", str(bool(spark_cfg["ui_enabled"])).lower())
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.streaming.numRecentProgressUpdates", "100000")
    )
    if with_kafka:
        builder = builder.config("spark.jars.packages", package)
    for key, value in (extra or {}).items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(str(spark_cfg["log_level"]))
    return spark
