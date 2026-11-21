"""Read-only DuckDB access to the Parquet lake (PRD §5: Spark writes, DuckDB reads).

``connect()`` returns an in-memory DuckDB connection with one view per lake layer that exists:

- ``raw_readings``: ``lake/raw`` with its hive partition columns (``source``, ``year``, ``date``,
  ``sensor_type``) typed explicitly;
- ``clean_readings``: ``lake/clean`` (same partitions, plus ``filled`` and ``stuck`` flags);
- ``baselines``: ``lake/baselines/hour_of_week`` (one row per series and hour of week);
- ``features`` / ``features_stream``: the batch and streaming feature tables (partitions
  ``source``, ``window``, ``date``);
- ``scores``: anomaly scores per window (partitions ``model``, ``origin``, ``date``);
- ``alert_events``: every alert version (append-only, partition ``origin``);
- ``work_orders``: work orders written by the copilot;
- ``alerts``: the current alerts = latest version per ``alert_id``, with the latest work order
  (``work_order_json``) joined in (PRD §6.3).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

from leakwatch.common import paths

RAW_HIVE_TYPES = {
    "source": "VARCHAR",
    "year": "INTEGER",
    "date": "VARCHAR",
    "sensor_type": "VARCHAR",
}


def has_parquet(directory: Path) -> bool:
    """True if ``directory`` exists and holds at least one Parquet file (stops at the first)."""
    return directory.exists() and next(directory.rglob("*.parquet"), None) is not None


FEATURE_HIVE_TYPES = {"source": "VARCHAR", "window": "VARCHAR", "date": "VARCHAR"}
SCORE_HIVE_TYPES = {"model": "VARCHAR", "origin": "VARCHAR", "date": "VARCHAR"}
ALERT_HIVE_TYPES = {"origin": "VARCHAR"}

#: Current alerts: latest version of each alert plus its latest work order (if any).
ALERTS_VIEW_SQL = """
SELECT a.* EXCLUDE (rn), w.work_order_json, w.generator AS work_order_generator
FROM (
    SELECT *, row_number() OVER (PARTITION BY alert_id ORDER BY version DESC, updated_at DESC) AS rn
    FROM alert_events
) a
LEFT JOIN (
    SELECT alert_id, work_order_json, generator,
           row_number() OVER (PARTITION BY alert_id ORDER BY generated_at DESC) AS wn
    FROM work_orders
) w ON w.alert_id = a.alert_id AND w.wn = 1
WHERE a.rn = 1
"""


def hive_parquet_sql(layer_dir: Path, hive_types: dict[str, str]) -> str:
    """``SELECT`` over a partitioned layer's Parquet files with typed partition columns."""
    types = ", ".join(f"'{k}': '{v}'" for k, v in hive_types.items())
    glob = (layer_dir / "**" / "*.parquet").as_posix().replace("'", "''")
    return (
        f"SELECT * FROM read_parquet('{glob}', hive_partitioning = true, "
        f"hive_types = {{{types}}}, union_by_name = true)"
    )


def raw_readings_sql(raw_dir: Path) -> str:
    """``SELECT`` over the raw layer's Parquet files with typed partition columns."""
    return hive_parquet_sql(raw_dir, RAW_HIVE_TYPES)


def query_df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    """Run ``sql`` and return a DataFrame with every timestamp column as ``datetime64[ns, UTC]``."""
    df = con.execute(sql).df()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.DatetimeTZDtype):
            df[col] = df[col].dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
    return df


def raw_layer_stats(lake_raw_dir: Path | None = None) -> dict:
    """Size of the raw layer: files and bytes on disk, rows per source/year (via DuckDB)."""
    raw_dir = lake_raw_dir or paths.lake_raw_dir()
    files = list(raw_dir.rglob("*.parquet")) if raw_dir.exists() else []
    total_bytes = sum(f.stat().st_size for f in files)
    rows: dict[str, int] = {}
    total_rows = 0
    if files:
        con = connect(raw_dir)
        for source, year, n in con.execute(
            "SELECT source, year, count(*) FROM raw_readings GROUP BY ALL ORDER BY ALL"
        ).fetchall():
            rows[f"{source}/{year}"] = int(n)
            total_rows += int(n)
    return {
        "parquet_files": len(files),
        "bytes": total_bytes,
        "megabytes": round(total_bytes / 1e6, 1),
        "rows_by_source_year": rows,
        "rows": total_rows,
        "bytes_per_reading": round(total_bytes / total_rows, 2) if total_rows else None,
    }


def layer_size(layer_dir: Path) -> dict:
    """Parquet files, bytes and rows of one lake layer directory (0 if it does not exist)."""
    files = list(layer_dir.rglob("*.parquet")) if layer_dir.exists() else []
    total_bytes = sum(f.stat().st_size for f in files)
    rows = 0
    if files:
        glob = (layer_dir / "**" / "*.parquet").as_posix().replace("'", "''")
        row = duckdb.sql(f"SELECT count(*) FROM read_parquet('{glob}')").fetchone()
        rows = int(row[0]) if row else 0
    return {
        "parquet_files": len(files),
        "bytes": total_bytes,
        "megabytes": round(total_bytes / 1e6, 1),
        "rows": rows,
    }


def lake_layers_summary(lake_root: Path | None = None) -> dict:
    """Size of every lake layer (raw, clean, baselines, features, features_stream)."""
    return {view: layer_size(d) for view, d in layer_dirs(lake_root).items()}


def main() -> int:
    """``make lake-stats``: write ``lake_raw_summary.json`` and ``lake_layers_summary.json``."""
    from leakwatch.common.logging import get_logger
    from leakwatch.common.metrics import write_metrics

    stats = raw_layer_stats()
    path = write_metrics("lake_raw_summary", stats)
    layers_path = write_metrics("lake_layers_summary", lake_layers_summary())
    get_logger("leakwatch.common.lake").info(
        "raw layer: %d rows in %d files (%.1f MB) -> %s",
        stats["rows"],
        stats["parquet_files"],
        stats["megabytes"],
        path,
    )
    get_logger("leakwatch.common.lake").info("per-layer sizes -> %s", layers_path)
    return 0


def layer_dirs(lake_root: Path | None = None) -> dict[str, Path]:
    """Directory of each lake view, under ``lake_root`` (default: the configured lake)."""
    default = {
        "raw_readings": paths.lake_raw_dir(),
        "clean_readings": paths.lake_clean_dir(),
        "baselines": paths.lake_baselines_dir(),
        "features": paths.lake_features_dir(),
        "features_stream": paths.lake_features_stream_dir(),
        "scores": paths.lake_scores_dir(),
        "alert_events": paths.lake_alerts_dir(),
        "work_orders": paths.lake_work_orders_dir(),
    }
    if lake_root is None:
        return default
    return {name: lake_root / d.relative_to(paths.lake_dir()) for name, d in default.items()}


def connect(
    lake_raw_dir: Path | None = None,
    lake_root: Path | None = None,
    only: Iterable[str] | None = None,
) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with views over the lake (timestamps shown in UTC).

    Args:
        lake_raw_dir: override the raw layer only (tests).
        lake_root: read every layer under this directory instead of the configured lake.
        only: create just these views (e.g. ``["features"]``). Creating a view reads the
            metadata of every Parquet file of its layer once, tens of seconds for the whole lake
            (13,688 raw files); a job that reads one table should not pay for all (D-047).
            ``alerts`` needs ``alert_events`` and ``work_orders``, which are added for it.

    """
    con = duckdb.connect(database=":memory:")
    con.execute("SET TimeZone = 'UTC'")
    dirs = layer_dirs(lake_root)
    if lake_raw_dir is not None:
        dirs["raw_readings"] = lake_raw_dir
    if only is not None:
        wanted = set(only)
        if "alerts" in wanted:
            wanted |= {"alert_events", "work_orders"}
        dirs = {k: v for k, v in dirs.items() if k in wanted}
    hive = {
        "raw_readings": RAW_HIVE_TYPES,
        "clean_readings": RAW_HIVE_TYPES,
        "features": FEATURE_HIVE_TYPES,
        "features_stream": FEATURE_HIVE_TYPES,
        "scores": SCORE_HIVE_TYPES,
        "alert_events": ALERT_HIVE_TYPES,
    }
    for view, layer_dir in dirs.items():
        if not has_parquet(layer_dir):
            continue
        if view in hive:
            sql = hive_parquet_sql(layer_dir, hive[view])
        else:
            glob = (layer_dir / "**" / "*.parquet").as_posix().replace("'", "''")
            sql = f"SELECT * FROM read_parquet('{glob}')"
        con.execute(f"CREATE VIEW {view} AS {sql}")
    views = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views()").fetchall()}
    if "alert_events" in views:
        if "work_orders" not in views:  # empty stand-in so the alerts view always has the column
            con.execute(
                "CREATE VIEW work_orders AS SELECT NULL::VARCHAR AS alert_id, "
                "NULL::VARCHAR AS work_order_json, NULL::VARCHAR AS generator, "
                "NULL::VARCHAR AS prompt_version, NULL::TIMESTAMPTZ AS generated_at WHERE false"
            )
        con.execute(f"CREATE VIEW alerts AS {ALERTS_VIEW_SQL}")
    return con


if __name__ == "__main__":
    raise SystemExit(main())
