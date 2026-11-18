"""Read-only DuckDB access to the Parquet lake (PRD §5: Spark writes, DuckDB reads).

``connect()`` returns an in-memory DuckDB connection with one view per lake layer that exists:

- ``raw_readings``: ``lake/raw`` with its hive partition columns (``source``, ``year``, ``date``,
  ``sensor_type``) typed explicitly;
- ``clean_readings``: ``lake/clean`` (same partitions, plus ``filled`` and ``stuck`` flags);
- ``baselines``: ``lake/baselines/hour_of_week`` (one row per series and hour of week);
- ``features`` / ``features_stream``: the batch and streaming feature tables (partitions
  ``source``, ``window``, ``date``).
"""

from __future__ import annotations

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


def _has_parquet(directory: Path) -> bool:
    return directory.exists() and next(directory.rglob("*.parquet"), None) is not None


FEATURE_HIVE_TYPES = {"source": "VARCHAR", "window": "VARCHAR", "date": "VARCHAR"}


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
    }
    if lake_root is None:
        return default
    return {name: lake_root / d.relative_to(paths.lake_dir()) for name, d in default.items()}


def connect(
    lake_raw_dir: Path | None = None, lake_root: Path | None = None
) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with views over the lake (timestamps shown in UTC).

    Args:
        lake_raw_dir: override the raw layer only (tests).
        lake_root: read every layer under this directory instead of the configured lake.

    """
    con = duckdb.connect(database=":memory:")
    con.execute("SET TimeZone = 'UTC'")
    dirs = layer_dirs(lake_root)
    if lake_raw_dir is not None:
        dirs["raw_readings"] = lake_raw_dir
    hive = {
        "raw_readings": RAW_HIVE_TYPES,
        "clean_readings": RAW_HIVE_TYPES,
        "features": FEATURE_HIVE_TYPES,
        "features_stream": FEATURE_HIVE_TYPES,
    }
    for view, layer_dir in dirs.items():
        if not _has_parquet(layer_dir):
            continue
        if view in hive:
            sql = hive_parquet_sql(layer_dir, hive[view])
        else:
            glob = (layer_dir / "**" / "*.parquet").as_posix().replace("'", "''")
            sql = f"SELECT * FROM read_parquet('{glob}')"
        con.execute(f"CREATE VIEW {view} AS {sql}")
    return con


if __name__ == "__main__":
    raise SystemExit(main())
