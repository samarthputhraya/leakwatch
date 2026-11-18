"""Read-only DuckDB access to the Parquet lake (PRD §5: Spark writes, DuckDB reads).

``connect()`` returns an in-memory DuckDB connection with one view per lake layer that exists:

- ``raw_readings``: ``lake/raw`` with its hive partition columns (``source``, ``year``, ``date``,
  ``sensor_type``) typed explicitly.
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


def raw_readings_sql(raw_dir: Path) -> str:
    """``SELECT`` over the raw layer's Parquet files with typed partition columns."""
    hive_types = ", ".join(f"'{k}': '{v}'" for k, v in RAW_HIVE_TYPES.items())
    glob = (raw_dir / "**" / "*.parquet").as_posix().replace("'", "''")
    return (
        f"SELECT * FROM read_parquet('{glob}', hive_partitioning = true, "
        f"hive_types = {{{hive_types}}}, union_by_name = true)"
    )


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


def main() -> int:
    """``make lake-stats``: write ``reports/metrics/lake_raw_summary.json``."""
    from leakwatch.common.logging import get_logger
    from leakwatch.common.metrics import write_metrics

    stats = raw_layer_stats()
    path = write_metrics("lake_raw_summary", stats)
    get_logger("leakwatch.common.lake").info(
        "raw layer: %d rows in %d files (%.1f MB) -> %s",
        stats["rows"],
        stats["parquet_files"],
        stats["megabytes"],
        path,
    )
    return 0


def connect(lake_raw_dir: Path | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with views over the lake (timestamps shown in UTC)."""
    con = duckdb.connect(database=":memory:")
    con.execute("SET TimeZone = 'UTC'")
    raw_dir = lake_raw_dir or paths.lake_raw_dir()
    if _has_parquet(raw_dir):
        con.execute(f"CREATE VIEW raw_readings AS {raw_readings_sql(raw_dir)}")
    return con


if __name__ == "__main__":
    raise SystemExit(main())
