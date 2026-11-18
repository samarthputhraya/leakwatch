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


def connect(lake_raw_dir: Path | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with views over the lake (timestamps shown in UTC)."""
    con = duckdb.connect(database=":memory:")
    con.execute("SET TimeZone = 'UTC'")
    raw_dir = lake_raw_dir or paths.lake_raw_dir()
    if _has_parquet(raw_dir):
        con.execute(f"CREATE VIEW raw_readings AS {raw_readings_sql(raw_dir)}")
    return con
