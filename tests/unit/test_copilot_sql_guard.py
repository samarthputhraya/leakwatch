from __future__ import annotations

import pytest

pytest.importorskip("sqlglot")

import duckdb

from leakwatch.copilot.sql_guard import SQLGuardError, guard_sql

VIEWS = {"alerts", "clean_readings", "features", "scores", "baselines"}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT area, count(*) AS n FROM alerts GROUP BY area",
        "WITH x AS (SELECT area FROM alerts) SELECT * FROM x",
        "SELECT 1 AS a UNION ALL SELECT 2",
        "FROM alerts",
        "SELECT top_sensors[1].sensor_id AS s FROM alerts",
        "SELECT unnest(top_sensors) AS s FROM alerts",
        'SELECT "window", window_start FROM features WHERE "window" = \'1h\'',
        "SELECT * FROM range(10) t(i)",
        "SELECT date_trunc('week', start_ts) AS w, count(*) FROM alerts GROUP BY ALL",
        "SELECT a.alert_id FROM alerts a JOIN (SELECT alert_id FROM alerts) b USING (alert_id)",
        "SELECT avg(value) FROM clean_readings WHERE date = '2019-01-15' AND sensor_id = 'n54'",
        "  select count(*) from alerts;  ",
    ],
)
def test_accepts_read_only_selects(query):
    guarded = guard_sql(query, VIEWS)
    assert "LIMIT" in guarded.sql.upper()


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("DROP TABLE alerts", "only SELECT"),
        ("INSERT INTO alerts VALUES (1)", "only SELECT"),
        ("CREATE TABLE x AS SELECT 1", "only SELECT"),
        ("COPY alerts TO 'x.csv'", "only SELECT"),
        ("ATTACH 'x.db'", "only SELECT"),
        ("INSTALL httpfs", "only SELECT"),
        ("LOAD httpfs", "only SELECT"),
        ("PRAGMA show_tables", "only SELECT"),
        ("SET threads = 1", "only SELECT"),
        ("DESCRIBE alerts", "only SELECT"),
        ("SHOW TABLES", "only SELECT"),
        ("SELECT 1; DROP TABLE alerts", "exactly one statement"),
        ("SELECT * INTO new_t FROM alerts", "INTO"),
        ("SELECT * FROM read_parquet('/etc/x.parquet')", "table function"),
        ("SELECT * FROM read_csv_auto('/etc/passwd')", "table function"),
        ("SELECT * FROM glob('*')", "table function"),
        ("SELECT * FROM query('SELECT 1')", "table function"),
        ("SELECT * FROM '/etc/hostname'", "unknown table"),
        ("SELECT getenv('HOME')", "not allowed"),
        ("SELECT current_setting('threads')", "not allowed"),
        ("SELECT (SELECT count(*) FROM read_json('x.json'))", "table function"),
        ("SELECT * FROM main.alerts", "unqualified"),
        ("SELECT * FROM information_schema.tables", "unqualified"),
        ("SELECT * FROM duckdb_settings()", "table function"),
        ("SELECT * FROM pg_settings", "unknown table"),
        ("SELECT * FROM secrets_table", "unknown table"),
        ("", "empty"),
        ("SELECT FROM WHERE", "parse"),
    ],
)
def test_rejects_unsafe_queries(query, reason):
    with pytest.raises(SQLGuardError, match=reason):
        guard_sql(query, VIEWS)


def test_rejects_long_queries():
    with pytest.raises(SQLGuardError, match="longer"):
        guard_sql("SELECT 1" + " + 1" * 30, VIEWS, max_chars=50)


def test_limit_is_added_lowered_kept_or_wrapped():
    assert guard_sql("SELECT * FROM alerts", VIEWS, max_rows=10).sql.endswith("LIMIT 10")
    lowered = guard_sql("SELECT * FROM alerts LIMIT 5000", VIEWS, max_rows=10)
    assert lowered.sql.endswith("LIMIT 10") and lowered.limit_changed
    kept = guard_sql("SELECT * FROM alerts LIMIT 3", VIEWS, max_rows=10)
    assert kept.sql.endswith("LIMIT 3") and not kept.limit_changed
    wrapped = guard_sql("SELECT * FROM alerts LIMIT 2 + 3", VIEWS, max_rows=10)
    assert wrapped.sql.startswith("SELECT * FROM (") and wrapped.sql.endswith("LIMIT 10")
    union = guard_sql("SELECT 1 AS a UNION ALL SELECT 2", VIEWS, max_rows=10)
    assert union.sql.endswith("LIMIT 10")


def test_guarded_sql_runs_on_duckdb_and_caps_rows():
    con = duckdb.connect()
    con.execute("CREATE VIEW alerts AS SELECT range AS i, 'A' AS area FROM range(50)")
    guarded = guard_sql("SELECT i FROM alerts ORDER BY i", {"alerts"}, max_rows=7)
    rows = con.execute(guarded.sql).fetchall()
    assert len(rows) == 7
    cte = guard_sql(
        "WITH t AS (SELECT area, count(*) AS n FROM alerts GROUP BY area) SELECT * FROM t",
        {"alerts"},
    )
    assert con.execute(cte.sql).fetchall() == [("A", 50)]
