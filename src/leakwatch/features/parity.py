"""Batch vs stream feature tables, window by window (``make features-parity``, PRD §11).

Joins ``lake/features`` (batch job) and ``lake/features_stream`` (streaming job) on source,
scenario and window start, and compares every quality count and feature column. Two cells agree
when both are null, or both are set and ``|a - b| <= rtol * max(1, |a|, |b|)``: the two jobs run
the same function on the same readings, so only floating-point summation order may differ.

Windows only one side has are listed. The stream cannot emit the last window(s) of its input: a
window needs data ``lateness`` past its last step (D-018).

Writes ``reports/metrics/features_parity_<source>.json``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import duckdb

from leakwatch.common import lake
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.features.spec import QUALITY_COLUMNS, FeatureSpec, load_feature_spec

log = get_logger("leakwatch.features.parity")


def _key(left: str, right: str) -> str:
    """Join condition on (source, scenario, window start); scenario may be null."""
    return (
        f"{left}.source = {right}.source "
        f"AND {left}.scenario_id IS NOT DISTINCT FROM {right}.scenario_id "
        f"AND {left}.window_start = {right}.window_start"
    )


KEY = _key("b", "s")


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def compare_tables(
    con: duckdb.DuckDBPyConnection,
    spec: FeatureSpec,
    source: str,
    rtol: float = 1e-9,
    list_limit: int = 20,
) -> dict[str, Any]:
    """Compare the ``features`` and ``features_stream`` views of ``con`` for one source."""
    where = f"source = '{source}' AND \"window\" = '{spec.window_label}'"  # WINDOW is reserved
    con.execute(f"CREATE OR REPLACE TEMP VIEW b AS SELECT * FROM features WHERE {where}")
    con.execute(f"CREATE OR REPLACE TEMP VIEW s AS SELECT * FROM features_stream WHERE {where}")

    counts = con.execute(
        f"""SELECT
              (SELECT count(*) FROM b),
              (SELECT count(*) FROM s),
              (SELECT count(*) FROM b JOIN s ON {KEY}),
              (SELECT count(*) FROM (SELECT source, scenario_id, window_start FROM s
                                     GROUP BY ALL HAVING count(*) > 1))"""
    ).fetchone()
    assert counts is not None
    batch_n, stream_n, matched, stream_dupes = (int(c) for c in counts)

    def one_sided(left: str, right: str) -> list[str]:
        rows = con.execute(
            f"""SELECT strftime({left}.window_start, '%Y-%m-%dT%H:%M:%SZ') FROM {left}
                ANTI JOIN {right} ON {_key(left, right)}
                ORDER BY 1 LIMIT {list_limit}"""
        ).fetchall()
        return [r[0] for r in rows]

    columns = [*QUALITY_COLUMNS, *spec.feature_columns()]
    exprs = []
    for c in columns:
        a, b = f"b.{_q(c)}", f"s.{_q(c)}"
        ok = (
            f"(({a} IS NULL AND {b} IS NULL) OR ({a} IS NOT NULL AND {b} IS NOT NULL AND "
            f"abs({a} - {b}) <= {rtol} * greatest(1.0, abs({a}), abs({b}))))"
        )
        exprs.append(f"sum(CASE WHEN {ok} THEN 0 ELSE 1 END)")
        exprs.append(f"max(abs({a} - {b}))")
    row = con.execute(f"SELECT {', '.join(exprs)} FROM b JOIN s ON {KEY}").fetchone()
    assert row is not None
    mismatches: dict[str, int] = {}
    max_diff, max_col = 0.0, None
    for i, c in enumerate(columns):
        bad, diff = row[2 * i] or 0, row[2 * i + 1]
        if bad:
            mismatches[c] = int(bad)
        if diff is not None and float(diff) > max_diff:
            max_diff, max_col = float(diff), c
    by_year = con.execute(
        f"""SELECT year(b.window_start) AS y, count(*) FROM b JOIN s ON {KEY}
            GROUP BY y ORDER BY y"""
    ).fetchall()
    return {
        "source": source,
        "window": spec.window_label,
        "tolerance": {"rtol": rtol, "rule": "|a-b| <= rtol * max(1, |a|, |b|); null == null"},
        "windows": {
            "batch": batch_n,
            "stream": stream_n,
            "matched": matched,
            "batch_only": batch_n - matched,
            "stream_only": stream_n - matched,
            "stream_duplicates": stream_dupes,
            "matched_by_year": {str(y): int(n) for y, n in by_year},
        },
        "batch_only_windows": one_sided("b", "s"),
        "stream_only_windows": one_sided("s", "b"),
        "columns_compared": len(columns),
        "cells_compared": matched * len(columns),
        "mismatched_cells": sum(mismatches.values()),
        "mismatches_by_column": mismatches,
        "max_abs_diff": max_diff,
        "max_abs_diff_column": max_col,
        "parity": bool(matched > 0 and not mismatches and stream_n == matched and not stream_dupes),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: compare the lake's batch and stream feature tables, write metrics."""
    parser = argparse.ArgumentParser(description="Compare batch and stream feature tables.")
    parser.add_argument("--source", default="battledim")
    parser.add_argument("--rtol", type=float, default=1e-9)
    args = parser.parse_args(argv)
    spec = load_feature_spec(load_config())
    con = lake.connect()
    result = compare_tables(con, spec, args.source, rtol=args.rtol)
    path = write_metrics(f"features_parity_{args.source}", result)
    log.info(
        "parity=%s: %d matched windows, %d mismatched cells, max |diff| %.3g -> %s",
        result["parity"],
        result["windows"]["matched"],
        result["mismatched_cells"],
        result["max_abs_diff"],
        path,
    )
    return 0 if result["parity"] else 1


if __name__ == "__main__":
    sys.exit(main())
