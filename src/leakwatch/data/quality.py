"""Data-quality checks on wide sensor tables (pandas reference implementation).

PRD §7.2 cleaning rules: forward-fill gaps of at most 2 steps, flag stuck sensors (same value for
12 or more steps), flag out-of-range values. The Spark streaming job (Phase 1–2) must agree with
these functions; parity tests compare the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def constant_runs(series: pd.Series) -> pd.DataFrame:
    """Return runs of identical consecutive values as ``start, end, length, value``.

    NaN breaks a run and never forms one.
    """
    values = series.to_numpy(dtype="float64")
    n = len(values)
    if n == 0:
        return pd.DataFrame(columns=["start", "end", "length", "value"])
    isna = np.isnan(values)
    starts = np.ones(n, dtype=bool)
    starts[1:] = (values[1:] != values[:-1]) | isna[1:] | isna[:-1]
    start_idx = np.flatnonzero(starts)
    end_idx = np.append(start_idx[1:] - 1, n - 1)
    keep = ~isna[start_idx]
    start_idx, end_idx = start_idx[keep], end_idx[keep]
    return pd.DataFrame(
        {
            "start": series.index[start_idx],
            "end": series.index[end_idx],
            "length": end_idx - start_idx + 1,
            "value": values[start_idx],
        }
    )


def stuck_summary(wide: pd.DataFrame, min_steps: int = 12) -> pd.DataFrame:
    """Per column: longest constant run, number of runs >= ``min_steps``, and steps they cover."""
    rows = []
    for column in wide.columns:
        runs = constant_runs(wide[column])
        long_runs = runs[runs["length"] >= min_steps]
        rows.append(
            {
                "column": column,
                "longest_run": int(runs["length"].max()) if len(runs) else 0,
                "stuck_runs": len(long_runs),
                "stuck_steps": int(long_runs["length"].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def stuck_flags(series: pd.Series, min_steps: int = 12) -> pd.Series:
    """Causal stuck flag: True once a value has repeated for ``min_steps`` consecutive steps.

    A reading is flagged when it and the ``min_steps - 1`` readings before it are all present and
    equal, i.e. from the ``min_steps``-th step of a constant run to its end. The first steps of a
    run are not flagged because a streaming job cannot know yet that the run will get long. NaN
    breaks a run (same rule as :func:`constant_runs`).
    """
    flags = np.zeros(len(series), dtype=bool)
    positions = {ts: i for i, ts in enumerate(series.index)}
    for run in constant_runs(series).itertuples(index=False):
        if run.length >= min_steps:
            first = positions[run.start] + min_steps - 1
            flags[first : positions[run.end] + 1] = True
    return pd.Series(flags, index=series.index, name=series.name)


def forward_fill_short_gaps(wide: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """Forward-fill NaN gaps of at most ``max_gap`` consecutive steps; longer gaps stay NaN."""
    filled = wide.copy()
    for column in wide.columns:
        s = wide[column]
        gap_id = s.notna().cumsum()
        gap_len = s.isna().groupby(gap_id).transform("sum")
        fill_mask = s.isna() & (gap_len <= max_gap)
        filled.loc[fill_mask, column] = s.ffill()[fill_mask]
    return filled
