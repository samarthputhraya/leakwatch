"""Residual sequences for the LSTM autoencoder (PRD §8.1).

Per 5-minute step and channel (33 pressure + 3 flow + 1 level sensors), the residual is the clean
value minus the hour-of-week baseline of that series (the same baselines as the K-Means features,
``lake/baselines``). A sequence is the ``seq_steps`` residual rows ending at the last step of a
K-Means window (windows start every ``stride_steps``), so LSTM-AE scores line up with K-Means
windows and the same alert rule and evaluation apply.

Two years of sequences would take ~0.75 GB as one tensor, so :class:`SequenceSet` keeps the
standardized residual matrix and the sequence end positions, and cuts sequences batch by batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from leakwatch.common import lake
from leakwatch.common.schemas import COLUMN_SEP


def residual_matrix(
    years: Sequence[int], series: Sequence[str], source: str = "battledim"
) -> pd.DataFrame:
    """Residuals (value − hour-of-week baseline): UTC index, one column per series, in order."""
    con = lake.connect()
    types = sorted({s.split(COLUMN_SEP)[0] for s in series})
    year_list = ", ".join(str(int(y)) for y in years)
    type_list = ", ".join(f"'{t}'" for t in types)
    long = lake.query_df(
        con,
        f"""SELECT ts, sensor_type || '{COLUMN_SEP}' || sensor_id AS series_id, value
            FROM clean_readings
            WHERE source = '{source}' AND year IN ({year_list}) AND sensor_type IN ({type_list})""",
    )
    wide = long.pivot_table(index="ts", columns="series_id", values="value", aggfunc="first")
    wide = wide.reindex(columns=list(series)).sort_index()
    base = lake.query_df(con, "SELECT series_id, hour_of_week, baseline FROM baselines")
    table = base.pivot(index="hour_of_week", columns="series_id", values="baseline")
    table = table.reindex(index=range(168), columns=list(series))
    how = (wide.index.dayofweek * 24 + wide.index.hour).to_numpy()
    return wide - table.to_numpy()[how]


def standardize(r: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(r − mean) / std per channel; missing values and zero-variance channels become 0."""
    scale = np.where(std > 0, 1.0 / np.where(std > 0, std, 1), 0.0)
    return np.nan_to_num((r - mean) * scale, nan=0.0).astype("float32")


@dataclass
class SequenceSet:
    """Sequence end positions over a standardized residual matrix (see module docstring)."""

    z: np.ndarray  # (steps, channels) float32
    ends: np.ndarray  # (n,) row index of each sequence's last step
    window_start: pd.DatetimeIndex  # (n,) K-Means window the sequence ends with
    window_end: pd.DatetimeIndex
    seq_steps: int

    def __len__(self) -> int:
        return len(self.ends)

    def batch(self, rows: np.ndarray) -> np.ndarray:
        """Sequences ``rows`` as an ``(len(rows), seq_steps, channels)`` array."""
        view = np.lib.stride_tricks.sliding_window_view(self.z, self.seq_steps, axis=0)
        return np.ascontiguousarray(view[self.ends[rows] - (self.seq_steps - 1)].transpose(0, 2, 1))

    def subset(self, mask: np.ndarray) -> SequenceSet:
        """Only the sequences where ``mask`` is True."""
        return SequenceSet(
            self.z, self.ends[mask], self.window_start[mask], self.window_end[mask], self.seq_steps
        )


def make_sequences(
    residuals: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    seq_steps: int,
    stride_steps: int,
    step_s: int,
    window_s: int,
) -> SequenceSet:
    """Every sequence that ends at the last step of a window and has a gap-free history."""
    z = standardize(residuals.to_numpy(dtype="float64"), mean, std)
    t = residuals.index.asi8 // 10**9
    win_steps = window_s // step_s
    window_start_s = t - (win_steps - 1) * step_s
    pos = np.arange(len(t))
    ok = (pos >= seq_steps - 1) & (window_start_s % (stride_steps * step_s) == 0)
    ends = pos[ok]
    contiguous = (t[ends] - t[ends - (seq_steps - 1)]) == (seq_steps - 1) * step_s
    ends = ends[contiguous]
    last = residuals.index[ends]
    step = pd.Timedelta(seconds=step_s)
    return SequenceSet(
        z=z,
        ends=ends,
        window_start=last - step * (win_steps - 1),
        window_end=last + step,
        seq_steps=seq_steps,
    )
