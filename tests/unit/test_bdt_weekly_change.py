"""Week-over-week change features (bdt.data.weekly_change)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leakwatch.bdt.data import weekly_change


def test_difference_to_same_window_a_week_earlier():
    starts = pd.date_range("2019-01-01", periods=8 * 96, freq="15min", tz="UTC")  # 8 days
    level = np.arange(len(starts), dtype=float)
    df = pd.DataFrame(
        {
            "source": "battledim",
            "scenario_id": None,
            "window_start": starts,
            "window_end": starts + pd.Timedelta(hours=1),
            "pressure__n1__mean": level,
            "flow__p1__residual": 2.0,
        }
    )
    out = weekly_change(df, ["pressure__n1__mean", "flow__p1__residual"])
    assert len(out) == 96  # only the 8th day has a reference a week earlier
    assert out["window_start"].iloc[0] == starts[7 * 96]
    assert (out["pressure__n1__mean"] == 7 * 96).all()  # level rises by 1 per window
    assert (out["flow__p1__residual"] == 0.0).all()
    assert list(out.columns[:4]) == ["source", "scenario_id", "window_start", "window_end"]


def test_scenarios_do_not_mix():
    starts = pd.date_range("2019-01-01", periods=7 * 96 + 1, freq="15min", tz="UTC")
    frames = []
    for sc, base in (("a", 0.0), ("b", 100.0)):
        frames.append(
            pd.DataFrame(
                {
                    "source": "sim",
                    "scenario_id": sc,
                    "window_start": starts,
                    "window_end": starts + pd.Timedelta(hours=1),
                    "x__y__mean": base + np.arange(len(starts)),
                }
            )
        )
    out = weekly_change(pd.concat(frames), ["x__y__mean"])
    assert sorted(out["scenario_id"]) == ["a", "b"]
    assert (out["x__y__mean"] == 7 * 96).all()
