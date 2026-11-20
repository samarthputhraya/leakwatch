"""The scaling benchmark scores feature rows inside Spark exactly like the numpy scorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakwatch.bdt.scaling import SCORE_SCHEMA, partition_scorer
from leakwatch.bdt.scorer import KMeansParams, KMeansScorer

pytestmark = pytest.mark.spark


def toy_params(columns: list[str]) -> KMeansParams:
    rng = np.random.default_rng(1)
    d = len(columns)
    return KMeansParams(
        feature_columns=tuple(columns),
        mean=np.zeros(d),
        std=np.ones(d),
        pc=np.eye(d)[:, :2],
        centers=rng.normal(size=(3, 2)),
        threshold=1.5,
        persistence_windows=1,
    )


def test_spark_scores_equal_the_numpy_scorer(spark):
    cols = ["a__mean", "b__mean", "a__residual"]
    params = toy_params(cols)
    rng = np.random.default_rng(2)
    pdf = pd.DataFrame(rng.normal(size=(40, 3)), columns=cols)
    pdf.insert(0, "source", "sim")
    pdf.insert(1, "scenario_id", "s1")
    pdf.insert(2, "window_start", pd.date_range("2020-01-06", periods=40, freq="15min", tz="UTC"))
    sdf = spark.createDataFrame(pdf)
    out = sdf.mapInPandas(partition_scorer(params), SCORE_SCHEMA).toPandas()
    ref, _ = KMeansScorer(params).score_frame(pdf)
    out = out.sort_values("window_start").reset_index(drop=True)
    np.testing.assert_allclose(out["score"], ref["score"], rtol=1e-12)
    assert list(out["cluster"]) == list(ref["cluster"])
    assert list(out["above"]) == list(ref["above"])


def test_rebaselined_series_are_scored_like_training(spark):
    from leakwatch.bdt.scaling import series_scorer
    from leakwatch.features.rebaseline import Rebaseline, rebaselined

    cols = ["a__residual", "b__residual"]
    params = toy_params(cols)
    rb = Rebaseline(window_days=0.5, gap_days=0.25)
    rng = np.random.default_rng(3)
    frames = []
    for scenario in ("s1", "s2"):
        pdf = pd.DataFrame(rng.normal(size=(200, 2)), columns=cols)
        pdf.insert(0, "source", "sim")
        pdf.insert(1, "scenario_id", scenario)
        ts = pd.date_range("2020-01-06", periods=200, freq="15min", tz="UTC")
        pdf.insert(2, "window_start", ts)
        frames.append(pdf)
    both = pd.concat(frames, ignore_index=True)
    out = (
        spark.createDataFrame(both.sample(frac=1.0, random_state=0))
        .groupBy("source", "scenario_id")
        .applyInPandas(series_scorer(params, rb), SCORE_SCHEMA)
        .toPandas()
        .sort_values(["scenario_id", "window_start"])
        .reset_index(drop=True)
    )
    ref = []
    for pdf in frames:  # each scenario re-baselined on its own, rows without a reference dropped
        r = rebaselined(pdf, cols, rb)
        r = r[r[cols].notna().all(axis=1)]
        scored, _ = KMeansScorer(params).score_frame(r)
        ref.append(scored.assign(scenario_id=r["scenario_id"].to_numpy()))
    ref = pd.concat(ref, ignore_index=True)
    assert len(out) == len(ref) < len(both)
    np.testing.assert_allclose(out["score"], ref["score"], rtol=1e-12)
    assert list(out["above"]) == list(ref["above"])


def test_series_scorer_without_a_reference_returns_no_rows():
    from leakwatch.bdt.scaling import series_scorer
    from leakwatch.features.rebaseline import Rebaseline

    cols = ["a__residual"]
    pdf = pd.DataFrame(
        {
            "source": "sim",
            "scenario_id": "s1",
            "window_start": pd.date_range("2020-01-06", periods=4, freq="15min", tz="UTC"),
            "a__residual": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = series_scorer(toy_params(cols), Rebaseline(7, 1))(pdf)
    assert out.empty and list(out.columns) == [f.name for f in SCORE_SCHEMA.fields]
