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
