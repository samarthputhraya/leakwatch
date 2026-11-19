"""Combined-index score (D-030) and the synthetic-leak helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from leakwatch.bdt.scorer import KMeansParams, KMeansScorer
from leakwatch.bdt.synthetic import alert_starts, window_means


def params(**meta) -> KMeansParams:
    # 3 features, PCA keeps the first 2 axes; one centre at the origin
    return KMeansParams(
        feature_columns=("a__residual", "b__residual", "c__residual"),
        mean=np.zeros(3),
        std=np.ones(3),
        pc=np.eye(3)[:, :2],
        centers=np.zeros((1, 2)),
        threshold=1.0,
        persistence_windows=1,
        meta=dict(meta),
    )


def test_distance_score_ignores_directions_outside_the_components():
    sc = KMeansScorer(params())
    res = sc.score_matrix(np.array([[0.0, 0.0, 5.0], [3.0, 4.0, 0.0]]))
    np.testing.assert_allclose(res["score"], [0.0, 5.0])
    np.testing.assert_allclose(res["spe"], [25.0, 0.0])
    np.testing.assert_allclose(res["distance"], [0.0, 5.0])


def test_combined_index_sees_the_residual_and_explains_it():
    sc = KMeansScorer(params(score="combined", d_ref=2.0, q_ref=4.0))
    res = sc.score_matrix(np.array([[0.0, 0.0, 5.0], [3.0, 4.0, 0.0]]))
    # (d / d_ref)^2 + q / q_ref
    np.testing.assert_allclose(res["score"], [25.0 / 4.0, 25.0 / 4.0])
    # the out-of-subspace feature c explains all of the first window's score
    np.testing.assert_allclose(res["series_share"][0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(res["series_share"][1].sum(), 1.0)


def test_with_meta_keeps_threshold_and_adds_references():
    p = params().with_meta(score="combined", d_ref=1.0, q_ref=1.0)
    assert p.threshold == 1.0 and p.meta["score"] == "combined"


def test_window_means_of_a_signature():
    spec = SimpleNamespace(step_s=300, window_s=900)  # 3-step windows
    start = pd.Timestamp("2018-11-14 06:00", tz="UTC")
    delta = pd.DataFrame({"pressure__n1": [-1.0, -2.0, -3.0]})  # 3 steps, then held at -3
    starts = pd.Series(
        pd.to_datetime(["2018-11-14 05:50", "2018-11-14 06:00", "2018-11-14 06:10"], utc=True)
    )
    out = window_means(delta, starts, start, spec)["pressure__n1"].to_numpy()
    # windows of 3 steps: [0, 0, -1] -> -1/3; [-1, -2, -3] -> -2; [-3, -3, -3] -> -3
    np.testing.assert_allclose(out, [-1.0 / 3.0, -2.0, -3.0])


@pytest.mark.parametrize(
    ("above", "contiguous", "m", "expected"),
    [
        ([0, 1, 1, 0, 0, 1, 1, 1], [1] * 8, 2, [2, 6]),  # closes after 2 below, reopens
        ([1, 1, 0, 1, 1], [1] * 5, 2, [1]),  # one below does not close it
        ([1, 1, 1, 1], [1, 1, 0, 1], 2, [1, 3]),  # a gap ends the alert; a new run reopens
        ([1, 0, 1, 1], [1, 1, 0, 1], 2, [3]),  # runs do not continue across a gap
        ([1, 0, 1], [1, 1, 1], 1, [0]),
    ],
)
def test_alert_starts(above, contiguous, m, expected):
    got = alert_starts(np.array(above, bool), np.array(contiguous, bool), m, close_after=2)
    assert got == expected
