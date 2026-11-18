"""Numpy scorer == Spark MLlib model; elbow rule; explanations (bdt.scorer, bdt.model)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakwatch.bdt.model import choose_k_elbow
from leakwatch.bdt.scorer import KMeansParams, KMeansScorer, series_of

COLS = [f"pressure__n{i}__{s}" for i in range(1, 5) for s in ("mean", "std")] + [
    "flow__p1__mean",
    "flow__p1__std",
]


def blobs(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 5, size=(3, len(COLS)))
    x = np.vstack([c + rng.normal(0, 0.5, size=(n // 3, len(COLS))) for c in centres])
    df = pd.DataFrame(x, columns=COLS)
    df["flow__p1__std"] = 2.0  # constant feature: Spark's scaler gives it std 0
    return df


def test_elbow():
    assert choose_k_elbow([2, 3, 4, 5, 6, 7], [100, 40, 20, 17, 15, 14]) == 4
    assert choose_k_elbow([2, 3], [5, 4]) == 2
    assert choose_k_elbow([2, 3, 4], [5, 5, 5]) == 2


def test_series_of():
    assert series_of("amr_sum__C__slope") == "amr_sum__C"


def test_contributions_explain_the_distance():
    p = KMeansParams(
        feature_columns=("a__x__mean", "a__x__std", "b__y__mean"),
        mean=np.zeros(3),
        std=np.ones(3),
        pc=np.eye(3),
        centers=np.zeros((1, 3)),
        threshold=1.0,
    )
    scorer = KMeansScorer(p)
    out, shares = scorer.score_frame(
        pd.DataFrame({"a__x__mean": [3.0], "a__x__std": [0.0], "b__y__mean": [4.0]})
    )
    assert out["score"].iloc[0] == pytest.approx(5.0)
    assert out["top_sensor"].iloc[0] == "b__y"
    share = shares[0]
    assert share == pytest.approx([9 / 25, 16 / 25])
    assert bool(out["above"].iloc[0])


def test_missing_values_count_as_the_mean():
    p = KMeansParams(
        ("a__x__mean",), np.array([10.0]), np.array([2.0]), np.eye(1), np.zeros((1, 1))
    )
    out, _ = KMeansScorer(p).score_frame(pd.DataFrame({"a__x__mean": [np.nan, 14.0]}))
    assert list(out["score"]) == pytest.approx([0.0, 2.0])


def test_save_load_round_trip(tmp_path):
    p = KMeansParams(
        ("a__x__mean",),
        np.array([1.0]),
        np.array([2.0]),
        np.eye(1),
        np.zeros((2, 1)),
        3.5,
        3,
        {"k": 2},
    )
    p.save(tmp_path)
    q = KMeansParams.load(tmp_path)
    assert q.threshold == 3.5 and q.persistence_windows == 3 and q.k == 2
    assert np.array_equal(q.centers, p.centers)


@pytest.mark.spark
def test_numpy_scorer_matches_spark_model(spark):
    from leakwatch.bdt.model import fit_kmeans

    train = blobs()
    cfg = {"pca_variance": 0.95, "k_min": 2, "k_max": 5, "max_iter": 50, "kmeans_init_steps": 2}
    trained = fit_kmeans(spark, train, COLS, cfg, seed=1)
    assert trained.k == 3  # three blobs
    test = blobs(seed=1, n=60)
    sdf = spark.createDataFrame(test[COLS])
    pred = trained.spark_model.transform(sdf).select("p", "prediction").collect()
    centers = trained.params.centers
    spark_cluster = np.array([r["prediction"] for r in pred])
    spark_dist = np.array(
        [np.linalg.norm(r["p"].toArray() - centers[r["prediction"]]) for r in pred]
    )
    ours, _ = KMeansScorer(trained.params).score_frame(test)
    assert np.array_equal(ours["cluster"].to_numpy(), spark_cluster)
    assert ours["score"].to_numpy() == pytest.approx(spark_dist, rel=1e-9, abs=1e-9)
    assert trained.params.std[COLS.index("flow__p1__std")] == 0.0
