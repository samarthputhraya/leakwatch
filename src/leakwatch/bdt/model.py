"""K-Means training with Spark MLlib (PRD §7.3).

Pipeline: ``VectorAssembler`` (the 228 feature columns) → ``StandardScaler`` (mean 0, std 1) →
``PCA`` (fewest components explaining ``pca_variance`` of the variance) → ``KMeans``.

k is chosen from ``k_min..k_max`` with the elbow of the within-cluster SSE curve (the "kneedle"
point: farthest below the straight line joining the curve's ends); the silhouette score of every
k is computed with ``ClusteringEvaluator`` and reported next to it (docs/DECISIONS.md D-020).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession

from leakwatch.bdt.scorer import KMeansParams
from leakwatch.common.logging import get_logger

log = get_logger("leakwatch.bdt.model")


@dataclass
class TrainedKMeans:
    """Result of :func:`fit_kmeans`."""

    params: KMeansParams
    spark_model: PipelineModel
    curve: list[dict[str, Any]]
    k: int
    n_components: int
    explained_variance: float


def choose_k_elbow(ks: Sequence[int], sse: Sequence[float]) -> int:
    """Elbow of a decreasing SSE curve: the k farthest below the line joining its end points."""
    k = np.asarray(ks, dtype=float)
    y = np.asarray(sse, dtype=float)
    if len(k) < 3 or y.max() == y.min():
        return int(k[0])
    xn = (k - k.min()) / (k.max() - k.min())
    yn = (y - y.min()) / (y.max() - y.min())
    gap = (1.0 - xn) - yn  # line from (0, 1) to (1, 0) minus the curve
    return int(k[int(np.argmax(gap))])


def to_spark(spark: SparkSession, pdf: pd.DataFrame, columns: Sequence[str]) -> DataFrame:
    """Feature columns of a pandas frame as a Spark DataFrame (Arrow transfer)."""
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    return spark.createDataFrame(pdf[list(columns)].astype("float64"))


def fit_kmeans(
    spark: SparkSession,
    train: pd.DataFrame,
    columns: Sequence[str],
    cfg: Mapping[str, Any],
    seed: int,
) -> TrainedKMeans:
    """Fit scaler, PCA and K-Means for every k; keep the elbow k (see module docstring)."""
    sdf = to_spark(spark, train, columns).cache()
    assembler = VectorAssembler(inputCols=list(columns), outputCol="x", handleInvalid="skip")
    scaler = StandardScaler(inputCol="x", outputCol="z", withMean=True, withStd=True)
    pre = Pipeline(stages=[assembler, scaler]).fit(sdf)
    z = pre.transform(sdf).select("z").cache()

    full = PCA(k=len(columns), inputCol="z", outputCol="p").fit(z)
    cum = np.cumsum(full.explainedVariance.toArray())
    n_comp = int(min(len(columns), np.searchsorted(cum, float(cfg["pca_variance"])) + 1))
    pca = PCA(k=n_comp, inputCol="z", outputCol="p").fit(z)
    p = pca.transform(z).select("p").cache()
    log.info(
        "PCA: %d of %d components explain %.3f of variance", n_comp, len(columns), cum[n_comp - 1]
    )

    evaluator = ClusteringEvaluator(featuresCol="p", predictionCol="prediction")
    curve, models = [], {}
    for k in range(int(cfg["k_min"]), int(cfg["k_max"]) + 1):
        km = KMeans(
            k=k,
            seed=seed,
            maxIter=int(cfg["max_iter"]),
            initSteps=int(cfg["kmeans_init_steps"]),
            featuresCol="p",
        ).fit(p)
        sil = evaluator.evaluate(km.transform(p))
        curve.append({"k": k, "sse": float(km.summary.trainingCost), "silhouette": float(sil)})
        models[k] = km
        log.info("k=%2d  SSE=%.1f  silhouette=%.3f", k, curve[-1]["sse"], sil)
    k_best = choose_k_elbow([c["k"] for c in curve], [c["sse"] for c in curve])
    km = models[k_best]
    scaler_model = pre.stages[1]
    params = KMeansParams(
        feature_columns=tuple(columns),
        mean=np.asarray(scaler_model.mean.toArray()),
        std=np.asarray(scaler_model.std.toArray()),
        pc=np.asarray(pca.pc.toArray()),
        centers=np.asarray(
            [c.toArray() if hasattr(c, "toArray") else c for c in km.clusterCenters()]
        ),
        meta={
            "k": k_best,
            "n_components": n_comp,
            "explained_variance": float(cum[n_comp - 1]),
            "train_windows": len(train),
            "seed": seed,
        },
    )
    spark_model = PipelineModel(stages=[*pre.stages, pca, km])
    for cached in (sdf, z, p):
        cached.unpersist()
    return TrainedKMeans(params, spark_model, curve, k_best, n_comp, float(cum[n_comp - 1]))
