"""K-Means anomaly scores and per-sensor explanations (PRD §7.3).

The model is trained with Spark MLlib (``bdt.model``): VectorAssembler → StandardScaler →
PCA → KMeans. Its fitted parameters (feature means and standard deviations, PCA loadings, cluster
centres) are exported to :class:`KMeansParams`, and :class:`KMeansScorer` applies exactly the same
arithmetic with numpy. Batch evaluation and the live stream both score through this class, and a
test checks it against Spark's own ``transform`` (same cluster, same distance).

- **score** = Euclidean distance from a window's PCA vector to the nearest cluster centre;
- **contribution** of a feature = its squared share of that distance, after mapping the PCA-space
  difference back to the standardized feature space (the shares sum to 1); a series' contribution
  is the sum over its statistics, so the top series explain the alert.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leakwatch.common.schemas import COLUMN_SEP


@dataclass(frozen=True)
class KMeansParams:
    """Everything needed to score a feature row, exported from the Spark model."""

    feature_columns: tuple[str, ...]
    mean: np.ndarray  # (n_features,)
    std: np.ndarray  # (n_features,) sample std; 0 -> feature ignored (Spark sets it to 0)
    pc: np.ndarray  # (n_features, n_components) PCA loadings, orthonormal columns
    centers: np.ndarray  # (k, n_components)
    threshold: float | None = None
    persistence_windows: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def k(self) -> int:
        """Number of clusters."""
        return int(self.centers.shape[0])

    def save(self, directory: Path) -> Path:
        """Write ``params.npz`` + ``params.json`` into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "params.npz", mean=self.mean, std=self.std, pc=self.pc, centers=self.centers
        )
        doc = {
            "feature_columns": list(self.feature_columns),
            "threshold": self.threshold,
            "persistence_windows": self.persistence_windows,
            "meta": self.meta,
        }
        path = directory / "params.json"
        path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> KMeansParams:
        """Read parameters written by :meth:`save`."""
        arrays = np.load(directory / "params.npz")
        doc = json.loads((directory / "params.json").read_text(encoding="utf-8"))
        return cls(
            feature_columns=tuple(doc["feature_columns"]),
            mean=arrays["mean"],
            std=arrays["std"],
            pc=arrays["pc"],
            centers=arrays["centers"],
            threshold=doc.get("threshold"),
            persistence_windows=doc.get("persistence_windows"),
            meta=doc.get("meta", {}),
        )

    def with_threshold(
        self, threshold: float, persistence_windows: int | None = None
    ) -> KMeansParams:
        """Copy with the alert threshold (and persistence) set."""
        return KMeansParams(
            self.feature_columns,
            self.mean,
            self.std,
            self.pc,
            self.centers,
            float(threshold),
            persistence_windows if persistence_windows is not None else self.persistence_windows,
            dict(self.meta),
        )


def series_of(feature_column: str) -> str:
    """``pressure__n54__mean`` -> ``pressure__n54``."""
    return feature_column.rsplit(COLUMN_SEP, 1)[0]


class KMeansScorer:
    """Score feature rows with exported K-Means parameters (see module docstring)."""

    def __init__(self, params: KMeansParams) -> None:
        self.params = params
        self._scale = np.where(params.std > 0, 1.0 / np.where(params.std > 0, params.std, 1), 0.0)
        series = [series_of(c) for c in params.feature_columns]
        self.series: list[str] = list(dict.fromkeys(series))
        index = {s: i for i, s in enumerate(self.series)}
        # (n_features, n_series) 0/1 matrix summing feature contributions per series
        self._to_series = np.zeros((len(series), len(self.series)))
        self._to_series[np.arange(len(series)), [index[s] for s in series]] = 1.0

    def standardize(self, x: np.ndarray) -> np.ndarray:
        """(x - mean) / std; features with std 0 and missing values become 0 (the mean)."""
        z = (x - self.params.mean) * self._scale
        return np.nan_to_num(z, nan=0.0)

    def score_matrix(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Score an ``(n, n_features)`` matrix in ``params.feature_columns`` order.

        Returns ``score`` (n,), ``cluster`` (n,), and ``series_share`` (n, n_series): each
        series' share of the squared distance (rows sum to 1, or 0 for a zero distance).
        """
        p = self.standardize(x) @ self.params.pc
        c = self.params.centers
        # |p - c|^2 = |p|^2 - 2 p.c + |c|^2 as an (n, k) matrix (no (n, k, d) temporary)
        d2 = (p**2).sum(axis=1)[:, None] - 2.0 * p @ c.T + (c**2).sum(axis=1)[None, :]
        cluster = d2.argmin(axis=1)
        diff = p - c[cluster]
        dist = np.sqrt((diff**2).sum(axis=1))  # exact distance to the chosen centre
        back = diff @ self.params.pc.T  # PCA-space difference in standardized feature space
        sq = back**2
        total = sq.sum(axis=1, keepdims=True)
        share = np.divide(sq, total, out=np.zeros_like(sq), where=total > 0) @ self._to_series
        return {"score": dist, "cluster": cluster, "series_share": share}

    def score_frame(self, features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Score a feature table.

        Returns the key columns plus ``score``, ``cluster``, ``top_sensor`` and, if a threshold is
        set, ``threshold`` and ``above``; and the ``(n, n_series)`` share matrix.
        """
        cols = list(self.params.feature_columns)
        missing = [c for c in cols if c not in features.columns]
        if missing:
            raise KeyError(f"feature table lacks {len(missing)} model columns, e.g. {missing[:3]}")
        x = features[cols].to_numpy(dtype="float64")
        res = self.score_matrix(x)
        keys = [c for c in ("source", "scenario_id", "window_start", "window_end") if c in features]
        out = features[keys].reset_index(drop=True).copy()
        out["score"] = res["score"]
        out["cluster"] = res["cluster"].astype("int32")
        top = res["series_share"].argmax(axis=1)
        out["top_sensor"] = [self.series[i] for i in top]
        if self.params.threshold is not None:
            out["threshold"] = float(self.params.threshold)
            out["above"] = out["score"] > out["threshold"]
        return out, res["series_share"]

    def top_series(self, share_row: np.ndarray, n: int) -> list[tuple[str, float]]:
        """The ``n`` series with the largest share in one row of ``series_share``."""
        order = np.argsort(share_row)[::-1][:n]
        return [(self.series[i], float(share_row[i])) for i in order]


def feature_matrix(features: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """``features[columns]`` as a float matrix (helper for training code)."""
    return features[list(columns)].to_numpy(dtype="float64")
