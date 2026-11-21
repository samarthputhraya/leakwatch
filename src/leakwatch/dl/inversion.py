"""Model-based leak localization: generalized least squares against the sensitivity matrix (D-044).

A leak at junction ``j`` changes the pressure at the 33 sensors by about ``a·s_j``: the
sensitivity signature ``s_j`` (``dl.gnn_data``: a 15 mm EPANET leak on the nominal model) times a
size factor ``a ≥ 0``. The observed residual ``x`` (mean of the two 1 h windows) also carries
noise that is strongly correlated across sensors: a day with a little more demand lowers every
pressure at once. On the simulated scenarios the largest eigenvector of that noise holds about
two thirds of its variance, which is why a plain cosine (``sensitivity_corr``) ranks so many
nearby junctions above the true one.

Each junction is scored by the weighted error of its best fit, a 1-D search over the leak size
with a closed form::

    a_j   = max(0, xᵀ Σ⁻¹ s_j / s_jᵀ Σ⁻¹ s_j)
    err_j = (x − a_j s_j)ᵀ Σ⁻¹ (x − a_j s_j)          log-likelihood_j = −err_j / 2

``Σ`` is the noise covariance, estimated either from the simulated training scenarios (the residual
left after fitting each one's true leak: perturbed roughness, background leaks, demand draws and
sensor noise) or from the real 2018 normal windows of the re-baselined residuals (no new leak, so
the residual is noise). A ridge ``λ·tr(Σ)/n·I`` keeps it well conditioned; λ is chosen on the
validation scenarios. The per-node signatures are cached, so scoring is one matrix product.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leakwatch.common import paths


def regularized(cov: np.ndarray, ridge: float) -> np.ndarray:
    """``cov + ridge · mean(diag(cov)) · I`` (well conditioned for any ridge > 0)."""
    cov = np.asarray(cov, dtype="float64")
    n = cov.shape[0]
    return cov + float(ridge) * np.eye(n) * np.trace(cov) / n


def whitener(cov: np.ndarray, ridge: float) -> np.ndarray:
    """Matrix ``W`` with ``Wᵀ W = Σ⁻¹`` for the regularized covariance (rows = whitened axes)."""
    inv = np.linalg.inv(regularized(cov, ridge))
    inv = 0.5 * (inv + inv.T)
    return np.linalg.cholesky(inv).T


def fit_residuals(x: np.ndarray, signatures: np.ndarray) -> np.ndarray:
    """``x − a·s`` with each row's own signature and the least-squares size ``a ≥ 0``.

    ``x`` and ``signatures`` are (B, sensors); used to estimate the simulated noise covariance from
    training scenarios whose leak node is known.
    """
    x = np.asarray(x, dtype="float64")
    s = np.asarray(signatures, dtype="float64")
    ss = (s**2).sum(axis=1)
    a = np.maximum((x * s).sum(axis=1) / np.where(ss > 0, ss, 1.0), 0.0)
    return x - a[:, None] * s


def second_moment(x: np.ndarray) -> np.ndarray:
    """``E[x xᵀ]`` over rows without NaN (noise about zero: the fit has no intercept)."""
    x = np.asarray(x, dtype="float64")
    x = x[~np.isnan(x).any(axis=1)]
    if len(x) < 2:
        raise ValueError("need at least two complete rows to estimate a covariance")
    return (x.T @ x) / len(x)


class Inversion:
    """GLS leak scores for every node from a sensitivity matrix and a noise covariance."""

    def __init__(
        self, sensitivity: np.ndarray, cov: np.ndarray, ridge: float, is_junction: np.ndarray
    ) -> None:
        """``sensitivity`` (N, sensors) raw pressure changes; ``cov`` (sensors, sensors)."""
        self.sensitivity = np.asarray(sensitivity, dtype="float64")
        self.cov = np.asarray(cov, dtype="float64")
        self.ridge = float(ridge)
        self.is_junction = np.asarray(is_junction, dtype=bool)
        if self.sensitivity.shape[1] != self.cov.shape[0]:
            raise ValueError("sensitivity and covariance disagree on the number of sensors")
        self.w = whitener(self.cov, self.ridge)
        self._sw = self.sensitivity @ self.w.T  # (N, sensors) whitened signatures
        self._ss = (self._sw**2).sum(axis=1)

    def fit(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(B, sensors) or (B, sensors, windows) residuals -> (size a, error) per node, (B, N)."""
        xm = np.asarray(x, dtype="float64")
        if xm.ndim == 3:
            xm = xm.mean(axis=2)
        xw = xm @ self.w.T
        dot = xw @ self._sw.T
        a = np.maximum(dot / np.where(self._ss > 0, self._ss, 1.0), 0.0)
        err = (xw**2).sum(axis=1, keepdims=True) - 2.0 * a * dot + a**2 * self._ss
        return a, np.maximum(err, 0.0)

    def loglik(self, x: np.ndarray) -> np.ndarray:
        """Gaussian log-likelihood up to a constant, −err/2; non-junctions get −inf (B, N)."""
        _, err = self.fit(x)
        ll = -0.5 * err
        ll[:, ~self.is_junction] = -np.inf
        return ll

    def save(self, path: Path) -> Path:
        """Write the matrices (``.npz``) so the live stream and the demo load the same model."""
        paths.ensure_dir(path.parent)
        np.savez_compressed(
            path,
            sensitivity=self.sensitivity,
            cov=self.cov,
            ridge=np.array(self.ridge),
            is_junction=self.is_junction,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> Inversion:
        """Read a model written by :meth:`save`."""
        z = np.load(path, allow_pickle=False)
        return cls(z["sensitivity"], z["cov"], float(z["ridge"]), z["is_junction"])


def inversion_path(kind: str) -> Path:
    """``data/models/localizer/inversion_<kind>.npz`` (kind: ``sim`` or ``real``)."""
    return paths.models_dir() / "localizer" / f"inversion_{kind}.npz"


def real_noise_windows(
    feats: pd.DataFrame, cols: Sequence[str], normal: np.ndarray, hour: pd.Timedelta | None = None
) -> np.ndarray:
    """Mean of two consecutive hourly windows (ending 1 h apart), both normal: (n, sensors).

    ``feats`` is indexed by ``window_end`` (one row per window, as ``gnn.rebaselined_pressure``);
    ``normal`` is the per-row normal mask. The pairs match the localizer's input exactly.
    """
    hour = hour or pd.Timedelta(hours=1)
    ends = pd.DatetimeIndex(feats.index)
    pos = pd.Series(np.arange(len(ends)), index=ends)
    prev = pos.reindex(ends - hour).to_numpy()
    has_prev = ~np.isnan(prev)
    cur = np.flatnonzero(normal & has_prev)
    prv = prev[normal & has_prev].astype(int)
    keep = normal[prv]
    vals = feats[list(cols)].to_numpy(dtype="float64")
    return 0.5 * (vals[cur[keep]] + vals[prv[keep]])


def ranks(scores: np.ndarray) -> np.ndarray:
    """Rank of every node per row, 0 = best (ties broken by index); −inf ranks last."""
    order = np.argsort(-scores, axis=1, kind="stable")
    out = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    out[rows, order] = np.arange(scores.shape[1])[None, :]
    return out


def rank_fusion(
    score_sets: Mapping[str, np.ndarray], is_junction: np.ndarray, k: float = 60.0
) -> np.ndarray:
    """Reciprocal rank fusion ``Σ 1 / (k + rank)`` of several (B, N) score arrays (Cormack 2009)."""
    parts = [
        1.0 / (k + 1.0 + ranks(np.asarray(scores, dtype="float64")))
        for scores in score_sets.values()
    ]
    if not parts:
        raise ValueError("rank_fusion needs at least one score array")
    total = np.sum(parts, axis=0).astype("float64")
    total[:, ~np.asarray(is_junction, dtype=bool)] = -np.inf
    return total


def interpolation_scores(
    dist: np.ndarray, sensor_idx: np.ndarray, x: np.ndarray, length_m: float
) -> np.ndarray:
    """Model-free cue: minus the distance-weighted mean residual of the sensors around each node.

    Weights ``exp(−d / length_m)`` along the pipes; a node near sensors whose pressure fell gets a
    high score. ``x`` is (B, sensors[, windows]).
    """
    xm = np.asarray(x, dtype="float64")
    if xm.ndim == 3:
        xm = xm.mean(axis=2)
    d = np.asarray(dist, dtype="float64")[:, np.asarray(sensor_idx)]
    w = np.where(np.isfinite(d), np.exp(-d / float(length_m)), 0.0)
    return -(xm @ w.T) / np.maximum(w.sum(axis=1), 1e-12)[None, :]


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax that tolerates −inf entries."""
    s = np.asarray(scores, dtype="float64")
    s = s - np.max(s, axis=1, keepdims=True)
    p = np.exp(s)
    return p / p.sum(axis=1, keepdims=True)


def describe(cfg: Mapping[str, Any]) -> str:
    """Human-readable name of a localizer config (``dl.localizer``)."""
    parts = []
    if cfg.get("gls"):
        parts.append(f"GLS ({cfg['gls']} noise)")
    if cfg.get("network"):
        parts.append("network")
    if cfg.get("sensor_prior_m"):
        parts.append(f"sensor prior {float(cfg['sensor_prior_m']):g} m")
    if cfg.get("interpolation"):
        parts.append("interpolation")
    joined = " + ".join(parts) or "none"
    return f"rank fusion of {joined}" if cfg.get("fusion") == "rank" else joined
