"""``make lstm-ae``: LSTM autoencoder detector and the K-Means comparison (PRD §8.1, Phase 5).

1. Residual sequences (``dl.sequences``): 6 h (72 steps) × 37 channels (33 pressure, 3 flow,
   1 level), standardized with means/stds of normal 2018 training steps.
2. Train on normal 2018 sequences (D-015; every step normal), split by UTC day exactly like the
   K-Means detector (train / threshold / validation days); early stopping on threshold-day loss.
3. Score = mean squared reconstruction error; τ = ``bdt.threshold_quantile`` of threshold-day
   scores; m tuned on 2018 validation days with the same rule as K-Means (``eval.alerting``).
4. Evaluate 2019 with ``eval.detection`` (+ chance level), and compare with K-Means at matched
   false-alarm rates: both detectors' detection rate vs false alarms per week over the same
   threshold quantiles at their own m; detection interpolated at 1, 2 and 4 false alarms/week.

Writes ``reports/metrics/dl_lstm_ae.json``, ``data/models/lstm_ae/``, and a comparison figure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from leakwatch.bdt.alerts import run_alerts
from leakwatch.bdt.data import load_features, series_area_map, split_roles
from leakwatch.bdt.evaluate import year_bounds
from leakwatch.bdt.scorer import KMeansParams, KMeansScorer
from leakwatch.bdt.train import feature_view, model_dir
from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.data.ground_truth import load_ground_truth
from leakwatch.dl.sequences import SequenceSet, make_sequences, residual_matrix
from leakwatch.eval.alerting import make_rule, operating_points, tune_persistence
from leakwatch.eval.detection import chance_detection, evaluate_detections, onset_contrast
from leakwatch.features.normal import intervals_from_config, is_normal
from leakwatch.features.spec import load_feature_spec

log = get_logger("leakwatch.dl.lstm_ae")
QUANTILES = [0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]
MATCHED_FA = [1.0, 2.0, 4.0]


class LSTMAutoencoder(nn.Module):
    """Sequence → latent vector → sequence (encoder/decoder LSTMs)."""

    def __init__(self, channels: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(channels, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102 (class doc)
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        d = self.from_latent(z).unsqueeze(1).expand(-1, x.shape[1], -1)
        y, _ = self.decoder(d.contiguous())
        return self.output(y)


def reconstruction(
    model: nn.Module, seqs: SequenceSet, batch: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sequence MSE and each channel's share of the squared error."""
    model.eval()
    scores, shares = [], []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            x = torch.from_numpy(seqs.batch(np.arange(i, min(i + batch, len(seqs)))))
            err = (model(x) - x) ** 2  # (b, T, C)
            per_ch = err.sum(dim=1)  # (b, C)
            total = per_ch.sum(dim=1, keepdim=True)
            scores.append((total[:, 0] / (x.shape[1] * x.shape[2])).numpy())
            shares.append((per_ch / total.clamp_min(1e-12)).numpy())
    return np.concatenate(scores), np.concatenate(shares)


def train_model(
    train: SequenceSet, val: SequenceSet, channels: int, cfg: Mapping[str, Any], seed: int
) -> tuple[LSTMAutoencoder, list[dict[str, float]]]:
    """Adam on MSE; keep the epoch with the lowest validation loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LSTMAutoencoder(channels, int(cfg["hidden"]), int(cfg["latent"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    best, best_state, history = np.inf, None, []
    bs = int(cfg["batch_size"])
    patience, bad = int(cfg["patience"]), 0
    for epoch in range(int(cfg["epochs"])):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        t0 = time.perf_counter()
        for i in range(0, len(order), bs):
            x = torch.from_numpy(train.batch(order[i : i + bs]))
            loss = ((model(x) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        val_loss = float(reconstruction(model, val)[0].mean())
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "wall_s": round(time.perf_counter() - t0, 1),
            }
        )
        log.info("epoch %d: train %.4f  val %.4f", epoch + 1, history[-1]["train_loss"], val_loss)
        if val_loss < best - 1e-5:
            best, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def interpolate_at(points: list[dict[str, Any]], fa: float) -> float | None:
    """Detection rate at a false-alarm rate, linearly interpolated over the operating points."""
    pts = sorted(
        (p["false_alarms_per_week"], p["detection_rate"])
        for p in points
        if p["false_alarms_per_week"] is not None
    )
    if not pts or fa < pts[0][0] or fa > pts[-1][0]:
        return None
    xs, ys = zip(*pts, strict=True)
    return round(float(np.interp(fa, xs, ys)), 4)


def plot_comparison(curves: dict[str, list[dict[str, Any]]], year: int) -> str:
    """Detection rate vs false alarms per week for each detector."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, pts in curves.items():
        pts = sorted(pts, key=lambda p: p["false_alarms_per_week"])
        ax.plot(
            [p["false_alarms_per_week"] for p in pts],
            [p["detection_rate"] for p in pts],
            "o-",
            label=name,
        )
    ax.axvline(2.0, color="grey", ls="--", lw=1, label="aim: ≤ 2 false alarms/week")
    ax.set_xlabel(f"false alarms per week ({year})")
    ax.set_ylabel(f"share of {year} leaks detected within 7 days")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("K-Means vs LSTM autoencoder (thresholds = quantiles of 2018 normal scores)")
    fig.tight_layout()
    out = paths.FIGURES_DIR / f"dl_detector_comparison_{year}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


def main(argv: list[str] | None = None) -> int:
    """Train, threshold, tune, evaluate and compare (see module docstring)."""
    argparse.ArgumentParser(description="LSTM autoencoder detector.").parse_args(argv)
    t_start = time.perf_counter()
    cfg = load_config()
    dl = cfg["dl"]["lstm"]
    bdt = cfg["bdt"]
    seed = int(cfg["project"]["seed"])
    torch.set_num_threads(int(dl["threads"]))
    spec = load_feature_spec(cfg)
    series = list(spec.sensor_series)
    gt = load_ground_truth()
    areas = series_area_map(spec)
    train_year, eval_year = int(bdt["train_year"]), int(bdt["eval_year"])

    resid = residual_matrix([train_year, eval_year], series)
    intervals = intervals_from_config(gt, cfg["normal"], rule="main")
    step_normal = is_normal(resid.index, intervals)
    in_train_year = resid.index.year == train_year
    roles_step = split_roles(pd.Series(resid.index), bdt)
    fit_rows = step_normal & in_train_year & (roles_step == "train")
    mean = np.nanmean(resid.to_numpy()[fit_rows], axis=0)
    std = np.nanstd(resid.to_numpy()[fit_rows], axis=0, ddof=1)
    seqs = make_sequences(
        resid,
        mean,
        std,
        int(dl["seq_steps"]),
        spec.slide_s // spec.step_s,
        spec.step_s,
        spec.window_s,
    )
    # a sequence is normal if all its steps are normal
    normal_csum = np.concatenate([[0], np.cumsum(~step_normal)])
    seq_normal = (normal_csum[seqs.ends + 1] - normal_csum[seqs.ends + 1 - seqs.seq_steps]) == 0
    roles = split_roles(pd.Series(seqs.window_start), bdt)
    year = seqs.window_start.year
    base = seq_normal & (year == train_year)
    train = seqs.subset(base & (roles == "train"))
    thr_set = seqs.subset(base & (roles == "threshold"))
    log.info("sequences: %d total, %d train, %d threshold", len(seqs), len(train), len(thr_set))

    model, history = train_model(train, thr_set, len(series), dl, seed)
    scores, shares = reconstruction(model, seqs)
    thr_scores = scores[base & (roles == "threshold")]
    tau = float(np.quantile(thr_scores, float(bdt["threshold_quantile"])))
    scored = pd.DataFrame(
        {
            "source": "battledim",
            "scenario_id": None,
            "window_start": seqs.window_start,
            "window_end": seqs.window_end,
            "score": scores,
        }
    )
    in_train = year == train_year
    validation_normal = (base & (roles == "validation"))[in_train]
    m, tuning = tune_persistence(
        tau,
        scored[in_train].reset_index(drop=True),
        shares[in_train],
        series,
        validation_normal,
        None,
        [],
        cfg,
        gt,
        areas,
        "lstm_ae",
        spec.window_s,
        spec.slide_s,
    )
    rule = make_rule(tau, m, cfg, areas, "lstm_ae")
    alerts = run_alerts(scored, shares, series, rule)
    period = year_bounds(eval_year)
    horizon = float(bdt["detection_horizon_days"])
    ev = evaluate_detections(alerts, gt, *period, horizon)
    ev["chance_level"] = chance_detection(ev["alerts"]["total"], gt, *period, horizon)
    ev["onset_contrast_24h"] = onset_contrast(scored, gt, *period)

    out_dir = paths.models_dir() / "lstm_ae"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "params.json").write_text(
        json.dumps(
            {
                "series": series,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "threshold": tau,
                "persistence_windows": m,
                "config": dl,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    # Matched false-alarm comparison with K-Means (both at their own 2018-tuned m).
    eval_mask = year == eval_year
    lstm_curve = operating_points(
        thr_scores,
        QUANTILES,
        m,
        scored[eval_mask].reset_index(drop=True),
        shares[eval_mask],
        series,
        cfg,
        gt,
        areas,
        "lstm_ae",
        period,
    )
    curves = {"LSTM-AE": lstm_curve}
    comparison: dict[str, Any] = {}
    feats = pd.concat([load_features(train_year), load_features(eval_year)], ignore_index=True)
    labels = {
        "kmeans": "K-Means (all PRD features)",
        "kmeans_residual": "K-Means (residual features)",
    }
    for model_name, label in labels.items():
        km_dir = model_dir(model_name)
        if not (km_dir / "params.json").exists():
            continue
        params = KMeansParams.load(km_dir)
        rows = feature_view(feats, spec, str(params.meta.get("feature_view", "absolute")))
        km_scorer = KMeansScorer(params)
        km_scored, km_shares = km_scorer.score_frame(rows)
        km_normal = is_normal(km_scored["window_start"], intervals) & is_normal(
            km_scored["window_start"] + pd.Timedelta(seconds=spec.window_s - spec.step_s), intervals
        )
        km_roles = split_roles(km_scored["window_start"], bdt)
        km_thr = km_scored.loc[
            km_normal
            & (km_roles == "threshold")
            & (km_scored["window_start"].dt.year == train_year),
            "score",
        ].to_numpy()
        km_eval = (km_scored["window_start"].dt.year == eval_year).to_numpy()
        curves[label] = operating_points(
            km_thr,
            QUANTILES,
            int(params.persistence_windows),
            km_scored[km_eval].reset_index(drop=True),
            km_shares[km_eval],
            km_scorer.series,
            cfg,
            gt,
            areas,
            model_name,
            period,
        )
    for name, pts in curves.items():
        comparison[name] = {
            f"detection_at_{fa:g}_fa_per_week": interpolate_at(pts, fa) for fa in MATCHED_FA
        }
    figure = plot_comparison(curves, eval_year)

    payload = {
        "series": len(series),
        "sequence_steps": int(dl["seq_steps"]),
        "sequences": {"total": len(seqs), "train": len(train), "threshold": len(thr_set)},
        "standardization": "per-channel mean/std of normal 2018 training steps",
        "training": {"epochs_run": len(history), "history": history},
        "threshold": tau,
        "persistence_windows": m,
        "persistence_tuning": tuning,
        "eval_2019": ev,
        "windows_above_threshold_2019": round(float((scores[eval_mask] > tau).mean()), 4),
        "operating_points_2019": curves,
        "matched_false_alarm_comparison": comparison,
        "figures": [figure],
        "model_dir": str(out_dir),
        "wall_s": round(time.perf_counter() - t_start, 1),
    }
    path = write_metrics("dl_lstm_ae", payload)
    log.info(
        "LSTM-AE 2019: detection %s, FA/week %s, precision %s -> %s",
        ev["detection"]["overall"],
        ev["false_alarms_per_week"],
        ev["precision"],
        path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
