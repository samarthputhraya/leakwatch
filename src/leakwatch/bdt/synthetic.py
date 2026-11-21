"""``make bdt-synthetic``: simulated leaks on real normal background (residual detectors, D-030).

The 2019 evaluation cannot tell detectors apart well (2019 is never leak-free and about 10 leaks
flow at once on average, so alerts are frequent and the chance level is near 100%). This test
asks a cleaner question on 2018 data only: on a normal stretch of BattLeDIM 2018 (no
significant leak, D-015), how often does a detector raise an alert for a leak of known size and
place, and how often does it alarm without one?

- Leaks: ``junctions`` junctions drawn at random (seeded), each with every diameter in
  ``diameters_m``, starting at one of ``starts`` random times in the background period. The
  signature is EPANET leak run minus no-leak run (``demo.signature``, the same as the live demo).
- Features: for detectors that use only the ``residual`` statistic, the leak adds exactly the
  window mean of the signature to each series' residual (residual = mean of reading minus
  baseline over the window's steps), so the background feature rows are shifted in numpy
  instead of re-running Spark. Detectors on other statistics are not covered.
- Detected: the alert rule (m consecutive windows above τ) fires on a window that ends within
  ``hours`` of the leak start; time to detection = end of that window − start.
- False alarms: the same rule on the unmodified background, alerts per week.

Writes ``reports/metrics/bdt_synthetic_leaks.json``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from leakwatch.bdt.alerts import fast_threshold
from leakwatch.bdt.data import load_features, normal_mask
from leakwatch.bdt.scorer import KMeansParams, KMeansScorer
from leakwatch.bdt.train import MODELS, VARIANTS, model_dir
from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.data.ground_truth import load_ground_truth
from leakwatch.features.spec import FeatureSpec, load_feature_spec

log = get_logger("leakwatch.bdt.synthetic")


def window_means(
    delta: pd.DataFrame,
    window_starts: pd.Series,
    start: pd.Timestamp,
    spec: FeatureSpec,
) -> pd.DataFrame:
    """Mean of the signature over each window's steps (0 before the start, last value after)."""
    steps = spec.window_s // spec.step_s
    ws = (window_starts - start).dt.total_seconds().to_numpy() // spec.step_s
    idx = ws[:, None].astype(int) + np.arange(steps)[None, :]  # (windows, steps) signature index
    values = delta.to_numpy()
    padded = np.vstack([np.zeros((1, values.shape[1])), values])  # row 0 = before the leak
    pos = np.where(idx < 0, 0, np.minimum(idx, len(values) - 1) + 1)
    return pd.DataFrame(padded[pos].mean(axis=1), columns=delta.columns)


def alert_starts(
    above: np.ndarray,
    contiguous: np.ndarray,
    m: int,
    close_after: int,
    fast: np.ndarray | None = None,
) -> list[int]:
    """Indices where the rule opens an alert: m consecutive above; closes after N below.

    ``contiguous[i]`` is False when window i does not directly follow window i-1 (a gap in the
    background); runs and open alerts do not continue across a gap. ``fast[i]`` True (the window
    is above the fast-path threshold, D-043) opens the alert at once.
    """
    starts, run, below, open_ = [], 0, 0, False
    for i, a in enumerate(above):
        if not contiguous[i]:
            run, below, open_ = 0, 0, False
        if a:
            run, below = run + 1, 0
            if not open_ and (run >= m or (fast is not None and fast[i])):
                starts.append(i)
                open_ = True
        else:
            run = 0
            below += 1
            if open_ and below >= close_after:
                open_ = False
    return starts


def residual_variants() -> list[str]:
    """Detector variants that use only residual features (the ones this test covers).

    The re-baselined variant (D-038) is included: its background windows are re-baselined over
    the whole training year first; a 6-hour leak does not reach its own reference (gap 1 day), so
    the leak signature adds to the re-baselined residual exactly as to the plain one.
    """
    return [v for v, spec in VARIANTS.items() if spec[1] in ("residual", "rebaselined")]


def draw(
    cfg: Mapping[str, Any], rng: np.random.Generator, wn: Any
) -> tuple[list[str], list[pd.Timestamp], pd.Timedelta]:
    """Leak junctions, start times and the observation horizon of the benchmark (seeded)."""
    scfg = cfg["bdt"]["synthetic"]
    lo, hi = (pd.Timestamp(t, tz="UTC") for t in scfg["background"])
    junctions = sorted(wn.junction_name_list)
    nodes = list(rng.choice(junctions, int(scfg["junctions"]), replace=False))
    # the leak keeps running after its simulated hours (signature held at its last value); a
    # detector gets ``observe_hours`` to fire (D-038: the re-baselined variant waits 8 h)
    horizon = pd.Timedelta(hours=float(scfg.get("observe_hours", scfg["hours"])))
    # starts: on the 5-minute grid, far enough from the period's ends
    grid = pd.date_range(
        lo + pd.Timedelta(hours=3), hi - horizon - pd.Timedelta(hours=3), freq="5min"
    )
    starts = sorted(grid[rng.choice(len(grid), int(scfg["starts"]), replace=False)])
    return nodes, starts, horizon


def synthetic_leaks(cfg: Mapping[str, Any], spec: FeatureSpec) -> Iterator[dict[str, Any]]:
    """The benchmark's leaks in its order: node, diameter, start and EPANET signature (D-048).

    The same draw as :func:`run` (same seed), so other detectors see exactly the same leaks.
    """
    from leakwatch.data import network
    from leakwatch.demo.signature import leak_signatures

    scfg = cfg["bdt"]["synthetic"]
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    nodes, starts, _ = draw(cfg, rng, network.load_network())
    diameters = [float(d) for d in scfg["diameters_m"]]
    for i, start in enumerate(starts):
        mine = [n for j, n in enumerate(nodes) if j % len(starts) == i]
        leaks = [(n, d) for n in mine for d in diameters]
        for (node, diameter), (delta, flow) in zip(
            leaks, leak_signatures(leaks, start, float(scfg["hours"])), strict=True
        ):
            yield {
                "node": node,
                "diameter_m": diameter,
                "start": start,
                "delta": delta,
                "leak_flow_m3h": flow,
            }


def run(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Simulate the leaks, score them with every residual variant, summarize."""
    from leakwatch.data import network
    from leakwatch.demo.signature import leak_signatures

    t0 = time.perf_counter()
    scfg = cfg["bdt"]["synthetic"]
    spec = load_feature_spec(cfg)
    gt = load_ground_truth()
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    lo, hi = (pd.Timestamp(t, tz="UTC") for t in scfg["background"])
    feats = load_features(int(cfg["bdt"]["train_year"]))
    feats = feats[(feats["window_start"] >= lo) & (feats["window_start"] < hi)]
    feats = feats[normal_mask(feats, spec, gt, cfg, "main")]
    bg = feats.sort_values("window_start").reset_index(drop=True)
    gap = bg["window_start"].diff() != pd.Timedelta(seconds=spec.slide_s)
    contiguous = (~gap).to_numpy()
    contiguous[0] = False

    wn = network.load_network()
    areas = network.node_areas(wn, cfg["network"]["area_anchors"])
    nodes, starts, horizon = draw(cfg, rng, wn)
    diameters = [float(d) for d in scfg["diameters_m"]]
    hours = float(scfg["hours"])

    variants = {}
    for v in residual_variants():
        d = model_dir(MODELS[v])
        if (d / "params.json").exists():
            variants[v] = KMeansScorer(KMeansParams.load(d))
    if not variants:
        raise FileNotFoundError("no trained residual detector: run `make bdt-train`")
    fast = {
        v: fast_threshold(cfg, MODELS[v], float(s.params.threshold))  # type: ignore[arg-type]
        for v, s in variants.items()
    }
    frames = {v: bg for v in variants}
    rebased = [v for v in variants if VARIANTS[v][1] == "rebaselined"]
    if rebased:
        from leakwatch.bdt.train import feature_view

        full = feature_view(load_features(int(cfg["bdt"]["train_year"])), spec, "rebaselined", cfg)
        bg_rb = full.set_index("window_start").reindex(bg["window_start"]).reset_index()
        bg_rb.index = bg.index
        for v in rebased:
            frames[v] = bg_rb

    scenarios: list[dict[str, Any]] = []
    for i, start in enumerate(starts):
        mine = [n for j, n in enumerate(nodes) if j % len(starts) == i]
        leaks = [(n, d) for n in mine for d in diameters]
        sigs = leak_signatures(leaks, start, hours)
        first = start - pd.Timedelta(seconds=spec.window_s - spec.step_s)
        rows = bg[(bg["window_start"] >= first) & (bg["window_start"] <= start + horizon)]
        win_end = rows["window_start"] + pd.Timedelta(seconds=spec.window_s)
        for (node, diameter), (delta, flow) in zip(leaks, sigs, strict=True):
            shift = window_means(delta, rows["window_start"], start, spec)
            rec: dict[str, Any] = {
                "node": node,
                "area": areas.get(node),
                "diameter_m": diameter,
                "start": start.isoformat(),
                "leak_flow_m3h": round(flow, 2),
            }
            for v, scorer in variants.items():
                cols = list(scorer.params.feature_columns)
                x = frames[v].loc[rows.index, cols].to_numpy("float64").copy()
                for j, c in enumerate(cols):
                    series = c.removesuffix("__residual")
                    if series in shift.columns:
                        x[:, j] += shift[series].to_numpy()
                score = scorer.score_matrix(x)["score"]
                above = score > float(scorer.params.threshold)
                fired = alert_starts(
                    above,
                    contiguous[rows.index.to_numpy()],
                    int(scorer.params.persistence_windows),
                    int(cfg["bdt"]["close_after_windows"]),
                    None if fast[v] is None else score > fast[v],
                )
                hits = [k for k in fired if win_end.iloc[k] > start]
                rec[v] = {
                    "detected": bool(hits),
                    "ttd_h": round((win_end.iloc[hits[0]] - start).total_seconds() / 3600, 2)
                    if hits
                    else None,
                }
            scenarios.append(rec)
        log.info("start %d/%d (%s): %d leaks", i + 1, len(starts), start, len(leaks))

    weeks = len(bg) * spec.slide_s / (7 * 86400)
    df = pd.DataFrame(scenarios)
    results = {}
    for v, scorer in variants.items():
        cols = list(scorer.params.feature_columns)
        score = scorer.score_matrix(frames[v][cols].to_numpy("float64"))["score"]
        fa = alert_starts(
            score > float(scorer.params.threshold),
            contiguous,
            int(scorer.params.persistence_windows),
            int(cfg["bdt"]["close_after_windows"]),
            None if fast[v] is None else score > fast[v],
        )
        det = df[v].map(lambda r: r["detected"])
        ttd = df[v].map(lambda r: r["ttd_h"]).dropna()
        results[v] = {
            "model": MODELS[v],
            "score": scorer.params.meta.get("score", "distance"),
            "threshold": float(scorer.params.threshold),
            "persistence_windows": int(scorer.params.persistence_windows),
            "fast_path_threshold": fast[v],
            "detection_rate": round(float(det.mean()), 4),
            "by_diameter_mm": {
                f"{d * 1000:g}": round(float(det[df["diameter_m"] == d].mean()), 4)
                for d in diameters
            },
            "by_area": {
                str(a): {
                    "leaks": int((df["area"] == a).sum()),
                    "rate": round(float(det[df["area"] == a].mean()), 4),
                }
                for a in sorted(df["area"].dropna().unique())
            },
            "median_ttd_h": round(float(ttd.median()), 2) if len(ttd) else None,
            "background_false_alarms": len(fa),
            "background_false_alarms_per_week": round(len(fa) / weeks, 3) if weeks else None,
        }
    return {
        "question": "simulated leaks added to normal BattLeDIM 2018 background (residual features)",
        "background": {
            "period": [lo.isoformat(), hi.isoformat()],
            "normal_windows": len(bg),
            "weeks": round(weeks, 2),
        },
        "scenarios": len(df),
        "junctions": len(nodes),
        "diameters_m": diameters,
        "hours": hours,
        "median_leak_flow_m3h": {
            f"{d * 1000:g} mm": round(
                float(df.loc[df["diameter_m"] == d, "leak_flow_m3h"].median()), 2
            )
            for d in diameters
        },
        "variants": results,
        "per_scenario": scenarios,
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the synthetic-leak test and write the metrics file."""
    argparse.ArgumentParser(description="Simulated leaks on normal 2018 background.").parse_args(
        argv
    )
    cfg = load_config()
    os.chdir(paths.private_scratch_dir())  # EPANET writes temp files into the working directory
    payload = run(cfg)
    path = write_metrics("bdt_synthetic_leaks", payload)
    for v, r in payload["variants"].items():
        log.info(
            "%s: detected %.1f%% of %d leaks; %s false alarms/week on the background -> %s",
            v,
            100 * r["detection_rate"],
            payload["scenarios"],
            r["background_false_alarms_per_week"],
            path,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
