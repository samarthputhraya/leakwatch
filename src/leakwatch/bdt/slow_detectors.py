"""``make bdt-slow``: slow detectors for small leaks, on the ``bdt-synthetic`` leaks (D-048).

The live K-Means model finds 8% of simulated 10 mm leaks (``bdt_synthetic_leaks.json``). Two
established slow detectors are tested on the *same* simulated leaks and normal 2018 background:

A. **Night inflow** (minimum-night-flow analysis): the mean total-inflow residual (reservoir
   inlets, ``bdt.inflow_series``) over the night windows of each day, minus the median of the
   previous ``reference_nights`` nights, divided by its robust spread (MAD) on the 2018 normal
   nights. Tested as a one-night (Shewhart) rule and as a one-sided CUSUM (allowance ``k``).
B. **Per-sensor pressure CUSUM**: a one-sided CUSUM per pressure sensor on its hourly
   re-baselined residual (minus, so a falling pressure counts up), scaled by its MAD on the 2018
   normal hours.

Thresholds are chosen on 2018 normal periods only (D-015): the largest statistic seen there times
``margin``, so each detector raises no false alarm on 2018 normal data by construction. Detection
is counted within the benchmark's ``observe_hours`` (24 h) and, for these slow rules, also within
72 h (the simulated leak keeps running at its last value). Writes
``reports/metrics/bdt_slow_detectors.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from leakwatch.bdt.data import load_features, normal_mask
from leakwatch.bdt.synthetic import synthetic_leaks, window_means
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.data.ground_truth import load_ground_truth
from leakwatch.features.spec import load_feature_spec

log = get_logger("leakwatch.bdt.slow_detectors")


def mad_scale(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """1.4826 × median absolute deviation (a robust standard deviation), NaN ignored."""
    med = np.nanmedian(x, axis=axis, keepdims=axis is not None)
    out = 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)
    return out


def cusum(z: np.ndarray, k: float) -> np.ndarray:
    """One-sided CUSUM ``S_t = max(0, S_{t-1} + z_t − k)`` along axis 0 (NaN counts as 0)."""
    z = np.nan_to_num(np.asarray(z, dtype="float64"))
    s = np.zeros(z.shape[1:]) if z.ndim > 1 else 0.0
    out = np.zeros_like(z)
    for i, row in enumerate(z):
        s = np.maximum(0.0, s + row - k)
        out[i] = s
    return out


def night_values(window_start: pd.Series, values: np.ndarray, hours: Sequence[int]) -> pd.Series:
    """Mean of ``values`` over windows starting in ``[hours[0], hours[1])`` (UTC), per day."""
    ws = pd.to_datetime(window_start, utc=True)
    h = ws.dt.hour.to_numpy()
    sel = (h >= int(hours[0])) & (h < int(hours[1]))
    day = ws[sel].dt.floor("D").to_numpy()
    return pd.Series(np.asarray(values)[sel], index=day).groupby(level=0).mean()


def night_residual(nightly: pd.Series, reference_nights: int) -> pd.Series:
    """Each night minus the median of the previous ``reference_nights`` nights (≥ half present)."""
    ref = nightly.rolling(reference_nights, min_periods=max(1, reference_nights // 2)).median()
    return nightly - ref.shift(1)


def run(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Thresholds on 2018 normal data, detection of the simulated leaks, false alarms."""
    t0 = time.perf_counter()
    scfg = cfg["bdt"]["slow"]
    spec = load_feature_spec(cfg)
    gt = load_ground_truth()
    year = int(cfg["bdt"]["train_year"])
    feats = load_features(year).sort_values("window_start").reset_index(drop=True)
    normal = normal_mask(feats, spec, gt, cfg, "main")
    inflow = [f"{s}__residual" for s in cfg["bdt"]["inflow_series"]]
    hours = [int(h) for h in scfg["night_hours"]]
    n_ref = int(scfg["reference_nights"])
    margin = float(scfg["margin"])
    k_night, k_press = float(scfg["night_cusum_k"]), float(scfg["pressure_cusum_k"])

    # A: night inflow on the real 2018 background
    total = feats[inflow].sum(axis=1).to_numpy()
    base_night = night_values(feats["window_start"], total, hours)
    r = night_residual(base_night, n_ref)
    day = pd.to_datetime(feats["window_start"], utc=True).dt.floor("D").to_numpy()
    night_ok = pd.Series(normal, index=day).groupby(level=0).all().reindex(r.index).fillna(False)
    rn = r[night_ok.to_numpy()].dropna()
    sigma = float(mad_scale(rn.to_numpy()))
    z = (r / sigma).to_numpy()
    c = cusum(z, k_night)
    thr_shewhart = float(np.nanmax(z[night_ok.to_numpy()])) * margin
    thr_cusum = float(np.nanmax(c[night_ok.to_numpy()])) * margin

    # B: per-sensor pressure CUSUM on hourly re-baselined residuals
    from leakwatch.bdt.train import feature_view

    rb = feature_view(feats, spec, "rebaselined", cfg)
    rb = rb.set_index("window_start").reindex(feats["window_start"]).reset_index()
    pcols = [c_ for c_ in rb.columns if c_.startswith("pressure__") and c_.endswith("__residual")]
    hourly = (pd.to_datetime(feats["window_start"], utc=True).dt.minute == 0).to_numpy()
    p_raw = -rb[pcols].to_numpy("float64")[hourly]
    p_sig = np.asarray(mad_scale(p_raw[normal[hourly]], axis=0), dtype="float64")
    p_sig = np.where(p_sig > 0, p_sig, 1.0)
    p_stat = cusum(p_raw / p_sig, k_press).max(axis=1)
    thr_press = float(np.nanmax(p_stat[normal[hourly]])) * margin
    hour_end = pd.to_datetime(feats["window_start"], utc=True)[hourly] + pd.Timedelta(
        seconds=spec.window_s
    )

    nights = pd.DatetimeIndex(r.index)
    nights = nights.tz_localize("UTC") if nights.tz is None else nights.tz_convert("UTC")
    night_end = nights + pd.Timedelta(hours=hours[1] + 1)
    horizons = [float(h) for h in scfg["horizons_h"]]
    rows: list[dict[str, Any]] = []
    for leak in synthetic_leaks(cfg, spec):
        shift = window_means(leak["delta"], feats["window_start"], leak["start"], spec)
        q = shift[[s for s in cfg["bdt"]["inflow_series"] if s in shift]].sum(axis=1).to_numpy()
        z_leak = (
            night_residual(night_values(feats["window_start"], total + q, hours), n_ref) / sigma
        ).to_numpy()
        c_leak = cusum(z_leak, k_night)
        ps = np.zeros_like(p_raw)
        for j, col in enumerate(pcols):
            series = col.removesuffix("__residual")
            if series in shift:
                ps[:, j] = -shift[series].to_numpy()[hourly]
        p_leak = cusum((p_raw + ps) / p_sig, k_press).max(axis=1)
        rec: dict[str, Any] = {
            "node": leak["node"],
            "diameter_m": leak["diameter_m"],
            "start": leak["start"].isoformat(),
            "leak_flow_m3h": round(float(leak["leak_flow_m3h"]), 2),
        }
        for hz in horizons:
            end = leak["start"] + pd.Timedelta(hours=hz)
            nights = (night_end > leak["start"]) & (night_end <= end)
            hrs = ((hour_end > leak["start"]) & (hour_end <= end)).to_numpy()
            rec[f"night_shewhart_{hz:g}h"] = bool(
                (np.nan_to_num(z_leak)[nights] > thr_shewhart).any()
            )
            rec[f"night_cusum_{hz:g}h"] = bool((c_leak[nights] > thr_cusum).any())
            rec[f"pressure_cusum_{hz:g}h"] = bool((p_leak[hrs] > thr_press).any())
        rows.append(rec)
    df = pd.DataFrame(rows)
    diameters = sorted(df["diameter_m"].unique())
    lo, hi = (pd.Timestamp(t, tz="UTC") for t in cfg["bdt"]["synthetic"]["background"])
    bg_nights = (night_end > lo) & (night_end <= hi)
    bg_hours = ((hour_end > lo) & (hour_end <= hi)).to_numpy()
    detectors: dict[str, Any] = {}
    for name in ("night_shewhart", "night_cusum", "pressure_cusum"):
        detectors[name] = {
            f"{hz:g}h": {
                "detection_rate": round(float(df[f"{name}_{hz:g}h"].mean()), 4),
                "by_diameter_mm": {
                    f"{d * 1000:g}": round(
                        float(df.loc[df["diameter_m"] == d, f"{name}_{hz:g}h"].mean()), 4
                    )
                    for d in diameters
                },
            }
            for hz in horizons
        }
    detectors["night_shewhart"]["background_alarms"] = int(
        (np.nan_to_num(z)[bg_nights] > thr_shewhart).sum()
    )
    detectors["night_cusum"]["background_alarms"] = int((c[bg_nights] > thr_cusum).sum())
    detectors["pressure_cusum"]["background_alarms"] = int((p_stat[bg_hours] > thr_press).sum())
    return {
        "question": "can a slow detector find the small simulated leaks the live model misses, "
        "with no false alarm on 2018 normal data?",
        "leaks": len(df),
        "night_inflow": {
            "night_hours_utc": hours,
            "reference_nights": n_ref,
            "normal_nights_2018": len(rn),
            "spread_mad_m3h": round(sigma, 2),
            "median_leak_flow_m3h": {
                f"{d * 1000:g} mm": round(
                    float(df.loc[df["diameter_m"] == d, "leak_flow_m3h"].median()), 2
                )
                for d in diameters
            },
            "thresholds": {"shewhart_z": round(thr_shewhart, 3), "cusum": round(thr_cusum, 3)},
            "cusum_k": k_night,
        },
        "pressure_cusum": {
            "sensors": len(pcols),
            "cusum_k": k_press,
            "threshold": round(thr_press, 3),
        },
        "threshold_rule": f"largest statistic on 2018 normal periods (D-015) x {margin:g}",
        "detectors": detectors,
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and write the metrics file."""
    argparse.ArgumentParser(description="Slow detectors for small leaks (D-048).").parse_args(argv)
    payload = run(load_config())
    path = write_metrics("bdt_slow_detectors", payload)
    log.info("slow detectors: %s -> %s", payload["detectors"], path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
