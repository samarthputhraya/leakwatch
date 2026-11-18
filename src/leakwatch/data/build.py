"""``make data``: download BattLeDIM, convert to Parquet, build and cross-check the ground truth.

Writes:
- ``data/raw/battledim/*``                 original files (MD5-verified)
- ``data/raw/battledim/parquet/*.parquet``  wide SCADA tables, leak flows, ground truth
- ``reports/metrics/data_summary.json``     counts and data-quality facts
- ``reports/metrics/network_stats.json``    L-Town network facts from ``L-TOWN.inp``
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.common.schemas import SENSOR_TYPES, split_sensor_column, wide_scada_schema
from leakwatch.data import battledim, ground_truth, network, quality

log = get_logger("leakwatch.data.build")  # explicit: __name__ is "__main__" under `python -m`


def scada_quality(year: int, wide: pd.DataFrame, stuck_min_steps: int) -> dict[str, Any]:
    """Counts and data-quality facts for one year of wide SCADA data."""
    steps = wide.index.to_series().diff().dropna()
    types = pd.Series([split_sensor_column(c)[0] for c in wide.columns], index=wide.columns)
    stuck = quality.stuck_summary(wide, min_steps=stuck_min_steps)
    per_type: dict[str, Any] = {}
    for sensor_type in SENSOR_TYPES:
        cols = types.index[types.eq(sensor_type)]
        block = wide[cols]
        per_type[sensor_type] = {
            "sensors": len(cols),
            "readings": int(block.notna().sum().sum()),
            "missing": int(block.isna().sum().sum()),
            "sensors_with_missing": int(block.isna().any().sum()),
            "min": float(block.min().min()),
            "max": float(block.max().max()),
            f"sensors_with_constant_runs_ge_{stuck_min_steps}": int(
                stuck.loc[cols, "stuck_runs"].gt(0).sum()
            ),
            "constant_run_steps": int(stuck.loc[cols, "stuck_steps"].sum()),
            "longest_constant_run_steps": int(stuck.loc[cols, "longest_run"].max()),
        }
    return {
        "year": year,
        "timesteps": len(wide),
        "first_ts": wide.index[0].isoformat(),
        "last_ts": wide.index[-1].isoformat(),
        "regular_5min_steps": bool(steps.eq(battledim.TIMESTEP).all()),
        "sensors": wide.shape[1],
        "readings": int(wide.notna().sum().sum()),
        "missing": int(wide.isna().sum().sum()),
        "by_type": per_type,
    }


def leak_activity(wide: pd.DataFrame, flows: pd.DataFrame) -> dict[str, Any]:
    """How much of the year has leaks flowing, and how big they are next to total inflow.

    Inflow = the two reservoir inlet meters (p227 + p235); PUMP_1 moves water inside the network.
    """
    flows = flows.reindex(wide.index, fill_value=0.0)
    n_active = flows.gt(0).sum(axis=1)
    total_leak = flows.sum(axis=1)
    inflow = wide["flow__p227"] + wide["flow__p235"]
    return {
        "share_of_steps_with_no_leak_flowing": float(n_active.eq(0).mean()),
        "steps_with_no_leak_flowing": int(n_active.eq(0).sum()),
        "max_simultaneous_leaks": int(n_active.max()),
        "mean_simultaneous_leaks": round(float(n_active.mean()), 2),
        "mean_total_leak_outflow_m3h": round(float(total_leak.mean()), 2),
        "mean_inflow_m3h": round(float(inflow.mean()), 2),
        "leak_share_of_inflow": round(float(total_leak.sum() / inflow.sum()), 4),
    }


def ground_truth_summary(gt: pd.DataFrame) -> dict[str, Any]:
    """Leak counts used by the pitch and by the Phase 0 Definition of Done."""
    out: dict[str, Any] = {
        "leaks_total": len(gt),
        "started_by_year": {str(k): int(v) for k, v in gt["year_started"].value_counts().items()},
        "repaired_in_data": int(gt["repaired_in_data"].sum()),
    }
    for year in battledim.YEARS:
        active = ground_truth.active_in_year(gt, year)
        out[f"active_in_{year}"] = {
            "total": len(active),
            "by_type": {str(k): int(v) for k, v in active["type"].value_counts().items()},
            "carried_over_from_previous_year": int((active["year_started"] < year).sum()),
            "repaired_within_year": int(active["end_ts"].dt.year.eq(year).sum()),
        }
    out["peak_outflow_m3h"] = {
        "min": float(gt["peak_outflow_m3h"].min()),
        "median": float(gt["peak_outflow_m3h"].median()),
        "max": float(gt["peak_outflow_m3h"].max()),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the full data build. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Download BattLeDIM, convert to Parquet, build and check the ground truth."
    )
    parser.add_argument("--skip-download", action="store_true", help="use files already on disk")
    args = parser.parse_args(argv)

    cfg = load_config()
    years = [int(y) for y in cfg["battledim"]["years"]]
    log.info("data dir: %s", paths.data_dir())

    if not args.skip_download:
        battledim.download_all(include_optional=bool(cfg["battledim"]["download_optional"]))
    battledim.convert_to_parquet(years)

    raw = paths.battledim_raw_dir()
    leak_flows = {y: battledim.load_leak_flows(y) for y in years}
    gt = ground_truth.build_ground_truth(raw / "dataset_configuration.yaml", leak_flows)
    problems = ground_truth.cross_check(
        gt, fixed_report=raw / "2018_Fixed_Leakages_Report.txt", leak_flows=leak_flows
    )
    if problems:
        for p in problems:
            log.error("ground-truth cross-check: %s", p)
        return 1
    gt.to_parquet(paths.ground_truth_path(), index=False)
    log.info("ground truth: %d leaks -> %s", len(gt), paths.ground_truth_path())

    stuck_min = int(cfg["cleaning"]["stuck_min_steps"])
    per_year, activity = [], {}
    for year in years:
        wide = battledim.load_scada(year)
        wide_scada_schema(expected_rows=battledim.STEPS_PER_YEAR).validate(wide)
        per_year.append(scada_quality(year, wide, stuck_min))
        activity[str(year)] = leak_activity(wide, leak_flows[year])
    summary_path = write_metrics(
        "data_summary",
        {
            "source": f"Zenodo record {battledim.ZENODO_RECORD} (doi:10.5281/zenodo.4017659)",
            "scada": per_year,
            "leak_activity": activity,
            "ground_truth": ground_truth_summary(gt),
            "cross_checks": {
                "fixed_report_2018": "agrees",
                "leak_flow_files": "agree",
                "epyt_flow_leak_configs": "agree",
            },
        },
    )
    log.info("wrote %s", summary_path)

    stats = network.network_stats(network.load_network())
    stats_path = write_metrics("network_stats", stats)
    log.info("wrote %s", stats_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
