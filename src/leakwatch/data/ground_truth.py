"""BattLeDIM leak ground truth as one tidy table (PRD §4.1 step 4).

Primary source: the official ``dataset_configuration.yaml`` (Zenodo), which lists all 33 leaks of
2018–2019 with link ID, start, end, diameter, type and peak time. A leak whose end time equals the
last timestamp of the data (2019-12-31 23:55) was never repaired within the data, so its
``end_ts`` is null. Peak outflow comes from the published leak-flow series (``*_Leakages.csv``).

Cross-checks (see :func:`cross_check`): the 2018 repair report and the per-year leak configs
shipped with epyt-flow must agree with the table.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import yaml

from leakwatch.common import paths
from leakwatch.common.logging import get_logger
from leakwatch.common.schemas import GROUND_TRUTH_SCHEMA

log = get_logger(__name__)

TS_FORMAT = "%Y-%m-%d %H:%M"
DATA_START = pd.Timestamp("2018-01-01 00:00", tz="UTC")
DATA_END = pd.Timestamp("2019-12-31 23:55", tz="UTC")
COLUMNS = list(GROUND_TRUTH_SCHEMA.columns)


def _ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(text.strip(), format=TS_FORMAT)).tz_localize("UTC")


def parse_leak_line(line: str) -> dict[str, object]:
    """Parse one leak line: ``pipe, start, end, diameter_m, type, peak`` (times ``%Y-%m-%d %H:%M``).

    Example: ``"p257, 2018-01-08 13:30, 2019-12-31 23:55, 0.011843, incipient, 2018-01-25 08:30"``.
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        raise ValueError(f"expected 6 fields, got {len(parts)}: {line!r}")
    pipe_id, start, end, diameter, leak_type, peak = parts
    if leak_type not in ("abrupt", "incipient"):
        raise ValueError(f"unknown leak type {leak_type!r} in {line!r}")
    return {
        "pipe_id": pipe_id,
        "start": _ts(start),
        "end": _ts(end),
        "diameter_m": float(diameter),
        "type": leak_type,
        "peak": _ts(peak),
    }


def parse_dataset_configuration(path: Path) -> list[dict[str, object]]:
    """Return the parsed ``leakages`` entries of the official dataset configuration."""
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # The first list item is a header comment ("- # linkID, ..."), which YAML reads as null.
    return [parse_leak_line(item) for item in cfg["leakages"] if isinstance(item, str)]


def parse_fixed_report(path: Path) -> pd.DataFrame:
    """Parse ``2018_Fixed_Leakages_Report.txt`` into ``pipe_id, end_ts``."""
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("-"):
            continue
        pipe_id, end = (p.strip() for p in line.lstrip("- ").split(",", 1))
        rows.append({"pipe_id": pipe_id, "end_ts": _ts(end)})
    return pd.DataFrame(rows, columns=["pipe_id", "end_ts"])


def peak_outflows(leak_flows: Mapping[int, pd.DataFrame]) -> dict[str, float]:
    """Maximum published outflow (m3/h) of each leaking pipe across the given years."""
    peaks: dict[str, float] = {}
    for flows in leak_flows.values():
        for pipe_id, series in flows.items():
            peaks[str(pipe_id)] = max(peaks.get(str(pipe_id), 0.0), float(series.max()))
    return peaks


def build_ground_truth(
    config_path: Path,
    leak_flows: Mapping[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build the validated ground-truth table.

    Args:
        config_path: official ``dataset_configuration.yaml``.
        leak_flows: optional ``{year: leak-flow frame}``; fills ``peak_outflow_m3h`` when given.

    Returns:
        One row per leak, columns as in ``GROUND_TRUTH_SCHEMA``, sorted by start time.

    """
    peaks = peak_outflows(leak_flows) if leak_flows else {}
    rows = []
    for leak in parse_dataset_configuration(config_path):
        start: pd.Timestamp = leak["start"]  # type: ignore[assignment]
        end: pd.Timestamp = leak["end"]  # type: ignore[assignment]
        repaired = end < DATA_END
        rows.append(
            {
                "leak_id": f"L{start.year}_{leak['pipe_id']}",
                "pipe_id": leak["pipe_id"],
                "start_ts": start,
                "end_ts": end if repaired else pd.NaT,
                "peak_ts": leak["peak"],
                "type": leak["type"],
                "diameter_m": leak["diameter_m"],
                "peak_outflow_m3h": peaks.get(str(leak["pipe_id"]), float("nan")),
                "year_started": int(start.year),
                "repaired_in_data": bool(repaired),
            }
        )
    gt = pd.DataFrame(rows, columns=COLUMNS)
    gt["end_ts"] = pd.to_datetime(gt["end_ts"], utc=True)
    gt["year_started"] = gt["year_started"].astype("int64")
    gt["peak_outflow_m3h"] = gt["peak_outflow_m3h"].astype("float64")
    gt = gt.sort_values(["start_ts", "pipe_id"]).reset_index(drop=True)
    return GROUND_TRUTH_SCHEMA.validate(gt)


def active_between(gt: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Leaks leaking at any moment in ``[start, end]`` (inclusive)."""
    mask = (gt["start_ts"] <= end) & (gt["end_ts"].isna() | (gt["end_ts"] >= start))
    return gt.loc[mask].reset_index(drop=True)


def active_in_year(gt: pd.DataFrame, year: int) -> pd.DataFrame:
    """Leaks leaking at any moment during calendar ``year`` (5-minute data ends at 23:55)."""
    start = pd.Timestamp(f"{year}-01-01 00:00", tz="UTC")
    end = pd.Timestamp(f"{year}-12-31 23:55", tz="UTC")
    return active_between(gt, start, end)


def _epyt_flow_configs() -> dict[int, list[dict[str, object]]]:
    # Imported lazily: importing epyt_flow takes a few seconds.
    from epyt_flow.data.benchmarks.battledim_data import (
        LEAKS_CONFIG_TEST,
        LEAKS_CONFIG_TRAIN,
    )

    return {
        2018: [parse_leak_line(s) for s in LEAKS_CONFIG_TRAIN.splitlines() if s.strip()],
        2019: [parse_leak_line(s) for s in LEAKS_CONFIG_TEST.splitlines() if s.strip()],
    }


def cross_check(
    gt: pd.DataFrame,
    fixed_report: Path | None = None,
    leak_flows: Mapping[int, pd.DataFrame] | None = None,
    epyt_flow_configs: Mapping[int, list[dict[str, object]]] | None = None,
) -> list[str]:
    """Compare the table with independent sources. Returns a list of problems (empty = agree).

    - ``2018_Fixed_Leakages_Report.txt``: every listed repair time equals ``end_ts``.
    - Leak-flow files: the leaking pipes of each year are exactly the leaks active that year.
    - epyt-flow per-year configs: same leaks, sizes, types; per-year start/end clipped to the year.
    """
    problems: list[str] = []
    by_pipe = gt.set_index("pipe_id")

    if fixed_report is not None:
        report = parse_fixed_report(fixed_report)
        for row in report.itertuples(index=False):
            if row.pipe_id not in by_pipe.index:
                problems.append(f"fixed report: {row.pipe_id} not in ground truth")
            elif by_pipe.at[row.pipe_id, "end_ts"] != row.end_ts:
                problems.append(f"fixed report: {row.pipe_id} end {row.end_ts} differs")
        repaired_2018 = set(gt.loc[gt["end_ts"].dt.year.eq(2018), "pipe_id"])
        if repaired_2018 != set(report["pipe_id"]):
            problems.append(
                f"fixed report lists {sorted(set(report['pipe_id']))}, "
                f"ground truth repaired in 2018: {sorted(repaired_2018)}"
            )

    if leak_flows is not None:
        for year, flows in leak_flows.items():
            expected = set(active_in_year(gt, year)["pipe_id"])
            if set(map(str, flows.columns)) != expected:
                problems.append(f"{year} leak flows columns differ from leaks active in {year}")

    configs = epyt_flow_configs if epyt_flow_configs is not None else _epyt_flow_configs()
    for year, entries in configs.items():
        year_start = pd.Timestamp(f"{year}-01-01 00:00", tz="UTC")
        year_end = pd.Timestamp(f"{year}-12-31 23:55", tz="UTC")
        active = active_in_year(gt, year).set_index("pipe_id")
        seen = set()
        for e in entries:
            pipe = str(e["pipe_id"])
            seen.add(pipe)
            if pipe not in active.index:
                problems.append(f"epyt-flow {year}: {pipe} not active in ground truth")
                continue
            row = active.loc[pipe]
            want_start = max(row["start_ts"], year_start)
            want_end = year_end if pd.isna(row["end_ts"]) else min(row["end_ts"], year_end)
            checks = {
                "start": (e["start"], want_start),
                "end": (e["end"], want_end),
                "diameter": (e["diameter_m"], row["diameter_m"]),
                "type": (e["type"], row["type"]),
            }
            for field, (got, want) in checks.items():
                if got != want:
                    problems.append(f"epyt-flow {year}: {pipe} {field} {got} != {want}")
        missing = set(active.index) - seen
        if missing:
            problems.append(f"epyt-flow {year}: missing {sorted(missing)}")
    return problems


def load_ground_truth() -> pd.DataFrame:
    """Load the ground-truth table written by ``make data``."""
    path = paths.ground_truth_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `make data`")
    return GROUND_TRUTH_SCHEMA.validate(pd.read_parquet(path))
