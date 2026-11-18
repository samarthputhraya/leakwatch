"""BattLeDIM 2020 benchmark: download, parse and convert the original files.

Source: Zenodo record 4017659 (doi:10.5281/zenodo.4017659), see docs/DECISIONS.md D-005 for why
we do not use ``water-benchmark-hub`` / ``epyt-flow`` loaders (their year selection is inverted).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from leakwatch.common import paths
from leakwatch.common.logging import get_logger
from leakwatch.common.schemas import UNITS, sensor_column, split_sensor_column

log = get_logger(__name__)

ZENODO_RECORD = "4017659"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
YEARS: tuple[int, ...] = (2018, 2019)
TIMESTEP = pd.Timedelta(minutes=5)
STEPS_PER_YEAR = 365 * 24 * 12  # 105,120 (neither 2018 nor 2019 is a leap year)

#: BattLeDIM file stem for each sensor type: ``{year}_SCADA_{stem}.csv``.
SCADA_FILE_STEMS: dict[str, str] = {
    "pressure": "Pressures",
    "flow": "Flows",
    "level": "Levels",
    "amr": "Demands",
}


@dataclass(frozen=True)
class RemoteFile:
    """A file in the Zenodo record, with the MD5 published by the Zenodo API."""

    name: str
    md5: str
    required: bool = True


FILES: tuple[RemoteFile, ...] = (
    RemoteFile("README.txt", "7ad5cdc2eebfe78197a3412e80d07ddc"),
    RemoteFile("L-TOWN.inp", "bfbb16b8b463e0576200ba6ad360f68e"),
    RemoteFile("dataset_configuration.yaml", "48486401f5b4d0447023f5ce5d242c52"),
    RemoteFile("2018_Fixed_Leakages_Report.txt", "50b1f8b7ba0db5a19cedbb060c63d912"),
    RemoteFile("2018_SCADA_Pressures.csv", "d389d8541350c19ff0bfc6b80f246d35"),
    RemoteFile("2018_SCADA_Flows.csv", "d0602c06946b46287e956f007e4264ee"),
    RemoteFile("2018_SCADA_Levels.csv", "92a512d6749ddf66fd6d99f2bdde6cc4"),
    RemoteFile("2018_SCADA_Demands.csv", "8803afcb1da2428de169237be3d911fb"),
    RemoteFile("2018_Leakages.csv", "c2c5fab90420da44f02e29775050abfe"),
    RemoteFile("2019_SCADA_Pressures.csv", "5ea1e46d3f2f0a89a3f98d6fd39a851d"),
    RemoteFile("2019_SCADA_Flows.csv", "28fc99fdcbf80fcd26079e7fe602d6dc"),
    RemoteFile("2019_SCADA_Levels.csv", "e5a8050bd38729e4b7648b9d52abd5b1"),
    RemoteFile("2019_SCADA_Demands.csv", "b3e111de397a1b06d33b8a9f13bc997d"),
    RemoteFile("2019_Leakages.csv", "e1f0a43683813a90a9fec8562dde599b"),
    # 177 MB model with the real (hidden) parameters; not needed by the pipeline.
    RemoteFile("L-TOWN_Real.inp", "40fb38de25ed692a5ddd30e0f5fae332", required=False),
)


class ChecksumError(RuntimeError):
    """A downloaded file does not match its published MD5."""


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex MD5 of a file."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    remote: RemoteFile,
    dest_dir: Path,
    session: requests.Session | None = None,
    retries: int = 5,
    timeout_s: float = 120.0,
    max_backoff_s: float = 60.0,
) -> Path:
    """Download one file into ``dest_dir`` unless a copy with the right MD5 is already there.

    Zenodo regularly answers with 502s and read timeouts, so failed attempts are retried with
    exponential backoff (2, 4, 8, ... seconds, capped at ``max_backoff_s``).
    """
    dest = dest_dir / remote.name
    if dest.exists() and md5sum(dest) == remote.md5:
        log.info("ok (cached)  %s", remote.name)
        return dest
    session = session or requests.Session()
    url = f"{BASE_URL}/{remote.name}?download=1"
    part = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout_s) as resp:
                resp.raise_for_status()
                with part.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            got = md5sum(part)
            if got != remote.md5:
                raise ChecksumError(f"{remote.name}: md5 {got} != expected {remote.md5}")
            part.replace(dest)
            log.info("downloaded   %s (%.1f MB)", remote.name, dest.stat().st_size / 1e6)
            return dest
        except (requests.RequestException, ChecksumError) as exc:
            last_error = exc
            log.warning("attempt %d/%d failed for %s: %s", attempt, retries, remote.name, exc)
            if attempt < retries:
                time.sleep(min(2.0**attempt, max_backoff_s))
    part.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {remote.name}") from last_error


def download_all(
    dest_dir: Path | None = None,
    include_optional: bool = False,
    retries: int = 5,
    timeout_s: float = 120.0,
) -> list[Path]:
    """Download every required BattLeDIM file (and optional ones if asked)."""
    target_dir: Path = paths.ensure_dir(dest_dir or paths.battledim_raw_dir())
    with requests.Session() as session:
        return [
            download_file(f, target_dir, session, retries=retries, timeout_s=timeout_s)
            for f in FILES
            if f.required or include_optional
        ]


# --------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------


def read_battledim_csv(path: Path) -> pd.DataFrame:
    """Read a BattLeDIM CSV (``;`` separator, ``,`` decimal) into a float frame with a UTC index.

    BattLeDIM timestamps carry no timezone; L-Town is fictional, so we define them as UTC.
    """
    df = pd.read_csv(path, sep=";", decimal=",", dtype={"Timestamp": str})
    ts = pd.to_datetime(df.pop("Timestamp"), format="%Y-%m-%d %H:%M:%S").dt.tz_localize("UTC")
    df.index = pd.DatetimeIndex(ts, name="ts")
    df.columns = [str(c).strip() for c in df.columns]
    return df.astype("float64")


def scada_csv_path(year: int, sensor_type: str, raw_dir: Path | None = None) -> Path:
    """Path of the original CSV for one year and sensor type."""
    raw_dir = raw_dir or paths.battledim_raw_dir()
    return raw_dir / f"{year}_SCADA_{SCADA_FILE_STEMS[sensor_type]}.csv"


def read_scada_year(year: int, raw_dir: Path | None = None) -> pd.DataFrame:
    """Read all four SCADA CSVs of one year into one wide frame (``pressure__n54`` columns)."""
    frames = []
    for sensor_type in SCADA_FILE_STEMS:
        df = read_battledim_csv(scada_csv_path(year, sensor_type, raw_dir))
        df.columns = [sensor_column(sensor_type, c) for c in df.columns]
        frames.append(df)
    wide = pd.concat(frames, axis=1, join="outer").sort_index()
    if len(wide) != len(frames[0]):
        raise ValueError(f"{year}: sensor files do not share the same timestamps")
    return wide


def read_leak_flows(year: int, raw_dir: Path | None = None) -> pd.DataFrame:
    """Per-leak outflow (m3/h) for one year: UTC index, one column per leaking pipe."""
    raw_dir = raw_dir or paths.battledim_raw_dir()
    return read_battledim_csv(raw_dir / f"{year}_Leakages.csv")


def sensor_catalog(columns: Iterable[str]) -> pd.DataFrame:
    """Describe wide-table columns: ``column, sensor_type, sensor_id, unit``."""
    rows = []
    for col in columns:
        sensor_type, sensor_id = split_sensor_column(col)
        rows.append(
            {
                "column": col,
                "sensor_type": sensor_type,
                "sensor_id": sensor_id,
                "unit": UNITS[sensor_type],
            }
        )
    return pd.DataFrame(rows, columns=["column", "sensor_type", "sensor_id", "unit"])


def to_long(
    wide: pd.DataFrame,
    source: str = "battledim",
    scenario_id: str | None = None,
) -> pd.DataFrame:
    """Convert a wide SCADA frame into long rows matching the ``scada.readings`` contract.

    Missing values are dropped (a missing reading is not published). Rows are ordered by
    timestamp, then by column order, which is the order the replayer publishes them in.
    """
    catalog = sensor_catalog(wide.columns).set_index("column")
    long = (
        wide.rename_axis("ts")
        .reset_index()
        .melt(id_vars="ts", var_name="column", value_name="value")
        .dropna(subset=["value"])
    )
    long["sensor_type"] = long["column"].map(catalog["sensor_type"])
    long["sensor_id"] = long["column"].map(catalog["sensor_id"])
    long["unit"] = long["sensor_type"].map(UNITS)
    long["source"] = source
    long["scenario_id"] = scenario_id
    long["year"] = long["ts"].dt.year.astype("int64")
    order = {col: i for i, col in enumerate(wide.columns)}
    long["_col_order"] = long["column"].map(order)
    long = long.sort_values(["ts", "_col_order"], kind="stable").reset_index(drop=True)
    return long[
        ["ts", "sensor_id", "sensor_type", "value", "unit", "source", "scenario_id", "year"]
    ]


# --------------------------------------------------------------------------------------------
# Parquet cache
# --------------------------------------------------------------------------------------------


def convert_to_parquet(years: Iterable[int] = YEARS, raw_dir: Path | None = None) -> list[Path]:
    """Convert each year's SCADA CSVs and leak flows to Parquet. Returns the written paths."""
    out_dir = paths.ensure_dir(paths.battledim_parquet_dir())
    written = []
    for year in years:
        wide = read_scada_year(year, raw_dir)
        target = paths.scada_parquet_path(year)
        wide.to_parquet(target, compression="zstd")
        written.append(target)
        flows = read_leak_flows(year, raw_dir)
        flows_target = paths.leak_flows_parquet_path(year)
        flows.to_parquet(flows_target, compression="zstd")
        written.append(flows_target)
        log.info("converted    %d: %d steps x %d sensors -> %s", year, *wide.shape, target.name)
    log.debug("parquet dir %s", out_dir)
    return written


def load_scada(year: int) -> pd.DataFrame:
    """Load the wide SCADA table for one year from the Parquet cache (run ``make data`` first)."""
    path = paths.scada_parquet_path(year)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `make data`")
    return pd.read_parquet(path)


def load_leak_flows(year: int) -> pd.DataFrame:
    """Load per-leak outflows (m3/h) for one year from the Parquet cache."""
    path = paths.leak_flows_parquet_path(year)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `make data`")
    return pd.read_parquet(path)
