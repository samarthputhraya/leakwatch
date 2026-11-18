from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leakwatch.common.schemas import READINGS_SCHEMA
from leakwatch.data import battledim


def test_read_csv_decimal_comma_and_utc_index(fixtures_dir: Path):
    df = battledim.read_battledim_csv(fixtures_dir / "sample_pressures.csv")
    assert list(df.columns) == ["n1", "n54", "n215"]
    assert str(df.index.tz) == "UTC"
    assert df.index.name == "ts"
    assert df.loc["2018-01-01 00:05:00+00:00", "n1"] == pytest.approx(28.94)
    assert df.loc["2018-01-01 00:15:00+00:00", "n1"] == pytest.approx(29.0)
    assert np.isnan(df.loc["2018-01-01 00:10:00+00:00", "n54"])
    assert (df.dtypes == "float64").all()


def test_to_long_matches_readings_contract(fixtures_dir: Path):
    wide = battledim.read_battledim_csv(fixtures_dir / "sample_pressures.csv")
    wide.columns = [f"pressure__{c}" for c in wide.columns]
    long = battledim.to_long(wide)
    READINGS_SCHEMA.validate(long)
    assert len(long) == wide.notna().sum().sum() == 11  # one missing value dropped
    # published in timestamp order, then column order
    assert long["ts"].is_monotonic_increasing
    first = long[long["ts"] == long["ts"].iloc[0]]
    assert list(first["sensor_id"]) == ["n1", "n54", "n215"]
    assert set(long["unit"]) == {"m"}
    assert set(long["year"]) == {2018}


def test_sensor_catalog():
    cat = battledim.sensor_catalog(["pressure__n1", "amr__n1", "flow__PUMP_1", "level__T1"])
    assert list(cat["sensor_type"]) == ["pressure", "amr", "flow", "level"]
    assert list(cat["unit"]) == ["m", "L/h", "m3/h", "m"]


def test_md5sum(tmp_path: Path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"leakwatch")
    assert battledim.md5sum(f) == hashlib.md5(b"leakwatch", usedforsecurity=False).hexdigest()


def test_download_skips_network_when_cached_copy_is_valid(tmp_path: Path):
    content = b"cached"
    (tmp_path / "f.txt").write_bytes(content)
    remote = battledim.RemoteFile("f.txt", hashlib.md5(content, usedforsecurity=False).hexdigest())

    class NoNetwork:
        def get(self, *args, **kwargs):
            raise AssertionError("network must not be used for a valid cached file")

    assert battledim.download_file(remote, tmp_path, session=NoNetwork()) == tmp_path / "f.txt"


def test_download_rejects_checksum_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(battledim.time, "sleep", lambda s: None)

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"tampered"

    class Session:
        def get(self, *args, **kwargs):
            return Resp()

    remote = battledim.RemoteFile("g.txt", "0" * 32)
    with pytest.raises(RuntimeError, match="could not download"):
        battledim.download_file(remote, tmp_path, session=Session(), retries=2)
    assert not (tmp_path / "g.txt").exists()
    assert not (tmp_path / "g.txt.part").exists()


def test_file_list_covers_both_years_and_all_sensor_types():
    names = {f.name for f in battledim.FILES if f.required}
    for year in battledim.YEARS:
        for stem in battledim.SCADA_FILE_STEMS.values():
            assert f"{year}_SCADA_{stem}.csv" in names
        assert f"{year}_Leakages.csv" in names
    assert {"L-TOWN.inp", "dataset_configuration.yaml", "2018_Fixed_Leakages_Report.txt"} <= names


def test_fixture_md5s_match_published(official_config: Path, fixed_report: Path):
    published = {f.name: f.md5 for f in battledim.FILES}
    assert battledim.md5sum(official_config) == published["dataset_configuration.yaml"]
    assert battledim.md5sum(fixed_report) == published["2018_Fixed_Leakages_Report.txt"]


def test_steps_per_year():
    idx = pd.date_range("2019-01-01", "2019-12-31 23:55", freq="5min")
    assert len(idx) == battledim.STEPS_PER_YEAR == 105_120
