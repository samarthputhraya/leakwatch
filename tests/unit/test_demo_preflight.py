"""Demo preflight: missing prerequisites stop the demo with the command that fixes them."""

from __future__ import annotations

from leakwatch.demo import preflight


def test_empty_data_dir_fails_and_names_the_fix(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("LEAKWATCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(preflight, "_kafka_up", lambda: True)
    with caplog.at_level("INFO"):
        assert preflight.main() == 1
    text = caplog.text
    assert "ok       Kafka broker" in text
    assert "make data" in text
    assert "make prepare" in text


def test_all_required_present_passes_without_the_optional_localizer(tmp_path, monkeypatch):
    from leakwatch.common import paths
    from leakwatch.common.config import load_config

    monkeypatch.setenv("LEAKWATCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(preflight, "_kafka_up", lambda: True)
    cfg = load_config()
    year = int(str(cfg["demo"]["background_start"])[:4])
    scada = paths.scada_parquet_path(year)
    scada.parent.mkdir(parents=True)
    scada.write_bytes(b"x")
    base = paths.lake_baselines_dir()
    base.mkdir(parents=True)
    (base / "part-0.parquet").write_bytes(b"x")
    model = paths.models_dir() / str(cfg["bdt"]["live_model"])
    model.mkdir(parents=True)
    (model / "params.json").write_text("{}")
    assert preflight.main() == 0
