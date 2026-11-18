from __future__ import annotations

from pathlib import Path

from leakwatch.common.config import deep_merge, env_overrides, load_config


def test_deep_merge_is_recursive_and_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}, "d": [1, 2]}
    override = {"a": {"b": 10}, "d": [3]}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 10, "c": 2}, "d": [3]}
    assert base == {"a": {"b": 1, "c": 2}, "d": [1, 2]}


def test_env_overrides_parse_nested_keys_and_yaml_values():
    env = {
        "LEAKWATCH__SIM__BENCH__DURATIONS_HOURS": "[24]",
        "LEAKWATCH__PROJECT__SEED": "7",
        "LEAKWATCH_DATA_DIR": "/ignored",  # single underscore: not an override
        "OTHER": "x",
    }
    assert env_overrides(env) == {
        "sim": {"bench": {"durations_hours": [24]}},
        "project": {"seed": 7},
    }


def test_load_config_defaults_are_laptop_safe():
    cfg = load_config(environ={})
    assert cfg["project"]["seed"] == 42
    assert cfg["battledim"]["years"] == [2018, 2019]
    assert cfg["battledim"]["download_optional"] is False
    assert cfg["network"]["area_anchors"] == {"A": "n54", "B": "n215", "C": "T1"}


def test_overlay_then_env_precedence(tmp_path: Path):
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("project:\n  seed: 1\nsim:\n  bench:\n    repeats: 5\n", encoding="utf-8")
    cfg = load_config(environ={"LEAKWATCH_CONFIG": str(overlay), "LEAKWATCH__PROJECT__SEED": "99"})
    assert cfg["sim"]["bench"]["repeats"] == 5  # from overlay
    assert cfg["project"]["seed"] == 99  # env beats overlay
    assert cfg["sim"]["bench"]["hydraulic_timestep_s"] == 300  # default kept


def test_full_overlay_merges():
    cfg = load_config("configs/full.yaml", environ={})
    assert cfg["battledim"]["download_optional"] is True
    assert cfg["sim"]["bench"]["durations_hours"] == [48, 168]
