"""Overlapping-leak benchmark of the localizer candidates (D-051)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from leakwatch.dl.multileak import choose, draw, report_metrics

CFG = {
    "dl": {
        "multileak": {
            "leak_counts": [1, 3],
            "scenarios_per_count": 4,
            "reference_pairs": 5,
            "seed_offset": 7,
        },
        "gnn_data": {"start_hours": [0, 12], "leak_diameter_m": [0.005, 0.03]},
    }
}


def test_draw_is_seeded_and_gives_distinct_leaks_per_scenario():
    junctions = [f"j{i}" for i in range(20)]
    a = draw(CFG, junctions, 3, 42)
    assert a == draw(CFG, junctions, 3, 42)
    pairs, todo = a
    assert len(todo) == 8 and all((w, s) in pairs for _, w, s, _ in todo)
    counts = [len(leaks) for *_, leaks in todo]
    assert counts == [1] * 4 + [3] * 4
    for *_, leaks in todo:
        assert len({n for n, _ in leaks}) == len(leaks)
        assert all(0.005 <= d <= 0.03 for _, d in leaks)
    assert len({t[0] for t in todo}) == len(todo)  # one seed per scenario


def test_report_metrics_counts_a_hit_on_any_new_leak():
    dist = np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]) * 200.0
    scores = np.array([[0, 0, 0, 0, 0, 9.0], [9.0, 0, 0, 0, 0, 0]])  # top-1: node 5, then node 0
    m = report_metrics(dist, scores, [[4, 0], [3, 5]], largest=[0, 3], within=300.0)
    assert m["within_m_of_a_leak_top1"] == 0.5  # 5 is 200 m from 4; 0 is 600 m from 3
    assert m["within_m_of_the_largest_top1"] == 0.0
    assert m["within_m_of_a_leak_best_of_top5"] == 1.0
    assert m["n"] == 2


def test_rule_prefers_top1_then_best_of_5_then_the_d044_rank():
    p = {
        "a": {"within_m_of_a_leak_top1": 0.4, "within_m_of_a_leak_best_of_top5": 0.5},
        "b": {"within_m_of_a_leak_top1": 0.5, "within_m_of_a_leak_best_of_top5": 0.5},
    }
    assert choose(p, {}) == "b"
    p["a"]["within_m_of_a_leak_top1"] = 0.5
    p["a"]["within_m_of_a_leak_best_of_top5"] = 0.6
    assert choose(p, {}) == "a"
    p["b"]["within_m_of_a_leak_best_of_top5"] = 0.6
    assert choose(p, {"a": 3.0, "b": 2.0}) == "b"
