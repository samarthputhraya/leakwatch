"""GNN scenario generator (D-024, D-038): perturbed worlds and scenario draws."""

from __future__ import annotations

import math

import numpy as np

from leakwatch.dl import gnn_data

CFG = {
    "leak_diameter_m": [0.005, 0.03],
    "start_hours": [0, 12],
    "roughness_noise": 0.1,
    "background_leaks": [0, 3],
    "background_diameter_m": [0.003, 0.01],
}


def test_worlds_are_reproducible_and_bounded():
    pipes, junctions = [f"p{i}" for i in range(50)], [f"n{i}" for i in range(20)]
    a = gnn_data.make_worlds(5, junctions, pipes, CFG, seed=1)
    b = gnn_data.make_worlds(5, junctions, pipes, CFG, seed=1)
    assert a == b
    for w in a:
        f = np.array(list(w["roughness"].values()))
        assert len(f) == 50 and f.min() >= 0.9 and f.max() <= 1.1
        assert 0 <= len(w["background"]) <= 3
        assert all(n in junctions and 0.003 <= d <= 0.01 for n, d in w["background"])


def test_scenarios_draw_world_node_size_and_start():
    rows = gnn_data.draw_scenarios(200, ["n1", "n2"], CFG, seed=3, worlds=4)
    seeds, worlds, nodes, diam, starts = zip(*rows, strict=True)
    assert len(set(seeds)) == 200 and set(worlds) <= {0, 1, 2, 3}
    assert set(nodes) <= {"n1", "n2"} and set(starts) <= {0, 12 * 3600}
    assert min(diam) >= 0.005 and max(diam) <= 0.03


def test_emitter_is_the_orifice_law():
    assert gnn_data.emitter(0.02, 0.75) == 0.75 * math.pi * 0.01**2 * math.sqrt(2 * 9.81)
