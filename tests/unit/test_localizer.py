"""System localizer (cue switches, fusion) and the D-044 selection rule."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from leakwatch.dl import gnn
from leakwatch.dl.inversion import Inversion
from leakwatch.dl.localizer import (
    choose,
    choose_official_only,
    mean_ranks,
    sim_metrics,
    wilson,
)


def line_graph(n: int = 6) -> gnn.GraphData:
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    sensors = np.array([0, n // 2, n - 1])
    d = np.abs(np.arange(n)[:, None] - sensors[None, :]) * 100.0
    return gnn.GraphData(
        {
            "sensitivity": (-np.exp(-d / 150.0)).astype("float32"),
            "nodes": np.array([f"j{i}" for i in range(n)]),
            "edge_index": np.array([src, dst]),
            "edge_length_m": np.full(len(src), 100.0, dtype="float32"),
            "edge_diameter_m": np.full(len(src), 0.2, dtype="float32"),
            "elevation_m": np.linspace(0, 10, n).astype("float32"),
            "base_demand_m3s": np.full(n, 0.001, dtype="float32"),
            "is_junction": np.ones(n, dtype=bool),
            "sensor_idx": sensors,
        }
    )


class Uniform(gnn.MLPLocalizer):
    """A 'network' with no opinion: equal logits everywhere."""

    def forward(self, sensor_x, **_):
        return 0.0 * super().forward(sensor_x)


def system(cfg, g=None):
    g = g or line_graph()
    raw = np.asarray(
        -np.exp(-np.abs(np.arange(g.n)[:, None] - g.sensor_idx[None, :]) * 100.0 / 150.0)
    )
    inv = Inversion(raw, np.eye(3), 1e-6, g.is_junction)
    model = Uniform(6, g.n, 4)
    return gnn.SystemLocalizer(cfg, loaded=(g, model, 1.0, {"model": "uniform"}), inversion=inv)


def leak_at(node: int, g: gnn.GraphData) -> np.ndarray:
    s = -np.exp(-np.abs(node - g.sensor_idx) * 100.0 / 150.0)
    return np.repeat(s[None, :, None], 2, axis=2).astype("float32")


def test_gls_alone_finds_the_leak_node():
    loc = system({"gls": "sim"})
    x = leak_at(2, loc.g)
    assert int(np.argmax(loc.scores(x))) == 2
    p = loc.probs(x)
    assert p.sum() == pytest.approx(1.0) and int(np.argmax(p)) == 2


def test_cues_follow_the_switches():
    loc = system({"gls": "sim", "network": True, "sensor_prior_m": 300, "interpolation": True})
    assert set(loc.cues(leak_at(1, loc.g))) == {"gls", "network", "prior", "interpolation"}
    with pytest.raises(ValueError, match="every cue off"):
        system({"gls": None, "network": False}).scores(leak_at(1, line_graph()))


def test_rank_fusion_weights_sum_to_one():
    loc = system({"gls": "sim", "sensor_prior_m": 300, "fusion": "rank"})
    p = loc.probs(leak_at(4, loc.g))
    assert p.sum() == pytest.approx(1.0) and (p >= 0).all()


def test_previous_posterior_is_the_default_config():
    cfg: dict = {"dl": {"gnn": {"sensor_prior_m": 300}}}
    assert gnn.localizer_config(cfg) == {
        "gls": None,
        "network": True,
        "sensor_prior_m": 300.0,
        "fusion": "product",
    }
    cfg["dl"]["localizer"] = {"gls": "sim"}
    assert gnn.localizer_config(cfg) == {"gls": "sim"}


def test_split_is_reproducible_and_holds_out_unseen_sizes():
    diam = np.array([0.005, 0.015, 0.025] * 40)
    gcfg = {"unseen_diameter_m": [0.014, 0.017]}
    a = gnn.split_indices(diam, gcfg, 7)
    b = gnn.split_indices(diam, gcfg, 7)
    for u, v in zip(a, b, strict=True):
        np.testing.assert_array_equal(u, v)
    tr, va, te, un = a
    assert len(un) == 40 and (diam[un] == 0.015).all()
    assert len(set(tr) | set(va) | set(te)) == 80 and not set(tr) & set(te)


def test_wilson_interval():
    lo, hi = wilson(0.7, 382)
    assert lo < 0.7 < hi and hi - lo < 0.1
    assert all(np.isnan(wilson(0.5, 0)))


def test_sim_metrics_has_the_prd_bucket():
    g = line_graph()
    sc = np.tile(np.arange(g.n, dtype="float64"), (3, 1))  # always ranks node 5 first
    y = np.array([5, 0, 5])
    diam = np.array([0.005, 0.012, 0.025])
    m = sim_metrics(g, sc, y, diam, [0.010, 0.020], 300.0)
    assert m["medium_large"]["n"] == 2 and m["medium_large"]["top5"] == pytest.approx(0.5)
    assert set(m["by_size"]) == {"small", "medium", "large"}
    assert len(m["medium_large"]["top5_ci95"]) == 2


def cand(sims: float, recorded: float, eur: float) -> dict:
    return {
        "sims": {"medium_large": {"top5": sims}},
        "recorded_2018": {"within_m_top1": recorded},
        "official_2018": {"total_score_eur": eur},
    }


def test_mean_rank_rule_and_the_official_only_rule():
    c = {
        "a": cand(0.70, 0.36, 75_000),  # ranks 1, 2, 2
        "b": cand(0.35, 0.29, 78_000),  # ranks 3, 3, 1
        "c": cand(0.63, 0.43, 45_000),  # ranks 2, 1, 3
    }
    assert mean_ranks(c) == {"a": 1.667, "b": 2.333, "c": 2.0}
    assert choose(c) == "a"
    assert choose_official_only(c) == "b"


def test_ties_share_a_rank_and_fall_back_to_the_official_score():
    c = {"a": cand(0.5, 0.5, 10.0), "b": cand(0.5, 0.5, 20.0)}
    assert mean_ranks(c) == {"a": 1.333, "b": 1.0}
    c = {"a": cand(0.6, 0.4, 10.0), "b": cand(0.4, 0.6, 10.0)}
    assert mean_ranks(c)["a"] == mean_ranks(c)["b"]
    assert choose(c) in ("a", "b")
