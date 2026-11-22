"""Map geometry and the pipe-tint weights of the control-room screen (D-052, D-053)."""

from __future__ import annotations

import json
import math

import networkx as nx
import pytest

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.dashboard import geometry


def line_graph() -> nx.Graph:
    """a -100 m- b -100 m- c -100 m- d, and a separate pair x - y."""
    g = nx.Graph()
    for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("x", "y")]:
        g.add_edge(u, v, length=100.0)
    return g


AREAS = {"a": "A", "b": "A", "c": "A", "d": "A", "x": "B", "y": "B"}


def weights(sensors: list[str], **kw) -> dict[str, list[list[float]]]:
    params = {"nearest": 4, "length_m": 400.0, "min_distance_m": 30.0, **kw}
    return geometry.tint_weights(line_graph(), AREAS, sensors, **params)


def test_a_sensor_junction_takes_its_own_measurement():
    w = weights(["a", "d"])
    own = dict((int(i), v) for i, v in w["a"])
    assert own[0] == pytest.approx(0.990, abs=0.002)  # 1/30^2 against 1/(300^2 + 30^2)
    assert sum(own.values()) == pytest.approx(1.0, abs=1e-4)  # no fall-off at distance 0


def test_weights_fade_with_the_pipe_distance_to_the_nearest_sensor():
    w = weights(["a", "d"])
    assert sum(v for _, v in w["b"]) == pytest.approx(math.exp(-100 / 400), abs=1e-4)
    assert sum(v for _, v in w["c"]) == pytest.approx(math.exp(-100 / 400), abs=1e-4)
    assert w["b"][0][0] == 0 and w["c"][0][0] == 1  # nearest sensor first
    assert w["b"][0][1] > w["b"][1][1]


def test_only_sensors_of_the_same_zone_and_area_count():
    w = weights(["a", "x"])
    assert [int(i) for i, _ in w["d"]] == [0]
    assert [int(i) for i, _ in w["y"]] == [1]
    assert "y" not in weights(["a"])  # a zone without a sensor gets no tint
    assert weights(["a", "d"], nearest=1)["b"] == [[0, pytest.approx(math.exp(-0.25), abs=1e-4)]]


def test_a_sensor_in_another_area_of_the_same_zone_is_ignored():
    areas = {**AREAS, "d": "C"}
    w = geometry.tint_weights(
        line_graph(), areas, ["a", "d"], nearest=4, length_m=400.0, min_distance_m=30.0
    )
    assert [int(i) for i, _ in w["b"]] == [0]
    assert [int(i) for i, _ in w["d"]] == [1]


@pytest.fixture(scope="module")
def built() -> geometry.Geometry:
    return geometry.build(load_config(environ={}))


@pytest.mark.needs_data
def test_the_map_has_the_network_the_reports_describe(built):
    stats = json.loads((paths.METRICS_DIR / "network_stats.json").read_text("utf-8"))
    p = built.payload
    assert p["counts"]["junctions"] == stats["counts"]["junctions"] == len(p["junctions"])
    assert p["counts"]["pipes"] == stats["counts"]["pipes"]
    assert len(p["pipes"]) == sum(stats["counts"][k] for k in ("pipes", "pumps", "valves"))
    assert p["counts"]["pressure_sensors"] == len(p["sensors"]) == 33
    assert set(p["area_labels"]) == {"A", "B", "C"}
    for pipe in p["pipes"]:
        assert pipe["a"] in p["nodes"] and pipe["b"] in p["nodes"]
    x0, y0, x1, y1 = p["bounds"]
    assert x1 - x0 > 2000 and y1 - y0 > 1000  # metres
    json.dumps(p)  # plain JSON


@pytest.mark.needs_data
def test_every_sensor_weighs_almost_only_itself_and_weights_never_exceed_one(built):
    p = built.payload
    for index, sensor in enumerate(p["sensors"]):
        own = dict((int(i), w) for i, w in p["tint"][sensor])
        assert own[index] > 0.9, sensor
    for node, pairs in p["tint"].items():
        assert node in p["nodes"]
        assert 0 < sum(w for _, w in pairs) <= 1.0001
        assert all(p["areas"][p["sensors"][int(i)]] == p["areas"][node] for i, _ in pairs)


@pytest.mark.needs_data
def test_pipe_distance_follows_the_pipes(built):
    assert built.pipe_distance_m("n470", "n470") == 0
    there, back = built.pipe_distance_m("n469", "n470"), built.pipe_distance_m("n470", "n469")
    assert there == back and 40 < there < 80  # 48.9 m apart in a straight line
    assert built.pipe_distance_m("n470", "no such node") is None


def test_pipe_ends_read_the_pipe_graph():
    g = nx.Graph()
    g.add_edge("n641", "n229", name="p701", length=39.0)
    geo = geometry.Geometry(payload={}, graph=g)
    assert set(geo.pipe_ends("p701")) == {"n229", "n641"}
    assert geo.pipe_ends("p999") is None


@pytest.mark.needs_data
def test_pipe_ends_name_the_two_junctions_of_a_pipe(built):
    assert set(built.pipe_ends("p701")) == {"n229", "n641"}  # 39 m: the owner's n641 run
    assert built.pipe_ends("no such pipe") is None


def test_outline_of_a_grid_runs_at_the_buffer_distance_around_it():
    grid = [(x, y) for x in range(0, 301, 50) for y in range(0, 201, 50)]
    rings = geometry.outline(grid, buffer_m=40, alpha_m=80)
    assert len(rings) == 1  # one outer ring, no holes
    ring = rings[0]
    assert ring[0] == ring[-1]  # closed
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    assert min(xs) == pytest.approx(-40, abs=0.1) and max(xs) == pytest.approx(340, abs=0.1)
    assert min(ys) == pytest.approx(-40, abs=1) and max(ys) == pytest.approx(240, abs=1)
    for x, y in ring:  # every outline point is about the buffer away from the nearest grid point
        nearest = min(math.hypot(x - gx, y - gy) for gx, gy in grid)
        assert nearest == pytest.approx(40, abs=0.5)


def test_outline_keeps_far_apart_groups_apart_and_drops_holes():
    left = [(x, y) for x in range(0, 101, 50) for y in range(0, 101, 50)]
    right = [(x + 1000, y) for x, y in left]
    assert len(geometry.outline(left + right, buffer_m=40, alpha_m=80)) == 2
    frame = [
        (x, y)
        for x in range(0, 601, 50)
        for y in range(0, 601, 50)
        if not 100 < x < 500 or not 100 < y < 500
    ]
    rings = geometry.outline(frame, buffer_m=40, alpha_m=80)
    assert len(rings) == 1 and min(p[0] for p in rings[0]) < 0  # the outer ring, not the hole
    assert geometry.outline([], buffer_m=40, alpha_m=80) == []


def test_area_points_follow_long_pipes_inside_the_area_only():
    nodes = {"a": [0.0, 0.0], "b": [200.0, 0.0], "c": [200.0, 500.0]}
    areas = {"a": "A", "b": "A", "c": "B"}
    pipes = [
        {"a": "a", "b": "b", "kind": "Pipe"},
        {"a": "b", "b": "c", "kind": "Pipe"},  # crosses into another area
    ]
    out = geometry.area_points(nodes, areas, pipes, ["a", "b", "c"], step_m=60)
    assert out["A"][:2] == [[0.0, 0.0], [200.0, 0.0]]
    assert [p[0] for p in out["A"][2:]] == pytest.approx([50.0, 100.0, 150.0])  # 3 points on 200 m
    assert out["B"] == [[200.0, 500.0]]


@pytest.mark.needs_data
def test_every_area_has_an_outline_around_its_junctions(built):
    p = built.payload
    assert set(p["outlines"]) == {"A", "B", "C"}
    for area, rings in p["outlines"].items():
        assert rings and all(r[0] == r[-1] and len(r) > 4 for r in rings), area


@pytest.mark.needs_data
def test_pipe_distance_equals_the_demo_checks_distance(built):
    from leakwatch.demo.check import distance_to

    candidates = ["n469", "n470", "n471", "n94", "n468"]
    expected = distance_to("n470", candidates)
    mine = [built.pipe_distance_m("n470", c) for c in candidates]
    assert mine[0] == pytest.approx(expected["top1_m"])
    assert min(mine) == pytest.approx(expected["best_of_top5_m"])
