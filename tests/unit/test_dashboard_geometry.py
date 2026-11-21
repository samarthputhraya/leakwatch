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
