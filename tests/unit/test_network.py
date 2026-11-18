"""L-Town network facts (needs `L-TOWN.inp` from `make data`)."""

from __future__ import annotations

import networkx as nx
import pytest

from leakwatch.data import network

pytestmark = pytest.mark.needs_data


@pytest.fixture(scope="module")
def wn():
    return network.load_network()


@pytest.fixture(scope="module")
def areas(wn):
    return network.node_areas(wn)


def test_counts(wn):
    assert wn.num_junctions == 782
    assert wn.num_pipes == 905
    assert (wn.num_reservoirs, wn.num_tanks, wn.num_pumps, wn.num_valves) == (2, 1, 1, 3)


def test_every_node_and_pipe_has_an_area(wn, areas):
    assert set(areas.index) == set(wn.node_name_list)
    assert set(areas.unique()) == {"A", "B", "C"}
    pipe_areas = network.pipe_areas(wn, areas)
    assert len(pipe_areas) == wn.num_pipes


def test_area_sizes_and_sensors(wn, areas):
    sensors = network.parse_sensor_config()
    junctions = set(wn.junction_name_list)
    sizes = {a: len(set(areas.index[areas.eq(a)]) & junctions) for a in "ABC"}
    # A = 657 junctions in the main zone + reservoir inlet nodes n303 and n336 (fed via PRV-1/2)
    assert sizes == {"A": 659, "B": 31, "C": 92}
    assert areas["n303"] == areas["n336"] == "A"
    pressure_by_area = areas.loc[sensors["pressure"]].value_counts().to_dict()
    assert pressure_by_area == {"A": 29, "B": 1, "C": 3}
    assert set(areas.loc[sensors["amr"]]) == {"C"}  # all 82 AMRs are in area C
    assert areas["T1"] == "C"
    assert areas["R1"] == areas["R2"] == "A"


def test_pipes_never_cross_areas(wn, areas):
    for name in wn.pipe_name_list:
        link = wn.get_link(name)
        assert areas[link.start_node_name] == areas[link.end_node_name], name


def test_anchors_in_same_zone_are_rejected(wn):
    with pytest.raises(ValueError, match="same zone"):
        network.node_areas(wn, anchors={"A": "n54", "B": "n105", "C": "T1"})


def test_pipe_graph_is_connected_and_distances_are_metres(wn):
    g = network.pipe_graph(wn)
    assert nx.is_connected(g)
    link = wn.get_link("p461")
    d = nx.shortest_path_length(g, link.start_node_name, link.end_node_name, weight="length")
    assert d == pytest.approx(link.length)


def test_network_stats_numbers_are_consistent(wn):
    stats = network.network_stats(wn)
    total = stats["pipe_length_m"]["total"]
    assert stats["total_pipe_length_km"] == pytest.approx(total / 1000, abs=1e-3)
    assert stats["pipe_length_m"]["mean"] == pytest.approx(total / 905, abs=0.01)
    search = stats["search_area"]
    assert search["pipe_length_to_inspect_m"] == pytest.approx(5 * total / 905, abs=0.1)
    assert stats["sensor_count_total"] == 119
