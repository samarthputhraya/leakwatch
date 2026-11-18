"""Pipes near a node or a sensor, for work orders (PRD §9.3) and the GNN-candidates tool.

Built from ``L-TOWN.inp`` through ``leakwatch.data.network`` (graph with pipe lengths, node areas),
or from any ``networkx`` graph in tests. Edges carry ``name``, ``kind``
(``Pipe``/``Pump``/``Valve``) and ``length`` (m; 0 for pumps and valves).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import networkx as nx

from leakwatch.common.logging import get_logger

log = get_logger("leakwatch.copilot.network_index")


def natural_key(name: str) -> tuple[str, int, str]:
    """Sort ``p2`` before ``p10``."""
    m = re.match(r"^(\D*)(\d+)", name)
    return (m.group(1), int(m.group(2)), name) if m else (name, -1, name)


class NetworkIndex:
    """Graph lookups: pipes at a node, nearest pipes by network distance, sensor locations."""

    def __init__(self, graph: nx.Graph, node_area: Mapping[str, str]) -> None:
        """Wrap a graph whose edges have ``name``, ``kind`` and ``length``."""
        self.graph = graph
        self.node_area = {str(k): str(v) for k, v in node_area.items()}
        self._links: dict[str, tuple[str, str, str]] = {
            str(d["name"]): (str(u), str(v), str(d.get("kind", "Pipe")))
            for u, v, d in graph.edges(data=True)
        }

    @classmethod
    def from_inp(cls, area_anchors: Mapping[str, str] | None = None) -> NetworkIndex:
        """L-Town from ``paths.network_inp_path()`` (FileNotFoundError before `make data`)."""
        from leakwatch.data import network

        wn = network.load_network()
        areas = network.node_areas(wn, area_anchors)
        return cls(network.pipe_graph(wn), areas.to_dict())

    def has_node(self, node: str) -> bool:
        """True if ``node`` is a junction, tank or reservoir of the network."""
        return node in self.graph

    def area_of(self, node: str) -> str | None:
        """Area (A/B/C) of a node, or None."""
        return self.node_area.get(node)

    def pipes_at(self, node: str) -> list[str]:
        """Pipes connected to ``node`` (pumps and valves excluded), naturally sorted."""
        if node not in self.graph:
            return []
        names = [
            str(d["name"])
            for _, _, d in self.graph.edges(node, data=True)
            if d.get("kind", "Pipe") == "Pipe"
        ]
        return sorted(names, key=natural_key)

    def locate_series(self, series_id: str) -> str | None:
        """Node where a feature series is measured.

        ``pressure__n54`` -> ``n54``; a flow on a link -> one end of that link;
        ``amr_sum__C`` -> None.
        """
        kind, _, ident = series_id.partition("__")
        ident = ident or kind
        if kind == "amr_sum":
            return None
        if ident in self.graph:
            return ident
        link = self._links.get(ident)
        return link[0] if link else None

    def nearest_pipes(self, node: str, k: int) -> list[tuple[str, float]]:
        """The ``k`` pipes closest to ``node`` by network distance (m) to their nearer end.

        Only pipes in the node's own area count: pumps and valves have length 0 in the graph, so
        without this the search would step into the next zone for free.
        """
        if node not in self.graph or k <= 0:
            return []
        area = self.node_area.get(node)
        dist = nx.single_source_dijkstra_path_length(self.graph, node, weight="length")
        found: list[tuple[str, float]] = []
        for name, (u, v, kind) in self._links.items():
            if kind != "Pipe" or (u not in dist and v not in dist):
                continue
            if area is not None and area not in (self.node_area.get(u), self.node_area.get(v)):
                continue
            found.append((name, min(dist.get(u, float("inf")), dist.get(v, float("inf")))))
        found.sort(key=lambda item: (item[1], natural_key(item[0])))
        return [(name, round(d, 1)) for name, d in found[:k]]


@lru_cache(maxsize=1)
def load_network_index() -> NetworkIndex:
    """L-Town, loaded once per process (needs `make data`)."""
    from leakwatch.common.config import load_config

    return NetworkIndex.from_inp(load_config()["network"]["area_anchors"])


def plan_pipes(
    alert: Mapping[str, Any],
    net: NetworkIndex | None,
    max_pipes: int,
    per_sensor: int,
) -> tuple[list[str], str]:
    """Ordered pipes to inspect for an alert and a sentence saying how they were chosen.

    With localizer candidates (``gnn_top5``): the pipes attached to each candidate node, most
    probable node first. Without them: the ``per_sensor`` pipes nearest each top sensor, in the
    order of the sensors' contributions.
    """
    if net is None:
        return [], "network model not available (run `make data`): no pipe list"
    pipes: list[str] = []
    gnn: Sequence[Mapping[str, Any]] = alert.get("gnn_top5") or []
    if gnn:
        for cand in sorted(gnn, key=lambda c: -float(c.get("prob") or 0.0)):
            pipes.extend(net.pipes_at(str(cand.get("node_id"))))
        basis = "pipes attached to the localizer's top-5 candidate nodes, most probable node first"
    else:
        located = []
        for sensor in alert.get("top_sensors") or []:
            sid = str(sensor.get("sensor_id") or sensor.get("series_id") or "")
            node = net.locate_series(sid)
            if node:
                located.append(node)
                pipes.extend(name for name, _ in net.nearest_pipes(node, per_sensor))
        where = ", ".join(located) if located else "none"
        basis = (
            "no localizer candidates for this alert: pipes nearest the top sensors by network "
            f"distance, top sensor first (sensor nodes: {where})"
        )
    ordered = list(dict.fromkeys(pipes))[:max_pipes]
    return ordered, basis
