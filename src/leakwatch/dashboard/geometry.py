"""L-Town geometry for the control-room map (D-052): coordinates, pipes, sensors, tint weights.

Built once from the network model (``data.network``) and sent to the page as JSON. The page draws
it with its own coordinates (metres); no basemap.

The pipe tint (D-053) is the only interpolated element on screen. Each junction gets weights for
the nearest pressure sensors **of its own area**, by distance along the pipes: inverse squared
distance, scaled by a fall-off with the distance to the nearest sensor. The page multiplies the
weights with the sensors' measured pressure drops. A sensor's own junction gets (almost exactly)
its measured value; a junction far from every sensor gets almost none.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import networkx as nx

from leakwatch.common.config import load_config
from leakwatch.data import network


@dataclass(frozen=True)
class Geometry:
    """The map payload and the pipe graph it was built from."""

    payload: dict[str, Any]
    graph: nx.Graph

    def pipe_distance_m(self, a: str, b: str) -> float | None:
        """Shortest distance along the pipes between two nodes (None if not connected)."""
        try:
            return float(nx.dijkstra_path_length(self.graph, a, b, weight="length"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


def tint_weights(
    zone_graph: nx.Graph,
    areas: Mapping[str, str],
    sensors: list[str],
    *,
    nearest: int,
    length_m: float,
    min_distance_m: float,
) -> dict[str, list[list[float]]]:
    """Per node: ``[[sensor index, weight], ...]`` for the nearest sensors of the node's area.

    Args:
        zone_graph: pipes only, so distances never cross a pump or valve into another zone.
        areas: area label of every node.
        sensors: pressure sensor node ids; the index in this list is what the page receives.
        nearest: sensors used per node.
        length_m: the weights are scaled by ``exp(-d_nearest / length_m)``.
        min_distance_m: added (squared) to the squared distance, keeps weights finite at a sensor.

    """
    reach: dict[str, list[tuple[float, int]]] = {}
    for index, sensor in enumerate(sensors):
        if sensor not in zone_graph:
            continue
        distances = nx.single_source_dijkstra_path_length(zone_graph, sensor, weight="length")
        for node, d in distances.items():
            if areas.get(node) == areas.get(sensor):
                reach.setdefault(node, []).append((float(d), index))
    out: dict[str, list[list[float]]] = {}
    for node, found in reach.items():
        close = sorted(found)[:nearest]
        raw = [1.0 / (d * d + min_distance_m * min_distance_m) for d, _ in close]
        fall_off = math.exp(-close[0][0] / length_m)
        total = sum(raw)
        out[node] = [
            [index, round(w / total * fall_off, 5)]
            for w, (_, index) in zip(raw, close, strict=True)
        ]
    return out


def build(cfg: Mapping[str, Any] | None = None) -> Geometry:
    """Read the network model and shape it for the map."""
    cfg = cfg or load_config()
    wn = network.load_network()
    coords = network.node_coordinates(wn)
    areas = network.node_areas(wn, cfg["network"]["area_anchors"]).to_dict()
    sensors = [s for s in network.parse_sensor_config()["pressure"] if s in coords.index]
    nodes = {n: [round(float(r.x), 1), round(float(r.y), 1)] for n, r in coords.iterrows()}
    junctions = sorted(wn.junction_name_list)
    pipes = [
        {
            "id": name,
            "a": link.start_node_name,
            "b": link.end_node_name,
            "kind": link.link_type,
            "d": round(float(link.diameter), 3) if link.link_type == "Pipe" else 0.0,
            "area": areas.get(link.start_node_name),
        }
        for name, link in wn.links()
    ]
    labels: dict[str, list[float]] = {}
    for area in sorted(set(areas.values())):
        members = [nodes[n] for n in junctions if areas.get(n) == area]
        if members:
            labels[area] = [
                round(sum(p[0] for p in members) / len(members), 1),
                round(sum(p[1] for p in members) / len(members), 1),
            ]
    tint = cfg["dashboard"]["tint"]
    xs, ys = [p[0] for p in nodes.values()], [p[1] for p in nodes.values()]
    payload = {
        "bounds": [min(xs), min(ys), max(xs), max(ys)],
        "nodes": nodes,
        "junctions": junctions,
        "areas": areas,
        "pipes": pipes,
        "sensors": sensors,
        "area_labels": labels,
        "tint": tint_weights(
            network.pipe_graph(wn, pipes_only=True),
            areas,
            sensors,
            nearest=int(tint["nearest"]),
            length_m=float(tint["length_m"]),
            min_distance_m=float(tint["min_distance_m"]),
        ),
        "counts": {
            "junctions": len(junctions),
            "pipes": len(wn.pipe_name_list),
            "pressure_sensors": len(sensors),
        },
    }
    return Geometry(payload=payload, graph=network.pipe_graph(wn))


@lru_cache(maxsize=1)
def load() -> Geometry:
    """The geometry of the configured network (built on first use, then kept)."""
    return build()
