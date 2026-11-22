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
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import networkx as nx
import numpy as np

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

    def pipe_ends(self, name: str) -> tuple[str, str] | None:
        """The two nodes a pipe joins (None for a name the graph does not have)."""
        for a, b, link in self.graph.edges(data="name"):
            if link == name:
                return str(a), str(b)
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


def _rings(edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Closed chains of point indices from boundary edges (each point meets an even number)."""
    adj: dict[int, set[int]] = defaultdict(set)
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    rings = []
    while adj:
        start = min(adj)
        ring, cur = [start], start
        while True:
            nxt = min(adj[cur])
            adj[cur].discard(nxt)
            adj[nxt].discard(cur)
            if not adj[cur]:
                del adj[cur]
            if not adj[nxt]:  # back at the start (or a dangling end)
                del adj[nxt]
                break
            if nxt == start:  # closed; the start has more edges (two rings touch there)
                break
            ring.append(nxt)
            cur = nxt
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def _shoelace(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def _inside(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Even-odd rule."""
    x, y = point
    xs, ys = polygon[:, 0], polygon[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    crosses = (ys > y) != (yn > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = xs + (y - ys) * (xn - xs) / (yn - ys)
    return bool(np.count_nonzero(crosses & (x < at)) % 2)


def outline(
    points: Sequence[Sequence[float]], *, buffer_m: float, alpha_m: float, sides: int = 8
) -> list[list[list[float]]]:
    """Outer rings around a cloud of points, about ``buffer_m`` outside the outermost ones.

    Every point is widened to a small circle (``sides`` points at ``buffer_m``); the alpha shape
    of all of them keeps the Delaunay triangles whose circumradius is at most ``alpha_m``, and
    its boundary is returned as closed rings (first point repeated). Holes are dropped.
    """
    from scipy.spatial import Delaunay

    base = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(base) == 0:
        return []
    turn = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    ring = np.stack([np.cos(turn), np.sin(turn)], axis=1) * buffer_m
    cloud = (base[:, None, :] + ring[None, :, :]).reshape(-1, 2)
    simplices = Delaunay(cloud).simplices
    a, b, c = (cloud[simplices[:, k]] for k in range(3))
    la, lb, lc = (np.linalg.norm(u - v, axis=1) for u, v in ((b, c), (a, c), (a, b)))
    area2 = np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = la * lb * lc / (2 * area2)
    kept = simplices[radius <= alpha_m]
    uses: Counter[tuple[int, int]] = Counter()
    for s in kept:
        for i, j in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            uses[(min(i, j), max(i, j))] += 1
    rings = [cloud[r] for r in _rings(e for e, n in uses.items() if n == 1)]
    rings.sort(key=lambda r: -abs(_shoelace(r)))
    outer: list[np.ndarray] = []
    for r in rings:
        if not any(_inside(r[0], o) for o in outer):
            outer.append(r)
    return [[[round(float(x), 1), round(float(y), 1)] for x, y in [*r, r[0]]] for r in outer]


def area_points(
    nodes: Mapping[str, Sequence[float]],
    areas: Mapping[str, str],
    pipes: Sequence[Mapping[str, Any]],
    junctions: Iterable[str],
    step_m: float,
) -> dict[str, list[list[float]]]:
    """Junctions per area, plus points every ``step_m`` along the pipes inside the area."""
    out: dict[str, list[list[float]]] = defaultdict(list)
    for j in junctions:
        if areas.get(j) is not None:
            out[areas[j]].append(list(nodes[j]))
    for p in pipes:
        area = areas.get(p["a"])
        if area is None or areas.get(p["b"]) != area or p["kind"] != "Pipe":
            continue
        (x0, y0), (x1, y1) = nodes[p["a"]], nodes[p["b"]]
        n = int(math.hypot(x1 - x0, y1 - y0) // step_m)
        out[area].extend(
            [x0 + (x1 - x0) * k / (n + 1), y0 + (y1 - y0) * k / (n + 1)] for k in range(1, n + 1)
        )
    return dict(out)


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
    shape = cfg["dashboard"]["outline"]
    points = area_points(nodes, areas, pipes, junctions, float(shape["step_m"]))
    outlines = {
        area: outline(pts, buffer_m=float(shape["buffer_m"]), alpha_m=float(shape["alpha_m"]))
        for area, pts in sorted(points.items())
    }
    xs, ys = [p[0] for p in nodes.values()], [p[1] for p in nodes.values()]
    payload = {
        "bounds": [min(xs), min(ys), max(xs), max(ys)],
        "nodes": nodes,
        "junctions": junctions,
        "areas": areas,
        "pipes": pipes,
        "sensors": sensors,
        "area_labels": labels,
        "outlines": outlines,
        "tint_full_m": float(tint["full_m"]),
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
