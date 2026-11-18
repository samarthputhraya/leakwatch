"""L-Town network model: loading, areas A/B/C, graph distances and headline network facts.

Areas are not stored in ``L-TOWN.inp``. They are the hydraulic zones you get when the network is
cut at its pump and pressure-reducing valves; each zone is named by an anchor node from config
(``network.area_anchors``). Zones without an anchor (the reservoir inlet stubs) take the name of
the zone their valve feeds.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import wntr
import yaml

from leakwatch.common import paths
from leakwatch.common.config import load_config

#: How many candidate pipes a crew walks if the localizer's top-5 is right (PRD §8.2).
TOP_K_CANDIDATES = 5


def load_network(inp_path: Path | None = None) -> wntr.network.WaterNetworkModel:
    """Load ``L-TOWN.inp`` (nominal parameters) as a WNTR model."""
    path = inp_path or paths.network_inp_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `make data`")
    return wntr.network.WaterNetworkModel(str(path))


def parse_sensor_config(config_path: Path | None = None) -> dict[str, list[str]]:
    """Sensor locations from the official ``dataset_configuration.yaml``, keyed by sensor type."""
    path = config_path or paths.battledim_raw_dir() / "dataset_configuration.yaml"
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return {
        "pressure": [str(s) for s in cfg["pressure_sensors"]],
        "flow": [str(s) for s in cfg["flow_sensors"]],
        "level": [str(s) for s in cfg["level_sensors"]],
        "amr": [str(s) for s in cfg["amrs"]],
    }


def pipe_graph(wn: wntr.network.WaterNetworkModel, pipes_only: bool = False) -> nx.Graph:
    """Undirected graph of the network; edges carry ``name``, ``kind`` and ``length`` (m).

    With ``pipes_only=True`` pumps and valves are left out, which splits the network into zones.
    Pumps and valves get length 0 so shortest-path distances are pipe lengths.
    """
    graph = nx.Graph()
    graph.add_nodes_from(wn.node_name_list)
    for name, link in wn.links():
        kind = link.link_type
        if pipes_only and kind != "Pipe":
            continue
        length = float(link.length) if kind == "Pipe" else 0.0
        graph.add_edge(
            link.start_node_name, link.end_node_name, name=name, kind=kind, length=length
        )
    return graph


def node_areas(
    wn: wntr.network.WaterNetworkModel,
    anchors: Mapping[str, str] | None = None,
) -> pd.Series:
    """Area label (A/B/C) for every node, from zones split at pumps and valves."""
    area_anchors: Mapping[str, str] = (
        anchors if anchors is not None else load_config()["network"]["area_anchors"]
    )
    zones = list(nx.connected_components(pipe_graph(wn, pipes_only=True)))
    zone_of = {node: i for i, zone in enumerate(zones) for node in zone}
    label: dict[int, str] = {}
    for area, anchor in area_anchors.items():
        zone = zone_of[anchor]
        if zone in label:
            raise ValueError(f"anchors {label[zone]} and {area} fall in the same zone")
        label[zone] = area
    # Unlabelled zones (reservoir inlet stubs) take the label of the zone their valve feeds.
    changed = True
    while changed:
        changed = False
        for _, link in wn.links():
            if link.link_type == "Pipe":
                continue
            up, down = zone_of[link.start_node_name], zone_of[link.end_node_name]
            if up not in label and down in label:
                label[up] = label[down]
                changed = True
    unlabelled = [i for i in range(len(zones)) if i not in label]
    if unlabelled:
        raise ValueError(f"{len(unlabelled)} zones could not be assigned an area")
    return pd.Series({n: label[z] for n, z in zone_of.items()}, name="area").sort_index()


def pipe_areas(wn: wntr.network.WaterNetworkModel, areas: pd.Series | None = None) -> pd.Series:
    """Area of every pipe (pipes never cross zone boundaries; pumps and valves do)."""
    areas = node_areas(wn) if areas is None else areas
    return pd.Series(
        {name: areas[wn.get_link(name).start_node_name] for name in wn.pipe_name_list},
        name="area",
    ).sort_index()


def node_coordinates(wn: wntr.network.WaterNetworkModel) -> pd.DataFrame:
    """``x, y`` map coordinates of every node (from the ``.inp`` [COORDINATES] section)."""
    coords = {name: node.coordinates for name, node in wn.nodes()}
    return pd.DataFrame.from_dict(coords, orient="index", columns=["x", "y"]).sort_index()


def network_stats(
    wn: wntr.network.WaterNetworkModel,
    sensors: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Headline facts about L-Town, measured from the model file (feeds the pitch numbers)."""
    sensor_map: Mapping[str, list[str]] = sensors or parse_sensor_config()
    lengths = wn.query_link_attribute("length", link_type=wntr.network.model.Pipe)
    areas = node_areas(wn)
    p_areas = pipe_areas(wn, areas)
    total_m = float(lengths.sum())
    mean_m = float(lengths.mean())
    junctions = set(wn.junction_name_list)

    per_area: dict[str, Any] = {}
    for area in sorted(areas.unique()):
        nodes = set(areas.index[areas.eq(area)])
        area_pipes = p_areas.index[p_areas.eq(area)]
        per_area[area] = {
            "junctions": len(nodes & junctions),
            "tanks": sorted(nodes & set(wn.tank_name_list)),
            "reservoirs": sorted(nodes & set(wn.reservoir_name_list)),
            "pressure_sensors": sorted(set(sensor_map["pressure"]) & nodes, key=_natural),
            "amrs": len(set(sensor_map["amr"]) & nodes),
            "pipes": len(area_pipes),
            "pipe_length_km": round(float(lengths[area_pipes].sum()) / 1000, 3),
        }
    for area_info in per_area.values():
        area_info["n_pressure_sensors"] = len(area_info["pressure_sensors"])

    search_m = TOP_K_CANDIDATES * mean_m
    return {
        "source": "L-TOWN.inp (nominal model, Zenodo record 4017659)",
        "counts": {
            "junctions": wn.num_junctions,
            "reservoirs": wn.num_reservoirs,
            "tanks": wn.num_tanks,
            "pipes": wn.num_pipes,
            "pumps": wn.num_pumps,
            "valves": wn.num_valves,
        },
        "valves": {
            v: {
                "type": wn.get_link(v).valve_type,
                "from": wn.get_link(v).start_node_name,
                "to": wn.get_link(v).end_node_name,
                "setting_m": float(wn.get_link(v).initial_setting),
            }
            for v in wn.valve_name_list
        },
        "pumps": {
            p: {"from": wn.get_link(p).start_node_name, "to": wn.get_link(p).end_node_name}
            for p in wn.pump_name_list
        },
        "pipe_length_m": {
            "total": round(total_m, 1),
            "mean": round(mean_m, 2),
            "median": round(float(lengths.median()), 2),
            "min": round(float(lengths.min()), 2),
            "max": round(float(lengths.max()), 2),
        },
        "total_pipe_length_km": round(total_m / 1000, 3),
        "areas": per_area,
        "sensors": {k: len(v) for k, v in sensor_map.items()},
        "sensor_count_total": sum(len(v) for v in sensor_map.values()),
        "options": {
            "demand_model": wn.options.hydraulic.demand_model,
            "headloss": wn.options.hydraulic.headloss,
            "pattern_timestep_s": int(wn.options.time.pattern_timestep),
            "patterns": list(wn.pattern_name_list),
        },
        "search_area": {
            "assumption": f"leak is in the top-{TOP_K_CANDIDATES} candidates; one pipe each",
            "top_k": TOP_K_CANDIDATES,
            "pipe_length_to_inspect_m": round(search_m, 1),
            "reduction_factor_vs_whole_network": round(total_m / search_m, 1),
        },
    }


def _natural(name: str) -> tuple[str, int]:
    digits = "".join(ch for ch in name if ch.isdigit())
    return (name.rstrip("0123456789"), int(digits) if digits else -1)
