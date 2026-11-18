"""A tiny lake and network for copilot tests (no BattLeDIM data, no Spark).

``build_tiny_lake(root)`` writes Parquet files with the contract columns
(``ALERT_LAKE_COLUMNS``, ``CLEAN_LAKE_COLUMNS``, ``BASELINES_SCHEMA``, feature rows) in the same
partition layout Spark uses, so ``common.lake.connect(lake_root=root)`` reads them like the real
lake. ``toy_network()`` is a seven-pipe network with areas A and C.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from leakwatch.common.schemas import ALERT_LAKE_COLUMNS
from leakwatch.copilot.network_index import NetworkIndex

#: Alert ids of the tiny lake (valid UUIDs).
ALERT_A1 = "11111111-aaaa-4aaa-8aaa-000000000001"  # area A, closed, no GNN candidates
ALERT_C2 = "22222222-cccc-4ccc-8ccc-000000000002"  # area C, open, GNN candidates
ALERT_A3 = "33333333-aaaa-4aaa-8aaa-000000000003"  # area A, closed, 2018, no loss estimate

_UTC = pa.timestamp("us", tz="UTC")
_TOP = pa.list_(pa.struct([("sensor_id", pa.string()), ("contribution", pa.float64())]))
_GNN = pa.list_(pa.struct([("node_id", pa.string()), ("prob", pa.float64())]))


def ts(text: str) -> pd.Timestamp:
    """UTC timestamp from text."""
    return pd.Timestamp(text, tz="UTC")


def alert_rows(written_at: str = "2026-01-01 00:00") -> list[dict[str, Any]]:
    """Alert versions of the tiny lake (A1 has two versions: open, then closed)."""
    w = ts(written_at)
    base = {
        "model": "kmeans",
        "source": "battledim",
        "scenario_id": None,
        "lstm_score_max": None,
        "created_at": w,
        "updated_at": w,
    }
    return [
        {
            **base,
            "alert_id": ALERT_A1,
            "version": 1,
            "start_ts": ts("2019-03-10 06:00"),
            "end_ts": None,
            "status": "open",
            "kmeans_score_max": 6.0,
            "threshold": 3.0,
            "top_sensors": [
                {"sensor_id": "pressure__n2", "contribution": 0.5},
                {"sensor_id": "pressure__n1", "contribution": 0.3},
                {"sensor_id": "flow__p1", "contribution": 0.1},
            ],
            "area": "A",
            "gnn_top5": None,
            "est_loss_m3h": 10.5,
            "windows_above": 3,
            "last_window_end": ts("2019-03-10 06:00"),
        },
        {
            **base,
            "alert_id": ALERT_A1,
            "version": 2,
            "start_ts": ts("2019-03-10 06:00"),
            "end_ts": ts("2019-03-10 18:00"),
            "status": "closed",
            "kmeans_score_max": 9.25,
            "threshold": 3.0,
            "top_sensors": [
                {"sensor_id": "pressure__n2", "contribution": 0.5},
                {"sensor_id": "pressure__n1", "contribution": 0.3},
                {"sensor_id": "flow__p1", "contribution": 0.1},
            ],
            "area": "A",
            "gnn_top5": None,
            "est_loss_m3h": 12.0,
            "windows_above": 40,
            "last_window_end": ts("2019-03-10 18:00"),
        },
        {
            **base,
            "alert_id": ALERT_C2,
            "version": 1,
            "start_ts": ts("2019-03-12 10:00"),
            "end_ts": None,
            "status": "open",
            "kmeans_score_max": 4.0,
            "threshold": 3.0,
            "top_sensors": [
                {"sensor_id": "pressure__n7", "contribution": 0.6},
                {"sensor_id": "level__T1", "contribution": 0.2},
            ],
            "area": "C",
            "gnn_top5": [{"node_id": "n6", "prob": 0.3}, {"node_id": "n7", "prob": 0.6}],
            "est_loss_m3h": 2.0,
            "windows_above": 17,
            "last_window_end": ts("2019-03-12 14:00"),
        },
        {
            **base,
            "alert_id": ALERT_A3,
            "version": 1,
            "start_ts": ts("2018-11-01 00:00"),
            "end_ts": ts("2018-11-01 05:00"),
            "status": "closed",
            "kmeans_score_max": 3.5,
            "threshold": 3.0,
            "top_sensors": [{"sensor_id": "pressure__n1", "contribution": 0.7}],
            "area": "A",
            "gnn_top5": None,
            "est_loss_m3h": None,
            "windows_above": 20,
            "last_window_end": ts("2018-11-01 05:00"),
        },
    ]


def write_alerts(
    root: Path, rows: list[dict[str, Any]], origin: str = "batch", name: str = "part-0"
) -> Path:
    """Alert versions as Spark writes them (``alerts/origin=…/``, no ``origin`` column)."""
    cols = [c for c in ALERT_LAKE_COLUMNS if c != "origin"]
    types = {
        "alert_id": pa.string(),
        "version": pa.int32(),
        "model": pa.string(),
        "source": pa.string(),
        "scenario_id": pa.string(),
        "start_ts": _UTC,
        "end_ts": _UTC,
        "status": pa.string(),
        "kmeans_score_max": pa.float64(),
        "threshold": pa.float64(),
        "top_sensors": _TOP,
        "area": pa.string(),
        "lstm_score_max": pa.float64(),
        "gnn_top5": _GNN,
        "est_loss_m3h": pa.float64(),
        "windows_above": pa.int32(),
        "last_window_end": _UTC,
        "created_at": _UTC,
        "updated_at": _UTC,
    }
    table = pa.table({c: pa.array([r.get(c) for r in rows], types[c]) for c in cols})
    out = root / "alerts" / f"origin={origin}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.parquet"
    pq.write_table(table, path)
    return path


def _readings(
    sensor_id: str, kind: str, day: str, base: float, amp: float, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(day, periods=288, freq="5min", tz="UTC")
    values = base + amp * np.sin(np.arange(288) / 288 * 2 * np.pi) + rng.normal(0, 0.01, 288)
    unit = {"pressure": "m", "flow": "m3/h", "level": "m"}[kind]
    return pd.DataFrame(
        {
            "ts": idx,
            "sensor_id": sensor_id,
            "value": values.round(4),
            "unit": unit,
            "scenario_id": pd.array([None] * 288, dtype="string"),
            "out_of_range": False,
            "filled": False,
            "stuck": False,
            "sensor_type": kind,
        }
    )


def build_tiny_lake(root: Path) -> Path:
    """Write the tiny lake under ``root`` (the lake root) and return it."""
    write_alerts(root, alert_rows())
    frames = [
        _readings(s, k, d, b, a, i)
        for i, (s, k, d, b, a) in enumerate(
            [
                ("n1", "pressure", "2019-03-10", 50.0, 2.0),
                ("n2", "pressure", "2019-03-10", 40.0, 1.5),
                ("n1", "pressure", "2019-03-11", 50.5, 2.0),
                ("n2", "pressure", "2019-03-11", 40.5, 1.5),
                ("p1", "flow", "2019-03-10", 80.0, 10.0),
                ("T1", "level", "2019-03-12", 3.0, 0.5),
            ]
        )
    ]
    for df in frames:
        day = df["ts"].iloc[0].strftime("%Y-%m-%d")
        kind = df["sensor_type"].iloc[0]
        out = (
            root
            / "clean"
            / "source=battledim"
            / "year=2019"
            / f"date={day}"
            / f"sensor_type={kind}"
        )
        out.mkdir(parents=True, exist_ok=True)
        part = df.drop(columns=["sensor_type"])
        n = len(list(out.glob("*.parquet")))
        pq.write_table(pa.Table.from_pandas(part, preserve_index=False), out / f"part-{n}.parquet")
    base = pd.DataFrame(
        [
            (s, h, b + (h % 24) * 0.01, 10)
            for s, b in (("pressure__n1", 50.0), ("pressure__n2", 40.0))
            for h in range(168)
        ],
        columns=["series_id", "hour_of_week", "baseline", "n_readings"],
    ).astype({"hour_of_week": "int32", "n_readings": "int64"})
    bdir = root / "baselines" / "hour_of_week"
    bdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(base, preserve_index=False), bdir / "part-0.parquet")
    starts = pd.date_range("2019-03-10 00:00", periods=8, freq="15min", tz="UTC")
    feat = pd.DataFrame(
        {
            "scenario_id": pd.array([None] * 8, dtype="string"),
            "window_start": starts,
            "window_end": starts + pd.Timedelta(hours=1),
            "n_readings": np.int64(12),
            "n_filled": np.int64(0),
            "n_stuck": np.int64(0),
            "n_out_of_range": np.int64(0),
            "n_series_missing": np.int64(0),
            "pressure__n1__mean": np.linspace(50, 51, 8),
            "pressure__n1__residual": np.linspace(-0.5, 0.2, 8),
        }
    )
    fdir = root / "features" / "source=battledim" / "window=1h" / "date=2019-03-10"
    fdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(feat, preserve_index=False), fdir / "part-0.parquet")
    return root


def tiny_retriever():  # -> rag.index.Retriever
    """Three passages (two from the SOP, one from a CPHEEO-style PDF) with the hashing embedder."""
    from leakwatch.copilot.rag.chunking import Chunk
    from leakwatch.copilot.rag.embed import HashingEmbedder
    from leakwatch.copilot.rag.index import Retriever, VectorIndex

    emb = HashingEmbedder()
    sop = ("kb_sop", "LeakWatch SOP", "LeakWatch SOP", "markdown", None)
    pdf = ("cpheeo", "CPHEEO O&M", "CPHEEO O&M Manual 2005", "pdf", "Ch. 15 Water Audit")
    chunks = [
        Chunk(
            "kb:0", *sop, "6. Confirm the leak before digging", None, None, None,
            "Never excavate on the alert alone. Listen at valves and hydrants with a listening "
            "stick.",
        ),
        Chunk(
            "kb:1", *sop, "4. Set the priority", None, None, None,
            "P1 when the estimated loss is high or the anomaly score is far above the threshold.",
        ),
        Chunk(
            "pdf:0", *pdf, "§15.9 Water Losses", 382, 382, "350",
            "Physical losses are leakage from pipes; non-physical losses are meter errors and "
            "theft.",
        ),
    ]  # fmt: skip
    vectors = emb.encode([f"{c.section}\n{c.text}" for c in chunks])
    index = VectorIndex(vectors, [c.to_dict() for c in chunks], {"embedder": emb.name})
    return Retriever(index, emb)


def tiny_context(lake_root: Path, **copilot_overrides: Any):  # -> tools.ToolContext
    """A ToolContext over a tiny lake with the toy network and retriever.

    The hashing embedder scores lower than the real model, so ``rag.min_score`` is 0 here.
    ``copilot_overrides`` replace whole sub-sections of the ``copilot`` config.
    """
    import copy

    from leakwatch.common.config import load_config
    from leakwatch.copilot.tools import ToolContext

    cfg = copy.deepcopy(load_config(environ={}))
    cfg["copilot"]["rag"]["min_score"] = 0.0
    for key, value in copilot_overrides.items():
        cfg["copilot"][key] = value
    return ToolContext(
        full_cfg=cfg, lake_root=lake_root, network=toy_network(), retriever=tiny_retriever()
    )


def toy_network() -> NetworkIndex:
    """R1-p1-n1-p2-n2-p3-n3-p4-n4, n2-p5-n5 (area A); n4-V1-n6-p6-n7-p7-T1 (area C)."""
    graph = nx.Graph()
    for name, u, v, length in [
        ("p1", "R1", "n1", 10.0),
        ("p2", "n1", "n2", 50.0),
        ("p3", "n2", "n3", 40.0),
        ("p4", "n3", "n4", 30.0),
        ("p5", "n2", "n5", 20.0),
        ("p6", "n6", "n7", 25.0),
        ("p7", "n7", "T1", 15.0),
    ]:
        graph.add_edge(u, v, name=name, kind="Pipe", length=length)
    graph.add_edge("n4", "n6", name="V1", kind="Valve", length=0.0)
    areas = {n: "A" for n in ("R1", "n1", "n2", "n3", "n4", "n5")} | {
        n: "C" for n in ("n6", "n7", "T1")
    }
    return NetworkIndex(graph, areas)
