"""``make gnn``: leak localization — nearest sensor, MLP, GCN, GAT (PRD §8.2, Phase 5).

Data: ``dl.gnn_data`` scenarios (one leak at a junction; two per-sensor residual values = first
and second hour of the leak). Node features per scenario: the two residuals at sensor nodes (0
elsewhere), ``is_sensor``, standardized elevation and base demand. Labels: the leak junction.

Models, evaluated identically on the same splits:

1. *nearest sensor*: rank nodes by pipe distance to the sensor with the largest pressure drop;
2. *MLP*: the 33×2 sensor residuals → one logit per node;
3. *GCN* (PyTorch Geometric ``GCNConv``, edge weight ∝ diameter / length);
4. *GAT* (``GATConv`` with edge features length and diameter).

Splits: scenarios whose leak diameter is in ``unseen_diameter_m`` are held out ("unseen sizes");
the rest are split at random 70/15/15 (fixed seed). Metrics: top-1 / top-5 accuracy, mean
shortest-path distance (m) from the top-1 node to the leak, share within ``within_m`` (top-1 and
best of top 5), per leak-size bucket.

BattLeDIM 2019 test: for each 2019 leak's first true-positive K-Means alert, the input is the
week-over-week change of each pressure sensor's ``residual`` feature in the two 1 h windows
before the alert fires (accumulated older leaks cancel out); the distance from the predicted
nodes to the leaking pipe (nearest end) is reported.

Writes ``reports/metrics/dl_gnn.json``, ``data/models/gnn/``, and adds ``gnn_top5`` to the batch
alerts (new alert versions).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch import nn

from leakwatch.common import lake, paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics

log = get_logger("leakwatch.dl.gnn")
MODEL_NAMES = ("nearest_sensor", "mlp", "gcn", "gat")


# ------------------------------------------------------------------------------------- data


class GraphData:
    """Static L-Town graph plus derived tensors and the all-pairs distance matrix (m)."""

    def __init__(self, graph: Mapping[str, np.ndarray]) -> None:
        self.nodes = [str(n) for n in graph["nodes"]]
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.n = len(self.nodes)
        self.sensor_idx = np.asarray(graph["sensor_idx"])
        self.is_junction = np.asarray(graph["is_junction"], dtype=bool)
        ei = np.asarray(graph["edge_index"])
        length = np.asarray(graph["edge_length_m"], dtype="float64")
        diameter = np.asarray(graph["edge_diameter_m"], dtype="float64")
        self.edge_index = torch.from_numpy(ei)
        w = diameter / np.maximum(length, 1.0)
        self.edge_weight = torch.from_numpy((w / w.mean()).astype("float32"))
        attr = np.stack([np.log1p(length), diameter / diameter.mean()], axis=1)
        spread = np.where(attr.std(0) > 0, attr.std(0), 1.0)  # constant columns -> 0, not NaN
        self.edge_attr = torch.from_numpy(((attr - attr.mean(0)) / spread).astype("float32"))
        adj = csr_matrix((np.maximum(length, 0.0), (ei[0], ei[1])), shape=(self.n, self.n))
        self.dist = shortest_path(adj, directed=False)
        elev = np.asarray(graph["elevation_m"], dtype="float64")
        dem = np.asarray(graph["base_demand_m3s"], dtype="float64")
        self.static = np.stack(
            [
                np.isin(np.arange(self.n), self.sensor_idx).astype("float64"),
                (elev - elev.mean()) / (elev.std() or 1.0),
                (dem - dem.mean()) / (dem.std() or 1.0),
            ],
            axis=1,
        ).astype("float32")

    def node_features(self, x: np.ndarray, scale: float) -> np.ndarray:
        """(B, sensors, 2) residuals -> (B, N, 2 + 3) node features."""
        b = x.shape[0]
        feats = np.zeros((b, self.n, x.shape[2]), dtype="float32")
        feats[:, self.sensor_idx, :] = x / scale
        static = np.broadcast_to(self.static, (b, self.n, self.static.shape[1]))
        return np.concatenate([feats, static], axis=2)


# ----------------------------------------------------------------------------------- models


class MLPLocalizer(nn.Module):
    """Sensor residuals -> logits over all nodes."""

    def __init__(self, inputs: int, nodes: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 4, hidden * 4),
            nn.ReLU(),
            nn.Linear(hidden * 4, nodes),
        )

    def forward(self, sensor_x: torch.Tensor, **_: Any) -> torch.Tensor:  # noqa: D102
        return self.net(sensor_x.flatten(1))


class GNNLocalizer(nn.Module):
    """Stacked GCN or GAT layers -> one logit per node (graphs in a batch share the topology)."""

    def __init__(self, kind: str, in_dim: int, hidden: int, layers: int, heads: int) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv, GCNConv

        self.kind = kind
        self.convs = nn.ModuleList()
        dim = in_dim
        for _ in range(layers):
            if kind == "gcn":
                self.convs.append(GCNConv(dim, hidden))
            else:
                self.convs.append(GATConv(dim, hidden, heads=heads, concat=False, edge_dim=2))
            dim = hidden
        self.head = nn.Sequential(
            nn.Linear(hidden + in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(  # noqa: D102
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        b, n, f = node_x.shape
        x0 = node_x.reshape(b * n, f)
        offsets = (torch.arange(b) * n).repeat_interleave(edge_index.shape[1])
        ei = edge_index.repeat(1, b) + offsets
        h = x0
        for conv in self.convs:
            if self.kind == "gcn":
                h = conv(h, ei, edge_weight.repeat(b))
            else:
                h = conv(h, ei, edge_attr.repeat(b, 1))
            h = torch.relu(h)
        return self.head(torch.cat([h, x0], dim=1)).reshape(b, n)


# --------------------------------------------------------------------------------- training


def masked(logits: torch.Tensor, is_junction: np.ndarray) -> torch.Tensor:
    """Only junctions can leak (tanks/reservoirs get -inf)."""
    mask = torch.from_numpy(~is_junction)
    return logits.masked_fill(mask, float("-inf"))


def forward(model: nn.Module, g: GraphData, x: np.ndarray, scale: float) -> torch.Tensor:
    """Logits (B, N) for residual inputs (B, sensors, 2)."""
    sensor_x = torch.from_numpy(x / scale)
    node_x = torch.from_numpy(g.node_features(x, scale))
    out = model(
        sensor_x=sensor_x,
        node_x=node_x,
        edge_index=g.edge_index,
        edge_weight=g.edge_weight,
        edge_attr=g.edge_attr,
    )
    return masked(out, g.is_junction)


def train_nn(
    model: nn.Module,
    g: GraphData,
    x: np.ndarray,
    y: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    scale: float,
    cfg: Mapping[str, Any],
    seed: int,
) -> list[dict[str, float]]:
    """Adam + cross-entropy over nodes; keep the epoch with the best validation top-5."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    bs = int(cfg["batch_size"])
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        order = rng.permutation(len(x))
        losses = []
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            logits = forward(model, g, x[idx], scale)
            loss = nn.functional.cross_entropy(logits, torch.from_numpy(y[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        top5 = topk_accuracy(predict(model, g, xv, scale), yv, 5)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_top5": top5})
        if top5 > best + 1e-4:
            best, bad = top5, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(
    model: nn.Module, g: GraphData, x: np.ndarray, scale: float, batch: int = 64
) -> np.ndarray:
    """Node probabilities (B, N)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(torch.softmax(forward(model, g, x[i : i + batch], scale), dim=1).numpy())
    return np.concatenate(out)


def nearest_sensor_scores(g: GraphData, x: np.ndarray) -> np.ndarray:
    """Heuristic: nodes close (in pipe metres) to the sensor with the largest pressure drop."""
    drop = x.mean(axis=2)  # (B, sensors), negative = pressure fell
    sensor = g.sensor_idx[np.argmin(drop, axis=1)]
    scores = -g.dist[sensor]  # (B, N)
    scores[:, ~g.is_junction] = -np.inf
    return scores


def topk_accuracy(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    """Share of rows whose label is among the k highest scores."""
    top = np.argsort(-scores, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def distance_metrics(
    g: GraphData, scores: np.ndarray, y: np.ndarray, within_m: float
) -> dict[str, Any]:
    """Top-1/top-5 accuracy and pipe distances (m) to the true leak node."""
    top = np.argsort(-scores, axis=1)[:, :5]
    d1 = g.dist[top[:, 0], y]
    d5 = g.dist[top, y[:, None]].min(axis=1)
    return {
        "n": len(y),
        "top1": round(float((top[:, 0] == y).mean()), 4),
        "top5": round(float((top == y[:, None]).any(axis=1).mean()), 4),
        "mean_distance_top1_m": round(float(np.mean(d1)), 1),
        "median_distance_top1_m": round(float(np.median(d1)), 1),
        "within_m_top1": round(float((d1 <= within_m).mean()), 4),
        "within_m_best_of_top5": round(float((d5 <= within_m).mean()), 4),
    }


def by_bucket(
    g: GraphData,
    scores: np.ndarray,
    y: np.ndarray,
    diam: np.ndarray,
    edges: list[float],
    within_m: float,
) -> dict[str, Any]:
    """:func:`distance_metrics` per leak-size bucket."""
    names = ["small", "medium", "large"]
    bins = np.digitize(diam, edges)
    return {
        names[b]: distance_metrics(g, scores[bins == b], y[bins == b], within_m)
        for b in range(len(names))
        if (bins == b).any()
    }


# -------------------------------------------------------------------- BattLeDIM 2019 inputs


def benchmark_inputs(g: GraphData, times: list[pd.Timestamp], sensors: list[str]) -> np.ndarray:
    """Inputs for real alerts: (n, sensors, 2) week-over-week residual changes.

    For each time, the change of each pressure sensor's ``residual`` feature over 7 days, in the
    two 1 h windows ending at that time (features table).
    """
    cols = [f"pressure__{s}__residual" for s in sensors]
    col_sql = ", ".join(f'"{c}"' for c in cols)
    con = lake.connect()
    feats = lake.query_df(
        con, f"SELECT window_end, {col_sql} FROM features WHERE source = 'battledim'"
    )
    feats = feats.set_index("window_end").sort_index()
    week, hour = pd.Timedelta(days=7), pd.Timedelta(hours=1)
    out = np.full((len(times), len(sensors), 2), np.nan, dtype="float32")
    for i, t in enumerate(times):
        for j, end in enumerate((t - hour, t)):
            if end in feats.index and end - week in feats.index:
                out[i, :, j] = (
                    feats.loc[end, cols].to_numpy() - feats.loc[end - week, cols].to_numpy()
                )
    return np.nan_to_num(out, nan=0.0)


def pipe_distance(g: GraphData, wn: Any, node_idx: np.ndarray, pipe: str) -> np.ndarray:
    """Distance (m) from nodes to the nearer end of a pipe."""
    link = wn.get_link(pipe)
    ends = [g.index[link.start_node_name], g.index[link.end_node_name]]
    return np.min(g.dist[np.ix_(node_idx, ends)], axis=1)


# ------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    """Train/evaluate all localizers, test on BattLeDIM 2019, write gnn_top5 to alerts."""
    argparse.ArgumentParser(description="GNN leak localizer.").parse_args(argv)
    t_start = time.perf_counter()
    cfg = load_config()
    gcfg = cfg["dl"]["gnn"]
    seed = int(cfg["project"]["seed"])
    torch.set_num_threads(int(cfg["dl"]["lstm"]["threads"]))
    data_dir = paths.sim_dir() / "gnn"
    ds = np.load(data_dir / "dataset.npz")
    g = GraphData(np.load(data_dir / "graph.npz", allow_pickle=False))
    x, y, diam = ds["x"].astype("float32"), ds["label"].astype("int64"), ds["diameter_m"]
    lo, hi = (float(v) for v in gcfg["unseen_diameter_m"])
    unseen = (diam >= lo) & (diam <= hi)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(np.flatnonzero(~unseen))
    n_tr, n_va = int(0.7 * len(idx)), int(0.15 * len(idx))
    tr, va, te = idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]
    un = np.flatnonzero(unseen)
    scale = float(np.std(x[tr])) or 1.0
    within = float(gcfg["within_m"])
    edges = [float(v) for v in gcfg["size_buckets_m"]]

    scores: dict[str, dict[str, np.ndarray]] = {}
    histories: dict[str, Any] = {}
    models: dict[str, nn.Module] = {}
    scores["nearest_sensor"] = {
        s: nearest_sensor_scores(g, x[ix]) for s, ix in (("test", te), ("unseen", un), ("val", va))
    }
    in_dim = 2 + g.static.shape[1]
    for name in ("mlp", "gcn", "gat"):
        t0 = time.perf_counter()
        if name == "mlp":
            model: nn.Module = MLPLocalizer(x.shape[1] * x.shape[2], g.n, int(gcfg["hidden"]))
        else:
            model = GNNLocalizer(
                name, in_dim, int(gcfg["hidden"]), int(gcfg["layers"]), int(gcfg["heads"])
            )
        hist = train_nn(model, g, x[tr], y[tr], x[va], y[va], scale, gcfg, seed)
        histories[name] = {
            "epochs": len(hist),
            "history": hist,
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        models[name] = model
        scores[name] = {
            s: predict(model, g, x[ix], scale)
            for s, ix in (("test", te), ("unseen", un), ("val", va))
        }
        log.info(
            "%s: val top5 %.3f test top5 %.3f",
            name,
            topk_accuracy(scores[name]["val"], y[va], 5),
            topk_accuracy(scores[name]["test"], y[te], 5),
        )

    results = {}
    for name in MODEL_NAMES:
        results[name] = {
            "test": distance_metrics(g, scores[name]["test"], y[te], within),
            "unseen_sizes": distance_metrics(g, scores[name]["unseen"], y[un], within)
            if len(un)
            else None,
            "test_by_size": by_bucket(g, scores[name]["test"], y[te], diam[te], edges, within),
            "val_top5": round(topk_accuracy(scores[name]["val"], y[va], 5), 4),
        }
    best = max(("mlp", "gcn", "gat"), key=lambda m: results[m]["val_top5"])

    # Save the chosen model.
    out_dir = paths.ensure_dir(paths.models_dir() / "gnn")
    torch.save(models[best].state_dict(), out_dir / "model.pt")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {"model": best, "scale": scale, "config": dict(gcfg), "nodes": g.nodes}, indent=1
        ),
        encoding="utf-8",
    )

    # BattLeDIM 2019: first true-positive alert per leak.
    from leakwatch.data import network
    from leakwatch.data.ground_truth import load_ground_truth

    wn = network.load_network()
    sensors = network.parse_sensor_config()["pressure"]
    gt = load_ground_truth()
    con = lake.connect()
    alerts = lake.query_df(
        con,
        "SELECT alert_id, start_ts, version FROM alerts "
        "WHERE origin = 'batch' AND model = 'kmeans'",
    )
    bench = []
    new_2019 = gt[gt["start_ts"].dt.year == 2019]
    for leak in new_2019.itertuples(index=False):
        hit = alerts[
            (alerts["start_ts"] >= leak.start_ts)
            & (alerts["start_ts"] <= leak.start_ts + pd.Timedelta(days=7))
        ]
        if hit.empty:
            continue
        t = hit["start_ts"].min()
        xb = benchmark_inputs(g, [t], sensors)
        sc = (
            predict(models[best], g, xb, scale)[0]
            if best != "nearest_sensor"
            else nearest_sensor_scores(g, xb)[0]
        )
        top = np.argsort(-sc)[:5]
        d = pipe_distance(g, wn, top, leak.pipe_id)
        heur = np.argsort(-nearest_sensor_scores(g, xb)[0])[:5]
        dh = pipe_distance(g, wn, heur, leak.pipe_id)
        bench.append(
            {
                "leak_id": leak.leak_id,
                "pipe_id": leak.pipe_id,
                "type": leak.type,
                "peak_outflow_m3h": float(leak.peak_outflow_m3h),
                "alert_start": t.isoformat(),
                "top5_nodes": [g.nodes[i] for i in top],
                "distance_top1_m": round(float(d[0]), 1),
                "distance_best_of_top5_m": round(float(d.min()), 1),
                "nearest_sensor_distance_top1_m": round(float(dh[0]), 1),
                "nearest_sensor_distance_best_of_top5_m": round(float(dh.min()), 1),
            }
        )
    bench_df = pd.DataFrame(bench)
    benchmark = {
        "leaks_localized": len(bench_df),
        "model": best,
        "within_m_top1": round(float((bench_df["distance_top1_m"] <= within).mean()), 4)
        if len(bench_df)
        else None,
        "within_m_best_of_top5": round(
            float((bench_df["distance_best_of_top5_m"] <= within).mean()), 4
        )
        if len(bench_df)
        else None,
        "median_distance_top1_m": float(bench_df["distance_top1_m"].median())
        if len(bench_df)
        else None,
        "median_distance_best_of_top5_m": float(bench_df["distance_best_of_top5_m"].median())
        if len(bench_df)
        else None,
        "nearest_sensor_median_distance_top1_m": float(
            bench_df["nearest_sensor_distance_top1_m"].median()
        )
        if len(bench_df)
        else None,
        "per_leak": bench,
        "input": "week-over-week change of pressure residual features, "
        "two 1 h windows before the alert",
    }

    # gnn_top5 for every batch alert (new alert versions).
    n_updated = write_gnn_top5(cfg, g, models[best], best, scale, sensors)

    payload = {
        "dataset": {
            "scenarios": len(y),
            "train": len(tr),
            "val": len(va),
            "test": len(te),
            "unseen_sizes": len(un),
            "unseen_diameter_m": [lo, hi],
        },
        "residual_scale_m": scale,
        "models": results,
        "training": histories,
        "chosen_model": best,
        "chosen_by": "validation top-5 accuracy",
        "gnn_beats_mlp_top5": max(results["gcn"]["test"]["top5"], results["gat"]["test"]["top5"])
        > results["mlp"]["test"]["top5"],
        "battledim_2019": benchmark,
        "alerts_with_gnn_top5": n_updated,
        "wall_s": round(time.perf_counter() - t_start, 1),
    }
    path = write_metrics("dl_gnn", payload)
    log.info(
        "chosen %s; test top5 %s; 2019 within %s m: %s -> %s",
        best,
        results[best]["test"]["top5"],
        within,
        benchmark["within_m_best_of_top5"],
        path,
    )
    return 0


def write_gnn_top5(
    cfg: Mapping[str, Any],
    g: GraphData,
    model: nn.Module,
    name: str,
    scale: float,
    sensors: list[str],
) -> int:
    """Append alert versions with ``gnn_top5`` for every current batch alert."""
    from leakwatch.bdt.data import write_alerts
    from leakwatch.stream.spark import get_spark

    con = lake.connect()
    alerts = lake.query_df(
        con,
        "SELECT * EXCLUDE (work_order_json, work_order_generator) FROM alerts "
        "WHERE origin = 'batch'",
    )
    if alerts.empty:
        return 0
    xb = benchmark_inputs(g, list(alerts["start_ts"]), sensors)
    probs = predict(model, g, xb, scale)
    top = np.argsort(-probs, axis=1)[:, :5]
    now = pd.Timestamp(datetime.now(UTC))
    alerts["gnn_top5"] = [
        [{"node_id": g.nodes[j], "prob": round(float(probs[i, j]), 4)} for j in top[i]]
        for i in range(len(alerts))
    ]
    alerts["version"] = alerts["version"] + 1
    alerts["updated_at"] = now
    spark = get_spark(cfg["spark"], "leakwatch-gnn-alerts")
    return write_alerts(spark, alerts, "batch", mode="append")


if __name__ == "__main__":
    sys.exit(main())


def make_localizer(model_dir: Any = None) -> Any:
    """Live localizer for ``bdt.live.LiveScoring.localize`` (None if no model is trained).

    Input at alert time: each pressure sensor's ``residual`` feature (metres, against the
    hour-of-week baseline) in the last two 1 h windows up to the alert — the quantity the model
    was trained on (``dl.gnn_data``).
    """
    from leakwatch.data import network

    model_dir = model_dir or paths.models_dir() / "gnn"
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    g = GraphData(np.load(paths.sim_dir() / "gnn" / "graph.npz", allow_pickle=False))
    cfg = meta["config"]
    if meta["model"] == "mlp":
        model: nn.Module = MLPLocalizer(len(g.sensor_idx) * 2, g.n, int(cfg["hidden"]))
    else:
        model = GNNLocalizer(meta["model"], 2 + g.static.shape[1], int(cfg["hidden"]), int(cfg["layers"]), int(cfg["heads"]))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    sensors = network.parse_sensor_config()["pressure"]
    cols = [f"pressure__{s}__residual" for s in sensors]
    scale = float(meta["scale"])

    def localize(alert: Mapping[str, Any], features: pd.DataFrame) -> list[dict[str, Any]] | None:
        if not set(cols) <= set(features.columns):
            return None
        rows = features[pd.to_datetime(features["window_end"], utc=True) <= pd.Timestamp(alert["start_ts"])]
        rows = rows.sort_values("window_end").tail(2)
        if rows.empty:
            return None
        x = rows[cols].to_numpy(dtype="float32").T  # (sensors, windows)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        probs = predict(model, g, np.nan_to_num(x)[None], scale)[0]
        top = np.argsort(-probs)[:5]
        return [{"node_id": g.nodes[j], "prob": round(float(probs[j]), 4)} for j in top]

    return localize
