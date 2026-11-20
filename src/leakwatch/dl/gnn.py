"""``make gnn``: leak localization — nearest sensor, MLP, GCN, GAT (PRD §8.2, Phase 5).

Data: ``dl.gnn_data`` scenarios (one leak at a junction; two per-sensor residual values = first
and second hour of the leak). Labels: the leak junction.

Node features per scenario (D-029): the two residuals at sensor nodes (0 elsewhere),
``is_sensor``, standardized elevation and base demand, and a *sensor-distance encoding*
``exp(-d(node, sensor_k) / sensor_distance_scale_m)`` for each of the 33 pressure sensors, so
nodes many hops from any sensor can still be told apart. The GNNs also get a *global sensor
context*: an MLP embedding of all 33×2 residuals, broadcast to every node before message passing.
A 3-layer GNN with residuals only at sensor nodes (the first design) cannot see a sensor more
than 3 hops away; ablation variants switch each addition off (``dl.gnn.variants``).
*Sensitivity features* (D-038, ``sensitivity: true``): each node also gets the cosine between the
observed sensor residuals (hour 1, hour 2, their mean) and that node's column of the sensitivity
matrix (``dl.gnn_data``) — the classic model-based method, given to the GNN as evidence to refine.

Models, evaluated identically on the same splits:

1. *nearest sensor*: rank nodes by pipe distance to the sensor with the largest pressure drop;
2. *sensitivity correlation*: rank nodes by that cosine alone (no learning);
3. *MLP*: the 33×2 sensor residuals → one logit per node;
4. *GCN* (PyTorch Geometric ``GCNConv``, edge weight ∝ diameter / length) and *GAT* (``GATConv``
   with edge features length and diameter), with and without the sensitivity features;
5. ablation: GCN with sensitivity but without the global context.

Splits: scenarios whose leak diameter is in ``unseen_diameter_m`` are held out ("unseen sizes");
the rest are split at random 70/15/15 (fixed seed). Metrics: top-1 / top-5 accuracy, mean
shortest-path distance (m) from the top-1 node to the leak, share within ``within_m`` (top-1 and
best of top 5), per leak-size bucket.

Real leaks (D-038): every BattLeDIM leak of 2018 (development) and 2019 (test) is localized at a
fixed time after it starts — 2 h after its outflow first reaches half its peak (at once for abrupt
leaks, later for slowly growing ones) — from each pressure sensor's *re-baselined* residual
(``features.rebaseline``: minus its median of the previous week) in the two 1 h windows ending
then; the distance from the predicted nodes to the leaking pipe (nearer end) is reported for every
localizer. The live localizer is the trained variant with the smallest median distance on the
2018 leaks (ties: validation top-5); 2019 is only reported.

Posterior localizer (D-042): the candidates written to alerts and used live are the chosen
network's probabilities times a prior ``exp(-d / sensor_prior_m)`` around the most-dropped
pressure sensor (:func:`posterior`); ``make gnn-prior`` picks the prior length on 2018.

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


# ------------------------------------------------------------------------------------- data


class GraphData:
    """Static L-Town graph plus derived tensors and the all-pairs distance matrix (m)."""

    def __init__(self, graph: Mapping[str, np.ndarray], sensor_distance_scale_m: float = 500.0):
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
        # (N, sensors): closeness of every node to every pressure sensor along the pipes
        d = self.dist[:, self.sensor_idx]
        d = np.where(np.isfinite(d), d, d[np.isfinite(d)].max())
        self.sensor_closeness = np.exp(-d / float(sensor_distance_scale_m)).astype("float32")
        self.node_static = np.concatenate([self.static, self.sensor_closeness], axis=1)
        # (N, sensors): unit sensitivity vector per node (0 for tanks/reservoirs), if generated
        self.sensitivity: np.ndarray | None = None
        if "sensitivity" in graph:
            s = np.asarray(graph["sensitivity"], dtype="float64")
            norm = np.linalg.norm(s, axis=1, keepdims=True)
            self.sensitivity = (s / np.where(norm > 0, norm, 1.0)).astype("float32")

    def sensitivity_corr(self, x: np.ndarray) -> np.ndarray:
        """(B, sensors, 2) residuals -> (B, N, 3) cosines with each node's sensitivity vector.

        The three cosines use hour 1, hour 2 and the mean of both hours.
        """
        if self.sensitivity is None:
            raise ValueError("graph.npz has no sensitivity matrix: run `make gnn-data`")
        out = []
        for v in (x[:, :, 0], x[:, :, 1], x.mean(axis=2)):
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            out.append((v / np.where(norm > 0, norm, 1.0)) @ self.sensitivity.T)
        return np.stack(out, axis=2).astype("float32")

    def node_dim(self, closeness: bool = True, sensitivity: bool = False) -> int:
        """Node feature width: 2 residuals + static features (+ closeness) (+ 3 cosines)."""
        static = self.node_static if closeness else self.static
        return 2 + static.shape[1] + (3 if sensitivity else 0)

    def node_features(
        self, x: np.ndarray, scale: float, closeness: bool = True, sensitivity: bool = False
    ) -> np.ndarray:
        """(B, sensors, 2) residuals -> (B, N, node_dim) node features."""
        b = x.shape[0]
        feats = np.zeros((b, self.n, x.shape[2]), dtype="float32")
        feats[:, self.sensor_idx, :] = x / scale
        base = self.node_static if closeness else self.static
        parts = [feats, np.broadcast_to(base, (b, self.n, base.shape[1]))]
        if sensitivity:
            parts.append(self.sensitivity_corr(x))
        return np.concatenate(parts, axis=2)


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
    """GCN or GAT over node features plus a global sensor context -> one logit per node.

    Each node starts from a projection of its own features (residuals, static features and, with
    ``closeness``, its pipe distance to every sensor) concatenated, with ``context``, with an
    embedding of all sensor residuals; ``layers`` message-passing layers with residual
    connections follow; the head scores each node. Graphs in a batch share the topology. With
    both switches off this is the first design (residuals only at sensor nodes), kept for the
    ablation.
    """

    def __init__(
        self,
        kind: str,
        node_dim: int,
        sensor_dim: int,
        hidden: int,
        layers: int,
        heads: int,
        closeness: bool = True,
        context: bool = True,
        sensitivity: bool = False,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv, GCNConv

        self.kind = kind
        self.closeness = closeness
        self.sensitivity = sensitivity
        self.use_context = context
        self.node_in = nn.Linear(node_dim, hidden)
        self.context = (
            nn.Sequential(
                nn.Linear(sensor_dim, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden)
            )
            if context
            else None
        )
        start_dim = hidden * 2 if context else hidden
        self.convs = nn.ModuleList()
        dim = start_dim
        for _ in range(layers):
            if kind == "gcn":
                self.convs.append(GCNConv(dim, hidden))
            else:
                self.convs.append(GATConv(dim, hidden, heads=heads, concat=False, edge_dim=2))
            dim = hidden
        self.head = nn.Sequential(
            nn.Linear(hidden + start_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(  # noqa: D102
        self,
        node_x: torch.Tensor,
        sensor_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        b, n, _ = node_x.shape
        h0 = torch.relu(self.node_in(node_x))  # (b, n, H)
        if self.context is not None:
            ctx = self.context(sensor_x.flatten(1))[:, None, :].expand(-1, n, -1)  # (b, n, H)
            h0 = torch.cat([h0, ctx], dim=2)
        start = h0.reshape(b * n, -1)
        offsets = (torch.arange(b) * n).repeat_interleave(edge_index.shape[1])
        ei = edge_index.repeat(1, b) + offsets
        h = start
        for i, conv in enumerate(self.convs):
            if self.kind == "gcn":
                out = conv(h, ei, edge_weight.repeat(b))
            else:
                out = conv(h, ei, edge_attr.repeat(b, 1))
            out = torch.relu(out)
            h = out if i == 0 else h + out
        return self.head(torch.cat([h, start], dim=1)).reshape(b, n)


def build_model(
    spec: Mapping[str, Any], g: GraphData, sensor_dim: int, gcfg: Mapping[str, Any]
) -> nn.Module:
    """A localizer from a variant spec: ``{kind: mlp|gcn|gat, closeness, context, sensitivity}``."""
    if spec["kind"] == "mlp":
        return MLPLocalizer(sensor_dim, g.n, int(gcfg["hidden"]))
    closeness = bool(spec.get("closeness", True))
    sensitivity = bool(spec.get("sensitivity", False))
    return GNNLocalizer(
        str(spec["kind"]),
        g.node_dim(closeness, sensitivity),
        sensor_dim,
        int(gcfg["hidden"]),
        int(gcfg["layers"]),
        int(gcfg["heads"]),
        closeness=closeness,
        context=bool(spec.get("context", True)),
        sensitivity=sensitivity,
    )


# --------------------------------------------------------------------------------- training


def masked(logits: torch.Tensor, is_junction: np.ndarray) -> torch.Tensor:
    """Only junctions can leak (tanks/reservoirs get -inf)."""
    mask = torch.from_numpy(~is_junction)
    return logits.masked_fill(mask, float("-inf"))


def forward(model: nn.Module, g: GraphData, x: np.ndarray, scale: float) -> torch.Tensor:
    """Logits (B, N) for residual inputs (B, sensors, 2)."""
    sensor_x = torch.from_numpy(x / scale)
    node_x = torch.from_numpy(
        g.node_features(
            x, scale, getattr(model, "closeness", True), getattr(model, "sensitivity", False)
        )
    )
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


def sensitivity_scores(g: GraphData, x: np.ndarray) -> np.ndarray:
    """Classic method: cosine between the mean residuals and each node's sensitivity vector."""
    scores = g.sensitivity_corr(x)[:, :, 2].astype("float64")
    scores[:, ~g.is_junction] = -np.inf
    return scores


def sensor_prior(g: GraphData, x: np.ndarray, length_m: float | None) -> np.ndarray:
    """Log-prior (B, N): −(pipe distance to the most-dropped sensor) / ``length_m`` (D-042).

    The physics half of the posterior localizer: a leak is most likely near the pressure sensor
    that fell most, with an exponential fall-off of ``length_m`` metres along the pipes. ``None``
    or 0 gives a flat prior (the network alone).
    """
    if not length_m:
        return np.zeros((x.shape[0], g.n))
    sensor = g.sensor_idx[np.argmin(x.mean(axis=2), axis=1)]
    return -g.dist[sensor] / float(length_m)


def posterior(probs: np.ndarray, g: GraphData, x: np.ndarray, length_m: float | None) -> np.ndarray:
    """The network's node probabilities times :func:`sensor_prior`, renormalized (B, N)."""
    logits = np.log(probs + 1e-9) + sensor_prior(g, x, length_m)
    logits[:, ~g.is_junction] = -np.inf
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


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


# ------------------------------------------------------------------- BattLeDIM leak inputs

_FEATURES: dict[tuple[str, ...], pd.DataFrame] = {}  # re-baselined pressure residuals, by columns


def rebaselined_pressure(sensors: list[str]) -> pd.DataFrame:
    """Re-baselined pressure residuals of 2018–2019 (``features.rebaseline``), by window end."""
    from leakwatch.features.rebaseline import Rebaseline, rebaselined

    cols = [f"pressure__{s}__residual" for s in sensors]
    key = tuple(cols)
    if key not in _FEATURES:
        col_sql = ", ".join(f'"{c}"' for c in cols)
        feats = lake.query_df(
            lake.connect(),
            f"SELECT window_start, window_end, {col_sql} FROM features WHERE source = 'battledim'",
        )
        rb = Rebaseline.from_config(load_config())
        _FEATURES[key] = rebaselined(feats, cols, rb).set_index("window_end").sort_index()
    return _FEATURES[key]


def benchmark_inputs(g: GraphData, times: list[pd.Timestamp], sensors: list[str]) -> np.ndarray:
    """Inputs at real times: (n, sensors, 2) re-baselined residuals (D-038).

    For each time, each pressure sensor's re-baselined ``residual`` in the two 1 h windows ending
    one hour apart at that time (missing values -> 0).
    """
    feats = rebaselined_pressure(sensors)
    cols = [f"pressure__{s}__residual" for s in sensors]
    hour = pd.Timedelta(hours=1)
    out = np.full((len(times), len(sensors), 2), np.nan, dtype="float32")
    for i, t in enumerate(times):
        for j, end in enumerate((t - hour, t)):
            if end in feats.index:
                out[i, :, j] = feats.loc[[end], cols].to_numpy()[-1]
    return np.nan_to_num(out, nan=0.0)


def pipe_distance(g: GraphData, wn: Any, node_idx: np.ndarray, pipe: str) -> np.ndarray:
    """Distance (m) from nodes to the nearer end of a pipe."""
    link = wn.get_link(pipe)
    ends = [g.index[link.start_node_name], g.index[link.end_node_name]]
    return np.min(g.dist[np.ix_(node_idx, ends)], axis=1)


def localization_times(year: int) -> pd.DataFrame:
    """Leaks starting in ``year`` and when to localize them.

    The time is 2 h after the outflow first reaches half its peak, on the 15-minute grid.
    """
    from leakwatch.data.ground_truth import load_ground_truth

    gt = load_ground_truth()
    flows = pd.read_parquet(paths.battledim_parquet_dir() / f"leak_flows_{year}.parquet")
    rows = []
    for leak in gt[gt["start_ts"].dt.year == year].itertuples(index=False):
        f = flows[leak.pipe_id]
        f = f[f.index >= leak.start_ts]
        half = f[f >= 0.5 * f.max()]
        t_half = half.index[0] if len(half) else leak.start_ts
        rows.append(
            {
                "leak_id": leak.leak_id,
                "pipe_id": leak.pipe_id,
                "type": leak.type,
                "start": leak.start_ts,
                "localize_at": (t_half + pd.Timedelta(hours=2)).ceil("15min"),
                "peak_outflow_m3h": float(leak.peak_outflow_m3h),
            }
        )
    return pd.DataFrame(rows)


def real_leak_benchmark(
    year: int,
    g: GraphData,
    localizers: Mapping[str, Any],
    within: float,
    sensors: list[str],
) -> dict[str, Any]:
    """Localize every leak of ``year`` with each localizer (``name -> scores(x)``)."""
    from leakwatch.data import network

    wn = network.load_network()
    times = localization_times(year)
    if times.empty:
        return {"leaks": 0}
    x = benchmark_inputs(g, list(times["localize_at"]), sensors)
    per_leak = [
        {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in r.items()}
        for r in times.to_dict("records")
    ]
    summary: dict[str, Any] = {}
    for name, scores_fn in localizers.items():
        sc = scores_fn(x)
        d1, d5 = [], []
        for i, leak in enumerate(times.itertuples(index=False)):
            top = np.argsort(-sc[i])[:5]
            d = pipe_distance(g, wn, top, leak.pipe_id)
            d1.append(float(d[0]))
            d5.append(float(d.min()))
            per_leak[i][f"{name}_top1"] = g.nodes[top[0]]
            per_leak[i][f"{name}_distance_top1_m"] = round(d1[-1], 1)
        summary[name] = {
            "median_distance_top1_m": round(float(np.median(d1)), 1),
            "mean_distance_top1_m": round(float(np.mean(d1)), 1),
            "within_m_top1": round(float(np.mean(np.array(d1) <= within)), 4),
            "within_m_best_of_top5": round(float(np.mean(np.array(d5) <= within)), 4),
        }
    return {
        "year": year,
        "role": "development" if year == 2018 else "test (reported only)",
        "leaks": len(times),
        "input": "re-baselined pressure residuals (D-038), two 1 h windows ending 2 h after the "
        "outflow first reaches half its peak",
        "localizers": summary,
        "per_leak": per_leak,
    }


# ------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    """Train/evaluate all localizers, test on BattLeDIM 2019, write gnn_top5 to alerts."""
    parser = argparse.ArgumentParser(description="GNN leak localizer.")
    parser.add_argument(
        "--apply-only",
        action="store_true",
        help="no training: re-localize the current batch alerts with the saved model",
    )
    if parser.parse_args(argv).apply_only:
        return apply_only()
    t_start = time.perf_counter()
    cfg = load_config()
    gcfg = cfg["dl"]["gnn"]
    seed = int(cfg["project"]["seed"])
    torch.set_num_threads(int(gcfg["threads"]))
    data_dir = paths.sim_dir() / "gnn"
    ds = np.load(data_dir / "dataset.npz")
    g = GraphData(
        np.load(data_dir / "graph.npz", allow_pickle=False),
        float(gcfg["sensor_distance_scale_m"]),
    )
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
    heuristics = ["nearest_sensor"]
    if g.sensitivity is not None:
        scores["sensitivity_corr"] = {
            s: sensitivity_scores(g, x[ix]) for s, ix in (("test", te), ("unseen", un), ("val", va))
        }
        heuristics.append("sensitivity_corr")
    variants: dict[str, dict[str, Any]] = {"mlp": {"kind": "mlp"}, **gcfg["variants"]}
    for name, spec in variants.items():
        t0 = time.perf_counter()
        model = build_model(spec, g, x.shape[1] * x.shape[2], gcfg)
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
    for name in [*heuristics, *variants]:
        results[name] = {
            "test": distance_metrics(g, scores[name]["test"], y[te], within),
            "unseen_sizes": distance_metrics(g, scores[name]["unseen"], y[un], within)
            if len(un)
            else None,
            "test_by_size": by_bucket(g, scores[name]["test"], y[te], diam[te], edges, within),
            "val_top5": round(topk_accuracy(scores[name]["val"], y[va], 5), 4),
        }
    from leakwatch.data import network

    sensors = network.parse_sensor_config()["pressure"]
    localizers: dict[str, Any] = {"nearest_sensor": lambda xb: nearest_sensor_scores(g, xb)}
    if g.sensitivity is not None:
        localizers["sensitivity_corr"] = lambda xb: sensitivity_scores(g, xb)
    for name in variants:
        localizers[name] = lambda xb, m=models[name]: predict(m, g, xb, scale)
    real = {str(y): real_leak_benchmark(y, g, localizers, within, sensors) for y in (2018, 2019)}
    dev = real["2018"]["localizers"]
    best = min(variants, key=lambda m: (dev[m]["median_distance_top1_m"], -results[m]["val_top5"]))

    # Save the chosen model.
    out_dir = paths.ensure_dir(paths.models_dir() / "gnn")
    torch.save(models[best].state_dict(), out_dir / "model.pt")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": best,
                "variant": variants[best],
                "scale": scale,
                "config": dict(gcfg),
                "nodes": g.nodes,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    gnn_names = [n for n, s in variants.items() if s["kind"] != "mlp"]
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
        "variants": variants,
        "models": results,
        "training": histories,
        "chosen_model": best,
        "chosen_by": "median top-1 distance on the 2018 BattLeDIM leaks (ties: validation top-5)",
        "gnn_beats_mlp_top5": max(results[n]["test"]["top5"] for n in gnn_names)
        > results["mlp"]["test"]["top5"],
        "real_leaks": real,
        "sensor_prior_m": prior_length(cfg),
        "alerts_with_gnn_top5": None,
        "wall_s": round(time.perf_counter() - t_start, 1),
    }
    # Results first, so a failure while updating the lake cannot lose the trained comparison.
    path = write_metrics("dl_gnn", payload)
    payload["alerts_with_gnn_top5"] = write_gnn_top5(cfg, g, models[best], best, scale, sensors)
    path = write_metrics("dl_gnn", payload)
    log.info(
        "chosen %s; test top5 %s; 2018 median %s m; 2019 median %s m -> %s",
        best,
        results[best]["test"]["top5"],
        dev[best]["median_distance_top1_m"],
        real["2019"]["localizers"][best]["median_distance_top1_m"],
        path,
    )
    return 0


def apply_only() -> int:
    """Re-localize the batch alerts with the saved model (after ``make bdt-eval``).

    ``make bdt-eval`` rewrites the batch alerts; this refreshes their ``gnn_top5`` without
    retraining and updates ``alerts_with_gnn_top5`` in ``dl_gnn.json``.
    """
    from leakwatch.common.metrics import read_metrics
    from leakwatch.data import network

    cfg = load_config()
    loaded = load_localizer()
    if loaded is None:
        raise FileNotFoundError("no saved localizer: run `make gnn` first")
    g, model, scale, meta = loaded
    best = str(meta["model"])
    sensors = network.parse_sensor_config()["pressure"]
    payload = read_metrics("dl_gnn")
    payload.pop("meta", None)
    payload["alerts_with_gnn_top5"] = write_gnn_top5(cfg, g, model, best, scale, sensors)
    payload["sensor_prior_m"] = prior_length(cfg)
    path = write_metrics("dl_gnn", payload)
    log.info("re-localized %s alerts with %s -> %s", payload["alerts_with_gnn_top5"], best, path)
    return 0


def prior_length(cfg: Mapping[str, Any]) -> float | None:
    """``dl.gnn.sensor_prior_m``: the posterior localizer's prior length (None: GNN alone)."""
    value = cfg["dl"]["gnn"].get("sensor_prior_m")
    return float(value) if value else None


def load_localizer(model_dir: Any = None) -> tuple[GraphData, nn.Module, float, dict] | None:
    """The saved localizer: graph, model (eval mode), input scale and ``meta.json`` (or None)."""
    model_dir = model_dir or paths.models_dir() / "gnn"
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    g = GraphData(
        np.load(paths.sim_dir() / "gnn" / "graph.npz", allow_pickle=False),
        float(meta["config"].get("sensor_distance_scale_m", 500.0)),
    )
    spec = meta.get("variant") or {"kind": meta["model"], "closeness": False, "context": False}
    model = build_model(spec, g, len(g.sensor_idx) * 2, meta["config"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return g, model, float(meta["scale"]), meta


def localize_times(
    times: list[pd.Timestamp], length_m: float | None, loaded: Any = None
) -> list[str]:
    """Top-1 node of the posterior localizer at each time (BattLeDIM re-baselined inputs)."""
    from leakwatch.data import network

    loaded = loaded or load_localizer()
    if loaded is None:
        raise FileNotFoundError("no saved localizer: run `make gnn` first")
    g, model, scale, _ = loaded
    if not times:
        return []
    x = benchmark_inputs(g, times, network.parse_sensor_config()["pressure"])
    p = posterior(predict(model, g, x, scale), g, x, length_m)
    return [g.nodes[j] for j in np.argmax(p, axis=1)]


def write_gnn_top5(
    cfg: Mapping[str, Any],
    g: GraphData,
    model: nn.Module,
    name: str,
    scale: float,
    sensors: list[str],
) -> int:
    """Append alert versions with ``gnn_top5`` for every current batch alert.

    The candidates are the posterior localizer's (network × sensor prior, ``dl.gnn.sensor_prior_m``,
    D-042); ``prob`` is the posterior probability.
    """
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
    probs = posterior(predict(model, g, xb, scale), g, xb, prior_length(cfg))
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

    Input: each pressure sensor's ``residual`` feature (metres, against the hour-of-week
    baseline) in two consecutive 1 h windows, ending one hour apart at the alert's latest window
    (``last_window_end``, else its start) — the quantity the model was trained on: the mean
    residual of the first and the second hour of a leak (``dl.gnn_data``). Falls back to the
    latest two windows when the earlier one is not in ``features`` (D-033).
    """
    from leakwatch.data import network

    loaded = load_localizer(model_dir)
    if loaded is None:
        return None
    g, model, scale, _ = loaded
    length_m = prior_length(load_config())
    sensors = network.parse_sensor_config()["pressure"]
    cols = [f"pressure__{s}__residual" for s in sensors]

    def localize(alert: Mapping[str, Any], features: pd.DataFrame) -> list[dict[str, Any]] | None:
        if not set(cols) <= set(features.columns):
            return None
        ref = pd.Timestamp(alert.get("last_window_end") or alert["start_ts"])
        ends = pd.to_datetime(features["window_end"], utc=True)
        hourly = [features[ends == ref - pd.Timedelta(hours=1)], features[ends == ref]]
        if all(len(h) for h in hourly):
            rows = pd.concat([h.tail(1) for h in hourly])
        else:
            rows = features[ends <= ref].sort_values("window_end").tail(2)
        if rows.empty:
            return None
        x = rows[cols].to_numpy(dtype="float32").T  # (sensors, windows)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        xb = np.nan_to_num(x)[None]
        probs = posterior(predict(model, g, xb, scale), g, xb, length_m)[0]
        top = np.argsort(-probs)[:5]
        return [{"node_id": g.nodes[j], "prob": round(float(probs[j]), 4)} for j in top]

    return localize
