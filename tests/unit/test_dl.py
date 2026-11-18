"""Deep-learning building blocks on synthetic data (dl.sequences, dl.lstm_ae, dl.gnn)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from leakwatch.dl.gnn import (  # noqa: E402
    GNNLocalizer,
    GraphData,
    MLPLocalizer,
    distance_metrics,
    forward,
    nearest_sensor_scores,
    topk_accuracy,
)
from leakwatch.dl.lstm_ae import LSTMAutoencoder, interpolate_at, reconstruction  # noqa: E402
from leakwatch.dl.sequences import make_sequences, standardize  # noqa: E402


def residual_frame(steps: int = 300, channels: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=steps, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(size=(steps, channels)), index=idx, columns=list("abc")[:channels]
    )


def test_sequences_end_at_window_ends_and_have_full_history():
    r = residual_frame()
    seqs = make_sequences(
        r, np.zeros(3), np.ones(3), seq_steps=24, stride_steps=3, step_s=300, window_s=3600
    )
    # windows start every 15 min; the sequence ends at the window's last step (start + 55 min)
    assert all((seqs.window_end - seqs.window_start) == pd.Timedelta(hours=1))
    assert all(ws.minute % 15 == 0 for ws in seqs.window_start)
    first = seqs.batch(np.array([0]))
    assert first.shape == (1, 24, 3)
    end = seqs.ends[0]
    np.testing.assert_allclose(first[0], r.to_numpy()[end - 23 : end + 1].astype("float32"))


def test_gaps_break_sequences():
    r = residual_frame().drop(index=residual_frame().index[100])
    seqs = make_sequences(r, np.zeros(3), np.ones(3), 24, 3, 300, 3600)
    t = r.index.asi8 // 10**9
    span = t[seqs.ends] - t[seqs.ends - 23]
    assert (span == 23 * 300).all()


def test_standardize_handles_nan_and_zero_std():
    z = standardize(
        np.array([[1.0, np.nan, 5.0]]), np.array([0.0, 0.0, 5.0]), np.array([2.0, 1.0, 0.0])
    )
    np.testing.assert_allclose(z, [[0.5, 0.0, 0.0]])


def test_lstm_autoencoder_scores_and_shares():
    torch.manual_seed(0)
    r = residual_frame()
    seqs = make_sequences(r, np.zeros(3), np.ones(3), 12, 3, 300, 3600)
    model = LSTMAutoencoder(3, hidden=8, latent=4)
    scores, shares = reconstruction(model, seqs, batch=7)
    assert scores.shape == (len(seqs),) and (scores > 0).all()
    np.testing.assert_allclose(shares.sum(axis=1), 1.0, rtol=1e-5)


def test_interpolate_at():
    pts = [
        {"false_alarms_per_week": 1.0, "detection_rate": 0.2},
        {"false_alarms_per_week": 3.0, "detection_rate": 0.6},
    ]
    assert interpolate_at(pts, 2.0) == pytest.approx(0.4)
    assert interpolate_at(pts, 5.0) is None


def line_graph(n: int = 6) -> GraphData:
    """Junctions 0..n-1 in a line, 100 m pipes; sensors at both ends and the middle."""
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return GraphData(
        {
            "nodes": np.array([f"j{i}" for i in range(n)]),
            "edge_index": np.array([src, dst]),
            "edge_length_m": np.full(len(src), 100.0, dtype="float32"),
            "edge_diameter_m": np.full(len(src), 0.2, dtype="float32"),
            "elevation_m": np.linspace(0, 10, n).astype("float32"),
            "base_demand_m3s": np.full(n, 0.001, dtype="float32"),
            "is_junction": np.ones(n, dtype=bool),
            "sensor_idx": np.array([0, n // 2, n - 1]),
        }
    )


def test_graph_distances_and_nearest_sensor():
    g = line_graph()
    assert g.dist[0, 5] == pytest.approx(500.0)
    x = np.zeros((1, 3, 2), dtype="float32")
    x[0, 2] = -1.0  # largest drop at the sensor on node 5
    scores = nearest_sensor_scores(g, x)
    assert int(np.argmax(scores[0])) == 5
    m = distance_metrics(g, scores, np.array([4]), within_m=150)
    assert m["top1"] == 0.0 and m["top5"] == 1.0
    assert m["mean_distance_top1_m"] == pytest.approx(100.0)
    assert m["within_m_top1"] == 1.0
    assert topk_accuracy(scores, np.array([0]), 5) == 0.0  # node 0 is the 6th nearest
    assert topk_accuracy(scores, np.array([0]), 6) == 1.0


@pytest.mark.parametrize("kind", ["mlp", "gcn", "gat"])
def test_localizers_give_one_logit_per_node(kind):
    torch.manual_seed(0)
    g = line_graph()
    x = np.random.default_rng(0).normal(size=(4, 3, 2)).astype("float32")
    model = MLPLocalizer(6, g.n, 8) if kind == "mlp" else GNNLocalizer(kind, 5, 8, 2, 2)
    out = forward(model, g, x, scale=1.0)
    assert out.shape == (4, g.n)
    loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert np.isfinite(float(loss))
