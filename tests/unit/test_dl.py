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
    build_model,
    distance_metrics,
    forward,
    nearest_sensor_scores,
    topk_accuracy,
)
from leakwatch.dl.lstm_ae import LSTMAutoencoder, reconstruction  # noqa: E402
from leakwatch.dl.sequences import make_sequences, standardize  # noqa: E402
from leakwatch.eval.alerting import best_within, threshold_sweep  # noqa: E402


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


def test_best_within_picks_the_best_point_inside_the_budget():
    pts = [
        {"level": "a", "false_alarms_per_week": 0.5, "detection_rate": 0.2},
        {"level": "b", "false_alarms_per_week": 1.8, "detection_rate": 0.6},
        {"level": "c", "false_alarms_per_week": 1.2, "detection_rate": 0.6},
        {"level": "d", "false_alarms_per_week": 3.0, "detection_rate": 0.9},
    ]
    best = best_within(pts, 2.0)
    assert best["within_budget"] and best["level"] == "c"  # tie on detection -> fewer alarms
    assert best_within(pts, 0.4) == {"within_budget": False, "lowest_false_alarms_per_week": 0.5}


def test_threshold_sweep_extends_past_the_largest_normal_score():
    scores = np.arange(1.0, 101.0)
    sweep = threshold_sweep(scores, [0.5, 0.99], [2.0])
    assert [label for label, _ in sweep] == ["q0.5", "q0.99", "2x max"]
    assert sweep[0][1] == pytest.approx(50.5) and sweep[-1][1] == pytest.approx(200.0)


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
    model = (
        MLPLocalizer(6, g.n, 8) if kind == "mlp" else GNNLocalizer(kind, g.node_dim(), 6, 8, 2, 2)
    )
    out = forward(model, g, x, scale=1.0)
    assert out.shape == (4, g.n)
    loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert np.isfinite(float(loss))


def test_sensor_closeness_tells_far_nodes_apart():
    g = line_graph()  # sensors on nodes 0, 3, 5; 100 m pipes; default scale 500 m
    assert g.sensor_closeness.shape == (g.n, 3)
    np.testing.assert_allclose(g.sensor_closeness[0], np.exp(-np.array([0, 300, 500]) / 500.0))
    assert len({tuple(row) for row in g.sensor_closeness.round(6)}) == g.n  # every node unique
    feats = g.node_features(np.ones((2, 3, 2), dtype="float32"), scale=1.0)
    assert feats.shape == (2, g.n, g.node_dim()) and g.node_dim() == 2 + 3 + 3
    assert feats[0, 1, 0] == 0.0 and feats[0, 3, 0] == 1.0  # residuals only at sensor nodes
    assert g.node_features(np.ones((1, 3, 2), "float32"), 1.0, closeness=False).shape[2] == 5


@pytest.mark.parametrize("closeness", [True, False])
@pytest.mark.parametrize("context", [True, False])
def test_ablation_variants_build_and_run(closeness, context):
    torch.manual_seed(0)
    g = line_graph()
    spec = {"kind": "gcn", "closeness": closeness, "context": context}
    cfg = {"hidden": 8, "layers": 2, "heads": 2}
    model = build_model(spec, g, 6, cfg)
    assert (model.context is not None) == context and model.closeness == closeness
    out = forward(model, g, np.ones((3, 3, 2), dtype="float32"), scale=1.0)
    assert out.shape == (3, g.n) and torch.isfinite(out).all()


def _seqs(seed: int, steps: int = 400):
    """Smooth periodic residuals (easy to reconstruct) as 12-step sequences."""
    idx = pd.date_range("2018-01-01", periods=steps, freq="5min", tz="UTC")
    t = np.arange(steps)
    rng = np.random.default_rng(seed)
    data = np.stack([np.sin(t / 6 + k) for k in range(3)], axis=1) + rng.normal(0, 0.05, (steps, 3))
    frame = pd.DataFrame(data, index=idx, columns=list("abc"))
    return make_sequences(frame, np.zeros(3), np.ones(3), 12, 3, 300, 3600)


def test_training_lowers_the_validation_error_and_keeps_the_best_epoch():
    from leakwatch.dl.lstm_ae import train_model

    train, val = _seqs(1), _seqs(2)
    cfg = {
        "hidden": 16,
        "latent": 8,
        "learning_rate": 0.01,
        "batch_size": 16,
        "epochs": 15,
        "patience": 15,
    }
    model, history = train_model(train, val, 3, cfg, seed=0)
    assert len(history) == 15
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    best = min(h["val_loss"] for h in history)
    # the returned model is the best epoch's, not the last one's
    assert float(reconstruction(model, val)[0].mean()) == pytest.approx(best, rel=1e-5)
    assert best < history[0]["val_loss"]


def test_training_stops_after_patience_epochs_without_improvement():
    from leakwatch.dl.lstm_ae import train_model

    cfg = {
        "hidden": 4,
        "latent": 2,
        "learning_rate": 0.0,
        "batch_size": 64,
        "epochs": 50,
        "patience": 3,
    }
    _, history = train_model(_seqs(1), _seqs(2), 3, cfg, seed=0)
    assert len(history) == 4  # epoch 1 sets the best; a zero learning rate never improves it


def test_comparison_plot_is_written(tmp_path, monkeypatch):
    from leakwatch.common import paths
    from leakwatch.dl import lstm_ae

    monkeypatch.setattr(paths, "FIGURES_DIR", tmp_path)
    pts = [
        {"false_alarms_per_week": 0.5, "detection_rate": 0.3, "chance_level": 0.6},
        {"false_alarms_per_week": None, "detection_rate": 0.0, "chance_level": 0.0},
        {"false_alarms_per_week": 2.5, "detection_rate": 0.9, "chance_level": 0.95},
    ]
    name = lstm_ae.plot_comparison({"LSTM-AE": pts, "K-Means": pts[:1]}, 2019)
    assert name == "dl_detector_comparison_2019.png"
    assert (tmp_path / name).stat().st_size > 0
