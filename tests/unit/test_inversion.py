"""Model-based inversion (GLS against the sensitivity matrix), cue fusion and the choice (D-044)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakwatch.dl.inversion import (
    Inversion,
    describe,
    fit_residuals,
    interpolation_scores,
    rank_fusion,
    ranks,
    real_noise_windows,
    regularized,
    second_moment,
    softmax_rows,
    whitener,
)


def signatures(n: int = 6, sensors: int = 3) -> np.ndarray:
    """A leak at node j lowers the pressure most at the sensor nearest to it (line network)."""
    pos = np.linspace(0, n - 1, sensors)
    d = np.abs(np.arange(n)[:, None] - pos[None, :])
    return -np.exp(-d / 1.5)


def test_whitener_inverts_the_regularized_covariance():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4))
    cov = a @ a.T + 0.1 * np.eye(4)
    w = whitener(cov, 0.01)
    np.testing.assert_allclose(w.T @ w, np.linalg.inv(regularized(cov, 0.01)), rtol=1e-8)
    assert np.trace(regularized(cov, 0.5)) == pytest.approx(1.5 * np.trace(cov))


def test_noise_free_leak_is_found_and_its_size_recovered():
    s = signatures()
    inv = Inversion(s, np.eye(3), 1e-6, np.ones(6, dtype=bool))
    x = 2.5 * s[[1, 4]]  # leaks 2.5x the template size at nodes 1 and 4
    a, err = inv.fit(x)
    assert list(np.argmin(err, axis=1)) == [1, 4]
    assert a[0, 1] == pytest.approx(2.5, rel=1e-4) and err[0, 1] == pytest.approx(0.0, abs=1e-6)
    ll = inv.loglik(x[:, :, None].repeat(2, axis=2))  # (B, sensors, windows) works too
    assert list(np.argmax(ll, axis=1)) == [1, 4]


def test_the_size_is_never_negative():
    s = signatures()
    inv = Inversion(s, np.eye(3), 1e-6, np.ones(6, dtype=bool))
    a, err = inv.fit(-s[[2]])  # pressure *rises* like an inverted leak
    assert (a >= 0).all()
    np.testing.assert_allclose(err[0], (s[2] ** 2).sum(), rtol=1e-6)  # best fit is no leak


def test_correlated_noise_is_discounted_by_the_covariance():
    """A common pressure drop at every sensor (demand) must not decide the leak node."""
    s = signatures(n=5, sensors=3)
    common = -np.ones(3)
    cov = 0.001 * np.eye(3) + 1.0 * np.outer(common, common)  # noise mostly along `common`
    x = 0.5 * s[[0]] + 2.0 * common  # small leak at node 0 plus a big demand dip
    plain = Inversion(s, np.eye(3), 1e-9, np.ones(5, dtype=bool))
    gls = Inversion(s, cov, 1e-9, np.ones(5, dtype=bool))
    assert int(np.argmax(plain.loglik(x))) != 0
    assert int(np.argmax(gls.loglik(x))) == 0


def test_non_junctions_are_never_chosen():
    s = signatures()
    junction = np.array([True, False, True, True, True, True])
    inv = Inversion(s, np.eye(3), 1e-6, junction)
    ll = inv.loglik(s[[1]])
    assert ll[0, 1] == -np.inf and int(np.argmax(ll)) != 1


def test_save_and_load_round_trip(tmp_path):
    s = signatures()
    inv = Inversion(s, 2 * np.eye(3), 0.03, np.ones(6, dtype=bool))
    back = Inversion.load(inv.save(tmp_path / "m" / "inv.npz"))
    assert back.ridge == pytest.approx(0.03)
    np.testing.assert_allclose(back.loglik(s), inv.loglik(s))


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="number of sensors"):
        Inversion(signatures(sensors=3), np.eye(4), 0.01, np.ones(6, dtype=bool))


def test_fit_residuals_remove_the_true_signature():
    s = signatures()
    x = 3.0 * s[[2, 5]]
    np.testing.assert_allclose(fit_residuals(x, s[[2, 5]]), 0.0, atol=1e-12)
    zero = fit_residuals(np.ones((1, 3)), np.zeros((1, 3)))  # no signature: residual is x
    np.testing.assert_allclose(zero, 1.0)


def test_second_moment_skips_incomplete_rows_and_needs_two():
    x = np.array([[1.0, 0.0], [np.nan, 1.0], [-1.0, 0.0]])
    np.testing.assert_allclose(second_moment(x), [[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="two complete rows"):
        second_moment(np.array([[1.0, np.nan], [2.0, 3.0]]))


def test_real_noise_windows_pair_consecutive_normal_hours():
    ends = pd.date_range("2018-01-01 01:00", periods=6, freq="15min", tz="UTC")  # 01:00..02:15
    feats = pd.DataFrame({"a": np.arange(6.0)}, index=ends)
    normal = np.ones(len(feats), dtype=bool)
    out = real_noise_windows(feats, ["a"], normal)
    # 02:00 pairs with 01:00 (values 4, 0) and 02:15 with 01:15 (5, 1)
    np.testing.assert_allclose(out, [[2.0], [3.0]])
    normal[0] = False  # 01:00 is not normal -> only the 02:15 pair is left
    np.testing.assert_allclose(real_noise_windows(feats, ["a"], normal), [[3.0]])


def test_ranks_and_rank_fusion():
    s = np.array([[3.0, 1.0, 2.0, -np.inf]])
    np.testing.assert_array_equal(ranks(s), [[0, 2, 1, 3]])
    fused = rank_fusion(
        {"a": s, "b": np.array([[0.0, 1.0, 4.0, 9.0]])}, np.array([True, True, True, False])
    )
    assert fused[0, 3] == -np.inf
    assert int(np.argmax(fused[0])) == 2  # 2nd in both beats 1st + 4th
    with pytest.raises(ValueError, match="at least one"):
        rank_fusion({}, np.ones(4, dtype=bool))


def test_interpolation_prefers_nodes_near_falling_sensors():
    dist = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :]) * 100.0
    x = np.array([[-1.0, 0.0]])  # sensors on nodes 0 and 4: the one on node 0 fell
    sc = interpolation_scores(dist, np.array([0, 4]), x, 100.0)
    assert int(np.argmax(sc[0])) == 0 and sc[0, 0] > sc[0, 2] > sc[0, 4]


def test_softmax_rows_and_describe():
    p = softmax_rows(np.array([[0.0, np.log(3.0), -np.inf]]))
    np.testing.assert_allclose(p, [[0.25, 0.75, 0.0]])
    assert describe({"gls": "sim"}) == "GLS (sim noise)"
    assert (
        describe({"gls": "real", "network": True, "sensor_prior_m": 300, "fusion": "rank"})
        == "rank fusion of GLS (real noise) + network + sensor prior 300 m"
    )
    assert describe({}) == "none"
