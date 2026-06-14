"""The CNN must train on the deep phases and recover a sane T_c crossing.

Kept deliberately tiny (L = 8, few epochs) so CI stays fast -- we assert that it
learns the confident region and that its P(disordered) curve crosses 1/2 in the
critical neighborhood, not that it nails the exact value.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ising import cnn, compare  # noqa: E402


def test_cnn_trains_and_crosses_in_range():
    r = cnn.estimate_tc_cnn(L=8, n_per_temp=24, epochs=20, seed=0)
    # It separates the deep-phase configurations it trained on...
    assert r["train_accuracy"] > 0.9
    # ...and its P(disordered) crossing is finite and inside the swept window.
    tc = r["tc_estimate"]
    assert np.isfinite(tc)
    assert 1.5 < tc < 3.2
    # The output actually rises through 1/2: cold phase low, hot phase high.
    p = r["p_disordered"]
    assert p[0] < 0.4 and p[-1] > 0.6


def test_cnn_is_translation_and_size_agnostic():
    """Global average pooling means the same net runs on any L and ignores shifts."""
    net = cnn.train_cnn(
        *_tiny_training_set(L=8), L=8, epochs=5, seed=0
    )
    rng = np.random.default_rng(0)
    config = rng.choice([-1.0, 1.0], size=(1, 64))
    base = cnn.cnn_proba(net, config, L=8)
    rolled = np.roll(config.reshape(8, 8), shift=3, axis=0).reshape(1, 64)
    shifted = cnn.cnn_proba(net, rolled, L=8)
    assert np.allclose(base, shifted, atol=1e-5)
    # The very same weights also accept a different lattice size without error.
    bigger = rng.choice([-1.0, 1.0], size=(2, 144))
    assert cnn.cnn_proba(net, bigger, L=12).shape == (2,)


def test_compare_returns_both_models():
    r = compare.compare(L=8, n_per_temp=24, epochs=15, seed=0)
    # Smoke test for the wiring: on a tiny lattice (where finite-size noise makes
    # the precise crossing wander) both models still train, learn the right
    # direction, and yield a finite crossing inside the swept window. The
    # near-critical precision bars live in the per-model tests at larger L.
    grid = r["temperatures"]
    for name in ("mlp", "cnn"):
        sub = r[name]
        assert sub["train_accuracy"] > 0.8
        tc = sub["tc_estimate"]
        assert np.isfinite(tc) and grid.min() <= tc <= grid.max()
        p = sub["p_disordered"]
        assert p[0] < p[-1]  # cold phase classed ordered, hot phase disordered
        assert len(p) == len(grid)


def _tiny_training_set(L: int):
    """A trivially separable ordered/disordered set for shape-level tests."""
    rng = np.random.default_rng(1)
    ordered = np.ones((10, L * L))
    ordered[5:] = -1.0  # both ground states
    disordered = rng.choice([-1.0, 1.0], size=(10, L * L))
    X = np.vstack([ordered, disordered])
    y = np.array([0] * 10 + [1] * 10)
    return X, y
