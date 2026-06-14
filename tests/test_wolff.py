"""The Wolff cluster sampler must reproduce the same physics as Metropolis,
and decorrelate without the critical slowing down that plagues single flips."""

import numpy as np

from ising import montecarlo as mc
from ising import wolff


def test_wolff_step_flips_a_cluster():
    rng = np.random.default_rng(0)
    spins = rng.choice([-1, 1], size=(8, 8)).astype(np.int8)
    before = spins.copy()
    size = wolff.wolff_step(spins, T=2.0, rng=rng)
    assert 1 <= size <= spins.size
    assert spins.dtype == np.int8
    # Exactly the cluster sites flipped sign; the rest are untouched.
    assert int((spins != before).sum()) == size


def test_wolff_orders_and_disorders():
    cold = wolff.simulate_wolff(L=16, T=1.5, n_eq=150, n_samples=80, seed=1)
    hot = wolff.simulate_wolff(L=16, T=3.5, n_eq=150, n_samples=80, seed=2)
    assert cold["abs_M"].mean() > 0.9
    assert hot["abs_M"].mean() < 0.3


def test_wolff_matches_metropolis_observables():
    """Same magnetization and energy as the Metropolis engine, within MC error."""
    T = 2.0
    m = mc.simulate(L=12, T=T, n_eq=600, n_samples=120, sample_every=4, seed=3)
    w = wolff.simulate_wolff(L=12, T=T, n_eq=150, n_samples=120, sample_every=2, seed=3)
    assert abs(m["abs_M"].mean() - w["abs_M"].mean()) < 0.1
    assert abs(m["energy"].mean() - w["energy"].mean()) < 0.1


def test_integrated_autocorr_time_estimator():
    # White noise is essentially uncorrelated: tau ~ 1/2.
    white = np.random.default_rng(0).standard_normal(4000)
    assert wolff.integrated_autocorr_time(white) < 1.0
    # A strongly correlated AR(1) series has a large tau.
    phi, x = 0.8, np.zeros(8000)
    rng = np.random.default_rng(1)
    for t in range(1, len(x)):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    assert wolff.integrated_autocorr_time(x) > 2.0


def test_wolff_has_no_critical_slowing_down():
    """At T_c the cluster update stays fast to decorrelate (tau is small)."""
    rng = np.random.default_rng(0)
    spins = rng.choice([-1, 1], size=(16, 16)).astype(np.int8)
    for _ in range(150):
        wolff.wolff_step(spins, mc.TC_EXACT, rng)
    series = np.empty(1500)
    for k in range(series.size):
        wolff.wolff_step(spins, mc.TC_EXACT, rng)
        series[k] = abs(spins.mean())
    assert wolff.integrated_autocorr_time(series) < 6.0
