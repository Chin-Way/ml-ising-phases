"""The Monte Carlo engine must reproduce known Ising physics."""

import numpy as np

from ising import montecarlo as mc


def test_critical_temperature_constant():
    # Onsager: T_c = 2 / ln(1 + sqrt2) ~ 2.2692
    assert abs(mc.TC_EXACT - 2.2692) < 1e-3


def test_ground_state_energy():
    """A fully aligned lattice has energy -2 per spin (4 bonds, each shared)."""
    spins = np.ones((8, 8), dtype=np.int8)
    assert np.isclose(mc.energy_per_spin(spins), -2.0)
    spins[:] = -1
    assert np.isclose(mc.energy_per_spin(spins), -2.0)


def test_magnetization_bounds():
    rng = np.random.default_rng(0)
    spins = rng.choice([-1, 1], size=(10, 10))
    assert -1.0 <= mc.magnetization_per_spin(spins) <= 1.0


def test_low_temperature_orders():
    """Well below T_c the system magnetizes (|M| close to 1)."""
    sim = mc.simulate(L=16, T=1.0, n_eq=500, n_samples=40, sample_every=4, seed=1)
    assert sim["abs_M"].mean() > 0.9


def test_high_temperature_disorders():
    """Well above T_c the magnetization is small."""
    sim = mc.simulate(L=16, T=4.0, n_eq=500, n_samples=40, sample_every=4, seed=2)
    assert sim["abs_M"].mean() < 0.2


def test_susceptibility_peaks_near_tc():
    """The magnetic susceptibility should peak in the critical region."""
    temps = np.linspace(1.5, 3.2, 12)
    thermo = mc.thermodynamics(L=16, temperatures=temps, n_eq=600, n_samples=120, seed=0)
    t_peak = temps[np.argmax(thermo["chi"])]
    assert abs(t_peak - mc.TC_EXACT) < 0.5
