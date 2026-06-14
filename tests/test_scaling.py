"""Finite-size scaling: helpers are exact; the small scan extrapolates sanely."""

import numpy as np

from ising import scaling
from ising.montecarlo import TC_EXACT


def test_parabolic_peak_recovers_vertex():
    x = np.array([0.0, 1.0, 2.0])
    y = -((x - 1.3) ** 2)  # parabola peaked at 1.3, grid max at the middle point
    assert abs(scaling._parabolic_peak(x, y) - 1.3) < 1e-9


def test_parabolic_peak_falls_back_at_boundary():
    x = np.linspace(0, 1, 5)
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # maximum at the left edge
    assert scaling._parabolic_peak(x, y) == x[0]


def test_extrapolate_tc_recovers_a_known_line():
    # T_c(L) = 2.27 + 1.5 / L exactly -> intercept 2.27, slope 1.5.
    sizes = np.array([8, 16, 24, 32])
    tc_of_L = 2.27 + 1.5 / sizes
    tc_inf, slope = scaling.extrapolate_tc(sizes, tc_of_L)
    assert abs(tc_inf - 2.27) < 1e-9
    assert abs(slope - 1.5) < 1e-9


def test_small_scan_extrapolates_near_onsager():
    temps = np.linspace(1.8, 3.0, 13)
    scan = scaling.finite_size_scan(sizes=(8, 16), temperatures=temps, n_eq=400, n_samples=80, seed=0)
    # Effective T_c(L) sits in the critical neighborhood, above the true value.
    for tc in scan["tc_of_L"]:
        assert 2.0 < tc < 3.0
    tc_inf, _ = scaling.extrapolate_tc(scan["sizes"], scan["tc_of_L"])
    # Only two small sizes -> a rough extrapolation, but it should land close.
    assert abs(tc_inf - TC_EXACT) < 0.4


def test_correct_exponents_collapse_better():
    """The hallmark of the universality class: beta = 1/8 collapses the data."""
    temps = np.linspace(1.8, 3.0, 13)
    scan = scaling.finite_size_scan(sizes=(8, 16), temperatures=temps, n_eq=400, n_samples=80, seed=0)
    good = scaling.collapse_spread(scan, beta=scaling.BETA, nu=scaling.NU)
    bad = scaling.collapse_spread(scan, beta=0.5, nu=scaling.NU)
    assert np.isfinite(good)
    assert good < bad
    # Shapes line up with the temperature grid.
    curves = scaling.collapse_curves(scan)
    for L, (x, y) in curves.items():
        assert x.shape == y.shape == temps.shape
