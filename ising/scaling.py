"""Finite-size scaling: push several lattice sizes toward the thermodynamic limit.

On a finite ``L x L`` lattice there is no true singularity -- the susceptibility
peak is rounded and sits at an *effective* critical temperature ``T_c(L)`` that
drifts with size. Finite-size scaling theory says that drift is itself
universal:

    T_c(L) = T_c(infinity) + a * L^{-1/nu},      nu = 1 for the 2-D Ising model,

so plotting ``T_c(L)`` against ``1/L`` and extrapolating to ``1/L -> 0`` recovers
Onsager's value. A second, sharper check is **data collapse**: the magnetization
obeys

    M(T, L) ~ L^{-beta/nu} * f((T - T_c) L^{1/nu}),

so rescaling the axes with the 2-D Ising exponents (``beta = 1/8``, ``nu = 1``)
should drop every size's curve onto one universal function -- a direct read-out
of the universality class.
"""

from __future__ import annotations

import numpy as np

from .montecarlo import TC_EXACT, thermodynamics

# 2-D Ising critical exponents (exact).
BETA = 0.125
NU = 1.0


def _parabolic_peak(x: np.ndarray, y: np.ndarray) -> float:
    """Sub-grid location of the maximum of ``y(x)`` via 3-point parabola.

    Fits a parabola through the grid maximum and its two neighbors and returns
    the vertex. Falls back to the grid maximum at the boundary or if the points
    are not concave (noise), and clamps the vertex to stay within the bracket.
    """
    i = int(np.argmax(y))
    if i == 0 or i == len(x) - 1:
        return float(x[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom >= 0:  # not a concave peak -- trust the grid point
        return float(x[i])
    h = x[i + 1] - x[i]
    delta = 0.5 * h * (y0 - y2) / denom
    delta = float(np.clip(delta, -h, h))
    return float(x[i] + delta)


def finite_size_scan(
    sizes=(8, 16, 24, 32),
    temperatures=None,
    n_eq: int = 1000,
    n_samples: int = 200,
    sample_every: int = 5,
    seed: int = 0,
):
    """Run :func:`thermodynamics` at each size and record the susceptibility peak.

    Returns a dict with the ``sizes`` and ``temperatures`` grids, the
    parabolically-refined ``tc_of_L`` (effective critical temperature from the
    chi peak), and the per-size ``M`` and ``chi`` curves for the data collapse.
    """
    if temperatures is None:
        temperatures = np.linspace(1.6, 3.2, 33)
    temperatures = np.asarray(temperatures, dtype=float)

    sizes = np.asarray(sizes, dtype=int)
    tc_of_L = np.empty(len(sizes))
    M = {}
    chi = {}
    for k, L in enumerate(sizes):
        thermo = thermodynamics(
            L=int(L),
            temperatures=temperatures,
            n_eq=n_eq,
            n_samples=n_samples,
            sample_every=sample_every,
            seed=seed + k,
        )
        tc_of_L[k] = _parabolic_peak(temperatures, thermo["chi"])
        M[int(L)] = thermo["M"]
        chi[int(L)] = thermo["chi"]

    return {
        "sizes": sizes,
        "temperatures": temperatures,
        "tc_of_L": tc_of_L,
        "M": M,
        "chi": chi,
    }


def extrapolate_tc(sizes: np.ndarray, tc_of_L: np.ndarray):
    """Linear fit ``T_c(L) = T_c_inf + a / L`` (since ``nu = 1``).

    Returns ``(tc_inf, slope)``: the ``1/L -> 0`` intercept is the estimate of
    the thermodynamic-limit critical temperature.
    """
    sizes = np.asarray(sizes, dtype=float)
    inv_L = 1.0 / sizes
    slope, intercept = np.polyfit(inv_L, np.asarray(tc_of_L, dtype=float), 1)
    return float(intercept), float(slope)


def collapse_curves(scan: dict, tc: float = TC_EXACT, beta: float = BETA, nu: float = NU):
    """Rescale each size's magnetization for a data-collapse plot.

    Returns ``{L: (x, y)}`` with ``x = (T - tc) L^{1/nu}`` and
    ``y = M L^{beta/nu}``. With the correct exponents the curves overlap.
    """
    temperatures = scan["temperatures"]
    out = {}
    for L in scan["sizes"]:
        L = int(L)
        x = (temperatures - tc) * L ** (1.0 / nu)
        y = scan["M"][L] * L ** (beta / nu)
        out[L] = (x, y)
    return out


def collapse_spread(scan: dict, tc: float = TC_EXACT, beta: float = BETA, nu: float = NU) -> float:
    """Quantify how well the rescaled magnetization curves overlap (lower = better).

    Interpolates every size's rescaled curve onto a common abscissa (the range
    all sizes share) and returns the mean spread across sizes. Useful as a
    quantitative check that the 2-D Ising exponents collapse the data better than
    wrong ones.
    """
    curves = collapse_curves(scan, tc=tc, beta=beta, nu=nu)
    xs = list(curves.values())
    lo = max(x.min() for x, _ in xs)
    hi = min(x.max() for x, _ in xs)
    if not hi > lo:
        return float("nan")
    grid = np.linspace(lo, hi, 60)
    stacked = []
    for x, y in xs:
        order = np.argsort(x)
        stacked.append(np.interp(grid, x[order], y[order]))
    stacked = np.vstack(stacked)
    return float(stacked.std(axis=0).mean())
