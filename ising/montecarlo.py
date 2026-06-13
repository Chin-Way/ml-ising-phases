"""2-D Ising model via Metropolis Monte Carlo.

The Hamiltonian (ferromagnet, no external field, in units of J = k_B = 1) is

    H = - sum_<ij> s_i s_j ,   s_i in {+1, -1}

on an L x L lattice with periodic boundary conditions. Onsager's exact result
gives a continuous phase transition at the critical temperature

    T_c = 2 / ln(1 + sqrt(2)) ~ 2.269 ,

below which the system spontaneously magnetizes. Updates use a vectorized
**checkerboard** sweep: all sites of one sublattice are independent (they are
never neighbors), so an entire color can be updated at once with NumPy.
"""

from __future__ import annotations

import numpy as np

TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ~ 2.2692


def _checkerboard(L: int):
    """Boolean masks for the two interpenetrating sublattices."""
    parity = (np.indices((L, L)).sum(axis=0)) % 2
    return parity == 0, parity == 1


def mc_sweep(spins: np.ndarray, T: float, rng: np.random.Generator, masks=None) -> np.ndarray:
    """One full Metropolis sweep (both sublattices), in place."""
    if masks is None:
        masks = _checkerboard(spins.shape[0])
    beta = 1.0 / T
    for mask in masks:
        neighbors = (
            np.roll(spins, 1, 0)
            + np.roll(spins, -1, 0)
            + np.roll(spins, 1, 1)
            + np.roll(spins, -1, 1)
        )
        dE = 2.0 * spins * neighbors  # cost of flipping each site
        accept = (rng.random(spins.shape) < np.exp(-beta * dE)) & mask
        spins[accept] *= -1
    return spins


def energy_per_spin(spins: np.ndarray) -> float:
    """Energy per spin; right/down neighbors count each bond exactly once."""
    bonds = spins * (np.roll(spins, 1, 0) + np.roll(spins, 1, 1))
    return float(-np.sum(bonds) / spins.size)


def magnetization_per_spin(spins: np.ndarray) -> float:
    """Net magnetization per spin in [-1, 1]."""
    return float(np.mean(spins))


def simulate(
    L: int,
    T: float,
    n_eq: int = 1000,
    n_samples: int = 100,
    sample_every: int = 5,
    seed: int | None = None,
):
    """Equilibrate at temperature ``T`` then collect ``n_samples`` configurations.

    Returns a dict with the raw ``configs`` (shape ``(n_samples, L, L)``) and the
    per-sample ``abs_M`` and ``energy`` arrays.
    """
    rng = np.random.default_rng(seed)
    masks = _checkerboard(L)
    spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)

    for _ in range(n_eq):
        mc_sweep(spins, T, rng, masks)

    configs = np.empty((n_samples, L, L), dtype=np.int8)
    abs_M = np.empty(n_samples)
    energy = np.empty(n_samples)
    for i in range(n_samples):
        for _ in range(sample_every):
            mc_sweep(spins, T, rng, masks)
        configs[i] = spins
        abs_M[i] = abs(magnetization_per_spin(spins))
        energy[i] = energy_per_spin(spins)

    return {"T": T, "configs": configs, "abs_M": abs_M, "energy": energy}


def thermodynamics(L: int, temperatures, n_eq=1000, n_samples=200, sample_every=5, seed=0):
    """Sweep temperatures and return mean observables (and their fluctuations).

    Susceptibility chi = N (<M^2> - <|M|>^2) / T and specific heat
    c = N (<E^2> - <E>^2) / T^2 both peak near T_c.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    N = L * L
    out = {k: np.empty(len(temperatures)) for k in ("M", "E", "chi", "C")}
    for j, T in enumerate(temperatures):
        sim = simulate(L, T, n_eq=n_eq, n_samples=n_samples, sample_every=sample_every, seed=seed + j)
        M, E = sim["abs_M"], sim["energy"]
        out["M"][j] = M.mean()
        out["E"][j] = E.mean()
        out["chi"][j] = N * M.var() / T
        out["C"][j] = N * E.var() / T**2
    out["T"] = temperatures
    return out
