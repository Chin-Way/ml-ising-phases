"""Wolff single-cluster Monte Carlo -- beating critical slowing down.

Right at ``T_c`` the Metropolis sweep grinds to a halt: correlated domains span
the lattice, single-spin flips barely move them, and the autocorrelation time
diverges like ``L^z`` with ``z ~ 2.17`` ("critical slowing down"). The Wolff
algorithm sidesteps this by flipping a whole correlated **cluster** at once.

Grow a cluster from a random seed spin, adding each aligned neighbor with
probability ``p_add = 1 - e^{-2/T}`` (the value that makes the move satisfy
detailed balance for the Ising Hamiltonian), then flip the entire cluster. Near
``T_c`` the typical cluster *is* the correlated domain, so one update decorrelates
the configuration -- the dynamical exponent drops to ``z ~ 0.25``.

This is pure NumPy: the cluster grows with a plain stack (it is inherently
sequential), while energy and magnetization reuse the vectorized observables
from :mod:`ising.montecarlo`. The sampler mirrors that module's
:func:`~ising.montecarlo.simulate` so the two are drop-in comparable.
"""

from __future__ import annotations

import numpy as np

from .montecarlo import energy_per_spin, magnetization_per_spin


def wolff_step(spins: np.ndarray, T: float, rng: np.random.Generator) -> int:
    """Grow one Wolff cluster and flip it, in place. Returns the cluster size."""
    L = spins.shape[0]
    p_add = 1.0 - np.exp(-2.0 / T)

    i0 = int(rng.integers(L))
    j0 = int(rng.integers(L))
    s0 = spins[i0, j0]

    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[i0, j0] = True
    stack = [(i0, j0)]
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni = (i + di) % L
            nj = (j + dj) % L
            if spins[ni, nj] == s0 and not in_cluster[ni, nj] and rng.random() < p_add:
                in_cluster[ni, nj] = True
                stack.append((ni, nj))

    spins[in_cluster] *= -1  # all cluster sites shared spin s0 -> flip to -s0
    return int(in_cluster.sum())


def simulate_wolff(
    L: int,
    T: float,
    n_eq: int = 200,
    n_samples: int = 100,
    sample_every: int = 1,
    seed: int | None = None,
):
    """Equilibrate with cluster moves, then collect ``n_samples`` configurations.

    Drop-in counterpart of :func:`ising.montecarlo.simulate`: same return dict
    (``T``, ``configs``, ``abs_M``, ``energy``). Because each cluster move
    decorrelates so effectively, ``n_eq`` and ``sample_every`` can be far smaller
    than their Metropolis equivalents.
    """
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)

    for _ in range(n_eq):
        wolff_step(spins, T, rng)

    configs = np.empty((n_samples, L, L), dtype=np.int8)
    abs_M = np.empty(n_samples)
    energy = np.empty(n_samples)
    for k in range(n_samples):
        for _ in range(sample_every):
            wolff_step(spins, T, rng)
        configs[k] = spins
        abs_M[k] = abs(magnetization_per_spin(spins))
        energy[k] = energy_per_spin(spins)

    return {"T": T, "configs": configs, "abs_M": abs_M, "energy": energy}


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation function ``rho(t)`` (``rho[0] = 1``) via FFT."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n == 0 or np.dot(x, x) == 0.0:
        return np.ones(max(n, 1))
    f = np.fft.rfft(x, 2 * n)
    acov = np.fft.irfft(f * np.conjugate(f))[:n]
    return acov / acov[0]


def integrated_autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time ``tau`` with Sokal automatic windowing.

    ``tau = 1/2 + sum_{t>=1} rho(t)``, truncated at the first window ``W`` with
    ``W >= c * tau(W)``. White noise gives ``tau ~ 1/2``; a strongly correlated
    series gives a large ``tau``. Units are sampling intervals.
    """
    rho = _autocorrelation(x)
    tau = 0.5
    for w in range(1, len(rho)):
        tau += rho[w]
        if w >= c * tau:
            break
    return float(max(tau, 0.5))
