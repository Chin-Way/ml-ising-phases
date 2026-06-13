"""Turn Monte Carlo configurations into a labeled machine-learning dataset.

Each example is a flattened L x L spin configuration; the label is the phase
(0 = ordered / ferromagnetic, 1 = disordered / paramagnetic) according to the
exact critical temperature. The phase label is only used for evaluation -- the
headline experiment (see ``ising.model``) trains *without* it near T_c.
"""

from __future__ import annotations

import numpy as np

from .montecarlo import TC_EXACT, simulate


def make_dataset(
    L: int = 16,
    temperatures=None,
    n_per_temp: int = 80,
    n_eq: int = 800,
    sample_every: int = 4,
    seed: int = 0,
):
    """Generate ``(X, y, T)`` across a temperature grid.

    ``X`` has shape ``(n_samples, L*L)`` with entries in {-1, +1}; ``y`` is the
    exact phase label; ``T`` is the temperature each sample was drawn at.
    """
    if temperatures is None:
        temperatures = np.linspace(1.0, 3.5, 26)
    temperatures = np.asarray(temperatures, dtype=float)

    X_blocks, y_blocks, T_blocks = [], [], []
    for j, T in enumerate(temperatures):
        sim = simulate(L, T, n_eq=n_eq, n_samples=n_per_temp, sample_every=sample_every, seed=seed + j)
        configs = sim["configs"].reshape(n_per_temp, -1).astype(np.float64)
        X_blocks.append(configs)
        y_blocks.append(np.full(n_per_temp, int(T > TC_EXACT)))
        T_blocks.append(np.full(n_per_temp, T))

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    T = np.concatenate(T_blocks)
    return X, y, T
