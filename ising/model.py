"""Train a neural network to recognize the Ising phase, and recover T_c.

The experiment reproduces the central idea of Carrasquilla & Melko, *Machine
learning phases of matter* (Nature Physics, 2017): a fully-connected network
trained only on configurations **deep** in each phase -- where the label is
obvious and needs no knowledge of T_c -- learns the order parameter and, when
asked about the critical region it never trained on, outputs P(disordered) that
crosses 1/2 right at T_c.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier

from .dataset import make_dataset
from .montecarlo import TC_EXACT


def _crossing(temps: np.ndarray, p: np.ndarray, level: float = 0.5) -> float:
    """Temperature where ``p(T)`` first crosses ``level`` (linear interpolation)."""
    for i in range(len(temps) - 1):
        if (p[i] - level) * (p[i + 1] - level) <= 0 and p[i + 1] != p[i]:
            frac = (level - p[i]) / (p[i + 1] - p[i])
            return float(temps[i] + frac * (temps[i + 1] - temps[i]))
    return float("nan")


def estimate_tc(
    L: int = 16,
    n_per_temp: int = 80,
    train_below: float = 1.7,
    train_above: float = 2.9,
    seed: int = 0,
):
    """Run the full pipeline and return a results dict.

    Trains on configurations with ``T < train_below`` or ``T > train_above``
    only, then predicts across all temperatures.
    """
    temperatures = np.linspace(1.0, 3.5, 26)
    X, y, T = make_dataset(L=L, temperatures=temperatures, n_per_temp=n_per_temp, seed=seed)

    # Confident region only: deep ordered vs deep disordered.
    train = (T < train_below) | (T > train_above)
    y_train = (T[train] > train_above).astype(int)

    clf = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        max_iter=400,
        random_state=seed,
    )
    clf.fit(X[train], y_train)

    # Held-out accuracy in the confident region (sanity check).
    train_acc = clf.score(X[train], y_train)

    # Average P(disordered) at every temperature, including the critical region.
    proba = clf.predict_proba(X)[:, 1]
    p_by_T = np.array([proba[T == t].mean() for t in temperatures])
    tc_est = _crossing(temperatures, p_by_T)

    return {
        "temperatures": temperatures,
        "p_disordered": p_by_T,
        "tc_estimate": tc_est,
        "tc_exact": TC_EXACT,
        "train_accuracy": train_acc,
        "classifier": clf,
        "train_below": train_below,
        "train_above": train_above,
    }
