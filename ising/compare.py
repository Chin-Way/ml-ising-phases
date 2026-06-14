"""A fair head-to-head: fully-connected MLP vs. convolutional net.

Both classifiers see the *same* Monte Carlo dataset and the *same* deep-phase
training split; the only difference is the inductive bias. The MLP flattens the
lattice and treats every spin as an independent feature; the CNN keeps the
lattice intact and looks at it locally with periodic padding. We then ask each
for the ``P(disordered) = 1/2`` crossing in the critical region neither trained
on, and compare both estimates against Onsager's exact ``T_c``.

The honest takeaway (see the README): the order parameter ``|M|`` is a global,
easily-learned feature, so the MLP baseline is already very good. The CNN's
spatial bias does not need to "rescue" anything here -- the interesting result
is that a *translation-invariant* network, which can only see the lattice
through local filters and a global average, recovers ``T_c`` just as well.
"""

from __future__ import annotations

import numpy as np

from . import cnn
from .dataset import make_dataset
from .model import average_by_temperature, crossing, train_mlp
from .montecarlo import TC_EXACT


def compare(
    L: int = 24,
    n_per_temp: int = 120,
    train_below: float = 1.7,
    train_above: float = 2.9,
    epochs: int = 40,
    seed: int = 0,
):
    """Train both models on one shared dataset and return their results.

    Returns a dict with a ``temperatures`` grid, the exact ``tc_exact``, and an
    ``mlp`` and ``cnn`` sub-dict each holding ``train_accuracy``, the per-T
    ``p_disordered`` curve, and the ``tc_estimate`` crossing.
    """
    temperatures = np.linspace(1.0, 3.5, 26)
    X, y, T = make_dataset(L=L, temperatures=temperatures, n_per_temp=n_per_temp, seed=seed)

    # Identical confident-region split for both models.
    train = (T < train_below) | (T > train_above)
    y_train = (T[train] > train_above).astype(int)

    def _summary(proba: np.ndarray, train_acc: float) -> dict:
        p_by_T = average_by_temperature(proba, T, temperatures)
        return {
            "train_accuracy": float(train_acc),
            "p_disordered": p_by_T,
            "tc_estimate": crossing(temperatures, p_by_T),
        }

    # --- fully-connected baseline -------------------------------------------
    clf = train_mlp(X[train], y_train, seed)
    mlp = _summary(clf.predict_proba(X)[:, 1], clf.score(X[train], y_train))

    # --- convolutional network ----------------------------------------------
    net = cnn.train_cnn(X[train], y_train, L, epochs=epochs, seed=seed)
    cnn_train_acc = ((cnn.cnn_proba(net, X[train], L) > 0.5).astype(int) == y_train).mean()
    cnn_res = _summary(cnn.cnn_proba(net, X, L), cnn_train_acc)

    return {
        "temperatures": temperatures,
        "tc_exact": TC_EXACT,
        "L": L,
        "mlp": mlp,
        "cnn": cnn_res,
        "classifier": clf,
        "network": net,
    }


def _rel_err(tc: float) -> float:
    return abs(tc - TC_EXACT) / TC_EXACT


def format_report(result: dict) -> str:
    """A compact text table summarizing a :func:`compare` run."""
    lines = [
        f"L = {result['L']}    T_c (exact, Onsager) = {result['tc_exact']:.3f}",
        f"{'model':<8}{'train acc':>11}{'T_c est':>10}{'rel err':>9}",
    ]
    for name in ("mlp", "cnn"):
        r = result[name]
        lines.append(
            f"{name.upper():<8}{r['train_accuracy']:>11.3f}{r['tc_estimate']:>10.3f}"
            f"{_rel_err(r['tc_estimate']) * 100:>8.1f}%"
        )
    return "\n".join(lines)
