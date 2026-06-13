"""The classifier must separate the phases and recover T_c."""

import numpy as np

from ising import dataset, model
from ising.montecarlo import TC_EXACT


def test_dataset_shapes_and_labels():
    temps = np.linspace(1.0, 3.5, 6)
    X, y, T = dataset.make_dataset(L=8, temperatures=temps, n_per_temp=10, n_eq=200, seed=0)
    assert X.shape == (60, 64)
    assert set(np.unique(y)) <= {0, 1}
    assert np.all((X == 1) | (X == -1))
    # labels follow the exact T_c
    assert np.all((T > TC_EXACT) == (y == 1))


def test_estimate_tc_is_close_and_confident():
    r = model.estimate_tc(L=16, n_per_temp=50, seed=0)
    # The network nails the deep-phase configurations...
    assert r["train_accuracy"] > 0.98
    # ...and the P=1/2 crossing lands near the exact T_c (finite-size tolerance).
    assert abs(r["tc_estimate"] - r["tc_exact"]) < 0.25
    # P(disordered) is monotone-ish: low in the cold phase, high in the hot one.
    p = r["p_disordered"]
    assert p[0] < 0.2 and p[-1] > 0.8
