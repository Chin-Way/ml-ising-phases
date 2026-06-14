"""ising -- 2-D Ising Monte Carlo plus phase classifiers.

* :mod:`ising.montecarlo` -- vectorized Metropolis simulation and observables.
* :mod:`ising.dataset`    -- labeled spin configurations across temperatures.
* :mod:`ising.model`      -- a fully-connected net that learns the phase, recovers T_c.
* :mod:`ising.cnn`        -- a small convolutional net on the raw lattice (PyTorch).
* :mod:`ising.compare`    -- a fair MLP-vs-CNN head-to-head on one dataset.

The Monte Carlo + analysis core is import-light (NumPy, scikit-learn). The
convolutional pieces pull in PyTorch, so :mod:`ising.cnn` and
:mod:`ising.compare` are imported lazily -- ``import ising`` stays cheap and
torch-free until you actually reach for them.
"""

import importlib

from . import dataset, model, montecarlo
from .montecarlo import TC_EXACT, simulate, thermodynamics

__all__ = [
    "montecarlo",
    "dataset",
    "model",
    "cnn",
    "compare",
    "simulate",
    "thermodynamics",
    "TC_EXACT",
]
__version__ = "0.2.0"

_LAZY = {"cnn", "compare"}


def __getattr__(name):
    """Import the torch-backed submodules on first access (PEP 562)."""
    if name in _LAZY:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
