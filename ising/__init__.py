"""ising -- 2-D Ising Monte Carlo plus a phase classifier.

* :mod:`ising.montecarlo` -- vectorized Metropolis simulation and observables.
* :mod:`ising.dataset`    -- labeled spin configurations across temperatures.
* :mod:`ising.model`      -- a neural net that learns the phase and recovers T_c.
"""

from . import dataset, model, montecarlo
from .montecarlo import TC_EXACT, simulate, thermodynamics

__all__ = ["montecarlo", "dataset", "model", "simulate", "thermodynamics", "TC_EXACT"]
__version__ = "0.1.0"
