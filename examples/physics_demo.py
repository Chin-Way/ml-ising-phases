"""Recover the Ising phase transition from Monte Carlo observables.

Sweeps temperature and plots magnetization, energy, susceptibility, and
specific heat -- the last two peak at T_c.

Run from the project root::

    python examples/physics_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ising import montecarlo as mc  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    L = 24
    temps = np.linspace(1.2, 3.4, 30)
    print(f"simulating L={L} lattice across {len(temps)} temperatures...")
    thermo = mc.thermodynamics(L=L, temperatures=temps, n_eq=1000, n_samples=250, seed=0)

    tc_chi = temps[np.argmax(thermo["chi"])]
    tc_c = temps[np.argmax(thermo["C"])]
    print(f"T_c (exact)              = {mc.TC_EXACT:.3f}")
    print(f"T_c (susceptibility peak) = {tc_chi:.3f}")
    print(f"T_c (specific-heat peak)  = {tc_c:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        ("M", "|magnetization| per spin", "C0"),
        ("E", "energy per spin", "C1"),
        ("chi", "susceptibility  χ", "C2"),
        ("C", "specific heat  c", "C3"),
    ]
    for ax, (key, label, color) in zip(axes.flat, panels):
        ax.plot(temps, thermo[key], "o-", color=color, ms=4)
        ax.axvline(mc.TC_EXACT, color="0.4", ls="--", lw=1, label=f"$T_c$ = {mc.TC_EXACT:.2f}")
        ax.set_xlabel("temperature T")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"2-D Ising model observables (L = {L})")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ising_observables.png")
    fig.savefig(out, dpi=130)
    print(f"figure written to {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
