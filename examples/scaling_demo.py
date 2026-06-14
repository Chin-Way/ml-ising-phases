"""Finite-size scaling of the 2-D Ising transition.

Runs several lattice sizes, locates each one's susceptibility-peak T_c(L),
extrapolates to the thermodynamic limit (nu = 1), and checks the universality
class by collapsing the magnetization with the exact exponents (beta = 1/8).
Writes a two-panel scaling figure.

Run from the project root::

    python examples/scaling_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ising import scaling  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    sizes = (8, 16, 24, 32)
    print(f"finite-size scan over L = {sizes} ...")
    scan = scaling.finite_size_scan(sizes=sizes, n_eq=1000, n_samples=250, seed=0)
    tc_inf, slope = scaling.extrapolate_tc(scan["sizes"], scan["tc_of_L"])

    print(f"{'L':>4}{'T_c(L) [chi peak]':>20}")
    for L, tc in zip(scan["sizes"], scan["tc_of_L"]):
        print(f"{L:>4}{tc:>20.3f}")
    print(f"\nextrapolated T_c(inf) = {tc_inf:.4f}  (1/L -> 0)")
    print(f"T_c exact (Onsager)   = {scaling.TC_EXACT:.4f}")
    print(f"relative error        = {abs(tc_inf - scaling.TC_EXACT) / scaling.TC_EXACT * 100:.2f}%")
    print(f"\ncollapse spread (beta=1/8, nu=1) = {scaling.collapse_spread(scan):.4f}")
    print(f"collapse spread (beta=1/2 wrong)  = {scaling.collapse_spread(scan, beta=0.5):.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- left: T_c(L) vs 1/L with linear extrapolation ----------------------
    inv_L = 1.0 / scan["sizes"].astype(float)
    ax1.plot(inv_L, scan["tc_of_L"], "o", color="C0", ms=7, label="susceptibility peak")
    xfit = np.linspace(0.0, inv_L.max() * 1.05, 50)
    ax1.plot(xfit, tc_inf + slope * xfit, "-", color="C0", lw=1.3,
             label=f"fit $\\to$ {tc_inf:.3f}")
    ax1.plot(0.0, tc_inf, "D", color="C1", ms=8, label="extrapolated $T_c(\\infty)$")
    ax1.axhline(scaling.TC_EXACT, color="0.3", ls="--", lw=1.2,
                label=f"Onsager {scaling.TC_EXACT:.3f}")
    for L, x, tc in zip(scan["sizes"], inv_L, scan["tc_of_L"]):
        ax1.annotate(f"L={L}", (x, tc), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax1.set_xlabel("1 / L")
    ax1.set_ylabel("effective $T_c(L)$")
    ax1.set_title("Extrapolating $T_c$ to the thermodynamic limit  ($\\nu = 1$)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # --- right: magnetization data collapse ---------------------------------
    curves = scaling.collapse_curves(scan)
    for (L, (x, y)), color in zip(curves.items(), ["C0", "C1", "C2", "C3"]):
        ax2.plot(x, y, "o-", color=color, ms=3, lw=1, label=f"L = {L}")
    ax2.axvline(0.0, color="0.6", ls=":", lw=1)
    ax2.set_xlabel(r"$(T - T_c)\, L^{1/\nu}$")
    ax2.set_ylabel(r"$|M|\, L^{\beta/\nu}$")
    ax2.set_title(r"Magnetization collapse with 2-D Ising exponents ($\beta=\frac{1}{8},\ \nu=1$)")
    ax2.set_xlim(-12, 12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ising_scaling.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nfigure written to {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
