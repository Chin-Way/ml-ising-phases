"""Does spatial structure help? CNN vs. MLP on the Ising transition.

Trains the fully-connected baseline (:mod:`ising.model`) and a small
convolutional net (:mod:`ising.cnn`) on the *same* deep-phase configurations,
then asks both for the ``P(disordered) = 1/2`` crossing in the critical region
neither saw. Prints a comparison table and writes a two-curve figure.

Run from the project root::

    python examples/cnn_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ising import compare  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    print("training MLP and CNN on the same deep-phase configurations...")
    result = compare.compare(L=24, n_per_temp=120, epochs=40, seed=0)
    print(compare.format_report(result))

    temps = result["temperatures"]
    tc_exact = result["tc_exact"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axvspan(temps.min(), 1.7, color="0.5", alpha=0.07)
    ax.axvspan(2.9, temps.max(), color="0.5", alpha=0.07)
    ax.text(temps.min() + 0.05, 0.92, "trained\nordered", fontsize=8, color="0.4")
    ax.text(2.95, 0.05, "trained\ndisordered", fontsize=8, color="0.4")

    for name, color, marker in [("mlp", "C0", "o"), ("cnn", "C3", "s")]:
        r = result[name]
        ax.plot(
            temps,
            r["p_disordered"],
            marker=marker,
            ls="-",
            color=color,
            ms=4,
            label=f"{name.upper()}  (T_c ≈ {r['tc_estimate']:.3f})",
        )
        ax.axvline(r["tc_estimate"], color=color, ls="-", lw=1, alpha=0.6)

    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.axvline(tc_exact, color="0.2", ls="--", lw=1.3, label=f"$T_c$ exact = {tc_exact:.3f}")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("P(disordered)")
    ax.set_title(f"MLP vs. CNN locating the Ising transition (L = {result['L']})")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "cnn_vs_mlp.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"figure written to {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
