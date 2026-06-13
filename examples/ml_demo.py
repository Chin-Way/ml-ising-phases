"""Machine learning the Ising phase transition (Carrasquilla & Melko 2017).

Train a small neural network on spin configurations drawn *deep* in each phase,
then ask it about the critical region it never saw. Its P(disordered) output
crosses 1/2 right at T_c.

Run from the project root::

    python examples/ml_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ising import model  # noqa: E402
from ising import montecarlo as mc  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def sample_configs_figure(L=24):
    """Show one ordered, one critical, and one disordered configuration."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, T, name in zip(axes, [1.2, mc.TC_EXACT, 3.4], ["ordered", "critical", "disordered"]):
        sim = mc.simulate(L, T, n_eq=800, n_samples=1, sample_every=1, seed=3)
        ax.imshow(sim["configs"][0], cmap="binary", interpolation="nearest")
        ax.set_title(f"{name}\nT = {T:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"2-D Ising spin configurations (L = {L})")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ising_configs.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"figure written to {os.path.normpath(out)}")


def main():
    print("training neural network on deep-phase configurations only...")
    r = model.estimate_tc(L=24, n_per_temp=120, seed=0)

    print(f"train accuracy (deep phases): {r['train_accuracy']:.3f}")
    print(f"T_c estimate (P = 1/2 crossing): {r['tc_estimate']:.3f}")
    print(f"T_c exact (Onsager):             {r['tc_exact']:.3f}")
    print(f"relative error: {abs(r['tc_estimate'] - r['tc_exact']) / r['tc_exact'] * 100:.1f}%")

    temps, p = r["temperatures"], r["p_disordered"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    # Shade the temperatures the network actually trained on.
    ax.axvspan(temps.min(), r["train_below"], color="C0", alpha=0.08)
    ax.axvspan(r["train_above"], temps.max(), color="C3", alpha=0.08)
    ax.plot(temps, p, "o-", color="C4", label="network output  P(disordered)")
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.axvline(r["tc_exact"], color="0.3", ls="--", lw=1.2, label=f"$T_c$ exact = {r['tc_exact']:.2f}")
    ax.axvline(r["tc_estimate"], color="C1", ls="-", lw=1.2, label=f"$T_c$ estimate = {r['tc_estimate']:.2f}")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("P(disordered)")
    ax.set_title("A neural net locates the Ising transition it never trained on")
    ax.text(temps.min() + 0.05, 0.9, "trained\nordered", fontsize=8, color="C0")
    ax.text(r["train_above"] + 0.05, 0.1, "trained\ndisordered", fontsize=8, color="C3")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ml_transition.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"figure written to {os.path.normpath(out)}")

    sample_configs_figure(L=24)


if __name__ == "__main__":
    main()
