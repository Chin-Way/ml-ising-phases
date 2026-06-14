"""Wolff cluster Monte Carlo vs. Metropolis.

Two things to show: (1) the cluster sampler reproduces the *same* magnetization
and energy curves as the single-spin Metropolis engine, and (2) at T_c it
decorrelates in a handful of updates while Metropolis suffers critical slowing
down -- its autocorrelation time climbs with lattice size. Writes a two-panel
figure.

Run from the project root::

    python examples/wolff_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ising import montecarlo as mc  # noqa: E402
from ising import wolff  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def validate_curves():
    """Magnetization and energy vs T from both samplers (they should overlap)."""
    temps = np.linspace(1.6, 3.2, 17)
    m_M, m_E, w_M, w_E = (np.empty(len(temps)) for _ in range(4))
    for j, T in enumerate(temps):
        m = mc.simulate(L=16, T=T, n_eq=1200, n_samples=200, sample_every=5, seed=j)
        w = wolff.simulate_wolff(L=16, T=T, n_eq=300, n_samples=200, sample_every=3, seed=j)
        m_M[j], m_E[j] = m["abs_M"].mean(), m["energy"].mean()
        w_M[j], w_E[j] = w["abs_M"].mean(), w["energy"].mean()
    return temps, m_M, m_E, w_M, w_E


def autocorr_vs_size(sizes=(8, 16, 24, 32), n_steps=4000):
    """Integrated autocorrelation time of |M| at T_c for both samplers."""
    Tc = mc.TC_EXACT
    tau_metro, tau_wolff = [], []
    for L in sizes:
        rng = np.random.default_rng(0)
        # Metropolis: one sample per sweep.
        spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)
        masks = mc._checkerboard(L)
        for _ in range(500):
            mc.mc_sweep(spins, Tc, rng, masks)
        s = np.empty(n_steps)
        for k in range(n_steps):
            mc.mc_sweep(spins, Tc, rng, masks)
            s[k] = abs(spins.mean())
        tau_metro.append(wolff.integrated_autocorr_time(s))
        # Wolff: one sample per cluster update.
        spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)
        for _ in range(200):
            wolff.wolff_step(spins, Tc, rng)
        s = np.empty(n_steps)
        for k in range(n_steps):
            wolff.wolff_step(spins, Tc, rng)
            s[k] = abs(spins.mean())
        tau_wolff.append(wolff.integrated_autocorr_time(s))
        print(f"  L={L:>2}  tau_metropolis={tau_metro[-1]:6.2f} sweeps   "
              f"tau_wolff={tau_wolff[-1]:5.2f} steps")
    return np.array(sizes), np.array(tau_metro), np.array(tau_wolff)


def main():
    print("validating Wolff against Metropolis observables...")
    temps, m_M, m_E, w_M, w_E = validate_curves()
    print("measuring autocorrelation time at T_c vs lattice size...")
    sizes, tau_m, tau_w = autocorr_vs_size()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- left: observable curves agree -------------------------------------
    ax1.plot(temps, m_M, "o-", color="C0", ms=4, label="Metropolis $|M|$")
    ax1.plot(temps, w_M, "s--", color="C1", ms=4, label="Wolff $|M|$")
    ax1.plot(temps, m_E / 2.0 + 1.0, "o-", color="C2", ms=4, label="Metropolis $E$ (scaled)")
    ax1.plot(temps, w_E / 2.0 + 1.0, "s--", color="C3", ms=4, label="Wolff $E$ (scaled)")
    ax1.axvline(mc.TC_EXACT, color="0.4", ls=":", lw=1)
    ax1.set_xlabel("temperature T")
    ax1.set_ylabel("observable")
    ax1.set_title("Wolff reproduces the Metropolis curves (L = 16)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # --- right: critical slowing down --------------------------------------
    ax2.plot(sizes, tau_m, "o-", color="C0", ms=6, label="Metropolis (per sweep)")
    ax2.plot(sizes, tau_w, "s-", color="C1", ms=6, label="Wolff (per cluster)")
    ax2.set_yscale("log")
    ax2.set_xlabel("lattice size L")
    ax2.set_ylabel(r"autocorrelation time $\tau$ at $T_c$")
    ax2.set_title("Wolff sidesteps critical slowing down")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "wolff_vs_metropolis.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"figure written to {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
