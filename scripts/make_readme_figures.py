"""Figures used only by the README.

    python scripts/make_readme_figures.py

docs/images/spacetime.png   |psi(r*, t)|^2 of the l = 1 production run: the
                            packet splits at the barrier into a transmitted
                            and a reflected part.
docs/images/circuit.png     the HW5 circuit (2 Strang steps) with each stage
                            boxed, exactly as it runs on hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bhtunnel.circuits import block_circuit  # noqa: E402
from bhtunnel.presets import AER_L1, HW5, HW_STEPS  # noqa: E402

OUT = ROOT / "docs" / "images"
INK, MUTED, GRID_C, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#eb6834"


def spacetime():
    exp = AER_L1
    w, u = np.linalg.eigh(exp.hamiltonian())
    c = u.conj().T @ exp.initial_state()
    ts = np.linspace(0, exp.time, 260)
    dens = np.abs(u @ (np.exp(-1j * np.outer(w, ts)) * c[:, None])).T ** 2 / exp.grid.dx
    x = exp.grid.x
    p_t = exp.transmitted_probability(u @ (np.exp(-1j * w * exp.time) * c))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 3.2], hspace=0.06))
    ax0.fill_between(x, exp.potential, color=ORANGE, alpha=0.25, lw=0)
    ax0.plot(x, exp.potential, color=ORANGE, lw=2)
    ax0.set_ylabel("$V_1(r_*)$", color=MUTED)
    ax0.text(0.01, 0.78, "Regge–Wheeler barrier (l = 1)", transform=ax0.transAxes,
             color=INK, fontsize=10)
    im = ax1.imshow(dens, origin="lower", aspect="auto", cmap="Blues",
                    norm=PowerNorm(0.45, vmin=0, vmax=dens.max()),
                    extent=[x[0], x[-1] + exp.grid.dx, 0, exp.time])
    ax1.set_xlabel("tortoise coordinate $r_*/M$", color=MUTED)
    ax1.set_ylabel("time $t/M$", color=MUTED)
    for text, xy, ha in (("incoming packet", (exp.x0 + 52, 30), "center"),
                         ("transmitted → horizon", (-35, exp.time * 0.9), "center"),
                         ("reflected → infinity", (150, exp.time * 0.9), "center")):
        ax1.text(*xy, text, ha=ha, va="center", color=INK, fontsize=10,
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    ax1.text(0.01, 0.03, f"8 qubits · 256 grid points · P_T = {p_t:.3f}",
             transform=ax1.transAxes, ha="left", color=MUTED, fontsize=9,
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    cb = fig.colorbar(im, ax=[ax0, ax1], pad=0.015, fraction=0.03)
    cb.set_label("$|\\psi|^2$", color=MUTED)
    cb.outline.set_visible(False)
    for ax in (ax0, ax1):
        ax.tick_params(colors=MUTED, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    ax0.grid(True, color=GRID_C, lw=0.8)
    fig.savefig(OUT / "spacetime.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def circuit():
    qc = block_circuit(HW5, HW_STEPS["HW5"])
    fig = qc.draw("mpl", fold=-1, scale=0.85, style={"backgroundcolor": "#ffffff"})
    fig.savefig(OUT / "circuit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    spacetime()
    circuit()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
