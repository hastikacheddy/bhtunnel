"""Error budget of the quantum scattering experiment, link by link.

    python scripts/error_budget.py

Gamma_exact -> Gamma_box (finite window) -> Gamma_grid (2^n grid, exact time
evolution) -> Gamma_Trotter (Strang steps) -> shot noise.

Walsh truncation of V is a cost trade-off (scripts/resource_study.py);
hardware noise is in scripts/noise_study.py.

Uses the numpy twin, which is verified gate-for-gate against Aer in the tests.
Writes data/error_budget_*.csv and figures/error_budget_*.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bhtunnel import greybody  # noqa: E402
from bhtunnel.grid import Experiment, Grid  # noqa: E402
from bhtunnel.window import Window, scatter_windowed  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"
TROTTER_STEPS = 60
SHOTS = 10_000

# One wrap-safe configuration per multipole (see Experiment.wrap_margin).
CONFIGS = {
    0: dict(grid=Grid(9, -60.0, 452.0), sigma=28.0, x0=200.0, k0s=(0.10, 0.15, 0.21)),
    1: dict(grid=Grid(8, -60.0, 196.0), sigma=14.0, x0=110.0, k0s=(0.24, 0.30, 0.37)),
    2: dict(grid=Grid(8, -60.0, 196.0), sigma=14.0, x0=110.0, k0s=(0.42, 0.49, 0.56)),
}


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(True, color=GRID_C, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_C)


def resolved_rows(l: int, cfg: dict) -> list[tuple]:
    rows = []
    for k0 in cfg["k0s"]:
        exp = Experiment(grid=cfg["grid"], l=l, sigma=cfg["sigma"], x0=cfg["x0"], k0=k0)
        assert exp.wrap_margin() > 0, (l, k0, exp.wrap_margin())
        k, g_grid, _ = exp.resolved_gamma(exp.evolve_exact(), rel_floor=0.15)
        _, g_trot, _ = exp.resolved_gamma(exp.evolve_strang(TROTTER_STEPS), rel_floor=0.15)
        for ki, gg, gt in zip(k, g_grid, g_trot):
            rows.append((l, k0, ki, greybody([ki], l)[0], scatter_windowed(ki, l), gg, gt))
    return rows


def chain(l: int = 1, k0: float = 0.30) -> list[tuple[str, float, float]]:
    """(link, P_T after this link, |change| caused by this link) for one packet."""
    cfg = CONFIGS[l]
    exp = Experiment(grid=cfg["grid"], l=l, sigma=cfg["sigma"], x0=cfg["x0"], k0=k0)
    p_exact = exp.reference_transmitted(lambda ks: greybody(ks, l))
    p_box = exp.reference_transmitted(
        lambda ks: np.array([scatter_windowed(x, l) for x in ks]))
    p_grid = exp.transmitted_probability(exp.evolve_exact())
    p_trot = exp.transmitted_probability(exp.evolve_strang(TROTTER_STEPS))
    shot = np.sqrt(p_trot * (1 - p_trot) / SHOTS)
    out, prev = [], p_exact
    for name, p in (("box window", p_box), ("grid (n=8)", p_grid),
                    (f"Trotter ({TROTTER_STEPS} steps)", p_trot)):
        out.append((name, p, abs(p - prev)))
        prev = p
    out.append((f"shot noise ({SHOTS:,} shots, 1 sd)", p_trot, shot))
    return [("exact", p_exact, 0.0)] + out


def main():
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                         "axes.titlesize": 11, "axes.titleweight": "bold"})

    rows = [r for l, cfg in CONFIGS.items() for r in resolved_rows(l, cfg)]
    arr = np.array(rows)
    np.savetxt(ROOT / "data/error_budget_resolved.csv", arr, delimiter=",",
               header="l,k0,k,gamma_exact,gamma_box,gamma_grid,gamma_trotter",
               comments="", fmt="%.10g")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    ks = np.linspace(0.04, 0.7, 200)
    for l in CONFIGS:
        sel = arr[:, 0] == l
        ax1.plot(ks, greybody(ks, l), lw=1.5, color=SERIES[l], label=f"l={l}")
        ax1.plot(arr[sel, 2], arr[sel, 6], "o", ms=4.5, color=SERIES[l],
                 mec="white", mew=0.8)
        ax2.semilogy(arr[sel, 2], np.abs(arr[sel, 4] - arr[sel, 3]), "s", ms=4,
                     color=SERIES[l], mfc="none")
        ax2.semilogy(arr[sel, 2], np.abs(arr[sel, 6] - arr[sel, 3]), "o", ms=4.5,
                     color=SERIES[l], mec="white", mew=0.8)
    ax1.plot([], [], "o", color=MUTED, label=f"packets, {TROTTER_STEPS} steps")
    ax1.set_title("Momentum-resolved Γ_l from wavepackets", loc="left")
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax1, "M ω", "Γ_l")
    ax2.plot([], [], "s", mfc="none", color=MUTED, label="window only  |Γ_box − Γ|")
    ax2.plot([], [], "o", color=MUTED, label="full pipeline  |Γ_trot − Γ|")
    ax2.set_title("Absolute error per frequency bin", loc="left")
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax2, "M ω", "|ΔΓ_l|")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/error_budget_resolved.png")
    plt.close(fig)

    links = chain()
    with open(ROOT / "data/error_budget_chain.csv", "w", encoding="utf-8") as fh:
        fh.write("link,P_T,abs_change\n")
        for name, p, d in links:
            fh.write(f"{name},{p:.10g},{d:.3e}\n")
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    names = [n for n, _, _ in links[1:]][::-1]
    vals = [d for _, _, d in links[1:]][::-1]
    ax.hlines(range(len(vals)), 1e-7, vals, color=SERIES[0], lw=2)
    ax.plot(vals, range(len(vals)), "o", color=SERIES[0], ms=7)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1e}", (v, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1)
    ax.set_yticks(range(len(vals)), names)
    ax.set_title("Error budget of P_T, one l=1 packet", loc="left")
    ax.text(0, 1.02, f"Mω ≈ 0.30, exact P_T = {links[0][1]:.4f}",
            transform=ax.transAxes, color=MUTED, fontsize=8.5)
    _style(ax, "|ΔP_T| contributed by each link", "")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/error_budget_chain.png")
    plt.close(fig)
    for name, p, d in links:
        print(f"{name:32s} P_T={p:.6f}  |delta|={d:.2e}")


if __name__ == "__main__":
    main()
