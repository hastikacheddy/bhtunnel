"""Generate the classical reference dataset and validation figures.

    python scripts/make_reference.py              # compute data (if missing) + plot
    python scripts/make_reference.py --recompute  # force recomputation

Writes data/*.csv (the ground truth every quantum run is scored against) and
figures/*.png.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bhtunnel import (T_HAWKING, absorption_cross_section,  # noqa: E402
                      blackbody_number_spectrum, greybody, number_spectrum,
                      potential, potential_peak, r_from_tortoise, total_power)
from bhtunnel.wkb import cross_section_sinc, greybody_wkb  # noqa: E402

DATA, FIGS = ROOT / "data", ROOT / "figures"
L_MAX = 5
OMEGAS = np.round(np.linspace(0.005, 1.2, 240), 6)
CS_OMEGAS = np.round(np.linspace(0.05, 1.5, 59), 6)

# Validated categorical slots 1-4 (light surface) + recessive ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _gamma_row(l):
    return l, greybody(OMEGAS, l)


def _cross_section_point(omega):
    return absorption_cross_section([omega])[0]


def compute():
    DATA.mkdir(exist_ok=True)
    with ProcessPoolExecutor() as pool:
        # Submit the slow high-frequency (many-l) points first so they overlap.
        sigma_it = pool.map(_cross_section_point, CS_OMEGAS[::-1])
        gammas = dict(pool.map(_gamma_row, range(L_MAX + 1)))
        sigma = np.array(list(sigma_it))[::-1]
    np.savetxt(DATA / "greybody_scalar.csv",
               np.column_stack([OMEGAS] + [gammas[l] for l in range(L_MAX + 1)]),
               delimiter=",", header="M_omega," + ",".join(
                   f"gamma_l{l}" for l in range(L_MAX + 1)), comments="")
    np.savetxt(DATA / "cross_section_scalar.csv",
               np.column_stack([CS_OMEGAS, sigma]), delimiter=",",
               header="M_omega,sigma_over_pi_M2", comments="")


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK, **kw)
    return leg


def plot():
    FIGS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                         "axes.titlesize": 11, "axes.titleweight": "bold"})
    gb = np.loadtxt(DATA / "greybody_scalar.csv", delimiter=",", skiprows=1)
    omegas, gammas = gb[:, 0], {l: gb[:, l + 1] for l in range(L_MAX + 1)}
    cs = np.loadtxt(DATA / "cross_section_scalar.csv", delimiter=",", skiprows=1)
    cs_omegas, sigma = cs[:, 0], cs[:, 1]

    # 1. Potential in the tortoise coordinate.
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rs = np.linspace(-15, 25, 800)
    r, _ = r_from_tortoise(rs)
    for l in range(4):
        v = potential(r, l)
        ax.plot(rs, v, lw=2, color=SERIES[l], label=f"l = {l}")
        i = np.argmax(v)
        ax.annotate(f"l={l}", (rs[i], v[i]), xytext=(0, 5),
                    textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_title("Scalar Regge–Wheeler barriers, Schwarzschild", loc="left")
    _style(ax, "tortoise coordinate  r* / M", "V_l   (M⁻²)")
    ax.set_ylim(top=0.55)
    ax.text(-14.5, 0.51, "← horizon", color=MUTED, fontsize=9)
    ax.text(24.5, 0.51, "infinity →", color=MUTED, fontsize=9, ha="right")
    _legend(ax, loc="center right")
    fig.tight_layout()
    fig.savefig(FIGS / "potential.png")
    plt.close(fig)

    # 2. Greybody factors, linear and log scale (the log panel shows the deep
    #    tunnelling tail that hardware noise will swallow).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))
    for l in range(4):
        ax1.plot(omegas, gammas[l], lw=2, color=SERIES[l], label=f"l = {l}")
        ax2.semilogy(omegas, gammas[l], lw=2, color=SERIES[l], label=f"l = {l}")
        if l > 0:
            ax1.plot(omegas, greybody_wkb(omegas, l), lw=1, ls="--",
                     color=SERIES[l], alpha=0.8)
        _, v0, _ = potential_peak(l)
        x = np.sqrt(v0)
        ax1.annotate(f"l={l}", (x, np.interp(x, omegas, gammas[l])),
                     xytext=(6, -10), textcoords="offset points",
                     color=INK, fontsize=9)
    ax1.set_title("Greybody factor Γ_l  (dashed: 1st-order WKB)", loc="left")
    ax2.set_title("Same, log scale: the tunnelling tail", loc="left")
    _style(ax1, "M ω", "Γ_l")
    _style(ax2, "M ω", "Γ_l")
    _legend(ax1, loc="lower right")
    _legend(ax2, loc="lower right")
    ax2.set_ylim(1e-12, 3)
    ax2.axhspan(1e-12, 1e-2, color=GRID, alpha=0.5, lw=0)
    ax2.text(0.62, 2e-3, "below ~1e-2: hard to resolve\non NISQ hardware",
             color=MUTED, fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(FIGS / "greybody.png")
    plt.close(fig)

    # 3. Absorption cross section.
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    fine = np.linspace(cs_omegas[0], cs_omegas[-1], 600)
    ax.plot(cs_omegas, sigma, lw=2, color=SERIES[0], label="exact (this solver)")
    ax.plot(fine, cross_section_sinc(fine), lw=1, ls="--", color=MUTED,
            label="photon-sphere sinc formula")
    for y, text in ((27, "27πM²  geometric optics"), (16, "16πM²  horizon area")):
        ax.axhline(y, color=MUTED, lw=0.8, ls=":")
        ax.text(cs_omegas[-1], y + 0.6, text, ha="right", color=MUTED, fontsize=8)
    ax.set_title("Scalar absorption cross section", loc="left")
    _style(ax, "M ω", "σ_abs / πM²")
    _legend(ax, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "cross_section.png")
    plt.close(fig)

    # 4. Hawking power spectrum: greybody-filtered vs. naive blackbody.
    mask = omegas <= 0.6
    w = omegas[mask]
    n_gb = number_spectrum(w, l_max=L_MAX, gammas={l: g[mask] for l, g in gammas.items()})
    n_bb = blackbody_number_spectrum(w)
    np.savetxt(DATA / "hawking_scalar.csv", np.column_stack([w, n_gb, n_bb]),
               delimiter=",", header="M_omega,dN_dtdw_greybody,dN_dtdw_blackbody27",
               comments="")
    p_gb, p_bb = total_power(w, n_gb), total_power(w, n_bb)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(w, 1e5 * w * n_gb, lw=2, color=SERIES[0],
            label=f"with greybody factors   P = {p_gb:.3e} M⁻²")
    ax.plot(w, 1e5 * w * n_bb, lw=2, color=SERIES[1],
            label=f"blackbody, σ = 27πM²   P = {p_bb:.3e} M⁻²")
    ax.set_title(f"Scalar Hawking power spectrum  (T_H = 1/8πM = {T_HAWKING:.4f}/M)",
                 loc="left")
    _style(ax, "M ω", "ω d²N/dt dω   (×10⁻⁵ M⁻²)")
    _legend(ax, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "hawking_spectrum.png")
    plt.close(fig)

    print(f"total scalar power, greybody:  {p_gb:.4e} / M^2  (Page/Elster: 7.44e-5)")
    print(f"total scalar power, blackbody: {p_bb:.4e} / M^2")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    if args.recompute or not (DATA / "greybody_scalar.csv").exists() \
            or not (DATA / "cross_section_scalar.csv").exists():
        compute()
    plot()


if __name__ == "__main__":
    main()
