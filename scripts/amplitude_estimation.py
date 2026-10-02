"""Amplitude estimation vs. sampling for tunnelling-tail probabilities.

    python scripts/amplitude_estimation.py

1. Monte-Carlo check of maximum-likelihood AE against plain sampling for a
   tail-sized probability, ideal device: relative RMSE vs. oracle calls.
   (The sin^2((2m+1) theta) law used here is verified on Aer circuits in the
   tests.)
2. Cramer-Rao advantage (sampling calls / MLAE calls at 10 % relative error)
   vs. per-oracle depolarising survival lambda, with the lambda measured for
   the HW5 circuit on noisy Aer (data/noise.json) marked.

Writes data/amplitude_estimation.json and figures/amplitude_estimation.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bhtunnel import greybody  # noqa: E402
from bhtunnel.ae import mlae_estimate, oracle_calls, p_model  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"
EPS = 0.1


def monte_carlo(p: float, rng, trials: int = 300, shots: int = 50):
    theta = np.arcsin(np.sqrt(p))
    out = {"sampling": [], "mlae": []}
    for j_max in range(0, 9):
        ms = [0] + [2**j for j in range(j_max)]
        calls = sum(shots * (2 * m + 1) for m in ms)
        est = []
        for _ in range(trials):
            hits = [rng.binomial(shots, p_model(theta, m)) for m in ms]
            est.append(mlae_estimate(hits, [shots] * len(ms), ms, grid=40_001))
        out["mlae"].append((calls, float(np.sqrt(np.mean((np.array(est) / p - 1) ** 2)))))
        samp = rng.binomial(calls, p, size=trials) / calls
        out["sampling"].append((calls, float(np.sqrt(np.mean((samp / p - 1) ** 2)))))
    return out


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(True, color=GRID_C, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_C)


def main():
    rng = np.random.default_rng(2026)
    p_tail = float(greybody([0.2], 1)[0])  # l=1 tail, Gamma ~ 3e-2
    p_deep = float(greybody([0.3], 2)[0])  # l=2 tail, Gamma ~ 1.5e-3
    mc = monte_carlo(p_deep, rng)

    lams = 1 - np.geomspace(1e-4, 0.8, 60)
    ps = (1e-1, 1e-2, 1e-3, 1e-4)
    adv = {p: [oracle_calls(p, EPS, lam)[0] / oracle_calls(p, EPS, lam)[1] for lam in lams]
           for p in ps}
    noise = json.loads((ROOT / "data/noise.json").read_text())
    hw = {f"HW5, {r['steps']} step(s)": r["lam"] for r in noise["runs"]}

    breakeven = {}
    for p in ps:
        a = np.array(adv[p])
        above = lams[a > 1]
        breakeven[p] = float(above.min()) if len(above) else None
    (ROOT / "data/amplitude_estimation.json").write_text(json.dumps({
        "p_tail_l1_w0.2": p_tail, "p_tail_l2_w0.3": p_deep, "monte_carlo": mc,
        "eps": EPS, "breakeven_lambda": {str(k): v for k, v in breakeven.items()},
        "hardware_lambda": hw}, indent=2))

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                         "axes.titlesize": 11, "axes.titleweight": "bold"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    for i, key in enumerate(("sampling", "mlae")):
        x, y = zip(*mc[key])
        ax1.loglog(x, y, "o-", lw=2, ms=5, color=SERIES[i],
                   label="sampling" if key == "sampling" else "max-likelihood AE")
    x = np.array([c for c, _ in mc["sampling"]], dtype=float)
    ax1.loglog(x, mc["sampling"][0][1] * (x / x[0]) ** -0.5, ":", color=MUTED, lw=1)
    ax1.loglog(x, mc["mlae"][-1][1] * (x / x[-1]) ** -1.0, ":", color=MUTED, lw=1)
    ax1.annotate("∝ N^-1/2", (x[-2], mc["sampling"][-2][1]), xytext=(0, 12),
                 textcoords="offset points", ha="center", color=MUTED, fontsize=8.5)
    ax1.annotate("∝ N^-1", (x[-2], mc["mlae"][-1][1] * 2), xytext=(-6, -14),
                 textcoords="offset points", ha="right", color=MUTED, fontsize=8.5)
    ax1.set_title(f"Ideal device, p = Γ₂(0.3) = {p_deep:.1e}", loc="left")
    ax1.legend(frameon=False, fontsize=8.5, loc="lower left")
    _style(ax1, "oracle calls (applications of A)", "relative RMSE of p")

    for i, p in enumerate(ps):
        ax2.semilogx(1 - lams, adv[p], lw=2, color=SERIES[i], label=f"p = {p:.0e}")
    ax2.set_yscale("log")
    ax2.axhline(1, color=INK, lw=1)
    for j, (name, lam) in enumerate(hw.items()):
        ax2.axvline(1 - lam, color=MUTED, ls="--", lw=1)
        ax2.text(1 - lam, 400 / 8**j, name.replace("HW5, ", "HW5 ").replace("(s)", "s" if j else ""),
                 ha="right", va="center", color=MUTED, fontsize=8,
                 bbox=dict(fc="white", ec="none", pad=1))
    ax2.text(1.2e-4, 1.15, "AE wins above this line", color=INK, fontsize=8.5)
    ax2.set_title("AE advantage at 10 % relative error", loc="left")
    ax2.legend(frameon=False, fontsize=8.5, loc="lower left")
    _style(ax2, "per-oracle depolarisation  1 − λ", "sampling calls / AE calls")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/amplitude_estimation.png")
    plt.close(fig)
    print(json.dumps({"breakeven_lambda": breakeven, "hardware_lambda": hw}, indent=1))


if __name__ == "__main__":
    main()
