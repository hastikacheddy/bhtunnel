"""Noisy-simulator study of the hardware-scale circuits, with error mitigation.

    python scripts/noise_study.py [--shots 20000]

Runs the HW5 circuit (1 and 2 Strang steps) on Aer with the FakeFez (IBM Heron)
noise model, and compares, against the classical twin at the same step count:

* raw estimate of P_T (sign-qubit probability),
* readout-mitigated (2x2 confusion matrix of the physical sign qubit),
* readout + zero-noise extrapolation (global folding A (A^dag A)^j, scale
  1/3/5, exponential fit of P - 1/2, which is what depolarising noise produces).

Also reports the effective depolarising survival lambda = (P_noisy - 1/2) /
(P_ideal - 1/2) of one application of A, which feeds the amplitude-estimation
break-even analysis. Writes data/noise.json and figures/noise.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeFez  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bhtunnel.circuits import scattering_circuit  # noqa: E402
from bhtunnel.presets import HW5  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"
SCALES = (1, 3, 5)


def folded(exp, steps: int, scale: int) -> QuantumCircuit:
    """A (A^dag A)^((scale-1)/2), then measure the sign qubit."""
    n = exp.grid.n
    a = scattering_circuit(exp, steps)
    qc = QuantumCircuit(n, 1)
    qc.compose(a, inplace=True)
    for _ in range((scale - 1) // 2):
        qc.barrier()
        qc.compose(a.inverse(), inplace=True)
        qc.barrier()
        qc.compose(a, inplace=True)
    qc.measure(n - 1, 0)
    return qc


def p_one(counts: dict) -> float:
    return counts.get("1", 0) / sum(counts.values())


def readout_matrix(sim, backend, phys: int, shots: int) -> np.ndarray:
    m = np.zeros((2, 2))
    for b in (0, 1):
        qc = QuantumCircuit(backend.num_qubits, 1)
        if b:
            qc.x(phys)
        qc.measure(phys, 0)
        t = transpile(qc, backend, initial_layout=list(range(backend.num_qubits)),
                      optimization_level=0)
        p1 = p_one(sim.run(t, shots=shots).result().get_counts())
        m[:, b] = (1 - p1, p1)
    return m


def unfold_readout(p1: float, m: np.ndarray) -> float:
    return float(np.clip(np.linalg.solve(m, [1 - p1, p1])[1], 0, 1))


def zne_exponential(scales, ps, shot_sd: float) -> float:
    """Fit |P_s - 1/2| = c exp(-g s), using only scale factors whose signal is
    resolvable (> 2 shot sd, same sign as s=1). NaN if fewer than two are."""
    y = np.array(ps) - 0.5
    sign = np.sign(y[0])
    keep = sign * y > 2 * shot_sd
    if keep.sum() < 2:
        return float("nan")
    g, logc = np.polyfit(np.array(scales)[keep], np.log(sign * y[keep]), 1)
    return float(np.clip(0.5 + sign * np.exp(logc), 0.0, 1.0))


def run(steps: int, shots: int, sim, backend) -> dict:
    exp = replace(HW5)
    p_ideal = exp.transmitted_probability(exp.evolve_strang(steps))
    out = {"steps": steps, "p_ideal_twin": p_ideal,
           "p_exact_grid": exp.transmitted_probability(exp.evolve_exact())}
    raw, layout_phys, czs = [], None, []
    for s in SCALES:
        t = transpile(folded(exp, steps, s), backend, optimization_level=3,
                      seed_transpiler=11)
        czs.append(t.count_ops().get("cz", 0))
        meas = [i for i in t.data if i.operation.name == "measure"][0]
        layout_phys = t.find_bit(meas.qubits[0]).index
        raw.append(p_one(sim.run(t, shots=shots, seed_simulator=5).result().get_counts()))
    m = readout_matrix(sim, backend, layout_phys, shots)
    ro = [unfold_readout(p, m) for p in raw]
    sd = float(np.sqrt(0.25 / shots))
    out.update(cz=czs, raw=raw, readout=ro, zne=zne_exponential(SCALES, ro, sd),
               lam=(ro[0] - 0.5) / (p_ideal - 0.5), shot_sd=sd)
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=20_000)
    args = ap.parse_args()
    backend = FakeFez()
    sim = AerSimulator.from_backend(backend)
    results = [run(s, args.shots, sim, backend) for s in (1, 2)]
    (ROOT / "data/noise.json").write_text(json.dumps(
        {"backend": backend.name, "shots": args.shots, "runs": results}, indent=2))

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                         "axes.titlesize": 11, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    labels = ["raw", "readout-mitigated", "readout + ZNE"]
    width = 0.22
    for j, r in enumerate(results):
        vals = [r["raw"][0], r["readout"][0], r["zne"]]
        for i, v in enumerate(vals):
            x = j + (i - 1) * width
            ax.bar(x, v, width * 0.9, color=SERIES[i], label=labels[i] if j == 0 else None)
        ax.hlines(r["p_ideal_twin"], j - 1.6 * width, j + 1.6 * width, color=INK, lw=2,
                  label="ideal (twin, same steps)" if j == 0 else None)
        ax.text(j, -0.1, f"{r['cz'][0]} CZ, λ≈{r['lam']:.2f}", ha="center",
                color=MUTED, fontsize=8.5, transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.text(1.45, 0.51, "fully depolarised", ha="right", color=MUTED, fontsize=8)
    ax.set_xticks([0, 1], ["1 Strang step", "2 Strang steps"])
    ax.set_ylim(0, 1)
    ax.set_title("HW5 on noisy Aer (FakeFez): P_T estimates", loc="left")
    ax.set_ylabel("P_T", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(True, axis="y", color=GRID_C, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/noise.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
