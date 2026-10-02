"""Gate-cost study of the split-operator circuit.

    python scripts/resource_study.py

1. Cost of one Strang step vs. qubit count n, split into QFT pair, kinetic
   phase (n(n+1)/2 rotations) and potential phase, with the potential either
   as an exact diagonal (Qiskit DiagonalGate) or as a CX-ladder Walsh series.
2. Accuracy/cost trade-off at n = 8: |Delta P_T| vs. two-qubit gate count over
   (Strang steps) x (Walsh terms kept).
3. Full transpiled cost of the hardware-scale presets on a Heron device
   (FakeFez coupling map), and the circuit fidelity implied by its reported
   two-qubit error rates.

Counts are after transpilation to {cz, rz, sx, x} at optimization level 2.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.circuit.library import DiagonalGate, QFTGate  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeFez  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bhtunnel.circuits import (diagonal_evolution, kinetic_terms,  # noqa: E402
                               scattering_circuit)
from bhtunnel.grid import Experiment, Grid, walsh_terms  # noqa: E402
from bhtunnel.presets import AER_L1, HW5, HW6, HW_STEPS  # noqa: E402

BASIS = ["cz", "rz", "sx", "x"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"


def two_q(qc: QuantumCircuit, **kw) -> tuple[int, int]:
    t = transpile(qc, basis_gates=BASIS, optimization_level=2, seed_transpiler=7, **kw)
    ops = t.count_ops()
    return ops.get("cz", 0), t.depth(lambda inst: inst.operation.num_qubits == 2)


def scaled_experiment(n: int) -> Experiment:
    """The AER_L1 physics box resampled on 2^n points."""
    return replace(AER_L1, grid=Grid(n, AER_L1.grid.a, AER_L1.grid.b), _cache={})


def per_step_costs(ns=range(4, 10)) -> list[dict]:
    rows = []
    for n in ns:
        exp = scaled_experiment(n)
        dt = exp.time / 50
        qft = QuantumCircuit(n)
        qft.append(QFTGate(n).inverse(), range(n))
        qft.append(QFTGate(n), range(n))
        kin = diagonal_evolution(n, kinetic_terms(exp), dt)
        v_terms = walsh_terms(exp.potential)
        v_walsh = diagonal_evolution(n, v_terms, dt)
        v_diag = QuantumCircuit(n)
        v_diag.append(DiagonalGate(list(np.exp(-1j * dt * exp.potential))), range(n))
        rows.append(dict(n=n, qft=two_q(qft)[0], kinetic=two_q(kin)[0],
                         v_exact_diag=two_q(v_diag)[0], v_walsh_all=two_q(v_walsh)[0],
                         walsh_terms=len(v_terms)))
        print(rows[-1])
    return rows


def tradeoff(steps_list=(10, 20, 40, 60, 100), walsh_list=(32, 64, 96, 128, 192, None)):
    exp = AER_L1
    p_ref = exp.transmitted_probability(exp.evolve_exact())
    n = exp.grid.n
    base = QuantumCircuit(n)
    base.append(QFTGate(n).inverse(), range(n))
    base.compose(diagonal_evolution(n, kinetic_terms(exp), 1.0), inplace=True)
    base.append(QFTGate(n), range(n))
    cost_qk = two_q(base)[0]
    v_cost = {}
    for K in walsh_list:
        e = replace(exp, walsh_terms=K, _cache={})
        if K is None:  # production circuits synthesise the exact diagonal
            qc = QuantumCircuit(n)
            qc.append(DiagonalGate(list(np.exp(-1j * e.potential))), range(n))
            v_cost[K] = two_q(qc)[0]
        else:
            v_cost[K] = two_q(diagonal_evolution(n, walsh_terms(e.potential), 1.0))[0]
    rows = []
    for K in walsh_list:
        e = replace(exp, walsh_terms=K, _cache={})
        for s in steps_list:
            err = abs(e.transmitted_probability(e.evolve_strang(s)) - p_ref)
            rows.append(dict(walsh=K if K else 255, steps=s,
                             cz=s * (cost_qk + v_cost[K]) + v_cost[K], err=err))
    return rows


def hardware_costs() -> dict:
    backend = FakeFez()
    target = backend.target
    cz_err = np.median([p.error for p in target["cz"].values() if p and p.error])
    sx_err = np.median([p.error for p in target["sx"].values() if p and p.error])
    ro_err = np.median([p.error for p in target["measure"].values() if p and p.error])
    out = {"backend": backend.name, "median_cz_error": cz_err,
           "median_sx_error": sx_err, "median_readout_error": ro_err}
    for name, exp in (("HW5", HW5), ("HW6", HW6)):
        qc = scattering_circuit(exp, HW_STEPS[name], measure="sign")
        t = transpile(qc, backend, optimization_level=3, seed_transpiler=7)
        ops = t.count_ops()
        n_cz, n_sx = ops.get("cz", 0), ops.get("sx", 0)
        fid = (1 - cz_err) ** n_cz * (1 - sx_err) ** n_sx
        out[name] = dict(qubits=exp.grid.n, steps=HW_STEPS[name], cz=n_cz, sx=n_sx,
                         depth=t.depth(), est_fidelity=fid)
        print(name, out[name])
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
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                         "axes.titlesize": 11, "axes.titleweight": "bold"})
    steps = per_step_costs()
    trade = tradeoff()
    hw = hardware_costs()
    (ROOT / "data/resources.json").write_text(json.dumps(
        {"per_step": steps, "tradeoff": trade, "hardware": hw}, indent=2, default=float))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    ns = [r["n"] for r in steps]
    for i, (key, label) in enumerate((("v_walsh_all", "V: Walsh CX ladders (all terms)"),
                                      ("v_exact_diag", "V: exact diagonal synthesis"),
                                      ("qft", "QFT + QFT†"),
                                      ("kinetic", "K: n(n+1)/2 ZZ/Z phases"))):
        ax1.semilogy(ns, [r[key] for r in steps], "o-", lw=2, ms=5, color=SERIES[i],
                     label=label)
    ax1.set_title("Two-qubit gates per Strang step", loc="left")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax1, "qubits n", "CZ count")

    for i, K in enumerate(sorted({r["walsh"] for r in trade}, reverse=True)[:4]):
        pts = sorted((r["cz"], r["err"]) for r in trade if r["walsh"] == K)
        ax2.loglog(*zip(*pts), "o-", lw=2, ms=5, color=SERIES[i],
                   label=f"{K} Walsh terms (CX ladders)" if K < 255
                   else "exact V (diagonal synthesis)")
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_title("Accuracy vs. cost, n = 8, l = 1", loc="left")
    ax2.legend(frameon=False, fontsize=8.5, loc="lower left")
    _style(ax2, "total CZ count (all steps)", "|ΔP_T| vs. exact grid evolution")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/resources.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
