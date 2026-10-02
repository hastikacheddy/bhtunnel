"""VQE for the 2p quasi-bound state of a massive scalar (M mu = 0.3, l = 1).

    python scripts/vqe_bound_state.py

The Hermitian Dirichlet-box problem of bhtunnel.bound on N = 32 points (5
qubits). Reports, against the exact grid ground state:

* the Pauli decomposition size of H and its qubit-wise-commuting groups,
* VQE energy / omega / state fidelity vs. ansatz depth (ideal statevector),
* a finite-shot energy estimate at the optimum, via the Aer Estimator,

and compares omega with the converged box value and the hydrogenic formula.
Writes data/vqe_bound_state.json and figures/vqe_bound_state.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from qiskit.circuit.library import real_amplitudes  # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402
from qiskit_aer.primitives import EstimatorV2 as AerEstimator  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bhtunnel.bound import BoundBox, hydrogenic_omega  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"
REPS = (1, 2, 3, 4, 6, 8)


def vqe(h_mat: np.ndarray, reps: int, rng, restarts: int = 3):
    n = int(np.log2(len(h_mat)))
    ansatz = real_amplitudes(n, reps=reps, entanglement="linear")

    def state(theta):
        return Statevector(ansatz.assign_parameters(theta)).data.real

    def energy(theta):
        v = state(theta)
        return float(v @ h_mat @ v)

    best = None
    for _ in range(restarts):
        x0 = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
        res = minimize(energy, x0, method="L-BFGS-B", options={"maxiter": 3000})
        if best is None or res.fun < best.fun:
            best = res
    return ansatz, best.x, best.fun, state(best.x)


def main():
    rng = np.random.default_rng(7)
    box = BoundBox(N=32)
    h_mat = box.hamiltonian()
    e_exact, v_exact = box.spectrum(1)
    e0, v0 = float(e_exact[0]), v_exact[:, 0]
    v0 = v0 * np.sign(v0.sum())  # eigenvector sign is arbitrary; plot it positive
    w_box = BoundBox(N=2048).omega(BoundBox(N=2048).spectrum(1)[0][0])
    w_hyd = hydrogenic_omega(1, 0.3)

    op = SparsePauliOp.from_operator(h_mat).simplify(atol=1e-12)
    groups = op.group_commuting(qubit_wise=True)
    print(f"H: {len(op)} Pauli terms in {len(groups)} qubit-wise commuting groups")

    rows = []
    for reps in REPS:
        ansatz, theta, e, v = vqe(h_mat, reps, rng)
        fid = float(abs(v @ v0) ** 2)
        rows.append(dict(reps=reps, params=ansatz.num_parameters, energy=e,
                         omega=box.omega(e), fidelity=fid,
                         rel_energy_error=abs(e - e0) / abs(e0)))
        print(rows[-1])
        if reps == REPS[-1]:
            best_ansatz, best_theta = ansatz, theta

    est = AerEstimator(options={"default_precision": 1e-4, "run_options": {"seed": 3}})
    res = est.run([(best_ansatz, op, best_theta)]).result()[0]
    shot = dict(energy=float(res.data.evs), std=float(res.data.stds),
                omega=box.omega(float(res.data.evs)))
    print("finite-shot estimate:", shot)

    summary = dict(l=1, mu=0.3, N=32, qubits=5, pauli_terms=len(op),
                   qwc_groups=len(groups), exact_grid_energy=e0,
                   exact_grid_omega=box.omega(e0), converged_box_omega=w_box,
                   hydrogenic_omega=w_hyd, vqe=rows, finite_shot=shot)
    (ROOT / "data/vqe_bound_state.json").write_text(json.dumps(summary, indent=2))

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                         "axes.titlesize": 11, "axes.titleweight": "bold"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax1.semilogy([r["params"] for r in rows], [r["rel_energy_error"] for r in rows],
                 "o-", color=SERIES[0], lw=2, ms=5, label="relative energy error")
    ax1.semilogy([r["params"] for r in rows], [1 - r["fidelity"] for r in rows],
                 "o-", color=SERIES[1], lw=2, ms=5, label="1 − state fidelity")
    for r in rows:
        ax1.annotate(f"reps={r['reps']}", (r["params"], r["rel_energy_error"]),
                     xytext=(4, 6), textcoords="offset points", fontsize=7.5, color=MUTED)
    ax1.set_title("VQE vs. ansatz size (5 qubits, ideal)", loc="left")
    ax1.legend(frameon=False, fontsize=8.5, loc="lower left")
    _style(ax1, "variational parameters", "error vs. exact grid ground state")

    x = box.x
    best = BoundBox(N=32)
    v_vqe = Statevector(best_ansatz.assign_parameters(best_theta)).data.real
    sign = np.sign(v_vqe @ v0)
    ax2.plot(x, v0 / np.sqrt(best.h), lw=2, color=SERIES[0], label="exact grid 2p state")
    ax2.plot(x, sign * v_vqe / np.sqrt(best.h), "o", ms=4, color=SERIES[1],
             mec="white", mew=0.6, label=f"VQE, reps={REPS[-1]}")
    ax2.set_title("Quasi-bound 2p state, Mμ = 0.3", loc="left")
    ax2.text(0.98, 0.55, f"ω: VQE {rows[-1]['omega']:.5f}\nbox (N→∞) {w_box:.5f}\n"
             f"hydrogenic {w_hyd:.5f}", transform=ax2.transAxes, ha="right", va="top",
             fontsize=8.5, color=INK)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
    _style(ax2, "tortoise coordinate r* / M", "ψ (normalised)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/vqe_bound_state.png")
    plt.close(fig)


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(True, color=GRID_C, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_C)


if __name__ == "__main__":
    main()
