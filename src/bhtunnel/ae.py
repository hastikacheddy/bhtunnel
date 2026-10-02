"""Amplitude estimation of small transmission probabilities.

Sampling the sign qubit estimates P_T with ~ (1-p)/(p eps^2) shots for relative
error eps. Amplitude estimation applies the Grover operator

    Q = A S_0 A^dag S_chi,     S_chi = Z on the sign qubit,  S_0 = I - 2|0><0|

so that after A Q^m the good-state probability is sin^2((2m+1) theta), with
p = sin^2 theta. Maximum-likelihood AE (Suzuki et al. 2020) combines shots at
several depths m without phase estimation.

Noise model: each application of A or A^dag is followed by depolarisation with
survival lambda, so the measured probability at depth m is

    P_m = lambda^(2m+1) sin^2((2m+1) theta) + (1 - lambda^(2m+1)) / 2.

The Cramer-Rao analysis below gives the oracle calls needed for relative error
eps, for sampling and for the best noise-aware MLAE depth.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def grover_operator(a: QuantumCircuit, good_qubit: int) -> QuantumCircuit:
    n = a.num_qubits
    a_gate = a.to_gate(label="A")
    qc = QuantumCircuit(n, name="Q")
    qc.z(good_qubit)                                    # S_chi
    qc.append(a_gate.inverse(), range(n))
    qc.x(range(n))                                      # S_0 = X MCZ X
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.append(a_gate, range(n))
    return qc


def ae_circuit(a: QuantumCircuit, good_qubit: int, m: int, measure: bool = True
               ) -> QuantumCircuit:
    n = a.num_qubits
    qc = QuantumCircuit(n, 1 if measure else 0)
    qc.compose(a, inplace=True)
    if m:
        q = grover_operator(a, good_qubit).to_gate(label="Q")
        for _ in range(m):
            qc.append(q, range(n))
    if measure:
        qc.measure(good_qubit, 0)
    return qc


def p_model(theta, m, lam: float = 1.0):
    d = lam ** (2 * m + 1)
    return d * np.sin((2 * m + 1) * theta) ** 2 + (1 - d) / 2


def mlae_estimate(hits, shots, ms, lam: float = 1.0, grid: int = 200_001) -> float:
    """Maximum-likelihood p from hits[i] good outcomes of shots[i] at depth ms[i]."""
    theta = np.linspace(1e-9, np.pi / 2 - 1e-9, grid)
    ll = np.zeros_like(theta)
    for h, n, m in zip(hits, shots, ms):
        p = np.clip(p_model(theta, m, lam), 1e-300, 1 - 1e-16)
        ll += h * np.log(p) + (n - h) * np.log1p(-p)
    i = int(np.argmax(ll))
    # parabolic refinement around the grid maximum
    if 0 < i < grid - 1:
        y0, y1, y2 = ll[i - 1], ll[i], ll[i + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom < 0 else 0.0
        th = theta[i] + shift * (theta[1] - theta[0])
    else:
        th = theta[i]
    return float(np.sin(th) ** 2)


def fisher_per_shot(theta, m, lam: float = 1.0):
    """Fisher information about theta from one shot at depth m."""
    k = 2 * m + 1
    p = p_model(theta, m, lam)
    dp = lam**k * k * np.sin(2 * k * theta)
    return dp * dp / (p * (1 - p))


def oracle_calls(p: float, eps: float, lam: float = 1.0, m_max: int = 20_000,
                 min_shots: int = 20, schedule_overhead: float = 2.0):
    """Oracle calls (applications of A or A^dag) to reach relative standard error
    eps on p at the Cramer-Rao bound, for sampling and for MLAE.

    MLAE is modelled as: the deepest level m carries the information, needs at
    least `min_shots` shots, and the shallower levels that resolve the
    ambiguity of sin^2 cost `schedule_overhead` x as much again. Both
    estimators are noise-aware (they know lambda). Returns
    (sampling_calls, mlae_calls, best_m).
    """
    theta = np.arcsin(np.sqrt(p))
    sd_theta = eps * p / np.sin(2 * theta)  # dp = sin(2 theta) dtheta
    sampling = 1.0 / (sd_theta**2 * fisher_per_shot(theta, 0, lam))
    ms = np.unique(np.round(np.geomspace(1, m_max, 600)).astype(int))
    # A real schedule cannot sit exactly on a zero of sin(2 k theta): average the
    # information over the local oscillation (+-1 depth).
    info = np.array([np.mean([fisher_per_shot(theta, mm, lam)
                              for mm in (max(m - 1, 0), m, m + 1)]) for m in ms])
    with np.errstate(divide="ignore", over="ignore"):
        shots = np.maximum(min_shots, 1.0 / (sd_theta**2 * info))
        calls = schedule_overhead * (2 * ms + 1) * shots
    j = int(np.argmin(calls))
    return sampling, float(calls[j]), int(ms[j])
