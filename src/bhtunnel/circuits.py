"""Qiskit circuits for the split-operator scattering experiment.

Gate-for-gate the same sequence as grid.Experiment.evolve_strang, so an Aer
statevector run must reproduce the classical twin to machine precision; any
deviation is a bug, not physics. See grid.py for the conventions.

    A = Prep(psi0) . [ e^{-iV dt/2} QFT D_K(dt) QFT^dag e^{-iV dt/2} ... ] . QFT^dag

after which qubit n-1 is the sign of the momentum: measuring it gives the
transmitted probability, measuring all qubits gives the momentum spectrum.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, QFTGate, StatePreparation

from .grid import Experiment, walsh_terms


def diagonal_evolution(n: int, terms: dict[int, float], theta: float,
                       label: str | None = None) -> QuantumCircuit:
    """exp(-i theta sum_S c_S Z_S) as CX-parity ladders + RZ.

    All Z strings commute, so this is exact. A string on qubits q_1 < ... < q_w
    costs 2(w-1) CX + 1 RZ before transpiler cancellation.
    """
    qc = QuantumCircuit(n, name=label)
    for mask, c in terms.items():
        qubits = [q for q in range(n) if mask >> q & 1]
        target = qubits[-1]
        for q in qubits[:-1]:
            qc.cx(q, target)
        qc.rz(2.0 * theta * c, target)
        for q in reversed(qubits[:-1]):
            qc.cx(q, target)
    return qc


def kinetic_terms(exp: Experiment) -> dict[int, float]:
    """k^2 in the momentum basis. k is linear in the Z's (two's complement), so
    k^2 has only 1- and 2-qubit Walsh terms: n(n+1)/2 rotations in total."""
    return walsh_terms(exp.grid.k ** 2)


def potential_evolution(exp: Experiment, theta: float) -> QuantumCircuit:
    """exp(-i theta V). The exact diagonal (Gray-code style synthesis, 2^n - 2
    CX) is cheaper than CX-ladder Walsh series once all terms are kept, so the
    Walsh form is only used when the experiment truncates V."""
    n = exp.grid.n
    if exp.walsh_terms is not None:
        return diagonal_evolution(n, walsh_terms(exp.potential), theta, "V")
    qc = QuantumCircuit(n, name="V")
    qc.append(DiagonalGate(list(np.exp(-1j * theta * exp.potential))), range(n))
    return qc


def evolution_circuit(exp: Experiment, steps: int) -> QuantumCircuit:
    n = exp.grid.n
    dt = exp.time / steps
    k_terms = kinetic_terms(exp)
    qft, iqft = QFTGate(n), QFTGate(n).inverse()

    qc = QuantumCircuit(n, name="U(t)")
    qc.compose(potential_evolution(exp, dt / 2), inplace=True)
    for i in range(steps):
        qc.append(iqft, range(n))
        qc.compose(diagonal_evolution(n, k_terms, dt, "K"), inplace=True)
        qc.append(qft, range(n))
        qc.compose(potential_evolution(exp, dt if i < steps - 1 else dt / 2),
                   inplace=True)
    return qc


def state_preparation(exp: Experiment) -> QuantumCircuit:
    qc = QuantumCircuit(exp.grid.n, name="prep")
    qc.append(StatePreparation(exp.initial_state()), range(exp.grid.n))
    return qc


def scattering_circuit(exp: Experiment, steps: int, measure: str | None = None
                       ) -> QuantumCircuit:
    """The state-preparation operator A of the experiment, ending in the momentum
    basis. measure: None, "sign" (top qubit only) or "all" (momentum spectrum)."""
    n = exp.grid.n
    qc = QuantumCircuit(n)
    qc.compose(state_preparation(exp), inplace=True)
    qc.compose(evolution_circuit(exp, steps), inplace=True)
    qc.append(QFTGate(n).inverse(), range(n))
    if measure == "sign":
        sign = QuantumCircuit(n, 1)
        sign.compose(qc, inplace=True)
        sign.measure(n - 1, 0)
        return sign
    if measure == "all":
        qc.measure_all()
    return qc


def momentum_statevector_twin(exp: Experiment, steps: int) -> np.ndarray:
    """What scattering_circuit(exp, steps) should output, from the numpy twin."""
    return np.fft.fft(exp.evolve_strang(steps)) / np.sqrt(exp.grid.N)
