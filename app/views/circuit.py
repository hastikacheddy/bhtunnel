from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import numpy as np
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector

from bhtunnel.circuits import (diagonal_evolution, kinetic_terms, potential_evolution,
                               scattering_circuit, state_preparation)
from bhtunnel.presets import HW5, HW6

st.title("The quantum circuit")
st.markdown(
    "The circuit that runs on the simulator and on IBM hardware. The top qubit, read "
    "after the final QFT†, is the **sign of the momentum**: 1 means the wave is moving "
    "toward the horizon, so P(top qubit = 1) *is* the transmitted probability."
)

presets = {"HW5: 5 qubits (hardware scale)": HW5, "HW6: 6 qubits (hardware scale)": HW6}
c1, c2 = st.columns([2, 1])
name = c1.selectbox("Experiment", list(presets))
steps = c2.slider("Strang steps", 1, 4, 2)
exp = replace(presets[name])
n, dt = exp.grid.n, exp.time / steps


def display_circuit(exp, steps):
    """Same operator sequence as scattering_circuit, with each block boxed."""
    n = exp.grid.n

    def qft(inverse=False):
        g = QFTGate(n).inverse() if inverse else QFTGate(n)
        g.label = "QFT†" if inverse else "QFT"
        return g

    qc = QuantumCircuit(n, 1)
    qc.append(state_preparation(exp).to_gate(label="ψ₀"), range(n))
    qc.append(potential_evolution(exp, dt / 2).to_gate(label="V/2"), range(n))
    k_gate = diagonal_evolution(n, kinetic_terms(exp), dt).to_gate(label="K")
    for i in range(steps):
        qc.append(qft(inverse=True), range(n))
        qc.append(k_gate, range(n))
        qc.append(qft(), range(n))
        last = i == steps - 1
        qc.append(potential_evolution(exp, dt / 2 if last else dt)
                  .to_gate(label="V/2" if last else "V"), range(n))
    qc.append(qft(inverse=True), range(n))
    qc.measure(n - 1, 0)
    return qc


st.pyplot(display_circuit(exp, steps).draw("mpl", fold=-1, scale=0.9), width="stretch")
st.caption("ψ₀: prepare the Gaussian packet · V, V/2: diagonal phase e^{−iV dt}, "
           "e^{−iV dt/2} · QFT†, K, QFT: kinetic phase e^{−ik² dt} in momentum space · "
           "final QFT† + measuring the top qubit = sign of the momentum.")


@st.cache_data(show_spinner="Transpiling for an IBM Heron device…")
def costs(name, steps):
    from qiskit_ibm_runtime.fake_provider import FakeFez
    e = replace(presets[name])
    qc = scattering_circuit(e, steps, measure="sign")
    flat = transpile(qc, basis_gates=["cz", "rz", "sx", "x"], optimization_level=2,
                     seed_transpiler=7)
    routed = transpile(qc, FakeFez(), optimization_level=3, seed_transpiler=7)
    return (flat.count_ops().get("cz", 0), routed.count_ops().get("cz", 0),
            routed.depth())


@st.cache_data(show_spinner="Simulating the circuit on Aer…")
def check(name, steps):
    e = replace(presets[name])
    sv = Statevector(scattering_circuit(e, steps))
    p_circ = float(sv.probabilities([e.grid.n - 1])[1])
    return (p_circ, e.transmitted_probability(e.evolve_strang(steps)),
            e.transmitted_probability(e.evolve_exact()))


cz_flat, cz_routed, depth = costs(name, steps)
p_circ, p_twin, p_exact = check(name, steps)

st.subheader("Cost and correctness")
m1, m2, m3 = st.columns(3)
m1.metric("CZ gates, all-to-all", cz_flat)
m2.metric("CZ gates on IBM Fez", cz_routed, f"depth {depth}", delta_color="off")
m3.metric("Est. circuit fidelity", f"{(1 - 2.8e-3) ** cz_routed:.2f}",
          "≈ 0.28 % error per CZ incl. routing", delta_color="off")
m1.metric("P(top qubit = 1), Aer circuit", f"{p_circ:.6f}")
m2.metric("Classical twin, same steps", f"{p_twin:.6f}",
          f"Δ = {p_circ - p_twin:+.1e}", delta_color="off")
m3.metric("Exact time evolution", f"{p_exact:.4f}",
          f"Trotter gap {p_twin - p_exact:+.3f}", delta_color="off")
st.markdown(
    f"""
The circuit and its classical twin agree to machine precision, so the circuit is
correct. At {steps} step(s), though, Trotter error moves P_T by
{abs(p_twin - p_exact):.2f} from the exact answer. Hardware-scale runs are therefore
**device benchmarks**: the device is scored against the twin at the same step count.
A faithful run needs 8–30 steps, which means thousands of CZ gates, beyond today's
hardware.
"""
)
with st.expander("Why is the kinetic term so cheap?"):
    st.markdown(
        "In the momentum basis k is *linear* in the qubits' Z operators (two's "
        "complement), so k² has only one- and two-qubit terms: n(n+1)/2 rotations. "
        "The potential, a localised barrier, needs the full diagonal (2ⁿ − 2 CZ). That "
        "is why the potential, not the QFT, dominates the cost from about 6 qubits."
    )
    st.code(f"kinetic Walsh terms for n = {n}: {len(kinetic_terms(exp))} "
            f"(max weight {max(bin(m).count('1') for m in kinetic_terms(exp))})")
