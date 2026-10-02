import streamlit as st

from common import REPO_URL

st.title("Black-hole tunnelling on a quantum computer")
st.markdown(
    f"""
A wave that escapes a black hole has to tunnel through the **Regge–Wheeler
barrier** around it. The fraction that gets through, the **greybody factor**
Γ_l(ω), is what turns Hawking's perfect blackbody into the spectrum a black hole
actually emits. This demo simulates that tunnelling with the same quantum
circuits used in the research code ([GitHub]({REPO_URL})).
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Scalar Hawking power", "7.437e-5 M⁻²", "literature 7.44e-5", delta_color="off")
c2.metric("Simulator error, l=1 packet", "≈ 4e-3", "8 qubits, 60 steps", delta_color="off")
c3.metric("Hardware-scale circuit", "390 CZ", "5 qubits, 2 steps", delta_color="off")
c4.metric("VQE quasi-bound state", "ω = 0.29622", "fidelity 0.99993", delta_color="off")

st.subheader("How the quantum simulation works")
st.markdown(
    r"""
1. **Encode space in qubits.** n qubits hold the wave on 2ⁿ points of the
   tortoise coordinate r*, which runs from the horizon (left) to far away (right).
2. **Evolve with split-operator steps.** The potential acts as a diagonal phase.
   The kinetic term acts in momentum space, after a quantum Fourier transform,
   where it is only a 2-qubit phase. The Hamiltonian is
   $H = -\partial^2_{r_*} + V_l(r_*)$, with eigenvalue $\omega^2$.
3. **Read out one qubit.** After a final QFT†, the top qubit is the *sign of the
   momentum*. Measuring it gives the probability that the wave went through
   toward the horizon. Measuring all qubits gives Γ at every frequency in the
   packet at once.
"""
)

st.subheader("What this demo does not claim")
st.markdown(
    """
* **No quantum advantage.** This is a 1-D problem a laptop solves exactly. The
  project measures what a quantum simulation *costs*, and where it breaks.
* **Hardware runs benchmark the device, not the physics.** A faithful simulation
  needs thousands of two-qubit gates. The 5-qubit hardware circuits are scored
  against a classical copy of the *same* circuit.
* **The Hawking spectrum is classical post-processing.** Only Γ_l comes from the
  quantum simulation. The thermal factor is an exact formula.
"""
)
st.caption("Units: G = c = ħ = 1, black-hole mass M = 1. Frequencies are Mω.")
