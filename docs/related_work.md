# Related work and positioning

A starting point for the paper's related-work section, not a finished survey.
Entries under "Found in a search (Oct 2026)" turned up in web searches. Read
each one before citing it. The classical references are standard results.
Their bibliographic details are given from memory and need checking against
the journals.

## Positioning (provisional)

Searches for digital quantum simulation of **Regge–Wheeler scattering,
greybody factors, or quasinormal modes** on gate-based hardware found nothing
directly on point. Nearby work exists in four areas:

* **Hawking-radiation toy models on IBM hardware.** These include qubit
  transport and scrambling models, Page-curve and entanglement studies, and a
  VQE-based "Hawking radiation" simulation. These encode
  *information-theoretic* models of evaporation, not the field equation on the
  Schwarzschild background.
* **Analogue black holes.** These include a superconducting-qubit chain with
  stimulated Hawking radiation, and BEC experiments. They are analogue
  simulations, not digital ones.
* **Particle production in curved spacetime on IBM devices**, here cosmological
  rather than black-hole.
* **Grid-based split-operator QFT methods and wavepacket scattering** in
  chemistry and lattice QFT. These are methodologically close but not applied
  to black-hole potentials.

So the defensible claim is narrow: *a digital, gate-level simulation of the
Schwarzschild greybody problem, with an end-to-end error budget and a
noise-aware resource analysis*. Repeat the search before submission. A
two-search gap is not proof of novelty.

## Found in a search (Oct 2026), read before citing

Hawking radiation / black holes on quantum computers
* Quantum Simulation of Hawking Radiation Using VQE Algorithm on IBM Quantum
  Computer. <https://www.researchgate.net/publication/357525798>
* Capturing the Page Curve and Entanglement Dynamics of Black Holes in Quantum
  Computers. arXiv:2412.15180. <https://arxiv.org/abs/2412.15180>
* Studying evaporating black hole using quantum computation algorithms on IBM
  quantum processor. AIP Advances 14, 125121.
  <https://pubs.aip.org/aip/adv/article/14/12/125121/3326985>
* Information retrieval from Hawking radiation in the non-isometric model of
  black hole interior. arXiv:2307.01454. <https://arxiv.org/abs/2307.01454>
* Quantum simulation of Hawking radiation and curved spacetime with a
  superconducting on-chip black hole (analogue).
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10241825/>
* Digital quantum simulation of cosmological particle creation with IBM
  quantum computers. Sci. Rep. (2025), arXiv:2410.02412.
  <https://www.nature.com/articles/s41598-025-87015-6>

Grid / split-operator / tunnelling / scattering methods
* Grid-based methods for chemistry simulations on a quantum computer. Sci. Adv.
  (2023), arXiv:2202.05864. <https://arxiv.org/abs/2202.05864>
* Quantum Simulation of Tunneling in Small Systems. Sci. Rep. 2, 597 (2012),
  arXiv:1202.1536. <https://www.nature.com/articles/srep00597>
* Quantum Tunneling: From Theory to Error-Mitigated Quantum Simulation.
  arXiv:2404.07034. <https://arxiv.org/abs/2404.07034>
* Scalable quantum simulations of scattering in scalar field theory on 120
  qubits (IBM Heron).
  <https://www.osti.gov/pages/biblio/2878986-scalable-quantum-simulations-scattering-scalar-field-theory-qubits>
* Fermionic wave packet scattering: a quantum computing approach. Quantum
  (2025). <https://quantum-journal.org/papers/q-2025-02-19-1638/>

Amplitude estimation under noise
* Noisy quantum amplitude estimation without noise estimation.
  arXiv:2110.04258. <https://arxiv.org/abs/2110.04258>

Classical greybody / quasinormal-mode methods
* Topical review: greybody factors and quasinormal modes for black holes in
  various theories. arXiv:2205.01771. <https://arxiv.org/abs/2205.01771>
* An efficient higher-order WKB code for quasinormal modes and greybody
  factors. arXiv:2603.12466. <https://arxiv.org/abs/2603.12466>

## Classical results this code checks against (verify details)

* D. N. Page, Phys. Rev. D 13, 198 (1976). Particle emission rates from a
  black hole.
* Massless-scalar total power ≈ 7.44×10⁻⁵ M⁻². Usually attributed to
  T. Elster (1983). **Confirm the source.**
* S. R. Das, G. Gibbons, S. D. Mathur, PRL 78, 417 (1997). Low-energy σ_abs
  equals the horizon area.
* Y. Décanini, G. Esposito-Farèse, A. Folacci, PRD 83, 044032 (2011).
  High-energy σ_abs (the sinc formula).
* B. F. Schutz, C. M. Will, ApJ 291, L33 (1985). WKB barrier transmission.
* Y. Suzuki et al., "Amplitude estimation without phase estimation",
  Quantum Inf. Process. 19, 75 (2020), arXiv:1904.10246. MLAE.
* S. Dolan, PRD 76, 084001 (2007). Quasi-bound states of massive scalars
  (Kerr). Use it for the Im(ω) comparison the Hermitian box can't give.
