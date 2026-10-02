# Quantum simulation of black-hole greybody factors

This project simulates wave scattering off the Schwarzschild Regge–Wheeler
barrier on a digital quantum computer. The target is the **greybody factors**
Γ_l(ω) that filter Hawking radiation. Alongside the simulation it builds an
end-to-end error budget, a gate-cost study, a noise and mitigation study,
an amplitude-estimation analysis for the tunnelling tail, and a VQE treatment
of massive-scalar quasi-bound states.

The quantum computer only ever estimates Γ_l. The Hawking spectrum is classical
post-processing, N_ω ∝ Σ(2l+1)Γ_l / (e^{ω/T_H} − 1), and is reported that way.
For positioning against prior work, see [docs/related_work.md](docs/related_work.md).

**Interactive demo:** [huggingface.co/spaces/Hastika06/bhtunnel](https://huggingface.co/spaces/Hastika06/bhtunnel).
It covers the wavepacket animation, greybody and Hawking curves, the live circuit, and the
amplitude-estimation break-even. To run it locally, see "Interactive demo" below.

## Main results

**1. Classical reference is validated** ([scripts/make_reference.py](scripts/make_reference.py))

| Check | Result |
|---|---|
| Flux conservation Γ + R = 1 | < 1e-9 (typ. 1e-12), s = 0, 1, 2 |
| Low-ω s-wave: σ_abs → 16πM² (horizon area) | ✓ |
| High-ω σ_abs vs. photon-sphere sinc formula | < 0.5 % at Mω ≥ 0.8 |
| Total scalar Hawking power | **7.437e-5 M⁻²**, literature 7.44e-5 |

**2. The algorithm reproduces Γ_l on a simulator** ([scripts/error_budget.py](scripts/error_budget.py))

The method is a split-operator wavepacket on a 2⁸-point r* grid. After QFT†,
the top qubit is the sign of the momentum, so **a single-qubit Z measurement
gives the transmitted probability**. Measuring the whole register gives Γ
resolved by frequency from one packet. The circuit matches the numpy twin to
a fidelity of 1 − 1e-14. Error budget for an l = 1 packet at Mω ≈ 0.3 (P_T ≈ 0.52):

| Link | ΔP_T |
|---|---|
| finite box (window at r* ∈ [−25, 30]) | 7e-4 |
| grid discretisation, n = 8 | 1.1e-3 |
| Trotter, 60 Strang steps | 3.9e-3 |
| shot noise, 10⁴ shots | 5e-3 |

The finite box costs little here, but it dominates in the tunnelling tail:
up to 97 % error for l = 2 at Mω = 0.1 with the window ending at r* = 30.

**3. Resources** ([scripts/resource_study.py](scripts/resource_study.py))

The kinetic term is only 2-body in the momentum basis (n(n+1)/2 rotations).
The potential is not sparse in the Walsh basis: a localised barrier needs
≳ 100 of 255 Walsh terms for 1e-3 accuracy. Exact diagonal synthesis
(2ⁿ − 2 CZ) is therefore the default, and it dominates the cost from n ≈ 6.

**4. Hardware is a device benchmark, not a physics measurement** ([scripts/noise_study.py](scripts/noise_study.py))

A faithful Trotterisation needs the packet to move less than one barrier width
per step. That means 8–30 steps and thousands of CZ, which is beyond current
devices. The 5-qubit, 2-step circuit already needs 390 CZ on Heron, for an
estimated fidelity of about 0.18. Hardware runs are therefore scored against the twin *at the
same step count*, with expected P_T far from ½, because a fully depolarised
device returns ½. Results on noisy Aer (FakeFez):

| circuit | twin | raw | readout + ZNE | per-oracle survival λ |
|---|---|---|---|---|
| HW5, 1 step (264 CZ) | 0.808 | 0.647 | 0.822 | 0.48 |
| HW5, 2 steps (399 CZ) | 0.756 | 0.571 | 0.715 | 0.28 |

**5. Amplitude estimation for the tunnelling tail** ([scripts/amplitude_estimation.py](scripts/amplitude_estimation.py))

The Grover circuits are verified on Aer to follow sin²((2m+1)θ). On an ideal
device, MLAE reaches ∝ N⁻¹ scaling and beats sampling by about 10–100× for
p = 1e-2…1e-4 at 10 % relative error. It only wins, though, while the
per-oracle survival is λ ≳ 0.63–0.93, depending on p. The measured
λ ≈ 0.3–0.5 for the hardware-scale oracle is **below break-even**, so today
sampling is the better estimator.

**6. Massive scalar, where VQE does fit** ([scripts/vqe_bound_state.py](scripts/vqe_bound_state.py))

For Mμ = 0.3, l = 1, a Dirichlet wall at the barrier top turns the 2p
quasi-bound state into a Hermitian ground state. VQE on 5 qubits
(real-amplitudes ansatz, 25 parameters) reaches a state fidelity of 0.99993.
The resulting ω = 0.29622 matches the exact grid value (0.29622), the
converged box (0.29624) and the hydrogenic formula (0.29662) to 0.13 %.
The Hamiltonian has 63 Pauli terms in 17 qubit-wise-commuting groups. The
Hermitian box cannot give Im ω.

## Conventions

* G = c = ħ = 1, **M = 1**; frequencies are the dimensionless Mω.
* ψ'' + (ω² − V_l)ψ = 0 in r* = r + 2 ln(r/2 − 1), with
  V_l = (1 − 2/r)[l(l+1)/r² + 2(1 − s²)/r³ (+ μ²)].
  So **H = −∂²_{r*} + V_l with eigenvalue ω²**, not −½∂² + V.
* Qubit order is little-endian, and the momentum register is fft(ψ)/√N = QFT†ψ.
* Γ_l depends on M and ω only through Mω.

## Layout

```
src/bhtunnel/
  schwarzschild.py  potential (incl. mass term), tortoise coordinate, barrier peak
  greybody.py       exact scattering solver, absorption cross section
  wkb.py            WKB / low-ω / sinc approximations (cross-checks)
  hawking.py        emission spectra from greybody factors
  window.py         finite-support potential, its exact transmission
  grid.py           classical twin: grid, Walsh tools, split-operator, observables
  circuits.py       Qiskit circuits, gate-for-gate equal to the twin
  presets.py        named experiment boxes (Aer-scale and hardware-scale)
  ae.py             Grover operator, MLAE, Cramér–Rao cost model under noise
  bound.py          massive-scalar quasi-bound states in a Dirichlet box
app/                Streamlit demo (views/: wavepacket, greybody, circuit, amplitude)
deploy/hf-space/    Dockerfile + Space card for Hugging Face
scripts/            one script per study → data/*.csv|json, figures/*.png
  run_ibm.py        IBM Runtime submission (dry run by default)
notebooks/results_overview.ipynb   all results, for presentations
tests/              physics + circuit-equivalence tests
docs/related_work.md
```

## Running

```bash
pip install -r requirements-lock.txt
```

```bash
pip install -e ".[dev,quantum]"
```

```bash
pytest -m "not slow"
```

```bash
python scripts/make_reference.py
```

Then run, in any order: `error_budget.py`, `resource_study.py`, `noise_study.py`,
`amplitude_estimation.py` (needs `data/noise.json`) and `vqe_bound_state.py`.
Each takes between about 20 seconds and 3 minutes.

### Interactive demo

```bash
pip install -e ".[quantum,app]"
```

```bash
streamlit run app/streamlit_app.py
```

The app reads precomputed greybody tables from `src/bhtunnel/data/`. Regenerate
them with `scripts/make_app_tables.py`. The Hugging Face Space is defined in
[deploy/hf-space/](deploy/hf-space/). It builds from this repository, so after
pushing to `main`, use **Factory rebuild** in the Space settings.

### IBM hardware

```bash
python scripts/run_ibm.py
```

This is a dry run on noisy Aer. The real run below uses your saved
`QiskitRuntimeService` account and **consumes QPU time**. It runs the Runtime
Estimator with resilience level 2, twirling and XY4 dynamical decoupling, and
saves results to `data/hardware/`.

```bash
python scripts/run_ibm.py --submit --backend ibm_fez
```

## Open items

* Run `run_ibm.py --submit` and add the hardware column to result 4.
* Im ω of the quasi-bound state, from a continued-fraction method (Leaver/Dolan)
  or a complex absorbing potential, to go beyond the Hermitian box.
* A Gray-code synthesis of *truncated* Walsh series, which could make
  truncation pay off at large n.
* Re-run the novelty search before submission (see docs/related_work.md).
