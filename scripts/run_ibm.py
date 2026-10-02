"""Run the hardware-scale scattering circuit on IBM Quantum.

    python scripts/run_ibm.py                       # dry run on FakeFez (default)
    python scripts/run_ibm.py --submit [--backend ibm_fez] [--steps 1 2]
    python scripts/run_ibm.py --retrieve <job_id>

The observable is Z on the sign qubit: P_T = (1 - <Z>) / 2. Mitigation is done
by the Runtime Estimator: resilience level 2 (TREX readout + ZNE), gate and
measurement twirling, XY4 dynamical decoupling.

Results are scored against the classical twin at the *same* Strang step count
(see bhtunnel.presets for why), and saved to data/hardware/<job_id>.json.

--submit uses the account saved with QiskitRuntimeService.save_account() and
consumes QPU time on that account. Nothing is submitted without the flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bhtunnel.circuits import scattering_circuit  # noqa: E402
from bhtunnel.presets import HW5  # noqa: E402

OUT = ROOT / "data" / "hardware"


def build_pubs(backend, steps_list, precision: float):
    exp = replace(HW5, _cache={})
    n = exp.grid.n
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3,
                                      seed_transpiler=11)
    obs = SparsePauliOp("Z" + "I" * (n - 1))  # Z on qubit n-1 (leftmost label)
    pubs, meta = [], []
    for steps in steps_list:
        isa = pm.run(scattering_circuit(exp, steps))
        pubs.append((isa, obs.apply_layout(isa.layout), None, precision))
        meta.append({"steps": steps,
                     "p_twin": exp.transmitted_probability(exp.evolve_strang(steps)),
                     "p_grid_exact": exp.transmitted_probability(exp.evolve_exact()),
                     "cz": isa.count_ops().get("cz", 0), "depth": isa.depth()})
    return pubs, meta


def estimator_options(level: int) -> dict:
    return {"resilience_level": level,
            "twirling": {"enable_gates": True, "enable_measure": True},
            "dynamical_decoupling": {"enable": True, "sequence_type": "XY4"}}


def report(meta, evs, stds):
    for m, ev, sd in zip(meta, evs, stds):
        p = (1 - ev) / 2
        m.update(p_hw=float(p), p_hw_sd=float(sd / 2),
                 rel_error=float(abs(p - m["p_twin"]) / m["p_twin"]))
        print(f"steps={m['steps']}  CZ={m['cz']:4d}  P_T(hw)={p:.4f} ± {sd / 2:.4f}"
              f"  P_T(twin)={m['p_twin']:.4f}  rel.err={m['rel_error']:.1%}")


def main():
    ap = argparse.ArgumentParser(description="IBM hardware run of the HW5 circuit")
    ap.add_argument("--submit", action="store_true", help="really submit (uses QPU time)")
    ap.add_argument("--retrieve", metavar="JOB_ID")
    ap.add_argument("--backend", default=None, help="default: least busy")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--precision", type=float, default=0.01,
                    help="target std of <Z> per PUB (sets shot count)")
    ap.add_argument("--resilience", type=int, default=2)
    args = ap.parse_args()

    if args.retrieve:
        from qiskit_ibm_runtime import QiskitRuntimeService
        job = QiskitRuntimeService().job(args.retrieve)
        saved = json.loads((OUT / f"{args.retrieve}.json").read_text())
        res = job.result()
        report(saved["pubs"], [float(r.data.evs) for r in res],
               [float(r.data.stds) for r in res])
        saved["status"] = "done"
        (OUT / f"{args.retrieve}.json").write_text(json.dumps(saved, indent=2))
        return

    if not args.submit:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator
        from qiskit_ibm_runtime.fake_provider import FakeFez
        backend = FakeFez()
        pubs, meta = build_pubs(backend, args.steps, args.precision)
        print("DRY RUN on noisy Aer (FakeFez), no Runtime mitigation. "
              "Pass --submit to run on IBM hardware.")
        est = AerEstimator.from_backend(AerSimulator.from_backend(backend))
        res = est.run(pubs).result()
        report(meta, [float(r.data.evs) for r in res], [float(r.data.stds) for r in res])
        return

    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
    service = QiskitRuntimeService()
    backend = (service.backend(args.backend) if args.backend
               else service.least_busy(operational=True, simulator=False))
    pubs, meta = build_pubs(backend, args.steps, args.precision)
    est = EstimatorV2(mode=backend, options=estimator_options(args.resilience))
    job = est.run(pubs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{job.job_id()}.json").write_text(json.dumps(
        {"job_id": job.job_id(), "backend": backend.name, "status": "submitted",
         "options": estimator_options(args.resilience), "pubs": meta}, indent=2))
    print(f"submitted {job.job_id()} to {backend.name}; "
          f"retrieve with: python scripts/run_ibm.py --retrieve {job.job_id()}")


if __name__ == "__main__":
    main()
