"""Quantum layer: windowed potential, grid twin, circuits, AE, bound states."""

from dataclasses import replace

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from bhtunnel import scatter
from bhtunnel.ae import ae_circuit, mlae_estimate, oracle_calls, p_model
from bhtunnel.bound import BoundBox, hydrogenic_omega
from bhtunnel.circuits import kinetic_terms, momentum_statevector_twin, scattering_circuit
from bhtunnel.grid import Experiment, Grid, fwht, ifwht, truncate_walsh
from bhtunnel.presets import AER_L1, HW5, HW6, HW_STEPS
from bhtunnel.window import Window, scatter_windowed

SMALL = Experiment(grid=Grid(5, -20.0, 44.0), window=Window(-16.0, 18.0, 5.0),
                   sigma=4.0, x0=32.0, k0=0.3)


# ------------------------------------------------------------------ window

def test_window_shape():
    w = Window(-10.0, 20.0, 4.0)
    assert w(-11.0) == 0.0 and w(21.0) == 0.0
    np.testing.assert_allclose(w(np.array([-6.0, 0.0, 16.0])), 1.0)


def test_windowed_transmission_converges_to_exact():
    exact = scatter(0.2, 0).gamma
    assert abs(scatter_windowed(0.2, 0, window=Window(-25, 400, 8)) / exact - 1) < 1e-4


# ------------------------------------------------------------------ grid twin

def test_walsh_roundtrip_and_truncation():
    v = np.random.default_rng(0).random(32)
    np.testing.assert_allclose(ifwht(fwht(v)), v, atol=1e-12)
    np.testing.assert_allclose(truncate_walsh(v, 31), v, atol=1e-12)
    assert np.count_nonzero(np.abs(fwht(truncate_walsh(v, 5))[1:]) > 1e-12) == 5


def test_replace_does_not_share_cache():
    base = replace(SMALL)
    v_full = base.potential                 # populate base's cache
    trunc = replace(base, walsh_terms=4)
    assert not np.array_equal(trunc.potential, v_full)
    np.testing.assert_array_equal(trunc.potential, truncate_walsh(v_full, 4))


def test_kinetic_operator_is_at_most_two_body():
    terms = kinetic_terms(SMALL)
    assert all(bin(mask).count("1") <= 2 for mask in terms)
    n = SMALL.grid.n
    assert len(terms) <= n * (n + 1) // 2


def test_strang_is_second_order():
    exact = SMALL.evolve_exact()
    err = [np.linalg.norm(SMALL.evolve_strang(s) - exact) for s in (40, 80)]
    assert 3.5 < err[0] / err[1] < 4.5


def test_production_box_is_wrap_safe_and_accurate():
    assert AER_L1.wrap_margin() > 0
    p_grid = AER_L1.transmitted_probability(AER_L1.evolve_exact())
    p_box = AER_L1.reference_transmitted(
        lambda ks: np.array([scatter_windowed(k, 1) for k in ks]))
    assert abs(p_grid - p_box) < 3e-3


# ------------------------------------------------------------------ circuits

@pytest.mark.parametrize("walsh", [None, 8])
def test_circuit_matches_twin(walsh):
    exp = replace(SMALL, walsh_terms=walsh)
    sv = Statevector(scattering_circuit(exp, 3)).data
    twin = momentum_statevector_twin(exp, 3)
    assert 1 - abs(np.vdot(sv, twin)) ** 2 < 1e-10
    # top qubit = sign of k = transmitted direction
    p_sign = Statevector(scattering_circuit(exp, 3)).probabilities([exp.grid.n - 1])[1]
    assert abs(p_sign - exp.transmitted_probability(exp.evolve_strang(3))) < 1e-10


def test_hardware_presets_are_distinguishable_from_depolarised():
    for name, exp in (("HW5", HW5), ("HW6", HW6)):
        p = exp.transmitted_probability(exp.evolve_strang(HW_STEPS[name]))
        assert abs(p - 0.5) > 0.2, name


# ------------------------------------------------------------------ AE

def test_grover_circuits_follow_sin2_law():
    exp = replace(HW5, k0=0.16)
    a = scattering_circuit(exp, 1)
    theta = np.arcsin(np.sqrt(exp.transmitted_probability(exp.evolve_strang(1))))
    for m in (0, 1, 2):
        p = Statevector(ae_circuit(a, exp.grid.n - 1, m, measure=False)).probabilities(
            [exp.grid.n - 1])[1]
        assert abs(p - p_model(theta, m)) < 1e-9


def test_mlae_recovers_probability():
    p = 2e-3
    theta = np.arcsin(np.sqrt(p))
    ms = [0, 1, 2, 4, 8, 16]
    shots = [10**6] * len(ms)
    hits = [round(n * p_model(theta, m)) for n, m in zip(shots, ms)]
    assert abs(mlae_estimate(hits, shots, ms) / p - 1) < 1e-3


def test_oracle_cost_model_limits():
    samp, ae_calls, _ = oracle_calls(1e-3, 0.1, 1.0)
    assert abs(samp / ((1 - 1e-3) / (1e-3 * 0.01)) - 1) < 1e-6
    assert ae_calls < samp / 10                    # ideal AE wins in the tail
    s2, a2, _ = oracle_calls(1e-3, 0.1, 0.3)
    assert a2 > s2                                 # ...but not at lambda = 0.3


# ------------------------------------------------------------------ bound states

def test_quasi_bound_2p_level():
    box = BoundBox(N=512)
    w = box.omega(box.spectrum(1)[0][0])
    assert abs(w / hydrogenic_omega(1, 0.3) - 1) < 3e-3   # O((M mu)^2) corrections
    small = BoundBox(N=32)
    assert abs(small.omega(small.spectrum(1)[0][0]) - w) < 1e-4
