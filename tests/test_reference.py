"""Validation of the classical reference layer against known physics."""

import numpy as np
import pytest

from bhtunnel import (absorption_cross_section, number_spectrum, potential_peak,
                      r_from_tortoise, scatter, total_power, tortoise)
from bhtunnel.wkb import cross_section_sinc


def test_tortoise_roundtrip():
    r = np.array([2.0 + 1e-9, 2.001, 2.5, 3.0, 10.0, 1e3, 1e6])
    r_back, y = r_from_tortoise(tortoise(r))
    np.testing.assert_allclose(r_back, r, rtol=1e-12)
    np.testing.assert_allclose(y[0], 1e-9, rtol=1e-6)


def test_tortoise_inverse_deep_near_horizon():
    _, y = r_from_tortoise(-60.0)
    # y + ln y = r*/2 - 1 with y = (r-2)/2, so r - 2 = 2 W(e^-31) ~ 2 e^-31.
    assert np.isclose(y, 2 * np.exp(-31.0), rtol=1e-12)


@pytest.mark.parametrize("omega,l,s", [(0.01, 0, 0), (0.2, 1, 0), (0.6, 2, 0),
                                       (1.2, 4, 0), (0.3, 1, 1), (0.5, 2, 2)])
def test_flux_conservation(omega, l, s):
    res = scatter(omega, l, s)
    assert 0.0 <= res.gamma <= 1.0
    assert res.flux_error < 1e-9


def test_result_independent_of_matching_radius():
    a = scatter(0.25, 1)
    b = scatter(0.25, 1, r_match=5 * a.r_match)
    assert abs(a.gamma / b.gamma - 1) < 1e-9


def test_low_frequency_s_wave_gives_horizon_area():
    # Gamma_0 -> 16 (M omega)^2  <=>  sigma_abs -> 16 pi M^2 (Das-Gibbons-Mathur).
    for omega in (0.001, 0.002):
        ratio = scatter(omega, 0).gamma / (16 * omega**2)
        assert abs(ratio - 1) < 7 * omega  # leading correction is O(M omega)


@pytest.mark.parametrize("l", [1, 2, 3, 5])
def test_barrier_top_transmission_approaches_half(l):
    # WKB says Gamma(omega^2 = V_max) = 1/2; the exact value approaches it from
    # above as l grows (the barrier becomes more parabolic relative to omega).
    _, v0, d2v = potential_peak(l)
    assert d2v < 0
    gamma = scatter(np.sqrt(v0), l).gamma
    assert 0.5 < gamma < 0.5 + 0.7 / (l + 1)


def test_known_barrier_peaks():
    # Scalar l=1 peak sits near r = 2.89 M; large-l peaks approach the photon
    # sphere r = 3M with V_max -> l(l+1)/27 M^2.
    r1, _, _ = potential_peak(1)
    assert abs(r1 - 2.886) < 1e-3
    r20, v20, _ = potential_peak(20)
    assert abs(r20 - 3.0) < 2e-3
    assert abs(v20 / (20 * 21 / 27) - 1) < 2e-3


@pytest.mark.slow
def test_high_frequency_cross_section_matches_photon_sphere_formula():
    omegas = np.array([0.8, 1.0])
    sigma = absorption_cross_section(omegas)
    np.testing.assert_allclose(sigma, cross_section_sinc(omegas), rtol=5e-3)


@pytest.mark.slow
def test_total_scalar_hawking_power():
    # Massless scalar emission from Schwarzschild: dE/dt = 7.44e-5 / M^2
    # (Page 1976; Elster 1983).
    omegas = np.linspace(2e-3, 0.6, 70)
    power = total_power(omegas, number_spectrum(omegas, l_max=4))
    assert abs(power / 7.44e-5 - 1) < 5e-3
