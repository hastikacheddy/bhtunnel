"""Hawking emission spectra built from greybody factors.

The quantum part of the project only ever estimates Gamma_l. Everything here is
classical post-processing: the thermal factor is an exact formula.

For one bosonic degree of freedom (per unit time, per unit M*omega):

    d^2N / dt d omega = (1/2 pi) sum_l (2l+1) Gamma_l(omega) / (exp(omega/T_H) - 1)

and the power spectrum is omega times that. T_H = 1/(8 pi M).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from .greybody import greybody
from .schwarzschild import T_HAWKING


def bose_factor(omegas):
    return 1.0 / np.expm1(np.asarray(omegas, dtype=float) / T_HAWKING)


def number_spectrum(omegas, s: int = 0, l_max: int = 8, gammas: dict | None = None):
    """d^2N/dt domega. Pass precomputed {l: Gamma_l(omegas)} via `gammas` to reuse
    (or substitute quantum-estimated) greybody factors."""
    omegas = np.asarray(omegas, dtype=float)
    total = np.zeros_like(omegas)
    for l in range(s, l_max + 1):
        g = gammas[l] if gammas is not None and l in gammas else greybody(omegas, l, s)
        total += (2 * l + 1) * g
    return total * bose_factor(omegas) / (2 * np.pi)


def blackbody_number_spectrum(omegas, sigma_over_pi: float = 27.0):
    """Same formula with a constant cross section (geometric-optics 27 pi M^2 by
    default): sum_l (2l+1) Gamma_l -> sigma omega^2 / pi."""
    omegas = np.asarray(omegas, dtype=float)
    return sigma_over_pi * omegas**2 * bose_factor(omegas) / (2 * np.pi)


def total_power(omegas, number_spec) -> float:
    """dE/dt = integral omega d^2N/dt domega, in units of M^-2."""
    return float(trapezoid(np.asarray(omegas) * number_spec, omegas))
