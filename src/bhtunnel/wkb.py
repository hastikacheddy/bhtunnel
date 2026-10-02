"""Approximate greybody factors used as cross-checks on the exact solver."""

from __future__ import annotations

import numpy as np

from .schwarzschild import potential_peak


def greybody_wkb(omegas, l: int, s: int = 0) -> np.ndarray:
    """Lowest-order WKB (Schutz-Will) transmission through the barrier top:

        Gamma ~ 1 / (1 + exp(2 pi (V0 - omega^2) / sqrt(-2 V0''))),

    with V0'' = d^2V/dr*^2 at the peak. Accurate near the peak and for large l,
    gives exactly 1/2 at omega^2 = V0, and fails badly in the deep-tunnelling tail.
    """
    _, v0, d2v = potential_peak(l, s)
    omegas = np.asarray(omegas, dtype=float)
    return 1.0 / (1.0 + np.exp(2 * np.pi * (v0 - omegas**2) / np.sqrt(-2 * d2v)))


def greybody_scalar_s_wave_low_freq(omegas) -> np.ndarray:
    """Leading low-frequency scalar s-wave result Gamma_0 -> 16 (M omega)^2, which
    makes sigma_abs equal the horizon area 16 pi M^2 (Das-Gibbons-Mathur)."""
    return 16.0 * np.asarray(omegas, dtype=float) ** 2


def cross_section_sinc(omegas) -> np.ndarray:
    """High-frequency scalar absorption cross section from the photon-sphere
    (Regge pole) analysis of Decanini, Esposito-Farese & Folacci (2011):

        sigma / (pi M^2) ~ 27 [1 - 8 pi exp(-pi) sinc(2 pi sqrt(27) M omega)].
    """
    x = 2 * np.pi * np.sqrt(27.0) * np.asarray(omegas, dtype=float)
    return 27.0 * (1.0 - 8 * np.pi * np.exp(-np.pi) * np.sin(x) / x)
