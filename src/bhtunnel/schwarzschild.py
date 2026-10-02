"""Schwarzschild geometry and Regge-Wheeler effective potentials.

Geometric units G = c = hbar = 1 with M = 1 throughout, so every length is in
units of M and every frequency is the dimensionless combination M*omega.

The radial equation for a massless field of spin s and multipole l is

    d^2 psi / dr*^2 + [omega^2 - V_l(r)] psi = 0,

    V_l(r) = f(r) [ l(l+1)/r^2 + 2(1 - s^2)/r^3 ],    f(r) = 1 - 2/r,

with tortoise coordinate r* = r + 2 ln(r/2 - 1). s = 0 is the scalar field,
s = 1 electromagnetic, s = 2 axial gravitational (Regge-Wheeler proper).

As an operator, H = -d^2/dr*^2 + V_l has eigenvalue omega^2 (note: no 1/2).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

R_HORIZON = 2.0
T_HAWKING = 1.0 / (8.0 * np.pi)


def lapse(r):
    """f(r) = 1 - 2/r."""
    return 1.0 - 2.0 / np.asarray(r, dtype=float)


def tortoise(r):
    """r*(r) = r + 2 ln(r/2 - 1), defined for r > 2."""
    r = np.asarray(r, dtype=float)
    return r + 2.0 * np.log(r / 2.0 - 1.0)


def r_from_tortoise(rs, tol: float = 1e-14, max_iter: int = 60):
    """Invert the tortoise coordinate.

    With y = r/2 - 1 the relation is y + ln y = r*/2 - 1, i.e. y = W(exp(r*/2 - 1)).
    Solved by Newton iteration on ln y so it neither overflows at large r* nor
    loses precision near the horizon. Returns r - 2 as well as r, because r - 2
    underflows relative to r deep in the near-horizon region.
    """
    z = np.asarray(rs, dtype=float) / 2.0 - 1.0
    # Unknown u = ln y solves u + exp(u) = z.
    u = np.where(z < 1.0, z, np.log(np.maximum(z, 1.0)))
    for _ in range(max_iter):
        eu = np.exp(u)
        step = (u + eu - z) / (1.0 + eu)
        u = u - step
        if np.all(np.abs(step) < tol):
            break
    y = np.exp(u)
    return 2.0 + 2.0 * y, 2.0 * y


def potential(r, l: int, s: int = 0, mu: float = 0.0):
    """Regge-Wheeler potential V_l(r) for spin s. mu > 0 adds a scalar mass
    term, f(r) mu^2 (only meaningful for s = 0)."""
    _check_ls(l, s)
    if mu and s != 0:
        raise ValueError("a mass term is only implemented for the scalar field")
    r = np.asarray(r, dtype=float)
    return lapse(r) * (l * (l + 1) / r**2 + 2.0 * (1 - s * s) / r**3 + mu * mu)


def potential_peak(l: int, s: int = 0):
    """Location and shape of the potential barrier top.

    Returns (r_peak, V_peak, d2V/dr*^2 at the peak). At an extremum dV/dr = 0, so
    d^2V/dr*^2 = f^2 d^2V/dr^2.
    """
    _check_ls(l, s)
    res = minimize_scalar(lambda r: -potential(r, l, s), bounds=(2.05, 10.0),
                          method="bounded", options={"xatol": 1e-12})
    r0 = float(res.x)
    h = 1e-4
    d2v_dr2 = (potential(r0 + h, l, s) - 2 * potential(r0, l, s)
               + potential(r0 - h, l, s)) / h**2
    return r0, float(potential(r0, l, s)), float(lapse(r0) ** 2 * d2v_dr2)


def _check_ls(l: int, s: int) -> None:
    if s not in (0, 1, 2):
        raise ValueError(f"spin s must be 0, 1 or 2, got {s}")
    if l < s:
        raise ValueError(f"multipole l={l} must satisfy l >= s={s}")
