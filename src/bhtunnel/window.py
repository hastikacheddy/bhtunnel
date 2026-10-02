"""Finite-support (windowed) versions of the Regge-Wheeler barrier.

A quantum register holds a finite box, so the potential the circuit actually
simulates is V_w(r*) = w(r*) V_l(r*), with w a smooth taper that is 1 on the
barrier and 0 outside [lo, hi]. The exact transmission of V_w on the infinite
line is the "box-truncated" link of the error budget:

    Gamma_exact  ->  Gamma_box (this module)  ->  Gamma_grid  ->  ...
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .schwarzschild import _check_ls, potential, r_from_tortoise


@dataclass(frozen=True)
class Window:
    """Smooth C^1 taper: 0 outside [lo, hi], cos^2 ramps of width `ramp`, 1 inside."""
    lo: float = -25.0
    hi: float = 30.0
    ramp: float = 8.0

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        up = np.clip((x - self.lo) / self.ramp, 0.0, 1.0)
        down = np.clip((self.hi - x) / self.ramp, 0.0, 1.0)
        return np.sin(0.5 * np.pi * up) ** 2 * np.sin(0.5 * np.pi * down) ** 2


def windowed_potential(rs, l: int, s: int = 0, window: Window = Window()):
    r, _ = r_from_tortoise(rs)
    return window(rs) * potential(r, l, s)


def scatter_windowed(k: float, l: int, s: int = 0, window: Window = Window(),
                     rtol: float = 1e-11) -> float:
    """Exact transmission probability of the windowed barrier at wavenumber k
    (k = M*omega for a massless field). Outside the window psi is a plane wave."""
    _check_ls(l, s)
    _, y0 = r_from_tortoise(window.lo)
    psi0 = np.exp(-1j * k * window.lo)
    state0 = np.array([psi0, -1j * k * psi0, y0], dtype=complex)
    L, c, k2 = l * (l + 1), 2.0 * (1 - s * s), k * k

    def rhs(x, st):
        psi, dpsi, y = st
        yr = y.real
        r = 2.0 + yr
        f = yr / r
        v = window(x) * f * (L / (r * r) + c / (r * r * r))
        return np.array([dpsi, (v - k2) * psi, f])

    sol = solve_ivp(rhs, (window.lo, window.hi), state0, method="DOP853",
                    rtol=rtol, atol=1e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    psi, dpsi, _ = sol.y[:, -1]
    x = window.hi
    # psi = A e^{-ikx} + B e^{ikx}  =>  A = (ik psi - dpsi) e^{ikx} / (2ik)
    a_in = (1j * k * psi - dpsi) * np.exp(1j * k * x) / (2j * k)
    return float(1.0 / abs(a_in) ** 2)
