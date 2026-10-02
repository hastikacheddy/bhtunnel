"""Exact (numerical) greybody factors for Schwarzschild.

Scattering setup, normalised to unit transmitted amplitude:

    psi ~ exp(-i w r*)                               r* -> -inf  (into horizon)
    psi ~ A_in exp(-i w r*) + A_out exp(+i w r*)     r* -> +inf

Then Gamma_l(w) = 1/|A_in|^2 and the reflection probability is |A_out/A_in|^2;
flux conservation demands Gamma + R = 1, which is reported as a self-check.

The ODE is integrated in r* from deep in the near-horizon region, where V is
exponentially small, out to a large radius r_match. There it is matched onto
the asymptotic series

    psi_eps = exp(i eps w r*) sum_k a_k r^-k,   eps = +/-1,
    2 i eps w (k+1) a_{k+1} = [k(k+1) - l(l+1)] a_k - 2 (k^2 - s^2) a_{k-1},

which captures the slowly decaying l(l+1)/r^2 tail exactly instead of
pretending the wave is already a plane wave at finite r.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .schwarzschild import _check_ls, r_from_tortoise, tortoise

# Start of the integration in r*. There r - 2 ~ 2e-13 and V ~ 1e-14, so the
# pure ingoing plane wave is exact to well below the integrator tolerance.
RS_START = -60.0


@dataclass(frozen=True)
class Scattering:
    omega: float
    l: int
    s: int
    gamma: float          # transmission (greybody) probability
    reflection: float     # reflection probability
    a_in: complex
    a_out: complex
    r_match: float

    @property
    def flux_error(self) -> float:
        return abs(1.0 - self.gamma - self.reflection)


def _asymptotic_series(r: float, omega: float, l: int, s: int, eps: int,
                       max_terms: int = 200):
    """u(r) and u'(r) for psi_eps = exp(i eps omega r*) u(r).

    The series is asymptotic, so it is summed only while the terms decrease.
    Individual coefficients can vanish exactly for k <= l+1 (e.g. a_1 = 0 for
    l = 0, a_{l+1} = 0 for l = s) without the series terminating, so the
    divergence test is only applied beyond that point.
    """
    L = l * (l + 1)
    a_prev, a = 0.0 + 0.0j, 1.0 + 0.0j
    u, du = a, 0.0 + 0.0j
    last = np.inf
    for k in range(max_terms):
        a_next = ((k * (k + 1) - L) * a - 2.0 * (k * k - s * s) * a_prev) / (
            2j * eps * omega * (k + 1))
        a_prev, a = a, a_next
        n = k + 1
        term = abs(a_next) * r ** (-n)
        if k > l + 1:
            if term >= last:
                break
            last = term
        u += a_next * r ** (-n)
        du += -n * a_next * r ** (-n - 1)
        if k > l + 1 and term < 1e-17 * abs(u):
            break
    return u, du


def default_r_match(omega: float, l: int) -> float:
    """Large enough that omega*r >> 1 and l(l+1)/(omega r) is small, so the
    asymptotic series converges to machine precision before it diverges."""
    return max(60.0, 30.0 / omega, 4.0 * (l * (l + 1) + 1) / omega)


def scatter(omega: float, l: int, s: int = 0, r_match: float | None = None,
            rtol: float = 1e-11) -> Scattering:
    """Solve the scattering problem at one frequency M*omega."""
    _check_ls(l, s)
    if omega <= 0:
        raise ValueError("omega must be positive")
    if r_match is None:
        r_match = default_r_match(omega, l)

    _, y0 = r_from_tortoise(RS_START)
    psi0 = np.exp(-1j * omega * RS_START)
    # State: psi, dpsi/dr*, and y = r - 2 (carried as a state so r stays accurate
    # near the horizon, where r - 2 is far below float resolution of r).
    state0 = np.array([psi0, -1j * omega * psi0, y0], dtype=complex)

    L, c, w2 = l * (l + 1), 2.0 * (1 - s * s), omega**2

    def rhs(_rs, st):
        # potential() inlined: this is called ~10^4 times per frequency.
        psi, dpsi, y = st
        yr = y.real
        r = 2.0 + yr
        f = yr / r
        v = f * (L / (r * r) + c / (r * r * r))
        return np.array([dpsi, (v - w2) * psi, f])

    rs_end = float(tortoise(r_match))
    sol = solve_ivp(rhs, (RS_START, rs_end), state0, method="DOP853",
                    rtol=rtol, atol=1e-13)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    psi, dpsi, y = sol.y[:, -1]
    r = 2.0 + y.real
    rs = rs_end
    f = 1.0 - 2.0 / r

    basis = []
    for eps in (-1, +1):
        u, du = _asymptotic_series(r, omega, l, s, eps)
        phase = np.exp(1j * eps * omega * rs)
        basis.append((phase * u, phase * (1j * eps * omega * u + f * du)))
    (pin, dpin), (pout, dpout) = basis
    a_in, a_out = np.linalg.solve(np.array([[pin, pout], [dpin, dpout]]),
                                  np.array([psi, dpsi]))
    return Scattering(omega=omega, l=l, s=s,
                      gamma=float(1.0 / abs(a_in) ** 2),
                      reflection=float(abs(a_out / a_in) ** 2),
                      a_in=complex(a_in), a_out=complex(a_out),
                      r_match=r_match)


def greybody(omegas, l: int, s: int = 0) -> np.ndarray:
    """Gamma_l on an array of frequencies M*omega."""
    return np.array([scatter(float(w), l, s).gamma for w in np.atleast_1d(omegas)])


def absorption_cross_section(omegas, s: int = 0, l_max: int | None = None):
    """sigma_abs(omega) / (pi M^2) = (1/omega^2) sum_l (2l+1) Gamma_l.

    l_max defaults to well past the geometric-optics cutoff l ~ 3 sqrt(3) M omega.
    """
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    if l_max is None:
        l_max = int(np.ceil(3 * np.sqrt(3) * omegas.max())) + 8
    total = np.zeros_like(omegas)
    for l in range(s, l_max + 1):
        total += (2 * l + 1) * greybody(omegas, l, s)
    return total / omegas**2
