"""Quasi-bound states of a massive scalar: where a ground-state method fits.

A scalar of mass mu sees V = f [l(l+1)/r^2 + 2/r^3 + mu^2], which tends to mu^2
at infinity and carries a -2 M mu^2 / r tail: a gravitational "atom" with
hydrogen-like levels behind the centrifugal barrier,

    omega_n ~ mu [1 - (M mu)^2 / (2 n^2)],     n = l + 1 + n_r.

Strictly these are resonances: they tunnel through the barrier into the
horizon, with Im(omega) suppressed by a high power of M mu (and for l = 0 at
M mu = 0.3 there is no barrier at all). Placing a Dirichlet wall at the barrier
top removes the horizon channel and leaves a Hermitian problem whose ground
state approximates Re(omega). Unlike the massless scattering problem, this is a
legitimate target for VQE.

Units: M = 1. Coordinate: tortoise r*. Energies: E = omega^2 - mu^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .schwarzschild import potential, r_from_tortoise


def hydrogenic_omega(l: int, mu: float, n_r: int = 0) -> float:
    n = l + 1 + n_r
    return mu * (1 - mu**2 / (2 * n * n))


def barrier_top(l: int, mu: float) -> float:
    """r* of the inner barrier maximum (l >= 1)."""
    res = minimize_scalar(lambda x: -potential(r_from_tortoise(x)[0], l, mu=mu),
                          bounds=(-5.0, 15.0), method="bounded",
                          options={"xatol": 1e-10})
    return float(res.x)


@dataclass(frozen=True)
class BoundBox:
    """Dirichlet box [x_lo, x_hi] in r* with N interior points."""
    l: int = 1
    mu: float = 0.3
    x_lo: float | None = None     # default: the barrier top
    x_hi: float = 260.0
    N: int = 32

    @property
    def lo(self) -> float:
        return barrier_top(self.l, self.mu) if self.x_lo is None else self.x_lo

    @property
    def h(self) -> float:
        return (self.x_hi - self.lo) / (self.N + 1)

    @property
    def x(self) -> np.ndarray:
        return self.lo + self.h * np.arange(1, self.N + 1)

    def hamiltonian(self) -> np.ndarray:
        """-d^2/dx^2 + V - mu^2, second-order finite differences."""
        N, h = self.N, self.h
        r, _ = r_from_tortoise(self.x)
        diag = 2.0 / h**2 + potential(r, self.l, mu=self.mu) - self.mu**2
        return (np.diag(diag) - np.diag(np.full(N - 1, 1.0 / h**2), 1)
                - np.diag(np.full(N - 1, 1.0 / h**2), -1))

    def spectrum(self, k: int = 3):
        e, v = np.linalg.eigh(self.hamiltonian())
        return e[:k], v[:, :k]

    def omega(self, energy: float) -> float:
        return float(np.sqrt(self.mu**2 + energy))
