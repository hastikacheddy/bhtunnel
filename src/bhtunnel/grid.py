"""Classical twin of the quantum simulation.

Everything here acts on the same 2^n-point periodic grid, with the same
operators, as the Qiskit circuits in `circuits.py`. Its job is to split the
quantum error budget into separately measurable links:

    Gamma_box   exact transmission of the windowed potential   (window.py)
    Gamma_grid  wavepacket on the grid, exact time evolution   (evolve_exact)
    Gamma_trot  same with Strang split-operator steps           (evolve_strang)
    Gamma_walsh same with a truncated Walsh expansion of V      (truncate_walsh)

Conventions (shared with circuits.py):
* basis index j = sum_q b_q 2^q (Qiskit little-endian), x_j = a + j dx;
* H = -d^2/dx^2 + V, eigenvalue k^2 = (M omega)^2 for a massless field;
* momentum amplitudes are fft(psi)/sqrt(N), which is exactly what the inverse
  QFT (QFT^dagger) produces, and index m has momentum 2 pi m'/L with m' the
  two's-complement value of m. The top qubit is therefore the sign of k:
  P(top qubit = 1) is the probability of moving toward the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .window import Window, windowed_potential


@dataclass(frozen=True)
class Grid:
    n: int
    a: float
    b: float

    @property
    def N(self) -> int:
        return 2**self.n

    @property
    def L(self) -> float:
        return self.b - self.a

    @property
    def dx(self) -> float:
        return self.L / self.N

    @property
    def x(self) -> np.ndarray:
        return self.a + self.dx * np.arange(self.N)

    @property
    def m_signed(self) -> np.ndarray:
        m = np.arange(self.N)
        return np.where(m < self.N // 2, m, m - self.N)

    @property
    def k(self) -> np.ndarray:
        return 2 * np.pi * self.m_signed / self.L


# ---------------------------------------------------------------- Walsh tools

def fwht(v: np.ndarray) -> np.ndarray:
    """Walsh coefficients c_S with v_j = sum_S c_S (-1)^{popcount(j & S)}, i.e.
    v = sum_S c_S Z_S where Z_S is the product of Z on the qubits in bitmask S."""
    c = np.array(v, dtype=float)
    h = 1
    while h < len(c):
        c = c.reshape(-1, 2, h)
        c = np.stack([c[:, 0] + c[:, 1], c[:, 0] - c[:, 1]], axis=1).reshape(-1)
        h *= 2
    return c / len(c)


def ifwht(c: np.ndarray) -> np.ndarray:
    return fwht(c) * len(c)


def truncate_walsh(v: np.ndarray, n_terms: int | None) -> np.ndarray:
    """Keep the identity plus the n_terms largest non-identity Walsh terms.
    The identity is a global phase and costs no gates."""
    if n_terms is None:
        return np.array(v, dtype=float)
    c = fwht(v)
    order = np.argsort(-np.abs(c[1:])) + 1
    kept = np.zeros_like(c)
    kept[0] = c[0]
    kept[order[:n_terms]] = c[order[:n_terms]]
    return ifwht(kept)


def walsh_terms(v: np.ndarray, tol: float = 1e-12) -> dict[int, float]:
    """Non-identity Walsh terms {bitmask S: c_S} with |c_S| > tol."""
    c = fwht(v)
    return {S: float(c[S]) for S in range(1, len(c)) if abs(c[S]) > tol}


# ---------------------------------------------------------------- experiment

@dataclass(frozen=True)
class Experiment:
    """A wavepacket scattering run, shared by the twin and the circuits.

    The packet starts at x0 in the potential-free zone right of the window,
    moving toward the horizon with mean wavenumber k0 and spatial width sigma.
    """
    grid: Grid = Grid(8, -60.0, 196.0)
    l: int = 1
    s: int = 0
    window: Window = Window()
    k0: float = 0.3
    sigma: float = 14.0
    x0: float = 110.0
    t: float | None = None           # default: see default_time()
    walsh_terms: int | None = None   # truncate V to this many Walsh terms
    # Not an init field, so dataclasses.replace() gives every copy a fresh cache
    # instead of sharing (and silently reusing) the original's potential.
    _cache: dict = field(default_factory=dict, init=False, compare=False, repr=False)

    @property
    def sigma_k(self) -> float:
        return 1.0 / (2.0 * self.sigma)

    @property
    def potential(self) -> np.ndarray:
        if "V" not in self._cache:
            v = windowed_potential(self.grid.x, self.l, self.s, self.window)
            self._cache["V"] = truncate_walsh(v, self.walsh_terms)
        return self._cache["V"]

    def default_time(self, x_clear: float = -8.0) -> float:
        """Long enough for the slow edge (k0 - 2 sigma_k) of the packet to cross
        the barrier and reach x_clear, at group velocity 2k."""
        k_slow = max(self.k0 - 2 * self.sigma_k, 0.25 * self.k0)
        return (self.x0 - x_clear + 2 * self.sigma) / (2 * k_slow)

    def wrap_margin(self, x_barrier: float = 1.5) -> float:
        """> 0 when the run is free of periodic wrap-around artefacts: the fast
        edge (k0 + 2 sigma_k) of the transmitted wave must not travel through the
        box boundary and back into the window before the run ends."""
        k_fast = self.k0 + 2 * self.sigma_k
        d_wrap = ((x_barrier - self.grid.a) + (self.grid.b - self.window.hi)
                  + (self.x0 - x_barrier))
        return 1.0 - 2 * k_fast * self.time / d_wrap

    @property
    def time(self) -> float:
        return self.default_time() if self.t is None else self.t

    def initial_state(self) -> np.ndarray:
        x = self.grid.x
        psi = np.exp(-((x - self.x0) ** 2) / (4 * self.sigma**2) - 1j * self.k0 * x)
        return psi / np.linalg.norm(psi)

    # ------------------------------------------------------------ evolution

    def hamiltonian(self) -> np.ndarray:
        N = self.grid.N
        f = np.fft.fft(np.eye(N), axis=0) / np.sqrt(N)
        kin = f.conj().T @ np.diag(self.grid.k**2) @ f
        return kin + np.diag(self.potential)

    def evolve_exact(self, psi: np.ndarray | None = None) -> np.ndarray:
        if "eig" not in self._cache:
            self._cache["eig"] = np.linalg.eigh(self.hamiltonian())
        w, u = self._cache["eig"]
        psi = self.initial_state() if psi is None else psi
        return u @ (np.exp(-1j * w * self.time) * (u.conj().T @ psi))

    def evolve_strang(self, steps: int, psi: np.ndarray | None = None) -> np.ndarray:
        """e^{-iV dt/2} [K(dt) e^{-iV dt}]^{steps-1} K(dt) e^{-iV dt/2},
        exactly the gate sequence built by circuits.evolution_circuit."""
        dt = self.time / steps
        psi = self.initial_state() if psi is None else psi.copy()
        half_v = np.exp(-0.5j * dt * self.potential)
        full_v = half_v * half_v
        kin = np.exp(-1j * dt * self.grid.k**2)
        psi = half_v * psi
        for i in range(steps):
            psi = np.fft.ifft(kin * np.fft.fft(psi))
            psi = (half_v if i == steps - 1 else full_v) * psi
        return psi

    # ------------------------------------------------------------ observables

    def momentum_probs(self, psi: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.fft(psi)) ** 2 / self.grid.N

    def transmitted_probability(self, psi: np.ndarray) -> float:
        """P(k < 0): the top-qubit-equals-1 probability after QFT^dagger."""
        return float(self.momentum_probs(psi)[self.grid.m_signed < 0].sum())

    def resolved_gamma(self, psi_final: np.ndarray, rel_floor: float = 1e-2):
        """Momentum-resolved transmission: Gamma(|k_m|) = P_final(m)/P_init(m)
        for incident bins (k < 0) carrying at least rel_floor of the peak."""
        p0 = self.momentum_probs(self.initial_state())
        p1 = self.momentum_probs(psi_final)
        sel = (self.grid.m_signed < 0) & (p0 > rel_floor * p0.max())
        return -self.grid.k[sel], p1[sel] / p0[sel], p0[sel]

    def reference_transmitted(self, gamma_of_k) -> float:
        """P(k<0) predicted by a transmission curve, using the exact initial
        momentum distribution on this grid. Positive-k leakage of the initial
        packet (which is reflected-like by construction) is excluded."""
        p0 = self.momentum_probs(self.initial_state())
        sel = self.grid.m_signed < 0
        return float(np.sum(p0[sel] * gamma_of_k(-self.grid.k[sel])))
