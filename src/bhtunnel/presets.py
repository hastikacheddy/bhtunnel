"""Named experiment configurations shared by scripts, tests and hardware runs.

AER_*: wrap-safe production boxes for the physics (n = 8); Trotter-converged
       runs reproduce Gamma_exact (see scripts/error_budget.py).

HW5, HW6: hardware-scale boxes. A *faithful* Trotterization of these needs
       dt small enough that the packet moves less than a barrier width per step,
       i.e. 8-30 steps and thousands of CZ, which is out of reach of current
       devices. Hardware runs therefore use 1-2 steps and are scored against the
       classical twin *at the same step count*: they benchmark how well the device
       executes this exact circuit, not the physics. k0 is chosen so the expected
       P_T is far from 1/2, the value a fully depolarised device returns.
"""

from __future__ import annotations

from .grid import Experiment, Grid
from .window import Window

AER_L1 = Experiment(grid=Grid(8, -60.0, 196.0), l=1, sigma=14.0, x0=110.0, k0=0.30)
AER_L2 = Experiment(grid=Grid(8, -60.0, 196.0), l=2, sigma=14.0, x0=110.0, k0=0.49)

HW5 = Experiment(grid=Grid(5, -20.0, 44.0), l=1, window=Window(-16.0, 18.0, 5.0),
                 sigma=4.0, x0=32.0, k0=0.24)
HW6 = Experiment(grid=Grid(6, -30.0, 98.0), l=1, window=Window(-25.0, 30.0, 8.0),
                 sigma=8.0, x0=64.0, k0=0.24)

# Step counts used on hardware (see module docstring): P_T(twin) ~ 0.76-0.81.
HW_STEPS = {"HW5": 2, "HW6": 2}
