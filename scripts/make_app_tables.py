"""Precompute the greybody tables the Streamlit app reads.

    python scripts/make_app_tables.py

Writes src/bhtunnel/data/greybody_table.npz with Gamma_l(M omega) for
s = 0, 1, 2 and l = s .. s+5 on a 300-point grid, so the app never runs the
ODE solver live. Takes a few minutes on a multi-core machine.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bhtunnel import greybody  # noqa: E402

OMEGAS = np.round(np.linspace(0.005, 1.5, 300), 6)
SPINS = (0, 1, 2)
N_L = 6


def _row(job):
    s, l = job
    return s, l, greybody(OMEGAS, l, s)


def main():
    jobs = [(s, l) for s in SPINS for l in range(s, s + N_L)]
    table = np.full((len(SPINS), N_L, len(OMEGAS)), np.nan)
    with ProcessPoolExecutor() as pool:
        for s, l, g in pool.map(_row, jobs):
            table[s, l - s] = g
    out = ROOT / "src/bhtunnel/data/greybody_table.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, omega=OMEGAS, gamma=table, spins=np.array(SPINS),
                        l_offset_is_s=True)
    print(f"wrote {out} {table.shape}")


if __name__ == "__main__":
    main()
