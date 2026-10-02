"""Shared helpers for the Streamlit app: palette, plot styling, cached physics."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from importlib import resources
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running from a plain checkout
    sys.path.insert(0, str(ROOT / "src"))

from bhtunnel.grid import Experiment, Grid  # noqa: E402

REPO_URL = "https://github.com/hastikacheddy/bhtunnel"

# Validated categorical slots (see the dataviz palette) + recessive ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
MUTED = "#8a8984"

# One wrap-safe box per multipole (same as scripts/error_budget.py); the qubit
# count only changes the grid spacing inside the box.
BOXES = {
    0: dict(a=-60.0, b=452.0, sigma=28.0, x0=200.0, k=(0.08, 0.26, 0.15)),
    1: dict(a=-60.0, b=196.0, sigma=14.0, x0=110.0, k=(0.20, 0.42, 0.30)),
    2: dict(a=-60.0, b=196.0, sigma=14.0, x0=110.0, k=(0.38, 0.62, 0.49)),
}


def style(fig, height: int = 380, **kw):
    has_title = bool(fig.layout.title.text) or "title" in kw
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=78 if has_title else 40, b=10),
                      title=dict(y=0.98, yanchor="top", x=0, xanchor="left",
                                 font=dict(size=15)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="left", x=0), **kw)
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    return fig


def experiment(l: int, n: int, k0: float) -> Experiment:
    b = BOXES[l]
    return Experiment(grid=Grid(n, b["a"], b["b"]), l=l, sigma=b["sigma"],
                      x0=b["x0"], k0=k0)


@st.cache_resource(show_spinner=False)
def eigensystem(l: int, n: int):
    """Eigen-decomposition of the grid Hamiltonian (independent of k0)."""
    exp = experiment(l, n, BOXES[l]["k"][2])
    return np.linalg.eigh(exp.hamiltonian())


@st.cache_data(show_spinner=False)
def greybody_table():
    path = resources.files("bhtunnel").joinpath("data/greybody_table.npz")
    with resources.as_file(path) as p:
        d = np.load(p)
        return d["omega"], d["gamma"]


def gamma_interp(l: int, s: int = 0):
    """Gamma_l(k) interpolated in log space from the precomputed exact table."""
    omega, table = greybody_table()
    g = np.log(np.clip(table[s, l - s], 1e-300, None))
    return lambda k: np.exp(np.interp(k, omega, g))


@st.cache_data(show_spinner=False)
def measured_lambdas() -> dict:
    try:
        runs = json.loads((ROOT / "data/noise.json").read_text())["runs"]
        return {f"HW5, {r['steps']} step{'s' if r['steps'] > 1 else ''}": r["lam"]
                for r in runs}
    except (OSError, KeyError, ValueError):
        return {"HW5, 1 step": 0.48, "HW5, 2 steps": 0.28}


__all__ = ["BOXES", "MUTED", "REPO_URL", "ROOT", "SERIES", "eigensystem",
           "experiment", "gamma_interp", "greybody_table", "measured_lambdas",
           "replace", "style"]
