"""Headless smoke tests of every app page across its widget settings."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))


def page(name: str) -> AppTest:
    at = AppTest.from_file(str(APP / "views" / f"{name}.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def test_home():
    page("home")


@pytest.mark.parametrize("l", [0, 1, 2])
def test_wavepacket_each_multipole(l):
    at = page("wavepacket")
    at.selectbox[0].set_value(l).run()
    assert not at.exception, at.exception
    by_label = {m.label: m.value for m in at.metric}
    p_run = float(by_label["Transmitted P_T (this run)"])
    p_ref = float(by_label["Exact reference"])
    assert abs(p_run - p_ref) < 0.02


def test_wavepacket_trotter_and_few_qubits():
    at = page("wavepacket")
    at.radio[0].set_value("Trotter (Strang)").run()
    at.select_slider[0].set_value(6).run()
    assert not at.exception, at.exception
    assert any("coarse grid" in i.value for i in at.info)


@pytest.mark.parametrize("s", [0, 1, 2])
def test_greybody_each_spin(s):
    at = page("greybody")
    at.selectbox[0].set_value(s).run()
    at.toggle[0].set_value(True).run()
    assert not at.exception, at.exception


@pytest.mark.slow
def test_circuit_page():
    at = page("circuit")
    by_label = {m.label: m.value for m in at.metric}
    p_circ = float(by_label["P(top qubit = 1), Aer circuit"])
    p_twin = float(by_label["Classical twin, same steps"])
    assert abs(p_circ - p_twin) < 1e-6


def test_amplitude_page():
    at = page("amplitude")
    at.select_slider[1].set_value(0.5).run()
    assert not at.exception, at.exception
    assert "sampling wins" in {m.label: m.delta for m in at.metric}["Advantage"]
