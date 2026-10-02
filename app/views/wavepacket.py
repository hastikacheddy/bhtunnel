import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import BOXES, MUTED, SERIES, eigensystem, experiment, gamma_interp, style

st.title("Wavepacket hitting the barrier")
st.markdown(
    "A wave packet starts far from the black hole and moves inward. Part of it "
    "tunnels through the barrier toward the horizon, and the rest reflects. This is "
    "exactly the state the quantum circuit prepares and evolves, computed here by its "
    "classical twin (verified against Aer to a fidelity of 1 − 10⁻¹⁴)."
)

c1, c2, c3, c4 = st.columns(4)
l = c1.selectbox("Multipole l", [0, 1, 2], index=1,
                 help="Angular momentum of the wave; higher l means a taller barrier.")
lo, hi, default = BOXES[l]["k"]
k0 = c2.slider("Mean frequency Mω", lo, hi, default, 0.01)
n = c3.select_slider("Qubits n", options=[6, 7, 8, 9, 10] if l == 0 else [6, 7, 8, 9],
                     value=9 if l == 0 else 8,
                     help="2ⁿ grid points in a fixed box: fewer qubits = coarser grid.")
mode = c4.radio("Time evolution", ["Exact", "Trotter (Strang)"], horizontal=False)
steps = None
if mode != "Exact":
    steps = st.slider("Strang steps", 2, 200, 60,
                      help="Each step = potential phase, QFT†, kinetic phase, QFT.")

exp = experiment(l, n, k0)
x, dx = exp.grid.x, exp.grid.dx
T, frames = exp.time, 48
psi0 = exp.initial_state()


@st.cache_data(show_spinner="Evolving the wave packet…")
def run(l, n, k0, steps, frames):
    e = experiment(l, n, k0)
    p0 = e.initial_state()
    if steps is None:
        w, u = eigensystem(l, n)
        c = u.conj().T @ p0
        ts = np.linspace(0, e.time, frames)
        snaps = u @ (np.exp(-1j * np.outer(w, ts)) * c[:, None])
        return snaps.T, ts
    # Strang, recording snapshots (same operator sequence as evolve_strang)
    dt = e.time / steps
    half_v = np.exp(-0.5j * dt * e.potential)
    kin = np.exp(-1j * dt * e.grid.k ** 2)
    keep = set(np.linspace(0, steps, frames).round().astype(int))
    psi, snaps, ts = p0.copy(), [p0], [0.0]
    for i in range(1, steps + 1):
        psi = half_v * np.fft.ifft(kin * np.fft.fft(half_v * psi))
        if i in keep:
            snaps.append(psi.copy())
            ts.append(i * dt)
    return np.array(snaps), np.array(ts)


snaps, ts = run(l, n, k0, steps, frames)
psi_final = snaps[-1]

p_t = exp.transmitted_probability(psi_final)
p_ref = exp.reference_transmitted(gamma_interp(l))
margin = exp.wrap_margin()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Transmitted P_T (this run)", f"{p_t:.4f}")
m2.metric("Exact reference", f"{p_ref:.4f}", f"Δ = {p_t - p_ref:+.4f}", delta_color="off")
m3.metric("Grid spacing", f"{dx:.2f} M", f"{2 ** n} points", delta_color="off")
m4.metric("Simulated time", f"{T:.0f} M")
if margin <= 0:
    st.warning("The fast edge of the packet wraps around the periodic box before the "
               "run ends; expect artefacts. Lower Mω or raise the box size.")
if n <= 6:
    st.info("Few qubits → coarse grid: the barrier is sampled by only a handful of "
            "points, which shows up as a discretisation error in P_T.")

# --- animation: potential (top) and probability density (bottom), shared x ---
dens = np.abs(snaps) ** 2 / dx
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.28, 0.72],
                    vertical_spacing=0.04)
fig.add_trace(go.Scatter(x=x, y=exp.potential, mode="lines", name=f"V_{l}(r*)",
                         line=dict(color=SERIES[1], width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=dens[0], mode="lines", name="|ψ|²", fill="tozeroy",
                         line=dict(color=SERIES[0], width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=dens[0], mode="lines", name="initial packet",
                         line=dict(color=MUTED, width=1, dash="dot")), row=2, col=1)
# Frames restate all three traces: frames that touch only the density trace make
# Plotly drop the potential from the upper panel during playback.
v_trace, init_trace = fig.data[0], fig.data[2]
fig.frames = [go.Frame(data=[v_trace, go.Scatter(x=x, y=d), init_trace],
                       traces=[0, 1, 2], name=f"{t:.0f}")
              for d, t in zip(dens, ts)]
fig.update_layout(
    updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.12, xanchor="left",
                      buttons=[dict(label="▶ Play", method="animate",
                                    args=[None, dict(frame=dict(duration=60, redraw=False),
                                                     fromcurrent=True, transition=dict(duration=0))]),
                               dict(label="❚❚ Pause", method="animate",
                                    args=[[None], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=False))])])],
    sliders=[dict(active=0, x=0.12, len=0.88, y=-0.08, ticklen=0, minorticklen=0,
                  font=dict(color="rgba(0,0,0,0)"),
                  currentvalue=dict(prefix="t = ", suffix=" M", font=dict(color="#8a8984")),
                  steps=[dict(label=f"{t:.0f}", method="animate",
                              args=[[f"{t:.0f}"], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=False))])
                         for t in ts])],
)
fig.update_yaxes(title_text="V (M⁻²)", row=1, col=1)
fig.update_yaxes(title_text="|ψ|²", range=[0, dens.max() * 1.1], row=2, col=1)
fig.update_xaxes(title_text="tortoise coordinate r* / M   (← horizon · infinity →)", row=2, col=1)
st.plotly_chart(style(fig, height=520), width="stretch")

# --- frequency-resolved transmission from the same run ---
st.subheader("Γ at every frequency in the packet, from one run")
st.markdown(
    "Measuring all qubits after the final QFT† gives the momentum spectrum. Incoming "
    "momenta point toward the horizon, so for each frequency bin, "
    r"$\Gamma(\omega) = P_\mathrm{after}(\mathrm{bin}) \,/\, P_\mathrm{before}(\mathrm{bin})$."
)
k, g, p0 = exp.resolved_gamma(psi_final, rel_floor=0.15)
kk = np.linspace(max(lo - 0.08, 0.02), hi + 0.08, 300)
c1, c2 = st.columns(2)
f2 = go.Figure()
f2.add_trace(go.Scatter(x=kk, y=gamma_interp(l)(kk), mode="lines", name="exact Γ",
                        line=dict(color=SERIES[0], width=2)))
f2.add_trace(go.Scatter(x=k, y=g, mode="markers", name="from this wave packet",
                        marker=dict(color=SERIES[1], size=9, line=dict(color="white", width=1))))
f2.update_xaxes(title_text="Mω")
f2.update_yaxes(title_text=f"Γ_{l}", range=[-0.03, 1.08])
c1.plotly_chart(style(f2), width="stretch")

pk0 = exp.momentum_probs(psi0)
pk1 = exp.momentum_probs(psi_final)
kgrid = exp.grid.k
order = np.argsort(kgrid)
sel = np.abs(kgrid[order]) < hi + 0.25
f3 = go.Figure()
f3.add_trace(go.Bar(x=kgrid[order][sel], y=pk0[order][sel], name="before",
                    marker_color=MUTED, opacity=0.6))
f3.add_trace(go.Bar(x=kgrid[order][sel], y=pk1[order][sel], name="after",
                    marker_color=SERIES[0]))
f3.update_layout(barmode="overlay", bargap=0.05)
f3.update_xaxes(title_text="momentum k  (k < 0: toward horizon = transmitted; k > 0: reflected)")
f3.update_yaxes(title_text="probability")
c2.plotly_chart(style(f3), width="stretch")
