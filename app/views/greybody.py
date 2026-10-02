import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import MUTED, ROOT, SERIES, greybody_table, style
from bhtunnel import T_HAWKING, potential, r_from_tortoise, total_power
from bhtunnel.hawking import blackbody_number_spectrum, bose_factor
from bhtunnel.wkb import greybody_wkb

st.title("Greybody factors and the Hawking spectrum")
st.markdown(
    "Exact classical results (ODE integration with asymptotic matching). They are the "
    "reference every quantum run is scored against. The curves come from a precomputed "
    "table, so this page responds instantly."
)

omega, table = greybody_table()
spin_names = {0: "scalar (s = 0)", 1: "electromagnetic (s = 1)", 2: "gravitational (s = 2)"}
c1, c2, c3 = st.columns([1.2, 2, 1])
s = c1.selectbox("Field", [0, 1, 2], format_func=spin_names.get)
ls = c2.multiselect("Multipoles l", list(range(s, s + 6)), default=list(range(s, s + 4)))
log = c3.toggle("Log scale", value=False)
show_wkb = c3.toggle("WKB overlay", value=True)

col1, col2 = st.columns(2)
fig = go.Figure()
for i, l in enumerate(ls):
    color = SERIES[i % len(SERIES)]
    fig.add_trace(go.Scatter(x=omega, y=table[s, l - s], mode="lines", name=f"l = {l}",
                             line=dict(color=color, width=2)))
    if show_wkb:
        fig.add_trace(go.Scatter(x=omega, y=greybody_wkb(omega, l, s), mode="lines",
                                 name=f"WKB l = {l}", showlegend=False, hoverinfo="skip",
                                 line=dict(color=color, width=1, dash="dash")))
fig.update_xaxes(title_text="Mω")
fig.update_yaxes(title_text="Γ_l", type="log" if log else "linear",
                 range=[-12, 0.2] if log else [-0.03, 1.05])
fig.update_layout(title="Greybody factor Γ_l(ω)" + ("  (dashed: 1st-order WKB)" if show_wkb else ""))
col1.plotly_chart(style(fig), width="stretch")

rs = np.linspace(-15, 30, 600)
r, _ = r_from_tortoise(rs)
fv = go.Figure()
for i, l in enumerate(ls):
    fv.add_trace(go.Scatter(x=rs, y=potential(r, l, s), mode="lines", name=f"l = {l}",
                            line=dict(color=SERIES[i % len(SERIES)], width=2)))
fv.update_xaxes(title_text="tortoise coordinate r* / M   (← horizon · infinity →)")
fv.update_yaxes(title_text="V_l (M⁻²)")
fv.update_layout(title="The barriers")
col2.plotly_chart(style(fv), width="stretch")

st.subheader("Hawking emission")
st.markdown(
    "Per degree of freedom: d²N/dt dω = (1/2π) Σ_l (2l+1) Γ_l / (e^{ω/T_H} − 1), with "
    f"T_H = 1/(8πM) = {T_HAWKING:.4f}/M. The quantum part only ever supplies Γ_l; the "
    "thermal factor is exact."
)
mask = omega <= 0.7
w = omega[mask]
n_gb = sum((2 * l + 1) * table[s, l - s][mask] for l in range(s, s + 6)) * bose_factor(w) / (2 * np.pi)
c1, c2 = st.columns(2)
fh = go.Figure()
fh.add_trace(go.Scatter(x=w, y=1e5 * w * n_gb, mode="lines", name="with greybody factors",
                        line=dict(color=SERIES[0], width=2)))
if s == 0:
    n_bb = blackbody_number_spectrum(w)
    fh.add_trace(go.Scatter(x=w, y=1e5 * w * n_bb, mode="lines", name="blackbody, σ = 27πM²",
                            line=dict(color=SERIES[1], width=2)))
fh.update_xaxes(title_text="Mω")
fh.update_yaxes(title_text="ω d²N/dt dω  (×10⁻⁵ M⁻²)")
fh.update_layout(title=f"Power spectrum, {spin_names[s]}")
c1.plotly_chart(style(fh), width="stretch")

p = total_power(w, n_gb)
c2.metric("Total power (this field, one degree of freedom)", f"{p:.3e} M⁻²")
if s == 0:
    c2.caption("Literature value for a massless scalar: 7.44 × 10⁻⁵ M⁻² (Page 1976 and follow-ups).")
    try:
        cs = np.loadtxt(ROOT / "data/cross_section_scalar.csv", delimiter=",", skiprows=1)
        fc = go.Figure()
        fc.add_trace(go.Scatter(x=cs[:, 0], y=cs[:, 1], mode="lines", name="σ_abs / πM²",
                                line=dict(color=SERIES[0], width=2)))
        for y, t in ((27, "27πM², geometric optics"), (16, "16πM², horizon area")):
            fc.add_hline(y=y, line=dict(color=MUTED, dash="dot", width=1),
                         annotation_text=t, annotation_position="bottom right")
        fc.update_xaxes(title_text="Mω")
        fc.update_yaxes(title_text="σ_abs / πM²")
        fc.update_layout(title="Absorption cross section (scalar)")
        c2.plotly_chart(style(fc, height=300), width="stretch")
    except OSError:
        pass
else:
    c2.caption("For s = 1, 2 multiply by the number of helicities (2) for the physical field.")
