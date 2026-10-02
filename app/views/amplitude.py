import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import MUTED, SERIES, measured_lambdas, style
from bhtunnel.ae import oracle_calls

st.title("Amplitude estimation vs. sampling")
st.markdown(
    "Deep in the tunnelling tail, Γ is tiny, and plain sampling needs ~1/(p ε²) shots. "
    "Amplitude estimation (maximum-likelihood, no phase estimation) can do it in "
    "~1/ε, *if* the circuit survives being repeated. Each repetition of the "
    "scattering circuit A loses fidelity, modelled as depolarisation with survival λ."
)

c1, c2, c3 = st.columns(3)
p_exp = c1.slider("Probability p (log₁₀)", -5.0, -1.0, -3.0, 0.5,
                  help="Γ_2 at Mω = 0.3 is ≈ 1.5e-3; Γ_1 at Mω = 0.2 is ≈ 3e-2.")
p = 10 ** p_exp
eps = c2.select_slider("Target relative error", [0.05, 0.1, 0.2], value=0.1,
                       format_func=lambda e: f"{e:.0%}")
one_minus = c3.select_slider("Per-oracle error 1 − λ",
                             options=[1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.2, 0.3,
                                      0.5, 0.7], value=0.01)
lam = 1 - one_minus


@st.cache_data(show_spinner=False)
def curve(p, eps):
    xs = np.geomspace(1e-4, 0.8, 50)
    return xs, np.array([np.divide(*oracle_calls(p, eps, 1 - x)[:2]) for x in xs])


samp, ae, m = oracle_calls(p, eps, lam)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Sampling: oracle calls", f"{samp:.2e}")
m2.metric("Amplitude estimation", f"{ae:.2e}", f"deepest Grover power m = {m}",
          delta_color="off")
m3.metric("Advantage", f"{samp / ae:.1f}×", "AE wins" if samp > ae else "sampling wins",
          delta_color="normal" if samp > ae else "inverse",
          help="Sampling calls ÷ AE calls. Mild noise can push this above the ideal "
               "value, because depolarisation also inflates the shots plain sampling "
               "needs (the signal p shrinks against a 1/2 background).")
m4.metric("Ideal device (λ = 1)", f"{np.divide(*oracle_calls(p, eps, 1.0)[:2]):.0f}×")

xs, adv = curve(p, eps)
fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=adv, mode="lines", name=f"p = {p:.0e}",
                         line=dict(color=SERIES[0], width=2)))
fig.add_trace(go.Scatter(x=[one_minus], y=[samp / ae], mode="markers", name="your choice",
                         marker=dict(color=SERIES[1], size=12, line=dict(color="white", width=1.5))))
fig.add_hline(y=1, line=dict(color=MUTED, width=1))
# On a log axis Plotly reads annotation y in log10 units, so y=0 is the line at 1.
fig.add_annotation(xref="paper", x=0, y=0, text="break-even", showarrow=False,
                   xanchor="left", yanchor="bottom", font=dict(color=MUTED, size=11))
for i, (label, lm) in enumerate(measured_lambdas().items()):
    color = SERIES[2 + i]
    fig.add_vline(x=1 - lm, line=dict(color=color, dash="dash", width=1.5))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"{label}, noisy Aer",
                             line=dict(color=color, dash="dash", width=1.5)))
fig.update_xaxes(type="log", dtick=1, exponentformat="power",
                 title_text="per-oracle depolarisation 1 − λ")
fig.update_yaxes(type="log", dtick=1, exponentformat="power",
                 title_text="sampling calls / AE calls")
fig.update_layout(title=f"Advantage at {eps:.0%} relative error")
st.plotly_chart(style(fig, height=430), width="stretch")

st.markdown(
    """
**Reading it.** On an ideal device AE wins by ~1/√p. Noise erodes it, and below a
break-even λ sampling is cheaper. The dashed lines mark the λ measured for the
5-qubit hardware-scale circuit on a noisy Heron model. Both sit on the losing side,
so on today's hardware sampling is the better estimator. That is the result, not a
failure: it gives a concrete fidelity target for when AE starts to pay.

*Model assumptions:* global depolarisation per oracle call, estimators that know λ,
at least 20 shots at the deepest level, and a 2× overhead for the shallower levels
of the MLAE schedule.
"""
)
