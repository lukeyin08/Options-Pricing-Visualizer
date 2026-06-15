#!/usr/bin/env python3
"""Streamlit dashboard: live Black-Scholes prices, Greeks, curves, and surfaces.

Run with::

    streamlit run app/streamlit_app.py

Move the sliders for S, K, T, r, q, sigma and watch the price, the full Greek
panel, the Greek-vs-strike curve, and the 3D Greek surface update live. The
heavy lifting is all in ``options_viz``; this file is only glue plus two small
plot helpers. The compute helpers are import-safe (no Streamlit calls at module
load) so they can be unit tested without a Streamlit runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

# Allow `streamlit run app/streamlit_app.py` without installing the package.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import plotly.graph_objects as go

from options_viz.black_scholes import call_price, implied_vol, put_price
from options_viz.greeks import all_greeks
from options_viz import visualize as viz


# --------------------------------------------------------------------------
# Pure compute / plot helpers (no Streamlit; unit-testable)
# --------------------------------------------------------------------------
def compute_summary(
    S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str
) -> Dict[str, float]:
    """Prices (call and put) plus the full Greek panel for the chosen type."""
    summary: Dict[str, float] = {
        "call": float(call_price(S, K, T, r, sigma, q)),
        "put": float(put_price(S, K, T, r, sigma, q)),
    }
    summary.update({k: float(v) for k, v in all_greeks(S, K, T, r, sigma, q, option_type).items()})
    return summary


def greek_curve_figure(
    name: str, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str
) -> go.Figure:
    """Greek-vs-strike line at the current maturity, with the live strike marked."""
    strikes = np.linspace(0.5 * S, 1.5 * S, 241)
    y = viz.eval_greek(name, S, strikes, T, r, sigma, q, option_type)
    fig = go.Figure()
    fig.add_scatter(x=strikes, y=np.asarray(y), mode="lines", name=name, line=dict(width=3))
    fig.add_vline(x=K, line_dash="dash", line_color="gray", annotation_text=f"K={K:g}")
    fig.update_layout(
        title=f"{viz.GREEK_LABELS.get(name, name)} vs strike (T={T:g}y, {option_type})",
        xaxis_title="Strike K",
        yaxis_title=viz.GREEK_LABELS.get(name, name),
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def greek_surface_figure(
    name: str, S: float, r: float, sigma: float, q: float, option_type: str
) -> go.Figure:
    """Interactive 3D Greek surface over (strike, time) at the current market."""
    base = viz.BaseCase(S=S, r=r, q=q, sigma=sigma)
    return viz.surface_plotly(name, base=base, option_type=option_type)


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def main() -> None:  # pragma: no cover - requires a Streamlit runtime
    import streamlit as st

    st.set_page_config(page_title="Options Pricing + Greeks", layout="wide")
    st.title("Options Pricing and Greeks Visualizer")
    st.caption("From-scratch Black-Scholes (no scipy in the core). Move the sliders.")

    with st.sidebar:
        st.header("Parameters")
        S = st.slider("Spot S", 1.0, 300.0, 100.0, 1.0)
        K = st.slider("Strike K", 1.0, 300.0, 100.0, 1.0)
        T = st.slider("Time to expiry T (years)", 0.01, 3.0, 1.0, 0.01)
        sigma = st.slider("Volatility sigma", 0.01, 1.50, 0.20, 0.01)
        r = st.slider("Risk-free rate r", -0.02, 0.15, 0.04, 0.005)
        q = st.slider("Dividend yield q", 0.0, 0.10, 0.0, 0.005)
        option_type = st.radio("Option type", ["call", "put"], horizontal=True)
        greek_name = st.selectbox("Greek to plot", ["delta", "gamma", "vega"])

    s = compute_summary(S, K, T, r, sigma, q, option_type)

    st.subheader("Price")
    c1, c2, c3 = st.columns(3)
    c1.metric("Call", f"{s['call']:.4f}")
    c2.metric("Put", f"{s['put']:.4f}")
    parity = s["call"] - s["put"] - (S * np.exp(-q * T) - K * np.exp(-r * T))
    c3.metric("Put-call parity residual", f"{parity:.2e}")

    st.subheader(f"Greeks ({option_type})")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta", f"{s['delta']:.4f}")
    g2.metric("Gamma", f"{s['gamma']:.5f}")
    g3.metric("Vega (per 1%)", f"{s['vega_pct']:.4f}")
    g4.metric("Theta (per day)", f"{s['theta_per_day']:.4f}")
    g5, g6, g7, g8 = st.columns(4)
    g5.metric("Rho (per 1%)", f"{s['rho_pct']:.4f}")
    g6.metric("Vanna", f"{s['vanna']:.4f}")
    g7.metric("Volga", f"{s['volga']:.4f}")
    g8.metric("Vega (per 1.00)", f"{s['vega']:.4f}")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(greek_curve_figure(greek_name, S, K, T, r, sigma, q, option_type),
                        use_container_width=True)
    with right:
        st.plotly_chart(greek_surface_figure(greek_name, S, r, sigma, q, option_type),
                        use_container_width=True)

    with st.expander("Implied volatility solver"):
        price_in = st.number_input("Observed option price", value=float(s[option_type]), step=0.10)
        try:
            iv = implied_vol(price_in, S, K, T, r, q, option_type)
            st.write(f"Implied volatility: **{iv:.4%}**")
        except ValueError as exc:
            st.write(f"No implied vol: {exc}")


if __name__ == "__main__":
    main()
