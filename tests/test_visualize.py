"""Tests for the visualizer.

Two layers:
  1. Numerical assertions that the *behaviors the figures are meant to show*
     actually hold (gamma's ATM spike, vega's ATM peak and growth with T, the
     delta S-curve steepening). This guarantees the figures are illuminating.
  2. A smoke test that every figure function runs headless and writes a file.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless before pyplot is imported anywhere

import numpy as np
import pytest

from options_viz import greeks
from options_viz import visualize as viz

S, R, Q, SIG = 100.0, 0.04, 0.0, 0.20


# ----------------------------- behaviors -----------------------------------
def test_gamma_peaks_at_the_money():
    T = 0.25
    atm = greeks.gamma(S, 100.0, T, R, SIG, Q)
    assert atm > greeks.gamma(S, 80.0, T, R, SIG, Q)
    assert atm > greeks.gamma(S, 120.0, T, R, SIG, Q)


def test_gamma_spikes_as_expiry_approaches():
    # ATM gamma blows up as T -> 0 (the price kink at K sharpens).
    g_short = greeks.gamma(S, 100.0, 0.05, R, SIG, Q)
    g_long = greeks.gamma(S, 100.0, 1.0, R, SIG, Q)
    assert g_short > 3.0 * g_long


def test_otm_gamma_vanishes_at_expiry():
    # Away from the money, gamma -> 0 as T -> 0 (opposite of the ATM spike).
    assert greeks.gamma(S, 120.0, 0.02, R, SIG, Q) < greeks.gamma(S, 120.0, 0.5, R, SIG, Q)


def test_vega_peaks_at_the_money_and_grows_with_maturity():
    T = 0.5
    atm = greeks.vega(S, 100.0, T, R, SIG, Q)
    assert atm > greeks.vega(S, 80.0, T, R, SIG, Q)
    assert atm > greeks.vega(S, 120.0, T, R, SIG, Q)
    # Grows with maturity at the money.
    assert greeks.vega(S, 100.0, 1.0, R, SIG, Q) > greeks.vega(S, 100.0, 0.1, R, SIG, Q)


def test_delta_is_monotone_scurve_in_strike():
    K = np.linspace(60, 140, 81)
    d = np.asarray(greeks.delta(S, K, 0.5, R, SIG, Q, "call"))
    # Call delta decreases with strike, bounded in [0, e^{-qT}] = [0, 1].
    assert np.all(np.diff(d) < 0)
    assert d[0] > 0.98 and d[-1] < 0.05
    assert np.all((d >= 0) & (d <= 1.0 + 1e-12))


def test_delta_steepens_near_expiry():
    # |dDelta/dS| at the money (= gamma) is larger for shorter maturities.
    steep_short = greeks.gamma(S, 100.0, 0.05, R, SIG, Q)
    steep_long = greeks.gamma(S, 100.0, 1.0, R, SIG, Q)
    assert steep_short > steep_long


# ----------------------------- smoke test ----------------------------------
@pytest.mark.parametrize("greek", ["delta", "gamma", "vega"])
def test_static_figures_render_and_save(tmp_path, greek):
    import matplotlib.pyplot as plt

    for fn in (viz.plot_greek_vs_strike, viz.plot_greek_vs_time, viz.surface_matplotlib, viz.heatmap):
        fig = fn(greek)
        out = tmp_path / f"{greek}_{fn.__name__}.png"
        fig.savefig(out)
        plt.close(fig)
        assert out.exists() and out.stat().st_size > 1000


@pytest.mark.parametrize("greek", ["delta", "gamma", "vega"])
def test_plotly_surface_builds(greek):
    fig = viz.surface_plotly(greek)
    # One Surface trace with a 2D z grid.
    assert len(fig.data) == 1
    assert np.asarray(fig.data[0].z).ndim == 2
