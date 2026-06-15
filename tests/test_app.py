"""Smoke tests for the Streamlit dashboard's import-safe helper functions.

The Streamlit UI in ``main()`` needs a running Streamlit server and is not
exercised here; the pure compute/plot helpers are.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _load_app():
    spec = importlib.util.spec_from_file_location("streamlit_app", APP)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_compute_summary_keys_and_values():
    m = _load_app()
    s = m.compute_summary(100.0, 100.0, 1.0, 0.04, 0.20, 0.0, "call")
    expected = {"call", "put", "delta", "gamma", "vega", "vega_pct",
                "theta", "theta_per_day", "rho", "rho_pct", "vanna", "volga"}
    assert expected.issubset(s)
    assert s["call"] > 0 and s["put"] > 0
    # Put-call parity holds (q = 0).
    residual = s["call"] - s["put"] - (100.0 - 100.0 * np.exp(-0.04))
    assert abs(residual) < 1e-9


def test_curve_and_surface_figures_build():
    m = _load_app()
    curve = m.greek_curve_figure("gamma", 100.0, 105.0, 1.0, 0.04, 0.2, 0.0, "call")
    assert len(curve.data) >= 1
    surface = m.greek_surface_figure("vega", 100.0, 0.04, 0.2, 0.0, "call")
    assert len(surface.data) == 1
    assert np.asarray(surface.data[0].z).ndim == 2


def test_streamlit_runtime_available_and_main_callable():
    pytest.importorskip("streamlit")
    m = _load_app()
    assert callable(m.main)
