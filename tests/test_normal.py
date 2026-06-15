"""Tests for the from-scratch standard normal (PDF, CDF, erf).

scipy is used here only as the trusted reference. It is never imported by the
implementation under test.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import special, stats

from options_viz.normal import erf, norm_cdf, norm_pdf

GRID = np.linspace(-6.0, 6.0, 2401)  # step 0.005


def test_pdf_matches_scipy():
    ours = np.asarray(norm_pdf(GRID))
    ref = stats.norm.pdf(GRID)
    # The PDF is a closed form, so agreement is to floating-point precision.
    assert np.max(np.abs(ours - ref)) < 1e-12


def test_cdf_matches_scipy_to_1e7():
    ours = np.asarray(norm_cdf(GRID))
    ref = stats.norm.cdf(GRID)
    max_err = float(np.max(np.abs(ours - ref)))
    # A&S 7.1.26 guarantees ~1.5e-7 on erf; the 1/2 in Phi halves it.
    assert max_err < 1e-7, f"max abs error {max_err:.2e} exceeds 1e-7"


def test_erf_matches_scipy():
    ours = np.asarray(erf(GRID))
    ref = special.erf(GRID)
    assert np.max(np.abs(ours - ref)) < 1.5e-7


def test_cdf_symmetry():
    # Phi(x) + Phi(-x) = 1 exactly because our erf is odd by construction.
    s = np.asarray(norm_cdf(GRID)) + np.asarray(norm_cdf(-GRID))
    assert np.max(np.abs(s - 1.0)) < 1e-12


def test_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert norm_cdf(1.959963985) == pytest.approx(0.975, abs=1e-6)
    assert norm_cdf(-1.959963985) == pytest.approx(0.025, abs=1e-6)


def test_pdf_known_value():
    assert norm_pdf(0.0) == pytest.approx(1.0 / np.sqrt(2 * np.pi), abs=1e-15)


def test_cdf_monotonic_and_bounded():
    c = np.asarray(norm_cdf(GRID))
    assert np.all(np.diff(c) >= 0.0)
    assert np.all(c >= 0.0) and np.all(c <= 1.0)


def test_scalar_returns_float_array_returns_array():
    assert isinstance(norm_cdf(0.3), float)
    out = norm_cdf(np.array([0.0, 1.0, 2.0]))
    assert isinstance(out, np.ndarray) and out.shape == (3,)
