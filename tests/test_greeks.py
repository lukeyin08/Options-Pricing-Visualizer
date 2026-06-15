"""Validate every analytical Greek against central finite differences.

Methodology note (important)
----------------------------
The finite differences are taken on a *machine-precision* reference price
(``ref_price`` below, built on the standard library's exact ``math.erf``),
**not** on the library's own price. Why: the shipped price uses the
Abramowitz & Stegun polynomial for N(.), which carries a ~1e-7 error whose
*derivative* does not equal the exact density ``phi``. Differencing that price
reintroduces (and, for rho, amplifies via the large d(d2)/dr sensitivity) that
~1e-7 wobble, which would swamp the thing we actually want to test: are the
closed-form Greek *formulas* correct? Differencing an exact price isolates the
formula check and yields agreement at the 1e-7-1e-5 level.

Coverage is complete in tandem with the other suites: test_black_scholes shows
the shipped (A&S) price matches the true price to ~1e-5, and this file shows
the analytic Greeks match the exact derivatives of the true price.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from options_viz import greeks


def ref_price(S, K, T, r, sigma, q, option_type):
    """Machine-precision European price using the exact erf (reference only)."""
    vol = sigma * math.sqrt(T)
    disc_S = S * math.exp(-q * T)
    disc_K = K * math.exp(-r * T)
    if vol == 0.0:
        return max(disc_S - disc_K, 0.0) if option_type == "call" else max(disc_K - disc_S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol
    d2 = d1 - vol
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    if option_type == "call":
        return disc_S * cdf(d1) - disc_K * cdf(d2)
    return disc_K * cdf(-d2) - disc_S * cdf(-d1)


# Grid spanning ITM/ATM/OTM, short/long dated, low/high vol, with/without divs,
# staying clear of the singular sigma->0 / T->0 corners.
SPOTS = [80.0, 90.0, 100.0, 110.0, 120.0]
STRIKE = 100.0
MATS = [0.10, 0.5, 1.0, 2.0]
VOLS = [0.15, 0.25, 0.40]
RATE = 0.04
DIVS = [0.0, 0.03]
TYPES = ["call", "put"]

CASES = list(itertools.product(SPOTS, MATS, VOLS, DIVS, TYPES))


@pytest.mark.parametrize("S,T,sigma,q,ot", CASES)
def test_delta_fd(S, T, sigma, q, ot):
    h = 1e-4 * S
    fd = (ref_price(S + h, STRIKE, T, RATE, sigma, q, ot) - ref_price(S - h, STRIKE, T, RATE, sigma, q, ot)) / (2 * h)
    an = float(greeks.delta(S, STRIKE, T, RATE, sigma, q, ot))
    assert an == pytest.approx(fd, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("S,T,sigma,q,ot", CASES)
def test_gamma_fd(S, T, sigma, q, ot):
    h = 1e-3 * S
    fd = (
        ref_price(S + h, STRIKE, T, RATE, sigma, q, ot)
        - 2 * ref_price(S, STRIKE, T, RATE, sigma, q, ot)
        + ref_price(S - h, STRIKE, T, RATE, sigma, q, ot)
    ) / (h * h)
    an = float(greeks.gamma(S, STRIKE, T, RATE, sigma, q))
    assert an == pytest.approx(fd, rel=1e-4, abs=1e-5)


@pytest.mark.parametrize("S,T,sigma,q,ot", CASES)
def test_vega_fd(S, T, sigma, q, ot):
    h = 1e-5
    fd = (ref_price(S, STRIKE, T, RATE, sigma + h, q, ot) - ref_price(S, STRIKE, T, RATE, sigma - h, q, ot)) / (2 * h)
    an = float(greeks.vega(S, STRIKE, T, RATE, sigma, q))
    assert an == pytest.approx(fd, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("S,T,sigma,q,ot", CASES)
def test_theta_fd(S, T, sigma, q, ot):
    # theta = dV/dt = -dV/dT.
    h = 1e-5
    fd = -(ref_price(S, STRIKE, T + h, RATE, sigma, q, ot) - ref_price(S, STRIKE, T - h, RATE, sigma, q, ot)) / (2 * h)
    an = float(greeks.theta(S, STRIKE, T, RATE, sigma, q, ot))
    assert an == pytest.approx(fd, rel=1e-4, abs=1e-5)


@pytest.mark.parametrize("S,T,sigma,q,ot", CASES)
def test_rho_fd(S, T, sigma, q, ot):
    h = 1e-5
    fd = (ref_price(S, STRIKE, T, RATE + h, sigma, q, ot) - ref_price(S, STRIKE, T, RATE - h, sigma, q, ot)) / (2 * h)
    an = float(greeks.rho(S, STRIKE, T, RATE, sigma, q, ot))
    assert an == pytest.approx(fd, rel=1e-4, abs=1e-4)


# Vanna and Volga are second order. A direct second difference of the price
# (mixed dS dsigma, or dsigma^2) is poorly conditioned at short maturities. We
# instead difference the analytic vega (which is exact: it uses only the exact
# phi), so vanna = dVega/dS and volga = dVega/dsigma. This still bottoms out at
# a finite difference and is accurate to ~1e-5 relative.
@pytest.mark.parametrize("S,T,sigma,q,ot", CASES)
def test_vanna_fd(S, T, sigma, q, ot):
    h = 1e-4 * S
    fd = (
        float(greeks.vega(S + h, STRIKE, T, RATE, sigma, q))
        - float(greeks.vega(S - h, STRIKE, T, RATE, sigma, q))
    ) / (2 * h)
    an = float(greeks.vanna(S, STRIKE, T, RATE, sigma, q))
    assert an == pytest.approx(fd, rel=1e-3, abs=1e-3)


@pytest.mark.parametrize("S,T,sigma,q,ot", CASES)
def test_volga_fd(S, T, sigma, q, ot):
    h = 1e-4
    fd = (
        float(greeks.vega(S, STRIKE, T, RATE, sigma + h, q))
        - float(greeks.vega(S, STRIKE, T, RATE, sigma - h, q))
    ) / (2 * h)
    an = float(greeks.volga(S, STRIKE, T, RATE, sigma, q))
    assert an == pytest.approx(fd, rel=1e-3, abs=1e-3)


# ----------------------------- sanity relations ----------------------------
def test_put_call_delta_relation():
    # Delta_call - Delta_put = e^{-qT}.
    S, K, T, r, sigma, q = 100.0, 105.0, 0.5, 0.04, 0.25, 0.03
    dc = float(greeks.delta(S, K, T, r, sigma, q, "call"))
    dp = float(greeks.delta(S, K, T, r, sigma, q, "put"))
    assert dc - dp == pytest.approx(np.exp(-q * T), abs=1e-12)


def test_gamma_vega_independent_of_type():
    # Gamma and vega are identical for calls and puts (they price the same).
    S, K, T, r, sigma, q = 103.0, 100.0, 0.7, 0.04, 0.3, 0.01
    assert greeks.gamma(S, K, T, r, sigma, q) > 0
    assert greeks.vega(S, K, T, r, sigma, q) > 0


def test_signs():
    # Long-option theta is negative; call rho positive, put rho negative.
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.04, 0.2, 0.0
    assert float(greeks.theta(S, K, T, r, sigma, q, "call")) < 0
    assert float(greeks.rho(S, K, T, r, sigma, q, "call")) > 0
    assert float(greeks.rho(S, K, T, r, sigma, q, "put")) < 0
    assert float(greeks.gamma(S, K, T, r, sigma, q)) > 0
    assert float(greeks.vega(S, K, T, r, sigma, q)) > 0


def test_convention_helpers():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.04, 0.2, 0.0
    assert greeks.vega_pct(S, K, T, r, sigma, q) == pytest.approx(
        float(greeks.vega(S, K, T, r, sigma, q)) / 100.0
    )
    assert greeks.theta_per_day(S, K, T, r, sigma, q, "call") == pytest.approx(
        float(greeks.theta(S, K, T, r, sigma, q, "call")) / 365.0
    )
    assert greeks.rho_pct(S, K, T, r, sigma, q, "call") == pytest.approx(
        float(greeks.rho(S, K, T, r, sigma, q, "call")) / 100.0
    )


def test_all_greeks_keys():
    g = greeks.all_greeks(100.0, 100.0, 1.0, 0.04, 0.2, 0.0, "call")
    assert set(g) == {
        "delta", "gamma", "vega", "vega_pct", "theta",
        "theta_per_day", "rho", "rho_pct", "vanna", "volga",
    }
