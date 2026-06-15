"""Analytical Black-Scholes Greeks, with continuous dividend yield q.

Each Greek is the exact partial derivative of the Black-Scholes price. Every
formula is validated against central finite differences of the price in
``tests/test_greeks.py`` (that comparison is the proof the formulas are right).

Conventions
-----------
* **Vega / Rho** are returned per **1.00** change in volatility / rate. Use
  :func:`vega_pct` and :func:`rho_pct` for the per-1% (per 0.01) convention
  that traders usually quote.
* **Theta** is returned per **year** and is the time decay dV/dt = -dV/dT;
  it is typically negative for long options. Use :func:`theta_per_day` for the
  per-calendar-day figure (divide by 365).
* **Vanna** = dDelta/dSigma = dVega/dSpot. **Volga (Vomma)** = dVega/dSigma.

All functions are vectorized and broadcast over array-like inputs.
"""
from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .black_scholes import _normalize_type
from .normal import norm_cdf, norm_pdf

__all__ = [
    "delta",
    "gamma",
    "vega",
    "vega_pct",
    "theta",
    "theta_per_day",
    "rho",
    "rho_pct",
    "vanna",
    "volga",
    "all_greeks",
]

FloatOrArray = Union[float, NDArray[np.float64]]

DAYS_PER_YEAR: float = 365.0


def _unwrap(x: np.ndarray) -> FloatOrArray:
    return x.item() if x.ndim == 0 else x


def _common(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike
) -> Tuple[np.ndarray, ...]:
    """Shared intermediate quantities for the Greeks.

    Returns S, K, T, r, sigma, q (as arrays), the total volatility
    ``vol = sigma*sqrt(T)``, d1, d2, and phi(d1). Where vol == 0 the d-terms
    are computed with a dummy denominator of 1 and the dependent Greeks are
    masked to finite limits by the callers.
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    vol = sigma * np.sqrt(T)
    safe_vol = np.where(vol > 0.0, vol, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / safe_vol
        d2 = d1 - safe_vol
    pdf_d1 = np.asarray(norm_pdf(d1), dtype=np.float64)
    return S, K, T, r, sigma, q, vol, d1, d2, pdf_d1


def delta(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
    q: ArrayLike = 0.0, option_type: str = "call",
) -> FloatOrArray:
    """Delta = dV/dS. Call: e^{-qT} N(d1). Put: e^{-qT} (N(d1) - 1).

    Shape across strike: a smooth S-curve from 0 (deep OTM) to ~e^{-qT}
    (deep ITM) that steepens as T -> 0.
    """
    S, K, T, r, sigma, q, vol, d1, d2, _ = _common(S, K, T, r, sigma, q)
    ot = _normalize_type(option_type)
    eqt = np.exp(-q * T)
    nd1 = np.asarray(norm_cdf(d1), dtype=np.float64)
    val = eqt * nd1 if ot == "call" else eqt * (nd1 - 1.0)
    return _unwrap(np.asarray(val))


def gamma(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0
) -> FloatOrArray:
    """Gamma = d^2V/dS^2 = e^{-qT} phi(d1) / (S sigma sqrt(T)). Same for call/put.

    Shape: a bell centered near the money that grows tall and narrow as T -> 0
    (the price's kink at K sharpens), so gamma spikes at expiry ATM.
    """
    S, K, T, r, sigma, q, vol, d1, d2, pdf_d1 = _common(S, K, T, r, sigma, q)
    eqt = np.exp(-q * T)
    denom = np.where(vol > 0.0, S * sigma * np.sqrt(T), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = eqt * pdf_d1 / denom
    val = np.where(vol > 0.0, val, 0.0)
    return _unwrap(np.asarray(val))


def vega(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0
) -> FloatOrArray:
    """Vega = dV/dsigma = S e^{-qT} phi(d1) sqrt(T) (per 1.00 vol). Same call/put.

    Shape: peaks near the money and *grows* with maturity (the sqrt(T) factor),
    so long-dated ATM options are the most vol-sensitive.
    """
    S, K, T, r, sigma, q, vol, d1, d2, pdf_d1 = _common(S, K, T, r, sigma, q)
    eqt = np.exp(-q * T)
    val = S * eqt * pdf_d1 * np.sqrt(T)
    val = np.where(vol > 0.0, val, 0.0)
    return _unwrap(np.asarray(val))


def vega_pct(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0
) -> FloatOrArray:
    """Vega per 1% (0.01) change in volatility = vega / 100."""
    return _unwrap(np.asarray(vega(S, K, T, r, sigma, q)) / 100.0)


def theta(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
    q: ArrayLike = 0.0, option_type: str = "call",
) -> FloatOrArray:
    """Theta = dV/dt = -dV/dT, per year.

    Call: -(S phi(d1) sigma e^{-qT})/(2 sqrt(T)) - r K e^{-rT} N(d2)
          + q S e^{-qT} N(d1).
    Put:  -(S phi(d1) sigma e^{-qT})/(2 sqrt(T)) + r K e^{-rT} N(-d2)
          - q S e^{-qT} N(-d1).
    """
    S, K, T, r, sigma, q, vol, d1, d2, pdf_d1 = _common(S, K, T, r, sigma, q)
    ot = _normalize_type(option_type)
    eqt = np.exp(-q * T)
    ert = np.exp(-r * T)
    safe_sqrt_T = np.where(T > 0.0, np.sqrt(T), 1.0)
    decay = -(S * pdf_d1 * sigma * eqt) / (2.0 * safe_sqrt_T)
    if ot == "call":
        val = decay - r * K * ert * np.asarray(norm_cdf(d2)) + q * S * eqt * np.asarray(norm_cdf(d1))
    else:
        val = decay + r * K * ert * np.asarray(norm_cdf(-d2)) - q * S * eqt * np.asarray(norm_cdf(-d1))
    val = np.where(vol > 0.0, val, 0.0)
    return _unwrap(np.asarray(val))


def theta_per_day(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
    q: ArrayLike = 0.0, option_type: str = "call",
) -> FloatOrArray:
    """Theta per calendar day = theta(per year) / 365."""
    return _unwrap(np.asarray(theta(S, K, T, r, sigma, q, option_type)) / DAYS_PER_YEAR)


def rho(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
    q: ArrayLike = 0.0, option_type: str = "call",
) -> FloatOrArray:
    """Rho = dV/dr, per 1.00 rate. Call: K T e^{-rT} N(d2). Put: -K T e^{-rT} N(-d2)."""
    S, K, T, r, sigma, q, vol, d1, d2, _ = _common(S, K, T, r, sigma, q)
    ot = _normalize_type(option_type)
    ert = np.exp(-r * T)
    if ot == "call":
        val = K * T * ert * np.asarray(norm_cdf(d2))
    else:
        val = -K * T * ert * np.asarray(norm_cdf(-d2))
    return _unwrap(np.asarray(val))


def rho_pct(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
    q: ArrayLike = 0.0, option_type: str = "call",
) -> FloatOrArray:
    """Rho per 1% (0.01) change in rate = rho / 100."""
    return _unwrap(np.asarray(rho(S, K, T, r, sigma, q, option_type)) / 100.0)


def vanna(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0
) -> FloatOrArray:
    """Vanna = dDelta/dSigma = dVega/dSpot = -e^{-qT} phi(d1) d2 / sigma.

    Uses the identity dd1/dsigma = -d2/sigma, so the chain rule on
    Delta = e^{-qT} N(d1) collapses to this compact form. Same for call/put.
    """
    S, K, T, r, sigma, q, vol, d1, d2, pdf_d1 = _common(S, K, T, r, sigma, q)
    eqt = np.exp(-q * T)
    safe_sigma = np.where(sigma > 0.0, sigma, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = -eqt * pdf_d1 * d2 / safe_sigma
    val = np.where(vol > 0.0, val, 0.0)
    return _unwrap(np.asarray(val))


def volga(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0
) -> FloatOrArray:
    """Volga / Vomma = dVega/dSigma = Vega * d1 d2 / sigma.

    Positive away from the money (vega is convex in sigma there), ~0 ATM where
    d1 d2 ~ 0. Same for call/put.
    """
    S, K, T, r, sigma, q, vol, d1, d2, pdf_d1 = _common(S, K, T, r, sigma, q)
    eqt = np.exp(-q * T)
    safe_sigma = np.where(sigma > 0.0, sigma, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vega_val = S * eqt * pdf_d1 * np.sqrt(T)
        val = vega_val * d1 * d2 / safe_sigma
    val = np.where(vol > 0.0, val, 0.0)
    return _unwrap(np.asarray(val))


def all_greeks(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
    q: ArrayLike = 0.0, option_type: str = "call",
) -> Dict[str, FloatOrArray]:
    """Return every Greek in one dict, using the documented conventions."""
    return {
        "delta": delta(S, K, T, r, sigma, q, option_type),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),               # per 1.00 vol
        "vega_pct": vega_pct(S, K, T, r, sigma, q),        # per 1% vol
        "theta": theta(S, K, T, r, sigma, q, option_type),  # per year
        "theta_per_day": theta_per_day(S, K, T, r, sigma, q, option_type),
        "rho": rho(S, K, T, r, sigma, q, option_type),     # per 1.00 rate
        "rho_pct": rho_pct(S, K, T, r, sigma, q, option_type),  # per 1% rate
        "vanna": vanna(S, K, T, r, sigma, q),
        "volga": volga(S, K, T, r, sigma, q),
    }
