"""Standard normal distribution N(0, 1) from first principles.

This module implements the probability density function (PDF) and the
cumulative distribution function (CDF) of the standard normal without any
statistics library. The CDF is built on a polynomial approximation of the
error function (Abramowitz & Stegun, Handbook of Mathematical Functions,
formula 7.1.26), which has a maximum absolute error of about 1.5e-7.

Everything is vectorized over numpy arrays. scipy is intentionally not used.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["norm_pdf", "erf", "norm_cdf"]

FloatOrArray = Union[float, NDArray[np.float64]]

_SQRT_2PI: float = float(np.sqrt(2.0 * np.pi))
_SQRT_2: float = float(np.sqrt(2.0))

# Abramowitz & Stegun 7.1.26 coefficients. The approximation is valid for
# x >= 0 with |error| <= 1.5e-7; we extend to x < 0 via the odd symmetry of erf.
_AS_P: float = 0.3275911
_AS_A1: float = 0.254829592
_AS_A2: float = -0.284496736
_AS_A3: float = 1.421413741
_AS_A4: float = -1.453152027
_AS_A5: float = 1.061405429


def norm_pdf(x: ArrayLike) -> FloatOrArray:
    """Standard normal PDF: phi(x) = exp(-x^2 / 2) / sqrt(2*pi).

    Args:
        x: Point(s) at which to evaluate the density.

    Returns:
        The density value(s); a float for scalar input, else an ndarray.
    """
    xa = np.asarray(x, dtype=np.float64)
    out = np.exp(-0.5 * xa * xa) / _SQRT_2PI
    return out.item() if out.ndim == 0 else out


def erf(x: ArrayLike) -> FloatOrArray:
    """Error function via Abramowitz & Stegun 7.1.26.

    The polynomial is only accurate for x >= 0, so we evaluate it on |x| and
    restore the sign using the identity erf(-x) = -erf(x). erf(0) = 0 falls out
    automatically because numpy's sign(0) = 0.

    Args:
        x: Point(s) at which to evaluate erf.

    Returns:
        erf(x); a float for scalar input, else an ndarray.
    """
    xa = np.asarray(x, dtype=np.float64)
    sign = np.sign(xa)
    ax = np.abs(xa)
    t = 1.0 / (1.0 + _AS_P * ax)
    # Horner form of (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5).
    poly = t * (_AS_A1 + t * (_AS_A2 + t * (_AS_A3 + t * (_AS_A4 + t * _AS_A5))))
    y = 1.0 - poly * np.exp(-ax * ax)
    out = sign * y
    return out.item() if out.ndim == 0 else out


def norm_cdf(x: ArrayLike) -> FloatOrArray:
    """Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).

    Inherits the ~1.5e-7 accuracy of the underlying erf approximation (the
    factor of 1/2 actually halves the worst-case error to ~7.5e-8).

    Args:
        x: Point(s) at which to evaluate the CDF.

    Returns:
        Phi(x) in [0, 1]; a float for scalar input, else an ndarray.
    """
    xa = np.asarray(x, dtype=np.float64)
    out = 0.5 * (1.0 + np.asarray(erf(xa / _SQRT_2), dtype=np.float64))
    return out.item() if out.ndim == 0 else out
