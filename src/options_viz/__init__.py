"""options_viz: from-scratch Black-Scholes pricing, Greeks, and Monte Carlo.

The public numeric API is re-exported at the package root, e.g.::

    from options_viz import call_price, delta, mc_price

Submodules can also be imported directly (``options_viz.visualize`` etc.).
"""
from __future__ import annotations

__version__ = "0.1.0"

from .normal import erf, norm_cdf, norm_pdf
from .black_scholes import (
    bs_price,
    call_price,
    d1_d2,
    implied_vol,
    put_call_parity_gap,
    put_price,
)
from .greeks import (
    all_greeks,
    delta,
    gamma,
    rho,
    rho_pct,
    theta,
    theta_per_day,
    vanna,
    vega,
    vega_pct,
    volga,
)
from .monte_carlo import (
    MCResult,
    convergence_study,
    mc_delta_pathwise,
    mc_gamma_crn,
    mc_gamma_lr,
    mc_price,
    mc_price_antithetic,
    mc_price_control_variate,
    variance_reduction_report,
)

__all__ = [
    "__version__",
    # normal
    "erf",
    "norm_cdf",
    "norm_pdf",
    # black_scholes
    "d1_d2",
    "bs_price",
    "call_price",
    "put_price",
    "put_call_parity_gap",
    "implied_vol",
    # greeks
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
    # monte_carlo
    "mc_price",
    "mc_price_antithetic",
    "mc_price_control_variate",
    "variance_reduction_report",
    "convergence_study",
    "mc_delta_pathwise",
    "mc_gamma_lr",
    "mc_gamma_crn",
    "MCResult",
]
