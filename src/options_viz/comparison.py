"""Black-Scholes vs Monte Carlo: comparison tables, MC-Greek checks, timing.

The reusable logic lives here (so tests can import it); ``scripts/run_comparison.py``
is a thin CLI that calls these functions, makes the plots, and prints the tables.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from . import greeks
from .black_scholes import bs_price
from .monte_carlo import (
    MCResult,
    mc_delta_pathwise,
    mc_gamma_crn,
    mc_gamma_lr,
    mc_price,
    mc_price_antithetic,
    mc_price_control_variate,
)

_ESTIMATORS: Dict[str, Callable[..., MCResult]] = {
    "naive": mc_price,
    "antithetic": mc_price_antithetic,
    "control_variate": mc_price_control_variate,
}


@dataclass(frozen=True)
class ComparisonRow:
    """One (strike, maturity) cell of the BS-vs-MC table."""

    S: float
    K: float
    T: float
    bs: float
    mc: float
    se: float
    abs_error: float
    in_ci95: bool


def compare_bs_mc(
    strikes: Sequence[float],
    maturities: Sequence[float],
    S: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.20,
    q: float = 0.0,
    option_type: str = "call",
    n: int = 200_000,
    seed: int = 0,
    method: str = "antithetic",
) -> List[ComparisonRow]:
    """Build the BS-vs-MC comparison rows over a strike x maturity grid.

    Each MC estimate uses an independent seed so the CI-coverage check is a
    fair (if small) sample. The grid loop is over a handful of cells only; the
    pricing inside each cell is vectorized.
    """
    estimator = _ESTIMATORS[method]
    rows: List[ComparisonRow] = []
    k = 0
    for T in maturities:
        for K in strikes:
            bs = float(bs_price(S, K, T, r, sigma, q, option_type))
            res = estimator(S, K, T, r, sigma, q, option_type, n=n, seed=seed + k)
            lo, hi = res.ci95
            rows.append(
                ComparisonRow(
                    S=S, K=K, T=T, bs=bs, mc=res.estimate, se=res.std_error,
                    abs_error=abs(res.estimate - bs), in_ci95=bool(lo <= bs <= hi),
                )
            )
            k += 1
    return rows


def comparison_table_markdown(rows: Sequence[ComparisonRow], method: str) -> str:
    """Render comparison rows as a GitHub-flavored markdown table."""
    header = (
        f"| K | T | BS price | MC price ({method}) | MC SE | abs error | BS in 95% CI |\n"
        "|---:|---:|---:|---:|---:|---:|:--:|"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r.K:g} | {r.T:g} | {r.bs:.4f} | {r.mc:.4f} | {r.se:.4f} | "
            f"{r.abs_error:.4f} | {'yes' if r.in_ci95 else 'no'} |"
        )
    return "\n".join(lines)


def coverage_fraction(rows: Sequence[ComparisonRow]) -> float:
    """Fraction of cells whose BS price lies inside the MC 95% CI."""
    return float(np.mean([r.in_ci95 for r in rows]))


@dataclass(frozen=True)
class GreekRow:
    name: str
    analytic: float
    mc: float
    se: float
    z: float


def greeks_comparison(
    S: float = 100.0, K: float = 100.0, T: float = 1.0, r: float = 0.05,
    sigma: float = 0.20, q: float = 0.0, seed: int = 0,
) -> List[GreekRow]:
    """Compare MC Greek estimators (pathwise delta, LR gamma, CRN gamma) to analytic."""
    d_an = float(greeks.delta(S, K, T, r, sigma, q, "call"))
    g_an = float(greeks.gamma(S, K, T, r, sigma, q))
    out: List[GreekRow] = []

    d_mc = mc_delta_pathwise(S, K, T, r, sigma, q, "call", n=400_000, seed=seed)
    out.append(GreekRow("delta (pathwise)", d_an, d_mc.estimate, d_mc.std_error,
                        (d_mc.estimate - d_an) / d_mc.std_error))

    g_lr = mc_gamma_lr(S, K, T, r, sigma, q, "call", n=1_000_000, seed=seed + 1)
    out.append(GreekRow("gamma (LR)", g_an, g_lr.estimate, g_lr.std_error,
                        (g_lr.estimate - g_an) / g_lr.std_error))

    g_crn = mc_gamma_crn(S, K, T, r, sigma, q, "call", n=400_000, seed=seed + 2)
    out.append(GreekRow("gamma (CRN bump)", g_an, g_crn.estimate, g_crn.std_error,
                        (g_crn.estimate - g_an) / g_crn.std_error))
    return out


def timing_comparison(
    S: float = 100.0, K: float = 100.0, T: float = 1.0, r: float = 0.05,
    sigma: float = 0.20, q: float = 0.0, mc_sizes: Sequence[int] = (10**4, 10**5, 10**6),
    bs_reps: int = 100_000, seed: int = 0,
) -> Dict[str, float]:
    """Wall-clock timing: one BS price vs MC at several sample sizes (seconds)."""
    out: Dict[str, float] = {}

    t0 = time.perf_counter()
    for _ in range(bs_reps):
        bs_price(S, K, T, r, sigma, q, "call")
    out["bs_per_price"] = (time.perf_counter() - t0) / bs_reps

    for N in mc_sizes:
        t0 = time.perf_counter()
        mc_price(S, K, T, r, sigma, q, "call", n=int(N), seed=seed)
        out[f"mc_n_{int(N)}"] = time.perf_counter() - t0
    return out
