#!/usr/bin/env python3
"""Black-Scholes vs Monte Carlo: print the tables, write the plots.

Usage::

    python scripts/run_comparison.py

Outputs:
  * figures/bs_vs_mc_errorbars.png  - BS curve vs MC points with 95% CI bars
  * figures/mc_convergence.png      - price +/- CI and half-width vs N (log-log)
  * results/bs_vs_mc_table.md       - the comparison table (for the README)
and prints the comparison table, the MC-Greek table, and the timing table.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from options_viz import comparison as cmp  # noqa: E402
from options_viz import visualize as viz  # noqa: E402
from options_viz.black_scholes import bs_price  # noqa: E402
from options_viz.monte_carlo import convergence_study, mc_price  # noqa: E402

BASE = dict(S=100.0, r=0.05, sigma=0.20, q=0.0)
STRIKES = [80.0, 90.0, 100.0, 110.0, 120.0]
MATURITIES = [0.25, 1.0, 2.0]


def errorbar_figure(outdir: Path) -> Path:
    """BS curve vs MC points with 95% CI error bars, across strikes at T=1."""
    T = 1.0
    strikes = np.linspace(80, 120, 17)
    bs = np.asarray(bs_price(BASE["S"], strikes, T, BASE["r"], BASE["sigma"], BASE["q"], "call"))
    mc_vals, mc_err = [], []
    for i, K in enumerate(strikes):
        res = mc_price(BASE["S"], float(K), T, BASE["r"], BASE["sigma"], BASE["q"], "call", n=40_000, seed=i)
        mc_vals.append(res.estimate)
        mc_err.append(res.ci_halfwidth)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(strikes, bs, color="#2166ac", lw=2, label="Black-Scholes (exact)", zorder=1)
    ax.errorbar(strikes, mc_vals, yerr=mc_err, fmt="o", ms=4, color="#b2182b",
                ecolor="0.5", capsize=3, label="Monte Carlo (95% CI), N=40k", zorder=2)
    ax.set_xlabel("Strike K  (S=100, T=1y)")
    ax.set_ylabel("Call price")
    ax.set_title("Black-Scholes vs Monte Carlo across strikes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "bs_vs_mc_errorbars.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def convergence_figure(outdir: Path) -> Path:
    """Two panels: price +/- CI converging to BS, and half-width's 1/sqrt(N) decay."""
    cs = convergence_study(BASE["S"], 100.0, 1.0, BASE["r"], BASE["sigma"], BASE["q"], "call",
                           n_grid=(10**2, 10**3, 10**4, 10**5, 10**6, 10**7), seed=0)
    bs = cs["bs_price"][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.axhline(bs, color="#2166ac", lw=2, label=f"BS = {bs:.4f}")
    ax1.errorbar(cs["N"], cs["price"], yerr=cs["ci_halfwidth"], fmt="o-", color="#b2182b",
                 ecolor="0.6", capsize=3, label="MC price (95% CI)")
    ax1.set_xscale("log")
    ax1.set_xlabel("samples N")
    ax1.set_ylabel("Call price")
    ax1.set_title("MC price converges onto BS")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.loglog(cs["N"], cs["ci_halfwidth"], "o-", color="#b2182b", label="MC 95% CI half-width")
    ref = cs["ci_halfwidth"][0] * np.sqrt(cs["N"][0] / cs["N"])  # slope -1/2 reference
    ax2.loglog(cs["N"], ref, "--", color="0.4", label=r"$\propto 1/\sqrt{N}$")
    ax2.set_xlabel("samples N")
    ax2.set_ylabel("CI half-width")
    ax2.set_title("Half-width decays like 1/sqrt(N)")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    p = outdir / "mc_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    figdir = _ROOT / "figures"
    resdir = _ROOT / "results"
    figdir.mkdir(exist_ok=True)
    resdir.mkdir(exist_ok=True)
    viz.apply_style()

    method = "antithetic"
    rows = cmp.compare_bs_mc(STRIKES, MATURITIES, n=200_000, seed=0, method=method, **BASE)
    table_md = cmp.comparison_table_markdown(rows, method)
    cov = cmp.coverage_fraction(rows)

    print("=" * 72)
    print(f"BLACK-SCHOLES vs MONTE CARLO ({method}, N=200,000, call)")
    print("=" * 72)
    print(table_md)
    print(f"\nBS inside MC 95% CI for {cov*100:.0f}% of the {len(rows)} cells "
          f"(expect ~95%).")
    (resdir / "bs_vs_mc_table.md").write_text(
        f"# Black-Scholes vs Monte Carlo ({method}, N=200,000, call)\n\n"
        + table_md
        + f"\n\nBS inside MC 95% CI for {cov*100:.0f}% of {len(rows)} cells.\n"
    )

    print("\n" + "=" * 72)
    print("MONTE CARLO GREEKS vs ANALYTIC  (S=100, K=100, T=1, r=0.05, sigma=0.20)")
    print("=" * 72)
    print(f"| {'Greek':18s} | {'analytic':>10s} | {'MC':>10s} | {'MC SE':>9s} | {'z':>6s} |")
    print(f"|{'-'*20}|{'-'*12}|{'-'*12}|{'-'*11}|{'-'*8}|")
    for g in cmp.greeks_comparison(seed=10):
        print(f"| {g.name:18s} | {g.analytic:10.5f} | {g.mc:10.5f} | {g.se:9.5f} | {g.z:+6.2f} |")

    print("\n" + "=" * 72)
    print("TIMING  (wall-clock; BS is one analytic evaluation)")
    print("=" * 72)
    t = cmp.timing_comparison()
    print(f"BS per price        : {t['bs_per_price']*1e6:8.3f} microseconds")
    for key in sorted(k for k in t if k.startswith("mc_n_")):
        N = int(key.split("_")[-1])
        print(f"MC price (N={N:>9,}): {t[key]*1e3:8.2f} ms")

    p1 = errorbar_figure(figdir)
    p2 = convergence_figure(figdir)
    print("\nWrote figures:")
    print(f"  - {p1.name}")
    print(f"  - {p2.name}")
    print(f"Wrote table: results/{(resdir / 'bs_vs_mc_table.md').name}")

    bs_us = t["bs_per_price"] * 1e6
    mc_lo, mc_hi = t["mc_n_10000"] * 1e3, t["mc_n_1000000"] * 1e3
    print("\n" + "-" * 72)
    print("TAKEAWAY: one BS price takes ~%.0f microseconds and is exact, but bakes in" % bs_us)
    print("constant volatility, a lognormal terminal law, and continuous frictionless")
    print("hedging. MC here costs %.2f ms (N=10^4) to %.1f ms (N=10^6), roughly %dx-%dx" % (
        mc_lo, mc_hi, round(mc_lo * 1e3 / bs_us), round(mc_hi * 1e3 / bs_us)))
    print("slower for one price, and it carries 1/sqrt(N) noise. What you buy: MC prices")
    print("path-dependent and exotic payoffs and arbitrary dynamics with no closed form.")
    print("Variance reduction buys back much of the gap: control variates cut the")
    print("ATM-call variance ~7x (standard error ~2.6x), i.e. ~7x fewer paths per CI.")


if __name__ == "__main__":
    main()
