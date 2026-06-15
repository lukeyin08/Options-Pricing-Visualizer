"""Tests for the Black-Scholes vs Monte Carlo comparison utilities."""
from __future__ import annotations

import numpy as np

from options_viz import comparison as cmp

STRIKES = [90.0, 100.0, 110.0]
MATS = [0.5, 1.0]


def test_compare_rows_agree_within_sampling_error():
    rows = cmp.compare_bs_mc(STRIKES, MATS, n=100_000, seed=0, method="antithetic")
    assert len(rows) == len(STRIKES) * len(MATS)
    for r in rows:
        assert np.isfinite(r.bs) and np.isfinite(r.mc) and np.isfinite(r.se)
        assert r.se > 0
        # MC must agree with BS to within a few standard errors.
        assert r.abs_error < 4.0 * r.se


def test_ci_coverage_is_reasonable():
    # Larger grid; nearly all cells should bracket BS in their 95% CI.
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    rows = cmp.compare_bs_mc(strikes, [0.25, 1.0, 2.0], n=150_000, seed=0, method="naive")
    assert cmp.coverage_fraction(rows) >= 0.7


def test_markdown_table_shape():
    rows = cmp.compare_bs_mc(STRIKES, MATS, n=20_000, seed=1)
    md = cmp.comparison_table_markdown(rows, "antithetic")
    lines = md.splitlines()
    # header + separator + one row per cell.
    assert lines[0].startswith("| K | T |")
    assert len(lines) == 2 + len(rows)


def test_mc_greeks_within_3_se_of_analytic():
    for g in cmp.greeks_comparison(seed=10):
        assert abs(g.z) < 3.0
        assert np.isfinite(g.mc) and g.se > 0


def test_timing_bs_cheaper_than_mc():
    t = cmp.timing_comparison(mc_sizes=(10**4,), bs_reps=2_000)
    assert t["bs_per_price"] > 0
    # One BS price is much cheaper than an N=10^4 Monte Carlo run.
    assert t["bs_per_price"] < t["mc_n_10000"]
