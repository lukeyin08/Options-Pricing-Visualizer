#!/usr/bin/env python3
"""Regenerate every figure into ``figures/`` in one run.

Usage::

    python scripts/generate_figures.py [--outdir figures]

Produces, for each of delta / gamma / vega:
  * <greek>_vs_strike.png   - 2D lines, one per maturity
  * <greek>_vs_time.png     - 2D lines, one per moneyness
  * <greek>_surface.png     - static 3D surface (matplotlib)
  * <greek>_surface.html    - interactive 3D surface (plotly)
  * <greek>_heatmap.png     - heatmap over (strike, time)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from options_viz import visualize as viz  # noqa: E402

GREEKS = ("delta", "gamma", "vega")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate all Greek figures.")
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[1] / "figures"),
        help="output directory (default: ./figures)",
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    viz.apply_style()
    written: list[str] = []

    for g in GREEKS:
        # A) vs strike (per maturity)
        fig = viz.plot_greek_vs_strike(g)
        p = outdir / f"{g}_vs_strike.png"
        fig.savefig(p)
        plt.close(fig)
        written.append(p.name)

        # B) vs time to expiry (per moneyness)
        fig = viz.plot_greek_vs_time(g)
        p = outdir / f"{g}_vs_time.png"
        fig.savefig(p)
        plt.close(fig)
        written.append(p.name)

        # C) static 3D surface
        fig = viz.surface_matplotlib(g)
        p = outdir / f"{g}_surface.png"
        fig.savefig(p)
        plt.close(fig)
        written.append(p.name)

        # C') interactive 3D surface (plotly HTML)
        pfig = viz.surface_plotly(g)
        p = outdir / f"{g}_surface.html"
        pfig.write_html(str(p), include_plotlyjs="cdn", full_html=True)
        written.append(p.name)

        # D) heatmap
        fig = viz.heatmap(g)
        p = outdir / f"{g}_heatmap.png"
        fig.savefig(p)
        plt.close(fig)
        written.append(p.name)

    print(f"Wrote {len(written)} figures to {outdir}:")
    for name in written:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
