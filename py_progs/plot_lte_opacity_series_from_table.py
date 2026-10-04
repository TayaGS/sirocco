#!/usr/bin/env python
"""Plot LTE opacity curves from a saved table.

The input table is expected to come from plot_lte_opacity_series.py and have
this header format:

    # r rho_g_cm3 temperature_K energy_ev bound_free_cm2_per_g free_free_cm2_per_g line_cm2_per_g total_cm2_per_g

Rows are grouped by the first three columns. The script plots the total opacity
for each unique r value on the same graph, using a scientific-notation label
rounded to two decimals.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
import seaborn as sns
sns.set_context("notebook", font_scale=1.5)

matplotlib.use("Agg")
# matplotlib.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
# })
import matplotlib.pyplot as plt


@dataclass
class CurveData:
    r: float
    rho: float
    temperature: float
    energy_ev: List[float]
    total: List[float]


def parse_table(path: str) -> List[CurveData]:
    curves: Dict[Tuple[float, float, float], CurveData] = {}
    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                r = float(parts[0])
                rho = float(parts[1])
                temperature = float(parts[2])
                energy_ev = float(parts[3])
                total = float(parts[7])
            except ValueError:
                continue
            key = (r, rho, temperature)
            curve = curves.get(key)
            if curve is None:
                curve = CurveData(r=r, rho=rho, temperature=temperature, energy_ev=[], total=[])
                curves[key] = curve
            curve.energy_ev.append(energy_ev)
            curve.total.append(total)
    return list(curves.values())


def plot_curves(curves: List[CurveData], outfile: str, title: str):
    if not curves:
        raise RuntimeError("No curves were found in the table")

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    colors = plt.cm.tab10([i / max(len(curves) - 1, 1) for i in range(len(curves))])

    all_positive = []
    for idx, curve in enumerate([curves[4], curves[6]]):
        label = fr"$r={curve.r:.2e},\ \tau={2*0.34*curve.r * curve.rho:.2e}$"
        energy_ev = curve.energy_ev
        total = curve.total
        ax.loglog(energy_ev, np.array(total)/0.34, color=colors[idx], label=label, alpha=0.7)
        positive = [value for value in total if value > 0.0]
        if positive:
            all_positive.extend(positive)

    ax.set_xlabel(r"Photon energy (eV)")
    ax.set_ylabel(r"$\kappa_{\rm abs}/\kappa_{\rm T}$ (cm$^2$ g$^{-1}$)")
    ax.set_title(title)
    ax.set_xlim(1.0, 20.0)

    if all_positive:
        ymin = max(min(all_positive) * 0.5, 1.0e-30)
        ymax = max(all_positive) * 2.0
        if ymax > ymin:
            ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_ylim(1.e-4, 1.e6)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)


def parse_args(argv: List[str]):
    parser = argparse.ArgumentParser(description="Plot LTE opacity series from a saved table.")
    parser.add_argument("table", help="Opacity table written by plot_lte_opacity_series.py")
    parser.add_argument(
        "--outfile",
        default="series_from_table.png",
        help="Output plot filename",
    )
    parser.add_argument(
        "--title",
        default="LTE opacity series from table",
        help="Plot title",
    )
    return parser.parse_args(argv)


def main(argv: List[str]):
    args = parse_args(argv)
    curves = parse_table(args.table)
    plot_curves(curves, args.outfile, args.title)
    print(f"Wrote {args.outfile}")


if __name__ == "__main__":
    main(sys.argv[1:])