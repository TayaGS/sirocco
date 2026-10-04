#!/usr/bin/env python
"""Plot multiple LTE opacity curves on one graph.

This helper reads rows of the form:

    r rho T_e

where r is used only for the legend label, rho is the mass density in g cm^-3,
and T_e is the LTE temperature in K.

The hydrogen number density for each row is derived from the composition in the
selected masterfile, so the same atomic dataset is used for every curve.

Example:

    python py_progs/plot_lte_opacity_series.py ./xdata/master_cno.dat \
        --profile-file series.txt --outfile series.png

You can also read the profile rows from standard input by passing
`--profile-file -`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_lte_opacity as opacity


@dataclass
class ProfileRow:
    r: float
    rho: float
    temperature: float


def parse_profile_rows(handle: Sequence[str]) -> List[ProfileRow]:
    rows: List[ProfileRow] = []
    for raw_line in handle:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            r = float(parts[0])
            rho = float(parts[1])
            temperature = float(parts[2])
        except ValueError:
            continue
        rows.append(ProfileRow(r=r, rho=rho, temperature=temperature))
    return rows


def read_profile_rows(path: str) -> List[ProfileRow]:
    if path == "-":
        return parse_profile_rows(sys.stdin)
    with open(path, "r") as handle:
        return parse_profile_rows(handle)


def compute_mass_per_h(masterfile: str) -> float:
    elements, _, _ = opacity.parse_atomic_data(masterfile)
    mass_per_h = 0.0
    for element in elements.values():
        mass_per_h += element["abun"] * element["atomic_weight"] * opacity.AMU
    return mass_per_h


def build_series_curves(masterfile: str, rows: List[ProfileRow], npoints: int):
    photo_records, line_records, gaunt_records = opacity.load_atomic_records(masterfile)
    if not photo_records:
        raise RuntimeError("No photoionization records were found in the selected atomic data")

    energy_grid_ev = opacity.build_energy_grid(photo_records, line_records, npoints)
    mass_per_h = compute_mass_per_h(masterfile)

    curves = []
    for row in rows:
        if row.rho <= 0.0:
            continue
        nh = row.rho / mass_per_h
        (
            ion_results,
            ion_density,
            ne,
            level_lookup,
            level_id_lookup,
            level_populations,
            element_mass,
            mass_density,
        ) = opacity.build_lte_state(masterfile, row.temperature, nh)
        kappa_bf, kappa_ff = opacity.compute_continuum_components(
            energy_grid_ev,
            row.temperature,
            ion_results,
            ion_density,
            photo_records,
            level_lookup,
            level_id_lookup,
            level_populations,
            gaunt_records,
        )
        kappa_line = opacity.compute_line_component(
            energy_grid_ev,
            row.temperature,
            ion_density,
            level_lookup,
            level_populations,
            line_records,
            element_mass,
        )
        kappa_bf = kappa_bf / mass_density
        kappa_ff = kappa_ff / mass_density
        kappa_line = kappa_line / mass_density
        total = kappa_bf + kappa_ff + kappa_line
        curves.append((row, energy_grid_ev, kappa_bf, kappa_ff, kappa_line, total))

    return curves


def write_table(filename: str, curves):
    with open(filename, "w") as handle:
        handle.write(
            "# r rho_g_cm3 temperature_K energy_ev bound_free_cm2_per_g free_free_cm2_per_g line_cm2_per_g total_cm2_per_g\n"
        )
        for row, energy_grid_ev, kappa_bf, kappa_ff, kappa_line, total in curves:
            for energy_ev, bf_value, ff_value, line_value, total_value in zip(
                energy_grid_ev, kappa_bf, kappa_ff, kappa_line, total
            ):
                handle.write(
                    f"{row.r:15.8e} {row.rho:15.8e} {row.temperature:15.8e} "
                    f"{energy_ev:15.8e} {bf_value:15.8e} {ff_value:15.8e} {line_value:15.8e} {total_value:15.8e}\n"
                )


def plot_series(curves, outfile: str, masterfile: str):
    if not curves:
        raise RuntimeError("No valid profile rows were found")

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    colors = plt.cm.tab10([i / max(len(curves) - 1, 1) for i in range(len(curves))])

    all_positive = []
    for idx, (row, energy_grid_ev, kappa_bf, kappa_ff, kappa_line, total) in enumerate(curves):
        label = f"r={row.r:.2e}"
        ax.loglog(energy_grid_ev, total, color=colors[idx], linewidth=1.7, label=label)
        positive = total[total > 0.0]
        if positive.size > 0:
            all_positive.append(positive)

    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Opacity (cm$^2$ g$^{-1}$)")
    ax.set_title(f"LTE opacity series from {os.path.basename(masterfile)}")
    ax.set_xlim(1.0, 20.0)

    if all_positive:
        merged = __import__("numpy").concatenate(all_positive)
        ymin = max(float(merged.min()) * 0.5, 1.0e-30)
        ymax = float(merged.max()) * 2.0
        if ymax > ymin:
            ax.set_ylim(bottom=ymin, top=ymax)

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)


def parse_args(argv: List[str]):
    parser = argparse.ArgumentParser(description="Plot several LTE opacity curves on one graph.")
    parser.add_argument("masterfile", help="Atomic data masterfile, for example ./xdata/master_cno.dat")
    parser.add_argument(
        "--profile-file",
        default="-",
        help="Text file containing rows of 'r rho T_e' (default: stdin)",
    )
    parser.add_argument(
        "--outfile",
        default="series_lte_opacity.png",
        help="Output plot filename",
    )
    parser.add_argument(
        "--table",
        default="",
        help="Optional ASCII table filename for the plotted curves",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=1400,
        help="Number of base grid points for the opacity curve",
    )
    return parser.parse_args(argv)


def main(argv: List[str]):
    args = parse_args(argv)
    rows = read_profile_rows(args.profile_file)
    curves = build_series_curves(args.masterfile, rows, args.npoints)
    if args.table:
        write_table(args.table, curves)
    plot_series(curves, args.outfile, args.masterfile)
    print(f"Wrote {args.outfile}")
    if args.table:
        print(f"Wrote {args.table}")


if __name__ == "__main__":
    main(sys.argv[1:])