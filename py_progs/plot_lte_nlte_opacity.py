#!/usr/bin/env python
"""Plot LTE and manually specified NLTE opacity curves side by side.

The departure-coefficient file contains two whitespace-separated columns:

    level_index  b_i

Levels omitted from that file use ``--default-b``.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from dense_spectrum import (
    load_sirocco_hydrogen_data,
    lte_opacity_for_state,
    nlte_opacity_for_state,
)
from nlte_opacity import read_departure_coefficients


PLANCK_ERG_S = 6.62607015e-27
EV2ERGS = 1.602176634e-12


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("masterfile", help="SIROCCO atomic-data master file")
    parser.add_argument("temperature_K", type=float, help="gas temperature in K")
    parser.add_argument("hydrogen_density_cm3", type=float, help="total hydrogen density in cm^-3")
    parser.add_argument(
        "--departure-coefficients",
        required=True,
        help="coefficient file with global indices or physical level IDs",
    )
    parser.add_argument("--default-b", type=float, default=1.0, help="b_i for omitted levels")
    parser.add_argument("--emin", type=float, default=1.0e-3, help="minimum photon energy in eV")
    parser.add_argument("--emax", type=float, default=1.0e4, help="maximum photon energy in eV")
    parser.add_argument("--npoints", type=int, default=1200, help="number of logarithmic energy points")
    parser.add_argument("--outfile", default="lte_nlte_opacity.png", help="output figure")
    parser.add_argument("--table", help="optional output table containing both curves")
    return parser.parse_args(argv)


def write_comparison_table(filename, energy_ev, lte_opacity, nlte_opacity):
    with open(filename, "w") as handle:
        handle.write("# energy_ev lte_opacity_cm2_g nlte_opacity_cm2_g\n")
        for energy, lte, nlte in zip(energy_ev, lte_opacity, nlte_opacity):
            handle.write(f"{energy:.8e} {lte:.8e} {nlte:.8e}\n")


def main(argv=None):
    args = parse_args(argv)
    if args.temperature_K <= 0.0 or args.hydrogen_density_cm3 <= 0.0:
        raise ValueError("temperature and hydrogen density must be positive")
    if args.emin <= 0.0 or args.emax <= args.emin or args.npoints < 2:
        raise ValueError("opacity energy range and point count are invalid")

    atomic_data = load_sirocco_hydrogen_data(args.masterfile)
    departure_coefficients = read_departure_coefficients(
        args.departure_coefficients,
        len(atomic_data["levels"]),
        args.default_b,
        atomic_data["levels"],
    )
    used_indices = np.flatnonzero(~np.isclose(departure_coefficients, args.default_b))[:5]
    print("first five non-default departure coefficients used:")
    for level_index in used_indices:
        level = atomic_data["levels"][level_index]
        print(
            f"  index={level_index} z={level['z']} istate={level['istate']} "
            f"level_id={level['level_id']} ex_ev={level['ex_ev']:.6g} "
            f"b_i={departure_coefficients[level_index]:.6g}"
        )
    energy_ev = np.geomspace(args.emin, args.emax, args.npoints)
    frequency_hz = energy_ev * EV2ERGS / PLANCK_ERG_S
    lte_opacity = lte_opacity_for_state(
        args.masterfile,
        args.temperature_K,
        args.hydrogen_density_cm3,
        frequency_hz,
        atomic_data,
    )
    nlte_opacity = nlte_opacity_for_state(
        args.masterfile,
        args.temperature_K,
        args.hydrogen_density_cm3,
        frequency_hz,
        departure_coefficients,
        atomic_data,
    )

    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    axis.loglog(energy_ev, np.maximum(lte_opacity, 1.0e-300), label="LTE")
    axis.loglog(energy_ev, np.maximum(nlte_opacity, 1.0e-300), label="NLTE")
    axis.set_xlabel("photon energy [eV]")
    axis.set_ylabel(r"absorption opacity [cm$^2$ g$^{-1}$]")
    axis.set_title("LTE and NLTE opacity")
    axis.legend()
    axis.grid(True, which="both", alpha=0.25)
    figure.suptitle(
        f"T={args.temperature_K:.3g} K, n_H={args.hydrogen_density_cm3:.3g} cm$^{{-3}}"
    )
    figure.savefig(args.outfile, dpi=180)
    if args.table:
        write_comparison_table(args.table, energy_ev, lte_opacity, nlte_opacity)
    print(f"wrote {args.outfile}")
    if args.table:
        print(f"wrote {args.table}")


if __name__ == "__main__":
    main()
