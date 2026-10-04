#!/usr/bin/env python
"""Calculate a single NLTE opacity curve from manually supplied b_i values.

The departure-coefficient file can use either global indices or physical IDs.

Global-index format:

    level_index  b_i

Physical-ID format for the tuple IDs used by ``LevTop``/``LevMacro``:

    z  istate  level_id_part1  level_id_part2  b_i

Blank lines and lines beginning with ``#`` are ignored. ``level_index`` is the
zero-based level index used by the parsed SIROCCO atomic data. Levels omitted
from the file use ``--default-b``.

Example:

    python py_progs/nlte_opacity.py zdata/h_only.dat 10000 1e12 \
        --departure-coefficients b_values.txt --outfile nlte_opacity.txt
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from dense_spectrum import (
    load_sirocco_hydrogen_data,
    nlte_opacity_for_state,
)


def read_departure_coefficients(filename: str, level_count: int, default_b: float, levels=None) -> np.ndarray:
    """Read sparse departure coefficients and return a complete b_i array."""
    if not np.isfinite(default_b) or default_b <= 0.0:
        raise ValueError("default departure coefficient must be positive and finite")

    coefficients = np.full(level_count, default_b, dtype=float)
    data = np.loadtxt(filename, comments="#", ndmin=2)
    if data.size == 0:
        return coefficients
    if data.shape[1] not in (2, 4, 5):
        raise ValueError("coefficient file must contain 2, 4, or 5 columns")

    physical_lookup = None
    if data.shape[1] in (4, 5):
        if levels is None:
            raise ValueError("atomic levels are required for physical-ID coefficients")
        physical_lookup = {
            (level["z"], level["istate"], level["level_id"]): index
            for index, level in enumerate(levels)
        }

    for row in data:
        if data.shape[1] == 2:
            level_index_value, coefficient = row
            if not np.isfinite(level_index_value) or level_index_value != int(level_index_value):
                raise ValueError("level indices must be finite integers")
            level_index = int(level_index_value)
        elif data.shape[1] == 4:
            z, istate, level_id, coefficient = row
            key = (int(z), int(istate), int(level_id))
            if key not in physical_lookup:
                raise ValueError(f"physical level ID {key} was not found")
            level_index = physical_lookup[key]
        else:
            z, istate, level_id_part1, level_id_part2, coefficient = row
            key = (int(z), int(istate), (int(level_id_part1), int(level_id_part2)))
            if key not in physical_lookup:
                raise ValueError(f"physical level ID {key} was not found")
            level_index = physical_lookup[key]
        if level_index < 0 or level_index >= level_count:
            raise ValueError(f"level index {level_index} is outside 0..{level_count - 1}")
        if not np.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError(f"b_i for level {level_index} must be positive and finite")
        coefficients[level_index] = coefficient

    return coefficients


def write_opacity_table(filename: str, frequency_hz: np.ndarray, opacity: np.ndarray) -> None:
    """Write frequency, energy, and total opacity columns."""
    energy_ev = frequency_hz * 6.62607015e-27 / 1.602176634e-12
    with open(filename, "w") as handle:
        handle.write("# frequency_hz energy_ev nlte_opacity_cm2_g\n")
        for frequency, energy, value in zip(frequency_hz, energy_ev, opacity):
            handle.write(f"{frequency:.8e} {energy:.8e} {value:.8e}\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("masterfile", help="SIROCCO atomic-data master file")
    parser.add_argument("temperature_K", type=float, help="gas temperature in K")
    parser.add_argument("hydrogen_density_cm3", type=float, help="total hydrogen density in cm^-3")
    parser.add_argument(
        "--departure-coefficients",
        required=True,
        help="two-column file containing level_index and b_i",
    )
    parser.add_argument("--default-b", type=float, default=1.0, help="b_i for omitted levels (default: 1)")
    parser.add_argument("--emin", type=float, default=1.0e-3, help="minimum photon energy in eV")
    parser.add_argument("--emax", type=float, default=1.0e4, help="maximum photon energy in eV")
    parser.add_argument("--npoints", type=int, default=1200, help="number of logarithmic energy points")
    parser.add_argument("--outfile", default="nlte_opacity.txt", help="output opacity table")
    return parser.parse_args(argv)


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
    energy_ev = np.geomspace(args.emin, args.emax, args.npoints)
    frequency_hz = energy_ev * 1.602176634e-12 / 6.62607015e-27
    opacity = nlte_opacity_for_state(
        args.masterfile,
        args.temperature_K,
        args.hydrogen_density_cm3,
        frequency_hz,
        departure_coefficients,
        atomic_data,
    )
    write_opacity_table(args.outfile, frequency_hz, opacity)
    print(f"loaded {len(atomic_data['levels'])} levels")
    print(f"wrote {args.outfile}")


if __name__ == "__main__":
    main()
