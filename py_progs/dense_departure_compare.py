#!/usr/bin/env python
"""Compare LTE tau_eff with tau_eff including dilution-factor departure coefficients.

Reads the LTE opacity and radial spectra tables written by dense_spectrum.py
(run with --opacity --spectrum, without --spectrum-nlte). The dilution factor is
W(r, nu) = L_nu / L_nu,BB, so the radiation field is J_nu = W * B_nu(T). Each
level gets b_i = Gamma_i* / Gamma_i from that field in a single pass (no
iteration), the departure-scaled populations give a new kappa_abs, and tau_eff
is recomputed and compared with the LTE table.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from dense_spectrum import (
    DEFAULT_KAPPA_T,
    HYDROGEN_MASS,
    RadialGrid,
    density_profile,
    departure_coefficients_from_spectra,
    effective_optical_depth,
    load_sirocco_hydrogen_data,
    nlte_opacity_for_state,
    outward_thomson_optical_depth,
    temperature_profile,
    write_departure_table,
)
from plot_dense_spectrum import (
    EV2ERGS,
    PLANCK_ERG_S,
    read_opacity_table,
    read_radial_spectra_table,
    select_log_radius_indices,
)


def dilution_factor(local_luminosity_nu: np.ndarray, blackbody_luminosity_nu: np.ndarray) -> np.ndarray:
    """Return W = L_nu / L_nu,BB, with zero where the blackbody reference is zero."""
    return np.divide(
        local_luminosity_nu,
        blackbody_luminosity_nu,
        out=np.zeros_like(local_luminosity_nu),
        where=blackbody_luminosity_nu > 0.0,
    )


def tau_eff_with_departures(
    masterfile: str,
    grid: RadialGrid,
    temperature_K: np.ndarray,
    tau_T: np.ndarray,
    frequency_hz: np.ndarray,
    dilution: np.ndarray,
    blackbody_luminosity_nu: np.ndarray,
    kappa_abs_lte: np.ndarray,
    kappa_T: float,
) -> tuple:
    """Return kappa_abs, tau_eff, b_i, Gamma_i* and Gamma_i from one dilution-factor pass."""
    atomic_data = load_sirocco_hydrogen_data(masterfile)
    nr = len(grid.radius_cm)
    kappa_abs = kappa_abs_lte.copy()
    b_values = np.ones((nr, len(atomic_data["levels"])))
    gamma_lte = np.zeros_like(b_values)
    gamma = np.zeros_like(b_values)
    # The outermost cell has tau_T = 0 and is left at its LTE values.
    for index in range(nr - 1):
        diluted_luminosity = dilution[index] * blackbody_luminosity_nu[index]
        b_values[index], gamma_lte[index], gamma[index] = departure_coefficients_from_spectra(
            frequency_hz, diluted_luminosity, blackbody_luminosity_nu[index], atomic_data
        )
        kappa_abs[index] = nlte_opacity_for_state(
            masterfile,
            temperature_K[index],
            grid.hydrogen_density_cm3[index],
            frequency_hz,
            b_values[index],
            atomic_data,
        )
    tau_eff = effective_optical_depth(kappa_abs, tau_T, kappa_T)
    return kappa_abs, tau_eff, b_values, gamma_lte, gamma, atomic_data["levels"]


def write_comparison_table(filename, radius_cm, frequency_hz, kappa_lte, kappa_nlte, tau_lte, tau_nlte):
    """Write LTE and departure-corrected opacity and tau_eff as a flat table."""
    with open(filename, "w") as handle:
        handle.write("# radius_cm frequency_hz kappa_lte kappa_dep tau_eff_lte tau_eff_dep\n")
        for r, k_l, k_d, t_l, t_d in zip(radius_cm, kappa_lte, kappa_nlte, tau_lte, tau_nlte):
            for nu, a, b, c, d in zip(frequency_hz, k_l, k_d, t_l, t_d):
                handle.write(f"{r:.8e} {nu:.8e} {a:.8e} {b:.8e} {c:.8e} {d:.8e}\n")


def plot_comparison(
    outfile, radius_cm, frequency_hz, dilution, b_values, levels, tau_lte, tau_nlte
):
    energy_ev = frequency_hz * PLANCK_ERG_S / EV2ERGS
    plot_radius = radius_cm[:-1]
    selected = select_log_radius_indices(plot_radius)
    ratio = np.divide(tau_nlte, tau_lte, out=np.full_like(tau_lte, np.nan), where=tau_lte > 0.0)

    figure, axes = plt.subplots(1, 4, figsize=(24, 5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(selected)))
    for color, index in zip(colors, selected):
        label = f"r={plot_radius[index]:.1e} cm"
        axes[0].loglog(energy_ev, np.maximum(dilution[index], 1.0e-30), color=color, label=label)
        axes[2].loglog(energy_ev, np.where(tau_lte[index] > 0.0, tau_lte[index], np.nan), color=color, label=label)
        axes[2].loglog(
            energy_ev, np.where(tau_nlte[index] > 0.0, tau_nlte[index], np.nan), color=color, linestyle="--"
        )
        axes[3].semilogx(energy_ev, ratio[index], color=color, label=label)
    # W diverges in the Wien tail where the local blackbody is ~0, so the range is fixed.
    axes[0].set_ylim(1.0e-6, 1.0e6)
    axes[0].set_xlabel("photon energy [eV]")
    axes[0].set_ylabel(r"dilution factor $W=L_\nu/L_{\nu,\rm BB}$")
    axes[0].set_title("Dilution factor")
    axes[0].legend(fontsize="small")

    hydrogen_levels = [i for i, level in enumerate(levels) if level["z"] == 1 and level["istate"] == 1]
    for level_index in hydrogen_levels[:6]:
        axes[1].loglog(plot_radius, b_values[:-1, level_index], label=f"level {levels[level_index]['level_id']}")
    axes[1].axhline(1.0, color="black", linestyle=":", alpha=0.6)
    axes[1].set_xlabel("radius [cm]")
    axes[1].set_ylabel(r"$b_i=\Gamma_i^*/\Gamma_i$")
    axes[1].set_title("Departure coefficients")
    axes[1].legend(fontsize="small")

    axes[2].axhline(1.0, color="black", linestyle=":", alpha=0.6)
    axes[2].set_xlabel("photon energy [eV]")
    axes[2].set_ylabel(r"$\tau_{\rm eff}$")
    axes[2].set_title(r"$\tau_{\rm eff}$: LTE (solid) vs departures (dashed)")

    axes[3].axhline(1.0, color="black", linestyle=":", alpha=0.6)
    axes[3].set_yscale("log")
    axes[3].set_xlabel("photon energy [eV]")
    axes[3].set_ylabel(r"$\tau_{\rm eff}^{\rm dep}/\tau_{\rm eff}^{\rm LTE}$")
    axes[3].set_title("Ratio")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    figure.savefig(outfile, dpi=150)
    print(f"wrote {outfile}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opacity_table", help="LTE opacity table from dense_spectrum.py --opacity")
    parser.add_argument("radial_spectra_table", help="radial spectra table from dense_spectrum.py --spectrum")
    parser.add_argument("--masterfile", default=os.path.join("zdata", "h20.dat"))
    parser.add_argument("--luminosity", type=float, required=True, help="source luminosity in erg s^-1")
    parser.add_argument("--alpha", type=float, required=True, help="density power-law index")
    parser.add_argument("--rho16", type=float, required=True, help="density at 1e16 cm in g cm^-3")
    parser.add_argument("--kappa-t", type=float, default=DEFAULT_KAPPA_T, help="Thomson opacity in cm^2 g^-1")
    parser.add_argument("--plot", default="dense_departure_compare.png", help="output figure")
    parser.add_argument("--out", default=None, help="optional output table of LTE and departure tau_eff")
    parser.add_argument("--departure-out", default=None, help="optional output table of b_i and photoionization rates")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    radius_cm, frequency_hz, kappa_lte, tau_lte = read_opacity_table(args.opacity_table)
    spectra_radius, spectra_frequency, local_luminosity, blackbody_luminosity = read_radial_spectra_table(
        args.radial_spectra_table
    )
    if not (np.array_equal(radius_cm, spectra_radius) and np.array_equal(frequency_hz, spectra_frequency)):
        raise ValueError("opacity and radial spectra tables use different grids; regenerate both together")

    density_g_cm3 = density_profile(radius_cm, args.alpha, args.rho16)
    grid = RadialGrid(radius_cm, density_g_cm3, density_g_cm3 / HYDROGEN_MASS)
    tau_T = outward_thomson_optical_depth(radius_cm, density_g_cm3, args.kappa_t)
    temperature_K = temperature_profile(args.luminosity, grid, tau_T, args.alpha)
    dilution = dilution_factor(local_luminosity, blackbody_luminosity)
    kappa_dep, tau_dep, b_values, gamma_lte, gamma, levels = tau_eff_with_departures(
        args.masterfile,
        grid,
        temperature_K,
        tau_T,
        frequency_hz,
        dilution,
        blackbody_luminosity,
        kappa_lte,
        args.kappa_t,
    )

    # The last radius has tau_T = 0 and is excluded, as in plot_dense_spectrum.py.
    valid = tau_lte[:-1] > 0.0
    ratio = tau_dep[:-1][valid] / tau_lte[:-1][valid]
    print(f"tau_eff(dep)/tau_eff(LTE): min {ratio.min():.4e}, median {np.median(ratio):.4e}, max {ratio.max():.4e}")
    print(f"tau_eff range, LTE: {tau_lte[:-1].min():.4e} - {tau_lte[:-1].max():.4e}")
    print(f"tau_eff range, dep: {tau_dep[:-1].min():.4e} - {tau_dep[:-1].max():.4e}")
    hydrogen_levels = [i for i, level in enumerate(levels) if level["z"] == 1 and level["istate"] == 1]
    for level_index in hydrogen_levels[:3]:
        column = b_values[:-1, level_index]
        print(f"b (level {levels[level_index]['level_id']}): {column.min():.4e} - {column.max():.4e}")

    if args.out:
        write_comparison_table(args.out, radius_cm, frequency_hz, kappa_lte, kappa_dep, tau_lte, tau_dep)
        print(f"comparison table: {args.out}")
    if args.departure_out:
        write_departure_table(args.departure_out, radius_cm, levels, b_values, gamma_lte, gamma)
        print(f"departure table: {args.departure_out}")
    plot_comparison(args.plot, radius_cm, frequency_hz, dilution[:-1], b_values, levels, tau_lte[:-1], tau_dep[:-1])


if __name__ == "__main__":
    main()
