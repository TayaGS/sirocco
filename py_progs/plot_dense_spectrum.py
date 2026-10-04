#!/usr/bin/env python
"""Plot radial LTE opacity and effective optical depth from dense_spectrum.py."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


PLANCK_ERG_S = 6.62607015e-27
EV2ERGS = 1.602176634e-12
IONIZATION_EDGES_EV = [13.6 / level**2 for level in range(1, 5)]


def read_opacity_table(filename: str):
    data = np.loadtxt(filename, comments="#")
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("opacity table must contain four numeric columns")

    radius_cm = np.unique(data[:, 0])
    frequency_hz = np.unique(data[:, 1])
    expected_rows = len(radius_cm) * len(frequency_hz)
    if len(data) != expected_rows:
        raise ValueError("opacity table does not contain a complete radius-frequency grid")

    order = np.lexsort((data[:, 1], data[:, 0]))
    ordered = data[order]
    kappa_abs = ordered[:, 2].reshape(len(radius_cm), len(frequency_hz))
    tau_eff = ordered[:, 3].reshape(len(radius_cm), len(frequency_hz))
    return radius_cm, frequency_hz, kappa_abs, tau_eff


def read_spectrum_table(filename: str):
    data = np.loadtxt(filename, comments="#")
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("spectrum table must contain two numeric columns")
    return data[:, 0], data[:, 1]


def read_radial_spectra_table(filename: str):
    data = np.loadtxt(filename, comments="#")
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("radial spectra table must contain four numeric columns")
    radius_cm = np.unique(data[:, 0])
    frequency_hz = np.unique(data[:, 1])
    expected_rows = len(radius_cm) * len(frequency_hz)
    if len(data) != expected_rows:
        raise ValueError("radial spectra table does not contain a complete grid")
    order = np.lexsort((data[:, 1], data[:, 0]))
    ordered = data[order]
    luminosity_nu = ordered[:, 2].reshape(len(radius_cm), len(frequency_hz))
    blackbody_luminosity_nu = ordered[:, 3].reshape(len(radius_cm), len(frequency_hz))
    return radius_cm, frequency_hz, luminosity_nu, blackbody_luminosity_nu


def select_log_radius_indices(radius_cm: np.ndarray, count: int = 10) -> np.ndarray:
    """Select at most ``count`` rows evenly spaced in log radius."""
    if radius_cm.ndim != 1 or len(radius_cm) == 0:
        raise ValueError("radius grid must be a non-empty one-dimensional array")
    selected_count = min(count, len(radius_cm))
    target_radii = np.geomspace(radius_cm[0], radius_cm[-1], selected_count)
    indices = np.searchsorted(radius_cm, target_radii)
    indices = np.clip(indices, 0, len(radius_cm) - 1)
    indices = np.unique(indices)
    if len(indices) < selected_count:
        candidates = np.argsort(np.abs(radius_cm[:, None] - target_radii), axis=0).ravel()
        indices = np.unique(np.concatenate((indices, candidates)))[:selected_count]
        indices.sort()
    return indices


def plot_opacity(
    filename: str,
    outfile: str,
    spectrum_filename: str = None,
    radial_spectra_filename: str = None,
):
    radius_cm, frequency_hz, kappa_abs, tau_eff = read_opacity_table(filename)
    plot_radius_cm = radius_cm[:-1]
    plot_kappa_abs = kappa_abs[:-1]
    plot_tau_eff = tau_eff[:-1]
    selected_indices = select_log_radius_indices(plot_radius_cm)
    energy_ev = frequency_hz * PLANCK_ERG_S / EV2ERGS
    positive_tau = np.where(plot_tau_eff > 0.0, plot_tau_eff, np.nan)

    panel_count = 1 + bool(spectrum_filename) + bool(radial_spectra_filename)
    figure, axes = plt.subplots(1, panel_count, figsize=(6.5 * panel_count, 5), constrained_layout=True, squeeze=False)
    axes = axes[0]
    for radius, tau in zip(plot_radius_cm[selected_indices], positive_tau[selected_indices]):
        axes[0].loglog(energy_ev, tau, label=f"r={radius:.1e} cm")
    axes[0].axhline(1.0, color="black", linestyle=":", alpha=0.6)
    for edge in IONIZATION_EDGES_EV:
        axes[0].axvline(edge, color="black", linestyle="--", alpha=0.35)
    axes[0].set_xlabel("photon energy [eV]")
    axes[0].set_ylabel(r"$\tau_{\rm eff}$")
    axes[0].set_title(r"Effective optical depth vs. energy")
    axes[0].legend(fontsize="small")
    axes[0].grid(True, which="both", alpha=0.25)

    if spectrum_filename:
        spectrum_frequency, luminosity_nu = read_spectrum_table(spectrum_filename)
        spectrum_energy = spectrum_frequency * PLANCK_ERG_S / EV2ERGS
        spectrum_values = np.maximum(luminosity_nu, 1.0e-300)
        axes[1].loglog(spectrum_energy, spectrum_values)
        axes[1].set_ylim(np.max(spectrum_values) / 1.0e6, np.max(spectrum_values) * 1.1)
        for edge in IONIZATION_EDGES_EV:
            axes[1].axvline(edge, color="black", linestyle="--", alpha=0.35)
        axes[1].set_xlabel("photon energy [eV]")
        axes[1].set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
        axes[1].set_title("Emergent spectrum")
        axes[1].grid(True, which="both", alpha=0.25)

    if radial_spectra_filename:
        radial_radius, radial_frequency, radial_luminosity, radial_blackbody = read_radial_spectra_table(
            radial_spectra_filename
        )
        if not np.array_equal(radius_cm, radial_radius):
            raise ValueError("opacity and radial spectra tables use different radial grids; regenerate both together")
        if not np.array_equal(frequency_hz, radial_frequency):
            raise ValueError("opacity and radial spectra tables use different frequency grids; regenerate both together")
        radial_energy = radial_frequency * PLANCK_ERG_S / EV2ERGS
        plot_radial_radius = radial_radius[:-1]
        plot_radial_luminosity = radial_luminosity[:-1]
        plot_radial_blackbody = radial_blackbody[:-1]
        radial_indices = select_log_radius_indices(plot_radial_radius)
        radial_axis = axes[-1]
        radial_values = np.maximum(
            np.maximum(plot_radial_luminosity[radial_indices], plot_radial_blackbody[radial_indices]), 1.0e-300
        )
        for radius, luminosity_nu, blackbody_luminosity_nu in zip(
            plot_radial_radius[radial_indices],
            plot_radial_luminosity[radial_indices],
            plot_radial_blackbody[radial_indices],
        ):
            radial_axis.loglog(radial_energy, np.maximum(luminosity_nu, 1.0e-300), label=f"r={radius:.1e} cm")
            radial_axis.loglog(
                radial_energy,
                np.maximum(blackbody_luminosity_nu, 1.0e-300),
                linestyle="--",
                color="black",
                alpha=0.45,
            )
        radial_axis.set_ylim(np.max(radial_values) / 1.0e6, np.max(radial_values) * 1.1)
        for edge in IONIZATION_EDGES_EV:
            radial_axis.axvline(edge, color="black", linestyle="--", alpha=0.35)
        radial_axis.set_xlabel("photon energy [eV]")
        radial_axis.set_ylabel(r"local $L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
        radial_axis.set_title("Spectrum at every radius")
        radial_axis.legend(fontsize="small")
        radial_axis.grid(True, which="both", alpha=0.25)

    figure.savefig(outfile, dpi=180)
    print(f"wrote {outfile}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", help="radial opacity table from dense_spectrum.py")
    parser.add_argument("--outfile", default="dense_spectrum_test.png")
    parser.add_argument("--spectrum-table", help="optional emergent spectrum table")
    parser.add_argument("--radial-spectra-table", help="optional spectra table for every radius")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plot_opacity(
        args.table,
        args.outfile,
        args.spectrum_table,
        args.radial_spectra_table,
    )


if __name__ == "__main__":
    main()