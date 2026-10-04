#!/usr/bin/env python
"""Foundations for a semi-analytic dense-gas spectrum model.

This module currently provides the logarithmic radial and frequency grids,
the requested power-law density profile, and loading of SIROCCO atomic data.
Temperature, opacity, and radiation transfer are added in later stages.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np

from plot_lte_opacity import (
    build_lte_state,
    compute_continuum_components,
    compute_line_component,
    interpolate_photo_xsection,
    load_atomic_records,
    parse_atomic_data,
)


CM_PER_REFERENCE_RADIUS = 1.0e16
HYDROGEN_MASS = 1.6735575e-24
BOLTZMANN_ERG_K = 1.380649e-16
PLANCK_ERG_S = 6.62607015e-27
ELECTRON_MASS = 9.1093837e-28
IONIZATION_ENERGY_H_ERG = 13.6 * 1.602176634e-12
STEFAN_BOLTZMANN = 5.670374419e-5
DEFAULT_KAPPA_T = 0.4


@dataclass
class RadialGrid:
    """Logarithmic radial grid and local density quantities."""

    radius_cm: np.ndarray
    density_g_cm3: np.ndarray
    hydrogen_density_cm3: np.ndarray


def logarithmic_grid(start: float, stop: float, count: int) -> np.ndarray:
    """Return ``count`` logarithmically spaced values from ``start`` to ``stop``."""
    if start <= 0.0 or stop <= 0.0:
        raise ValueError("logarithmic grid bounds must be positive")
    if stop < start:
        raise ValueError("logarithmic grid stop must not be below start")
    if count < 2:
        raise ValueError("logarithmic grid requires at least two points")
    return np.geomspace(start, stop, count)


def density_profile(radius_cm: np.ndarray, alpha: float, rho16: float) -> np.ndarray:
    """Evaluate rho(r) = rho16 * (r / 1e16 cm)^(-alpha)."""
    radius_cm = np.asarray(radius_cm, dtype=float)
    if np.any(radius_cm <= 0.0) or rho16 <= 0.0:
        raise ValueError("radii and rho16 must be positive")
    return rho16 * (radius_cm / CM_PER_REFERENCE_RADIUS) ** (-alpha)


def build_radial_grid(rin_cm: float, rout_cm: float, alpha: float, rho16: float, nr: int = 256) -> RadialGrid:
    """Build a logarithmic radial grid with hydrogen number density."""
    radius_cm = logarithmic_grid(rin_cm, rout_cm, nr)
    density_g_cm3 = density_profile(radius_cm, alpha, rho16)
    return RadialGrid(radius_cm, density_g_cm3, density_g_cm3 / HYDROGEN_MASS)


def frequency_grid(nu_min_hz: float, nu_max_hz: float, nnu: int = 512) -> np.ndarray:
    """Build the logarithmic frequency grid used by later transfer stages."""
    return logarithmic_grid(nu_min_hz, nu_max_hz, nnu)


def saha_ionized_fraction(temperature_K: float, hydrogen_density_cm3: float) -> float:
    """Return the LTE hydrogen ionized fraction from the ground-state Saha equation."""
    if temperature_K <= 0.0 or hydrogen_density_cm3 <= 0.0:
        raise ValueError("temperature and hydrogen density must be positive")
    saha_factor = 2.0 * (2.0 * np.pi * ELECTRON_MASS * BOLTZMANN_ERG_K * temperature_K / PLANCK_ERG_S**2) ** 1.5
    ratio = saha_factor * np.exp(-IONIZATION_ENERGY_H_ERG / (BOLTZMANN_ERG_K * temperature_K)) / hydrogen_density_cm3
    return float(2.0 * ratio / (ratio + np.sqrt(ratio**2 + 4.0 * ratio)))


def outward_thomson_optical_depth(radius_cm, density_g_cm3, kappa_T=DEFAULT_KAPPA_T):
    """Integrate Thomson optical depth from every radius to the outer edge."""
    radius_cm = np.asarray(radius_cm, dtype=float)
    density_g_cm3 = np.asarray(density_g_cm3, dtype=float)
    if radius_cm.ndim != 1 or density_g_cm3.shape != radius_cm.shape:
        raise ValueError("radius and density must be one-dimensional arrays of equal length")
    if len(radius_cm) < 2 or np.any(np.diff(radius_cm) <= 0.0):
        raise ValueError("radius must contain at least two increasing points")
    if np.any(density_g_cm3 <= 0.0) or kappa_T <= 0.0:
        raise ValueError("density and kappa_T must be positive")
    tau_T = np.zeros_like(radius_cm)
    shell_integral = kappa_T * 0.5 * (density_g_cm3[:-1] + density_g_cm3[1:]) * np.diff(radius_cm)
    tau_T[:-1] = np.cumsum(shell_integral[::-1])[::-1]
    return tau_T


def radiative_temperature_profile(luminosity_erg_s, radius_cm, tau_T, alpha):
    """Evaluate the diffusion temperature law with the free-streaming limit T^4 = L / (16 pi r^2 sigma)."""
    radius_cm = np.asarray(radius_cm, dtype=float)
    tau_T = np.asarray(tau_T, dtype=float)
    if luminosity_erg_s <= 0.0:
        raise ValueError("luminosity must be positive")
    if alpha <= 1.0:
        raise ValueError("the supplied temperature relation requires alpha > 1")
    if radius_cm.shape != tau_T.shape or np.any(radius_cm <= 0.0) or np.any(tau_T < 0.0):
        raise ValueError("radius and tau_T must be matching positive arrays")

    prefactor = 0.75 * (alpha - 1.0) / (alpha + 1.0)
    return (
        luminosity_erg_s / (4.0 * np.pi * radius_cm**2 * STEFAN_BOLTZMANN) * (prefactor * tau_T + 0.25)
    ) ** 0.25


def freeze_temperature_at_half_ionization(radiative_temperature_K, ionized_fraction) -> np.ndarray:
    """Hold T constant outward of the first cell where hydrogen is at most 50% ionized, if any.

    ``ionized_fraction(index, temperature_K)`` returns the hydrogen ionized fraction in that cell.
    """
    temperature_K = np.array(radiative_temperature_K, dtype=float)
    for index, temperature in enumerate(temperature_K):
        if ionized_fraction(index, temperature) <= 0.5:
            temperature_K[index:] = temperature
            break
    return temperature_K


def temperature_profile(luminosity_erg_s, radial_grid: RadialGrid, tau_T, alpha) -> np.ndarray:
    """LTE temperature: energy-density law, held constant once the local Saha ionization reaches 50%."""
    radiative_temperature_K = radiative_temperature_profile(
        luminosity_erg_s, radial_grid.radius_cm, tau_T, alpha
    )
    return freeze_temperature_at_half_ionization(
        radiative_temperature_K,
        lambda index, temperature: saha_ionized_fraction(
            temperature, radial_grid.hydrogen_density_cm3[index]
        ),
    )


def hydrogen_ionized_fraction(
    masterfile: str,
    temperature_K: float,
    hydrogen_density_cm3: float,
    departure_coefficients: np.ndarray,
) -> float:
    """Return 1 - n(H I)/n_H with the LTE level populations scaled by b_i."""
    level_populations = build_lte_state(masterfile, temperature_K, hydrogen_density_cm3)[5]
    neutral_hydrogen = sum(
        population * departure_coefficients[key[2]]
        for key, population in level_populations.items()
        if key[0] == 1 and key[1] == 1
    )
    return 1.0 - neutral_hydrogen / hydrogen_density_cm3


def lte_opacity_for_state(
    masterfile: str,
    temperature_K: float,
    hydrogen_density_cm3: float,
    frequency_hz: np.ndarray,
    atomic_data: Dict[str, object] = None,
) -> np.ndarray:
    """Calculate LTE absorption opacity for one radial state in cm^2 g^-1."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if np.any(frequency_hz <= 0.0):
        raise ValueError("frequencies must be positive")

    if atomic_data is None:
        atomic_data = load_sirocco_hydrogen_data(masterfile)
    photo_records = atomic_data["photo_records"]
    gaunt_records = atomic_data["gaunt_records"]
    energy_grid_ev = frequency_hz * PLANCK_ERG_S / 1.602176634e-12
    state = build_lte_state(masterfile, temperature_K, hydrogen_density_cm3)
    (
        ion_results,
        ion_density,
        _ne,
        level_lookup,
        level_id_lookup,
        level_populations,
        _element_mass,
        mass_density,
    ) = state
    if mass_density <= 0.0:
        raise ValueError("LTE state returned a non-positive mass density")
    kappa_bf, kappa_ff = compute_continuum_components(
        energy_grid_ev,
        temperature_K,
        ion_results,
        ion_density,
        photo_records,
        level_lookup,
        level_id_lookup,
        level_populations,
        gaunt_records,
    )
    return (kappa_bf + kappa_ff) / mass_density


def nlte_opacity_for_state(
    masterfile: str,
    temperature_K: float,
    hydrogen_density_cm3: float,
    frequency_hz: np.ndarray,
    departure_coefficients: np.ndarray,
    atomic_data: Dict[str, object] = None,
) -> np.ndarray:
    """Calculate hydrogen NLTE opacity using level-aligned departure coefficients."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    departure_coefficients = np.asarray(departure_coefficients, dtype=float)
    if np.any(frequency_hz <= 0.0):
        raise ValueError("frequencies must be positive")
    if departure_coefficients.ndim != 1 or np.any(~np.isfinite(departure_coefficients)):
        raise ValueError("departure coefficients must be a finite one-dimensional array")
    if np.any(departure_coefficients <= 0.0):
        raise ValueError("departure coefficients must be positive")

    if atomic_data is None:
        atomic_data = load_sirocco_hydrogen_data(masterfile)
    photo_records = atomic_data["photo_records"]
    gaunt_records = atomic_data["gaunt_records"]
    energy_grid_ev = frequency_hz * PLANCK_ERG_S / 1.602176634e-12
    state = build_lte_state(masterfile, temperature_K, hydrogen_density_cm3)
    (
        ion_results,
        ion_density,
        _ne_lte,
        level_lookup,
        level_id_lookup,
        level_populations_lte,
        _element_mass,
        mass_density,
    ) = state
    level_indices = [key[2] for key in level_populations_lte]
    if level_indices and max(level_indices) >= len(departure_coefficients):
        raise ValueError("departure coefficient array is shorter than the parsed level array")

    level_populations = {
        key: population * departure_coefficients[key[2]]
        for key, population in level_populations_lte.items()
    }
    neutral_hydrogen = sum(
        population for (z, istate, _), population in level_populations.items() if z == 1 and istate == 1
    )
    electron_density = hydrogen_density_cm3 - neutral_hydrogen
    if electron_density <= 0.0:
        raise ValueError("departure-scaled hydrogen populations leave no positive electron density")

    ion_density = dict(ion_density)
    if (1, 2) in ion_density:
        ion_density[(1, 2)] = electron_density

    kappa_bf, kappa_ff = compute_continuum_components(
        energy_grid_ev,
        temperature_K,
        ion_results,
        ion_density,
        photo_records,
        level_lookup,
        level_id_lookup,
        level_populations,
        gaunt_records,
        electron_density,
    )
    return (kappa_bf + kappa_ff) / mass_density


def effective_optical_depth(kappa_abs: np.ndarray, tau_T: np.ndarray, kappa_T: float) -> np.ndarray:
    """Return tau_eff = sqrt(tau_abs * (tau_abs + tau_T)) with tau_abs = kappa_abs / kappa_T * tau_T."""
    tau_abs = np.maximum(kappa_abs, 0.0) / kappa_T * tau_T[:, np.newaxis]
    return np.sqrt(tau_abs * (tau_abs + tau_T[:, np.newaxis]))


def radial_lte_opacity(
    masterfile: str,
    radial_grid: RadialGrid,
    temperature_K: np.ndarray,
    tau_T: np.ndarray,
    frequency_hz: np.ndarray,
    kappa_T: float = DEFAULT_KAPPA_T,
) -> tuple:
    """Calculate kappa_abs and tau_eff for every radial-frequency cell."""
    temperature_K = np.asarray(temperature_K, dtype=float)
    tau_T = np.asarray(tau_T, dtype=float)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if temperature_K.shape != radial_grid.radius_cm.shape or tau_T.shape != temperature_K.shape:
        raise ValueError("temperature and tau_T must match the radial grid")
    if kappa_T <= 0.0:
        raise ValueError("kappa_T must be positive")

    atomic_data = load_sirocco_hydrogen_data(masterfile)
    kappa_abs = np.empty((len(radial_grid.radius_cm), len(frequency_hz)))
    for index, (temperature, hydrogen_density) in enumerate(
        zip(temperature_K, radial_grid.hydrogen_density_cm3)
    ):
        kappa_abs[index] = lte_opacity_for_state(
            masterfile,
            temperature,
            hydrogen_density,
            frequency_hz,
            atomic_data,
        )
    tau_eff = effective_optical_depth(kappa_abs, tau_T, kappa_T)
    return kappa_abs, tau_eff


def photoionization_rate_from_luminosity(
    frequency_hz: np.ndarray,
    luminosity_nu: np.ndarray,
    photo_record,
) -> float:
    """Integrate a bound-free rate using L_nu and one photoionization record."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    luminosity_nu = np.asarray(luminosity_nu, dtype=float)
    if frequency_hz.ndim != 1 or luminosity_nu.shape != frequency_hz.shape:
        raise ValueError("frequency and luminosity arrays must have the same one-dimensional shape")
    if np.any(frequency_hz <= 0.0) or np.any(luminosity_nu < 0.0):
        raise ValueError("frequency must be positive and luminosity must be non-negative")

    energy_ev = frequency_hz * PLANCK_ERG_S / 1.602176634e-12
    sigma = interpolate_photo_xsection(photo_record, energy_ev)
    threshold_ev = max(photo_record.threshold_ev, energy_ev[0])
    valid = (energy_ev >= threshold_ev) & (sigma > 0.0)
    if np.count_nonzero(valid) < 2:
        return 0.0

    photon_rate_density = luminosity_nu[valid] / (energy_ev[valid] * 1.602176634e-12)
    return float(np.trapezoid(photon_rate_density * sigma[valid], frequency_hz[valid]))


def _photo_record_level_index(photo_record, level_lookup, level_id_lookup):
    """Resolve a photoionization record to its parsed global level index."""
    level_index = None
    if photo_record.lower_level_id is not None:
        level_index = level_id_lookup.get(
            (photo_record.z, photo_record.istate, photo_record.lower_level_id)
        )
    if level_index is None:
        from plot_lte_opacity import find_matching_threshold_level

        level_index = find_matching_threshold_level(
            photo_record.z, photo_record.istate, photo_record.threshold_ev, level_lookup
        )
    return level_index


def departure_coefficients_from_spectra(
    frequency_hz: np.ndarray,
    luminosity_nu: np.ndarray,
    blackbody_luminosity_nu: np.ndarray,
    atomic_data: Dict[str, object],
) -> tuple:
    """Calculate b_i = Gamma_i^*/Gamma_i for one radial cell."""
    levels = atomic_data["levels"]
    coefficients = np.ones(len(levels), dtype=float)
    gamma = np.zeros(len(levels), dtype=float)
    gamma_lte = np.zeros(len(levels), dtype=float)
    level_lookup = {}
    level_id_lookup = {}
    for index, level in enumerate(levels):
        level_lookup.setdefault((level["z"], level["istate"]), []).append((index, level))
        level_id_lookup[(level["z"], level["istate"], level["level_id"])] = index

    mapped_records = set()
    for photo_record in atomic_data["photo_records"]:
        level_index = _photo_record_level_index(photo_record, level_lookup, level_id_lookup)
        if level_index is None:
            continue
        mapped_records.add(level_index)
        gamma[level_index] += photoionization_rate_from_luminosity(
            frequency_hz, luminosity_nu, photo_record
        )
        gamma_lte[level_index] += photoionization_rate_from_luminosity(
            frequency_hz, blackbody_luminosity_nu, photo_record
        )

    for level_index in mapped_records:
        if gamma[level_index] <= 0.0 or gamma_lte[level_index] <= 0.0:
            raise ValueError(f"zero photoionization rate for level index {level_index}")
        coefficients[level_index] = gamma_lte[level_index] / gamma[level_index]

    return coefficients, gamma_lte, gamma


def _nlte_temperature(masterfile, radial_grid, radiative_temperature_K, b_values) -> np.ndarray:
    """Freeze T where the b_i-scaled level populations first give <= 50% hydrogen ionization."""
    return freeze_temperature_at_half_ionization(
        radiative_temperature_K,
        lambda index, temperature: hydrogen_ionized_fraction(
            masterfile, temperature, radial_grid.hydrogen_density_cm3[index], b_values[index]
        ),
    )


def radial_nlte_spectra(
    masterfile: str,
    radial_grid: RadialGrid,
    radiative_temperature_K: np.ndarray,
    tau_T: np.ndarray,
    frequency_hz: np.ndarray,
    kappa_T: float = DEFAULT_KAPPA_T,
    max_iterations: int = 20,
    tolerance: float = None,
    damping: float = 0.5,
) -> tuple:
    """Iterate local spectra, photoionization rates, b_i, temperature, and NLTE opacity.

    ``radiative_temperature_K`` is the unfloored energy-density temperature. It is held
    constant outward of the first cell where hydrogen is at most 50% ionized, which is
    re-evaluated with the current b_i on every iteration.
    ``damping`` in (0, 1] is the log-space step fraction taken toward the new b_i.
    ``tolerance`` defaults to 0.3 * Delta nu / nu of the frequency grid.
    """
    if tolerance is None:
        tolerance = 0.3 * (np.max(frequency_hz[1:] / frequency_hz[:-1]) - 1.0)
    if max_iterations < 1 or tolerance <= 0.0 or not 0.0 < damping <= 1.0:
        raise ValueError("NLTE iteration controls are invalid")
    atomic_data = load_sirocco_hydrogen_data(masterfile)
    b_values = np.ones((len(radial_grid.radius_cm), len(atomic_data["levels"])))
    temperature_K = _nlte_temperature(masterfile, radial_grid, radiative_temperature_K, b_values)
    kappa_abs, tau_eff = radial_lte_opacity(
        masterfile, radial_grid, temperature_K, tau_T, frequency_hz, kappa_T
    )
    local_spectra, blackbody_spectra = radial_spectra(
        1.0, radial_grid, temperature_K, tau_T, tau_eff, frequency_hz
    )
    active_cells = range(max(1, len(radial_grid.radius_cm) - 1))

    for iteration in range(1, max_iterations + 1):
        next_b = b_values.copy()
        gamma_lte = np.zeros_like(b_values)
        gamma = np.zeros_like(b_values)
        for index in active_cells:
            next_b[index], gamma_lte[index], gamma[index] = departure_coefficients_from_spectra(
                frequency_hz, local_spectra[index], blackbody_spectra[index], atomic_data
            )

        relative_change = np.max(
            np.abs(next_b - b_values) / np.maximum(np.abs(next_b), 1.0e-30)
        )
        b_values = b_values ** (1.0 - damping) * next_b**damping
        new_temperature_K = _nlte_temperature(masterfile, radial_grid, radiative_temperature_K, b_values)
        relative_change = max(
            relative_change, np.max(np.abs(new_temperature_K - temperature_K) / new_temperature_K)
        )
        temperature_K = new_temperature_K
        for index in active_cells:
            temperature = temperature_K[index]
            hydrogen_density = radial_grid.hydrogen_density_cm3[index]
            kappa_abs[index] = nlte_opacity_for_state(
                masterfile,
                temperature,
                hydrogen_density,
                frequency_hz,
                b_values[index],
                atomic_data,
            )
        tau_eff = effective_optical_depth(kappa_abs, tau_T, kappa_T)
        local_spectra, blackbody_spectra = radial_spectra(
            1.0, radial_grid, temperature_K, tau_T, tau_eff, frequency_hz
        )
        if relative_change < tolerance:
            break

    else:
        raise RuntimeError(
            f"NLTE population iteration did not converge after {max_iterations} iterations"
        )

    return (
        local_spectra,
        blackbody_spectra,
        kappa_abs,
        tau_eff,
        b_values,
        gamma_lte,
        gamma,
        iteration,
        temperature_K,
    )


def write_radial_opacity_table(filename: str, radius_cm, frequency_hz, kappa_abs, tau_eff):
    """Write radial opacity and effective optical depth as a flat ASCII table."""
    with open(filename, "w") as handle:
        handle.write("# radius_cm frequency_hz kappa_abs_cm2_g tau_eff\n")
        for radius, row_kappa, row_tau in zip(radius_cm, kappa_abs, tau_eff):
            for frequency, kappa, tau in zip(frequency_hz, row_kappa, row_tau):
                handle.write(f"{radius:.8e} {frequency:.8e} {kappa:.8e} {tau:.8e}\n")


def planck_nu(frequency_hz: np.ndarray, temperature_K: float) -> np.ndarray:
    """Return the Planck function in erg s^-1 cm^-2 Hz^-1 sr^-1."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if np.any(frequency_hz <= 0.0) or temperature_K <= 0.0:
        raise ValueError("frequency and temperature must be positive")
    exponent = PLANCK_ERG_S * frequency_hz / (BOLTZMANN_ERG_K * temperature_K)
    result = np.zeros_like(frequency_hz)
    valid = exponent < 700.0
    valid_exponent = exponent[valid]
    log_denominator = valid_exponent + np.log1p(-np.exp(-valid_exponent))
    log_numerator = (
        np.log(2.0 * PLANCK_ERG_S / 2.99792458e10**2)
        + 3.0 * np.log(frequency_hz[valid])
    )
    result[valid] = np.exp(log_numerator - log_denominator)
    return result


def normalized_blackbody_luminosity(
    luminosity_erg_s: float,
    frequency_hz: np.ndarray,
    temperature_K: float,
) -> np.ndarray:
    """Return a blackbody-shaped L_nu normalized to the supplied luminosity."""
    if luminosity_erg_s <= 0.0:
        raise ValueError("luminosity must be positive")
    spectrum = np.pi * planck_nu(frequency_hz, temperature_K)
    normalization = np.trapezoid(spectrum, frequency_hz)
    if normalization <= 0.0:
        raise ValueError("frequency grid has zero blackbody power")
    return luminosity_erg_s * spectrum / normalization


def emergent_spectrum(
    luminosity_erg_s: float,
    radial_grid: RadialGrid,
    temperature_K: np.ndarray,
    kappa_abs: np.ndarray,
    tau_eff: np.ndarray,
    frequency_hz: np.ndarray,
    kappa_T: float = DEFAULT_KAPPA_T,
) -> np.ndarray:
    """Form an approximate emergent L_nu from diluted and thermalized radiation.

    The central blackbody is attenuated by the outward effective optical depth.
    Each shell contributes thermal emission using the effective absorption
    coefficient and is attenuated by the effective depth above that shell.
    """
    temperature_K = np.asarray(temperature_K, dtype=float)
    kappa_abs = np.asarray(kappa_abs, dtype=float)
    tau_eff = np.asarray(tau_eff, dtype=float)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    expected_shape = (len(radial_grid.radius_cm), len(frequency_hz))
    if temperature_K.shape != radial_grid.radius_cm.shape:
        raise ValueError("temperature must match the radial grid")
    if kappa_abs.shape != expected_shape or tau_eff.shape != expected_shape:
        raise ValueError("opacity arrays must have shape (nr, nnu)")
    if kappa_T <= 0.0:
        raise ValueError("kappa_T must be positive")

    radii = radial_grid.radius_cm
    shell_width = np.empty_like(radii)
    shell_width[1:-1] = 0.5 * (radii[2:] - radii[:-2])
    shell_width[0] = 0.5 * (radii[1] - radii[0])
    shell_width[-1] = 0.5 * (radii[-1] - radii[-2])

    core_spectrum = normalized_blackbody_luminosity(luminosity_erg_s, frequency_hz, temperature_K[0])
    emergent = core_spectrum * np.exp(-tau_eff[0])
    effective_absorption = np.sqrt(np.maximum(kappa_abs, 0.0) * kappa_T)
    for index, (radius, width, temperature) in enumerate(zip(radii, shell_width, temperature_K)):
        shell_tau = effective_absorption[index] * width
        shell_source = 4.0 * np.pi**2 * radius**2 * planck_nu(frequency_hz, temperature)
        emergent += shell_source * (-np.expm1(-shell_tau)) * np.exp(-tau_eff[index])
    normalization = np.trapezoid(emergent, frequency_hz)
    if normalization <= 0.0:
        raise ValueError("transfer calculation produced no emergent luminosity")
    return luminosity_erg_s * emergent / normalization


def radial_spectra(
    luminosity_erg_s: float,
    radial_grid: RadialGrid,
    temperature_K: np.ndarray,
    tau_T: np.ndarray,
    tau_eff: np.ndarray,
    frequency_hz: np.ndarray,
) -> tuple:
    """Return local L_nu and local-temperature blackbody references.

    A frequency is thermalized while tau_eff >= 1 and follows the local
    blackbody luminosity. Once it becomes optically thin, its luminosity is
    propagated outward using the ratio of successive Thomson depths.
    """
    temperature_K = np.asarray(temperature_K, dtype=float)
    tau_T = np.asarray(tau_T, dtype=float)
    tau_eff = np.asarray(tau_eff, dtype=float)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    expected_shape = (len(radial_grid.radius_cm), len(frequency_hz))
    if temperature_K.shape != radial_grid.radius_cm.shape:
        raise ValueError("temperature must match the radial grid")
    if tau_T.shape != temperature_K.shape or tau_eff.shape != expected_shape:
        raise ValueError("tau_T and tau_eff must match the radial grid")
    local_spectra = np.empty_like(tau_eff)
    blackbody_spectra = np.empty_like(tau_eff)
    radii = radial_grid.radius_cm
    for index, (radius, temperature) in enumerate(zip(radii, temperature_K)):
        thermal_spectrum = 4.0 * np.pi**2 * radius**2 * planck_nu(frequency_hz, temperature)
        blackbody_spectra[index] = thermal_spectrum
    local_spectra[0] = blackbody_spectra[0]
    for frequency_index in range(len(frequency_hz)):
        for index in range(1, len(radii)):
            if tau_eff[index, frequency_index] >= 1.0:
                local_spectra[index, frequency_index] = blackbody_spectra[index, frequency_index]
                continue
            previous_tau = tau_T[index - 1]
            if previous_tau > 0.0:
                local_spectra[index, frequency_index] = (
                    local_spectra[index - 1, frequency_index]
                    * tau_T[index]
                    / previous_tau
                )
            else:
                local_spectra[index, frequency_index] = 0.0
    return local_spectra, blackbody_spectra


def write_spectrum_table(filename: str, frequency_hz, luminosity_nu):
    """Write the emergent spectrum as a two-column ASCII table."""
    with open(filename, "w") as handle:
        handle.write("# frequency_hz luminosity_per_hz_erg_s^-1_Hz^-1\n")
        for frequency, luminosity in zip(frequency_hz, luminosity_nu):
            handle.write(f"{frequency:.8e} {luminosity:.8e}\n")


def write_radial_spectra_table(filename: str, radius_cm, frequency_hz, luminosity_nu, blackbody_luminosity_nu):
    """Write one local spectrum for each radial cell as a flat table."""
    with open(filename, "w") as handle:
        handle.write("# radius_cm frequency_hz local_luminosity_per_hz_erg_s^-1_Hz^-1 blackbody_luminosity_per_hz_erg_s^-1_Hz^-1\n")
        for radius, spectrum, blackbody in zip(radius_cm, luminosity_nu, blackbody_luminosity_nu):
            for frequency, luminosity, blackbody_value in zip(frequency_hz, spectrum, blackbody):
                handle.write(f"{radius:.8e} {frequency:.8e} {luminosity:.8e} {blackbody_value:.8e}\n")


def write_departure_table(filename, radius_cm, levels, b_values, gamma_lte, gamma):
    """Write converged departure coefficients and photoionization rates."""
    with open(filename, "w") as handle:
        handle.write("# radius_cm level_index z istate level_id excitation_ev b_i gamma_lte gamma\n")
        for radius, b_row, lte_row, gamma_row in zip(radius_cm, b_values, gamma_lte, gamma):
            for index, (level, b_i, rate_lte, rate) in enumerate(
                zip(levels, b_row, lte_row, gamma_row)
            ):
                handle.write(
                    f"{radius:.8e} {index} {level['z']} {level['istate']} "
                    f"{level['level_id']} {level['ex_ev']:.8e} {b_i:.8e} "
                    f"{rate_lte:.8e} {rate:.8e}\n"
                )


def load_sirocco_hydrogen_data(masterfile: str) -> Dict[str, object]:
    """Load atomic records referenced by a SIROCCO master file.

    ``zdata/h_only.dat`` is a master file rather than a single atomic table;
    the existing opacity parser resolves and reads the files it references.
    """
    masterfile = os.path.abspath(masterfile)
    elements, ions, levels = parse_atomic_data(masterfile)
    photo_records, line_records, gaunt_records = load_atomic_records(masterfile)
    return {
        "masterfile": masterfile,
        "elements": elements,
        "ions": ions,
        "levels": levels,
        "photo_records": photo_records,
        "line_records": line_records,
        "gaunt_records": gaunt_records,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--masterfile",
        default=os.path.join("zdata","h20.dat"),
        help="SIROCCO atomic-data master file",
    )
    parser.add_argument("--rin", type=float, required=True, help="inner radius in cm")
    parser.add_argument("--rout", type=float, required=True, help="outer radius in cm")
    parser.add_argument("--luminosity", type=float, required=True, help="source luminosity in erg s^-1")
    parser.add_argument("--alpha", type=float, required=True, help="density power-law index")
    parser.add_argument("--rho16", type=float, required=True, help="density at 1e16 cm in g cm^-3")
    parser.add_argument("--kappa-t", type=float, default=DEFAULT_KAPPA_T, help="Thomson opacity in cm^2 g^-1")
    parser.add_argument("--nr", type=int, default=256, help="number of radial grid points")
    parser.add_argument("--nu-min", type=float, default=1.0e12, help="minimum frequency in Hz")
    parser.add_argument("--nu-max", type=float, default=1.0e18, help="maximum frequency in Hz")
    parser.add_argument("--nnu", type=int, default=512, help="number of frequency grid points")
    parser.add_argument("--spectrum-nlte", action="store_true", help="iterate radial spectra and NLTE level populations")
    parser.add_argument("--nlte-max-iterations", type=int, default=20, help="maximum NLTE population iterations")
    parser.add_argument("--nlte-tolerance", type=float, default=None, help="relative NLTE convergence tolerance (default: 0.3 * Delta nu / nu of the frequency grid)")
    parser.add_argument("--nlte-damping", type=float, default=0.5, help="log-space under-relaxation factor for b_i in (0, 1]")
    parser.add_argument(
        "--departure-out",
        default="dense_spectrum_departures.txt",
        help="output converged departure coefficients and photoionization rates",
    )
    parser.add_argument("--opacity", action="store_true", help="calculate radial LTE opacity and effective optical depth")
    parser.add_argument(
        "--opacity-out",
        default="dense_spectrum_opacity.txt",
        help="output table for radial opacity when --opacity is used",
    )
    parser.add_argument("--spectrum", action="store_true", help="calculate local spectra at every radius")
    parser.add_argument(
        "--emergent-spectrum",
        action="store_true",
        help="also calculate the outgoing spectrum",
    )
    parser.add_argument("--spectrum-out", default="dense_spectrum.txt", help="output outgoing spectrum table")
    parser.add_argument(
        "--radial-spectra-out",
        default="dense_spectrum_radial.txt",
        help="output local spectrum at every radius when --spectrum is used",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    grid = build_radial_grid(args.rin, args.rout, args.alpha, args.rho16, args.nr)
    frequencies_hz = frequency_grid(args.nu_min, args.nu_max, args.nnu)
    atomic_data = load_sirocco_hydrogen_data(args.masterfile)
    tau_T = outward_thomson_optical_depth(grid.radius_cm, grid.density_g_cm3, args.kappa_t)
    temperature_K = temperature_profile(args.luminosity, grid, tau_T, args.alpha)
    print(f"loaded {len(atomic_data['levels'])} levels from {args.masterfile}")
    print(f"radial grid: {len(grid.radius_cm)} points")
    print(f"frequency grid: {len(frequencies_hz)} points")
    print(f"rho(1e16 cm): {density_profile(np.array([CM_PER_REFERENCE_RADIUS]), args.alpha, args.rho16)[0]:.6e} g cm^-3")
    print(f"tau_T at Rin: {tau_T[0]:.6e}")
    print(f"temperature range: {temperature_K.min():.6f} - {temperature_K.max():.6f} K")
    if args.opacity or args.spectrum or args.spectrum_nlte or args.emergent_spectrum:
        if args.spectrum_nlte:
            (
                local_luminosity_nu,
                blackbody_luminosity_nu,
                kappa_abs,
                tau_eff,
                b_values,
                gamma_lte,
                gamma,
                iterations,
                temperature_K,
            ) = radial_nlte_spectra(
                args.masterfile,
                grid,
                radiative_temperature_profile(args.luminosity, grid.radius_cm, tau_T, args.alpha),
                tau_T,
                frequencies_hz,
                args.kappa_t,
                args.nlte_max_iterations,
                args.nlte_tolerance,
                args.nlte_damping,
            )
            write_departure_table(
                args.departure_out,
                grid.radius_cm,
                atomic_data["levels"],
                b_values,
                gamma_lte,
                gamma,
            )
            print(f"NLTE population iterations: {iterations}")
            print(f"NLTE temperature range: {temperature_K.min():.6f} - {temperature_K.max():.6f} K")
            print(f"departure table: {args.departure_out}")
        else:
            kappa_abs, tau_eff = radial_lte_opacity(
                args.masterfile,
                grid,
                temperature_K,
                tau_T,
                frequencies_hz,
                args.kappa_t,
            )
        if args.opacity:
            write_radial_opacity_table(args.opacity_out, grid.radius_cm, frequencies_hz, kappa_abs, tau_eff)
            print(f"opacity table: {args.opacity_out}")
        print(f"kappa_abs range: {kappa_abs.min():.6e} - {kappa_abs.max():.6e} cm^2 g^-1")
        print(f"tau_eff range: {tau_eff.min():.6e} - {tau_eff.max():.6e}")
        if args.spectrum or args.spectrum_nlte:
            local_luminosity_nu, blackbody_luminosity_nu = radial_spectra(
                args.luminosity,
                grid,
                temperature_K,
                tau_T,
                tau_eff,
                frequencies_hz,
            )
            write_radial_spectra_table(
                args.radial_spectra_out,
                grid.radius_cm,
                frequencies_hz,
                local_luminosity_nu,
                blackbody_luminosity_nu,
            )
            print(f"radial spectra table: {args.radial_spectra_out}")
        if args.emergent_spectrum:
            luminosity_nu = emergent_spectrum(
                args.luminosity,
                grid,
                temperature_K,
                kappa_abs,
                tau_eff,
                frequencies_hz,
                args.kappa_t,
            )
            write_spectrum_table(args.spectrum_out, frequencies_hz, luminosity_nu)
            print(f"outgoing spectrum table: {args.spectrum_out}")
            print(f"outgoing L_nu range: {luminosity_nu.min():.6e} - {luminosity_nu.max():.6e} erg s^-1 Hz^-1")


if __name__ == "__main__":
    main()
    main()