# Dense Spectrum NLTE Plan

## Agreed Scope

The NLTE work will be implemented incrementally, beginning with hydrogen and a single-pass diagnostic calculation.

- Remove the existing ground-state-only NLTE implementation first.
- Start the generalized implementation with hydrogen.
- Compute departure coefficients for every included level with a photoionization cross-section.
- Missing cross-sections are an error to diagnose and resolve, not a reason to silently use `b_i = 1`.
- Keep LTE ion densities and the atomic mass/electron bookkeeping available as the starting state.
- Modify level populations with `n_i = b_i n_i^LTE` for opacity.
- Use the local luminosity spectrum `L_nu` directly in photoionization-rate integrals; omit geometric dilution because it cancels in the departure-coefficient ratio.
- Use the blackbody luminosity spectrum as the LTE reference radiation field for `Gamma_i^*`.
- Use each photoionization record's first tabulated frequency as its integration threshold.
- Perform one pass initially: calculate rates and `b_i`, recalculate opacity and spectra once, and defer radiation/opacity iteration.
- Write departure-coefficient diagnostics to a separate radial-level table.

## Stage 1: Remove Legacy Ground-State NLTE

1. Remove `solve_ground_state_nlte()` and `radial_ground_state_nlte()`.
2. Remove the `--nonlte-ground` and `--ground-state-out` command-line options.
3. Remove ground-state population overrides from `lte_opacity_for_state()` and restore its LTE-only interface.
4. Remove the ground-state departure-coefficient argument and spectral modification from `radial_spectra()`.
5. Remove the ground-state diagnostic table writer and its main-program handling.
6. Remove ground-state diagnostic reading, plotting, and CLI support from `plot_dense_spectrum.py`.
7. Remove imports and constants that are used only by the deleted legacy path.
8. Preserve the existing LTE opacity, radial spectra, emergent spectrum, and outer-cell plotting behavior.

## Stage 2: NLTE Opacity API

Status: implemented as `nlte_opacity_for_state()` in `py_progs/dense_spectrum.py`.

Add a separate NLTE opacity path rather than changing the LTE function's behavior.

- Build the LTE state normally.
- Accept a departure-coefficient array aligned with the parsed levels.
- Set each opacity level population to `b_i * n_i^LTE`.
- Keep ion densities and electron density fixed for the first generalized opacity implementation.
- Reuse the existing bound-free, free-free, and line-opacity calculations.
- Validate coefficient length, finiteness, positivity, and level mapping.
- Verify that all coefficients equal to one reproduce the LTE opacity.

The implementation also applies the requested hydrogen charge closure and passes
the resulting electron density into the free-free opacity calculation. With all
`b_i = 1`, the NLTE path matches the LTE opacity to machine precision in the
focused validation case.

## Stage 3: Photoionization Rates and Departure Coefficients

Status: implemented in `dense_spectrum.py` with radial H20/Hydrogen rate integration.

For each radial cell and included level, calculate:

`Gamma_i = integral[L_nu / (h nu) * sigma_i(nu) dnu]`

and the LTE reference:

`Gamma_i^* = integral[L_nu^* / (h nu) * sigma_i(nu) dnu]`

where `L_nu^*` is the local blackbody luminosity spectrum. Integrate over the overlap of the model frequency grid and the tabulated cross-section range, beginning at the cross-section threshold. Then set:

`b_i = Gamma_i^* / Gamma_i`

Invalid or zero rates should initially raise an explicit error. Write radius, level identity, `b_i`, `Gamma_i^*`, and `Gamma_i` to a diagnostic table.

## Stage 4: NLTE Radial Spectrum Option

Status: implemented as `--spectrum-nlte` with fixed-point opacity/transfer iteration.

Add an option such as `--spectrum-nlte` that:

1. Calculates the baseline local spectra.
2. Calculates radial departure coefficients.
3. Recalculates opacity with the NLTE level populations.
4. Recalculates effective optical depth and radial spectra.
5. Writes the NLTE spectra and departure-coefficient diagnostics.

The first version is single-pass. Iteration between radiation field, departure coefficients, opacity, and transfer can be added later.

## Electron-Density Closure

For the generalized hydrogen NLTE state, use the departure-scaled neutral-level populations in the hydrogen charge balance:

`sum_i(n_i * b_i) + n_e = n_H`

This changes the recombination-front location. The implementation should make this closure explicit and test it independently before enabling it in the radial NLTE spectrum path. The initial hydrogen-only interpretation is that the sum is over H I levels and `n_H` is the total hydrogen density.

If later work requires strict conservation of each ion's total population, the scaled level populations should be renormalized within each ion. The initial design follows the direct `n_i = b_i n_i^LTE` prescription.

## Validation

- Compile all modified Python modules.
- Run the existing LTE opacity/radial-spectrum command.
- Confirm stage 1 produces the same LTE outputs apart from removal of the old optional ground-state path.
- Add focused tests for rate integration, `b_i = 1` LTE equivalence, NLTE opacity changes, charge closure, and radial diagnostic output as each later stage is implemented.
