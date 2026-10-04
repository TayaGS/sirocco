# Semi-analytic dense-gas spectrum

This document is the implementation plan for a semi-analytic spectral
formation model using SIROCCO atomic physics. The initial atomic-data input is
`zdata/h_only.dat`.

## Model inputs

The required physical parameters are:

- `L`: source luminosity in erg s^-1
- `Rin`: inner radius in cm
- `alpha`: density power-law index
- `rho16`: density at 10^16 cm in g cm^-3

The density profile is

```text
rho(r) = rho16 * (r / 1e16 cm)^(-alpha).
```

Numerical controls such as the outer radius, number of radial cells, number of
frequency cells, and maximum atomic level should be optional command-line
arguments rather than additional physical parameters.

## Implementation stages

### 1. Atomic-data and numerical foundations

- Confirm and document the columns and referenced files in `zdata/h_only.dat`.
- Reuse the SIROCCO atomic-data readers already implemented in
  `py_progs/plot_lte_opacity.py`.
- Add a logarithmic radial grid beginning at `Rin`.
- Add a logarithmic frequency grid with no special edge adjustments.
- Implement the density profile and hydrogen number density.
- Add unit checks and an analytic test of the density and optical-depth
  integrals.

### 2. Optical depths and temperature profile

- Calculate the outward Thomson optical depth,

  ```text
  tau_T(r) = integral_r^Rout kappa_T * rho(r') dr'.
  ```

- Use the supplied temperature relation with a free-streaming term so that
  T stays finite where `tau_T -> 0`,

  ```text
  T(r)^4 = L / (4*pi*r^2*sigma_sb)
           * [(3/4) * (alpha - 1)/(alpha + 1) * tau_T(r) + 1/4].
  ```

- Handle `alpha = 1` explicitly because the supplied prefactor vanishes in
  that limit.
- Hold T constant outward of the first cell where hydrogen is at most 50
  percent ionized, evaluated cell by cell at the local density. There is no
  single `Trec`. The LTE path uses the local Saha equation; the NLTE path uses
  the b_i-scaled level populations and re-evaluates the freeze point on every
  iteration.

### 3. LTE opacity

- Use `py_progs/plot_lte_opacity.py` as the starting point for the opacity
  calculation rather than implementing a separate atomic-data parser.
- Reuse its LTE population calculation and opacity contributions:
  - bound-free absorption,
  - free-free absorption,
  - line opacity where the available atomic data support it.
- Extend the calculation from one `(T, n_H)` state to every radial cell and
  frequency.
- Produce `kappa_abs(r, nu)` in cm^2 g^-1.
- Calculate the effective optical depth,

  ```text
  tau_eff(r, nu) = sqrt(kappa_abs(r, nu) / kappa_T) * tau_T(r).
  ```

### 4. Spectrum formation

- Use the Planck function `B_nu[T(r)]` as the thermal source function.
- Define a transfer prescription that approaches a diluted radiation field
  when `tau_eff` is small and a thermalized spectrum when `tau_eff` is large.
- Integrate the radial contributions to obtain the emergent `L_nu`.
- Check both limits explicitly:
  - `tau_eff << 1`: radiation is diluted;
  - `tau_eff >> 1`: radiation approaches the local thermal spectrum.

### 5. Photoionization/recombination populations

- For each level, calculate the ionizing photon rate above its threshold,

  ```text
  Gamma_i(r) = 1/(4*pi*r^2) * integral_nu_i^infinity
               [L_nu / (h*nu)] * sigma_i(nu) dnu.
  ```

- Solve the requested balance,

  ```text
  n_i * Gamma_i = n_e * n_p * alpha_i.
  ```

- Use the SIROCCO photoionization cross sections and recombination data where
  available. Document any level for which a cross section or recombination
  coefficient is missing.
- Keep this photoionization-equilibrium calculation distinct from LTE
  populations so the two assumptions are not silently mixed.

### 6. Command-line driver and outputs

- Add a script such as `py_progs/dense_spectrum.py`.
- Accept the four physical parameters as required arguments.
- Expose grid resolution and outer-radius settings as optional arguments.
- Write machine-readable tables containing radius, density, temperature,
  ionization state, `tau_T`, and frequency-dependent `tau_eff`.
- Write the emergent spectrum and level populations to separate output files.
- Add a small reproducible example and plotting helper after the core model is
  validated.

## Validation sequence

1. Verify that the radial density profile reproduces the requested `rho16`.
2. Compare numerical Thomson optical depths with analytic power-law integrals.
3. Verify that the Saha ionized fraction is at most 0.5 at the freeze cell and above 0.5 in the cell before it.
4. Check that the temperature becomes constant beyond the freeze cell.
5. Compare the single-zone LTE opacity against `plot_lte_opacity.py`.
6. Check the optically thin and optically thick spectral limits.
7. Verify photon-rate integration and the level-population balance.

The first coding milestone is the radial grid, density profile, Saha
ionized-fraction function, and outward Thomson optical depth. LTE
opacity is the next milestone and should be built by extracting or adapting
the existing logic in `plot_lte_opacity.py`.