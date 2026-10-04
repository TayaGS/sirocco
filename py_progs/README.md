Description
###########

This directory contains python scripts for use with the C code python.

The basic python modules required can be found in `requirements.txt`.

Using `hydro_2_python` requires the `pyhdf` module, which itelf requires the `libhdf4-dev` package.
This can be installed manually or via your package manager or from the site [here](https://support.hdfgroup.org/downloads/index.html).

`convert_restart_files.py` can update the fixed-size header in a `.spec_save` file after changing `NWAVE_IONIZ` and rebuilding the code. It does not convert `.wind_save` files.

`plot_lte_opacity.py` combines LTE level populations with bound-free, free-free, and thermally broadened line opacity from the included atomic data.

Dense-gas spectrum model
########################

`dense_spectrum.py` is a semi-analytic model of spectral formation in a dense hydrogen wind, `rho ~ r^-alpha`. The design is in [docs/notes/dense-spectrum-implementation-plan.md](../docs/notes/dense-spectrum-implementation-plan.md) and [docs/dense_spectrum_nlte_plan.md](../docs/dense_spectrum_nlte_plan.md). `plot_dense_spectrum.py` plots its tables. Run both from the repository root; they need numpy, scipy and matplotlib.

Required arguments:

| Argument | Meaning |
| --- | --- |
| `--rin`, `--rout` | inner and outer radius (cm) |
| `--luminosity` | source luminosity (erg/s) |
| `--alpha` | density power-law index |
| `--rho16` | density at r = 1e16 cm (g/cm^3) |

Optional grid and physics arguments: `--nr` (radial points, default 256), `--nnu`, `--nu-min`, `--nu-max` (frequency grid, default 512 points, 1e12-1e18 Hz), `--kappa-t` (Thomson opacity), and `--masterfile` (default `zdata/h20.dat`).

Outputs are selected with flags. Without any of them the script only prints the profile summary (grid sizes, tau_T at the inner radius, temperature range).

| Flag | Output file option | Contents |
| --- | --- | --- |
| `--opacity` | `--opacity-out` | radius, frequency, kappa_abs, tau_eff |
| `--spectrum` | `--radial-spectra-out` | radius, frequency, local L_nu, blackbody L_nu |
| `--spectrum-nlte` | `--departure-out` | radius, level, b_i, gamma_lte, gamma (also writes the radial spectra) |
| `--emergent-spectrum` | `--spectrum-out` | outgoing L_nu (off by default) |

`--spectrum-nlte` replaces the LTE opacity with level populations from photoionization equilibrium, iterated up to `--nlte-max-iterations` times. Each iteration moves b_i a fraction `--nlte-damping` (default 0.5) of the way toward its new value in log space. `--nlte-tolerance` is the largest allowed relative change in b_i or T; its default is 0.3 * (nu_{i+1}/nu_i - 1), because one frequency bin crossing the tau_eff = 1 threshold changes the rates by about that fraction, so a tighter value need not converge.

Example (LTE, small test grid):

```
python py_progs/dense_spectrum.py --rin 1e15 --rout 1e16 --luminosity 1e44 \
    --alpha 2 --rho16 1e-14 --nr 4 --nnu 64 \
    --opacity --opacity-out /tmp/o.txt \
    --spectrum --radial-spectra-out /tmp/r.txt
python py_progs/plot_dense_spectrum.py /tmp/o.txt \
    --radial-spectra-table /tmp/r.txt --outfile /tmp/out.png
```

`plot_dense_spectrum.py` takes the opacity table as its positional argument. `--radial-spectra-table` adds local spectra at up to 10 log-spaced radii with the blackbody as a dashed line, and `--spectrum-table` adds the outgoing spectrum. Tables used together must come from the same run (same `--rout`, `--nr`, `--nnu`); the plotter rejects mismatched grids.

Notes:

- Local L_nu is a blackbody at the local temperature where tau_eff >= 1. Outward of that it is scaled by the ratio of Thomson depths, tau_T(i)/tau_T(i-1).
- The temperature follows the radiation energy density, T^4 = L/(4 pi r^2 sigma) [ (3/4)(alpha-1)/(alpha+1) tau_T + 1/4 ], where the 1/4 is the free-streaming limit. tau_T is computed for fully ionized gas. T is then held constant outward of the first cell where hydrogen is at most 50% ionized: from the local Saha equation in the LTE path, and in the NLTE path from the b_i-scaled level populations, re-evaluated every iteration. If hydrogen never reaches 50% ionization on the grid, T is not frozen.
- The last radius has tau_T = 0, so its values are a boundary artifact.
