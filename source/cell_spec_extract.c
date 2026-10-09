/***********************************************************/
/** @file  cell_spec_extract.c
 * @date   October 2026
 *
 * @brief  High resolution spectra (J_nu) of each wind cell, accumulated
 * during the spectral cycles
 *
 * The cell spectra stored in the plasma structure (cell_spec_flux) are only
 * accumulated during ionization cycles, have a resolution fixed at compile
 * time (NBINS_IN_CELL_SPEC) and are saved in the windsave file. The routines
 * here instead accumulate J_nu in each cell during the spectral cycles, when
 * the wind is fixed, with a number of bins and a wavelength range read from
 * the .pf file. The arrays are allocated at run time and are not part of the
 * windsave file, so they can be used when restarting (-r) from a converged
 * model to run only the spectral cycles.
 *
 * The option is turned on with the command line switch -cell_spec, which
 * causes the following to be read from the .pf file
 *
 *   Spectrum.cell_spec.nbins
 *   Spectrum.cell_spec.wavemin(Angstroms)
 *   Spectrum.cell_spec.wavemax(Angstroms)
 *
 * The bins are uniform in log(frequency) and refer to the comoving frame.
 * Photons are only generated between Spectrum.wavemin and Spectrum.wavemax
 * in the spectral cycles, so the cell spectra are only complete inside that
 * range. Only photons transported through the wind contribute; the virtual
 * photons created for extract mode do not.
 *
 * The results are written to root.cell_spec.txt after every spectral cycle,
 * normalised as in the ionization cycle cell spectra, i.e. in
 * erg/s/cm^2/Hz/sr.
 *
 ***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "atomic.h"
#include "sirocco.h"

static double *cell_spec_j = NULL;      /* NPLASMA x cell_spec_nbins, indexed nplasma * cell_spec_nbins + i */
static int cell_spec_nbins = 0;
static double cell_spec_lfmin, cell_spec_dlf;   /* log10 of the minimum frequency and the log bin width */
static int cell_spec_ncycles = 0;       /* number of spectral cycles accumulated in this run */
static int cell_spec_counting = FALSE;  /* TRUE only while a real (not extracted) photon is being translated */



/**********************************************************/
/**
 * @brief      Read the parameters for the spectral cycle cell spectra
 *
 * @return     Always returns 0
 *
 * @details
 * Called from main, only if the -cell_spec switch was given and spectral
 * cycles remain to be run, including when a restart continues the spectral
 * cycles. The default wavelength range is that of the detailed spectra,
 * geo.swavemin to geo.swavemax.
 *
 **********************************************************/

int
cell_spec_extract_read_params (void)
{
  double wavemin, wavemax;

  cell_spec_nbins = 10000;
  wavemin = geo.swavemin;
  wavemax = geo.swavemax;

  rdpar_comment ("Parameters for the cell spectra accumulated in the spectral cycles");
  rdint ("Spectrum.cell_spec.nbins", &cell_spec_nbins);
  rddoub ("Spectrum.cell_spec.wavemin(Angstroms)", &wavemin);
  rddoub ("Spectrum.cell_spec.wavemax(Angstroms)", &wavemax);

  if (cell_spec_nbins < 1)
  {
    Error ("cell_spec_extract_read_params: Spectrum.cell_spec.nbins must be positive, got %d\n", cell_spec_nbins);
    Exit (EXIT_FAILURE);
  }
  if (wavemin <= 0 || wavemax <= wavemin)
  {
    Error ("cell_spec_extract_read_params: need 0 < Spectrum.cell_spec.wavemin < wavemax, got %g %g\n", wavemin, wavemax);
    Exit (EXIT_FAILURE);
  }
  if (wavemin < geo.swavemin || wavemax > geo.swavemax)
  {
    Error ("cell_spec_extract_read_params: cell spectra range %g-%g A extends beyond Spectrum.wavemin/max %g-%g A,\n",
           wavemin, wavemax, geo.swavemin, geo.swavemax);
    Error ("cell_spec_extract_read_params: no photons are generated there in the spectral cycles\n");
  }

  cell_spec_lfmin = log10 (VLIGHT / (wavemax * ANGSTROM));
  cell_spec_dlf = (log10 (VLIGHT / (wavemin * ANGSTROM)) - cell_spec_lfmin) / cell_spec_nbins;

  Log ("Cell spectra in spectral cycles: %d bins from %g to %g A (R = %.0f)\n", cell_spec_nbins, wavemin, wavemax,
       1. / (cell_spec_dlf * log (10.)));

  return (0);
}



/**********************************************************/
/**
 * @brief      Allocate and zero the spectral cycle cell spectra
 *
 * @return     Always returns 0
 *
 * @details
 * Called before the first spectral cycle of this run, once NPLASMA is known.
 *
 **********************************************************/

int
cell_spec_extract_init (void)
{
  double size_mb;

  if (!modes.cell_spec_extract || cell_spec_j != NULL || cell_spec_nbins < 1)
    return (0);

  size_mb = (double) NPLASMA * cell_spec_nbins * sizeof (double) / 1048576.;
  Log ("Cell spectra in spectral cycles: allocating %.1f MB per rank for %d cells\n", size_mb, NPLASMA);

  cell_spec_j = calloc ((size_t) NPLASMA * cell_spec_nbins, sizeof (double));
  if (cell_spec_j == NULL)
  {
    Error ("cell_spec_extract_init: unable to allocate %.1f MB for cell spectra\n", size_mb);
    Exit (EXIT_FAILURE);
  }
  cell_spec_ncycles = 0;

  return (0);
}



/**********************************************************/
/**
 * @brief      Turn the accumulation on or off
 *
 * @param [in] int on   TRUE while a real photon is being translated
 *
 * @details
 * trans_phot_single turns this on around its call to translate, so the
 * virtual photons translated in extract_one are not counted.
 *
 **********************************************************/

void
cell_spec_extract_counting (int on)
{
  cell_spec_counting = on;
}



/**********************************************************/
/**
 * @brief      Add the contribution of a photon path segment in a cell
 *
 * @param [in] PhotPtr p   The photon at the start of the segment (observer frame)
 * @param [in] double ds   The length of the segment in the observer frame
 * @param [in] double w_ave   The average weight of the photon along the segment
 *
 * @return     Always returns 0
 *
 * @details
 * As for the ionization cycle estimators, the photon is moved to the middle
 * of the segment and transformed to the local frame, and w ds is added to
 * the bin containing the comoving frequency.
 *
 **********************************************************/

int
cell_spec_extract_increment (PhotPtr p, double ds, double w_ave)
{
  struct photon phot_mid, phot_mid_cmf;
  double ds_cmf, w_cmf;
  int nplasma, i;

  if (cell_spec_j == NULL || !cell_spec_counting || geo.ioniz_or_extract != CYCLE_EXTRACT)
    return (0);

  if (p->grid < 0 || p->grid >= geo.ndim2)
    return (0);

  nplasma = wmain[p->grid].nplasma;
  if (nplasma < 0 || nplasma >= NPLASMA)
    return (0);

  stuff_phot (p, &phot_mid);
  move_phot (&phot_mid, 0.5 * ds);
  observer_to_local_frame (&phot_mid, &phot_mid_cmf);
  ds_cmf = observer_to_local_frame_ds (&phot_mid, ds);
  w_cmf = w_ave * phot_mid_cmf.freq / phot_mid.freq;

  i = (int) floor ((log10 (phot_mid_cmf.freq) - cell_spec_lfmin) / cell_spec_dlf);
  if (i < 0 || i >= cell_spec_nbins)
    return (0);

  cell_spec_j[(size_t) nplasma * cell_spec_nbins + i] += w_cmf * ds_cmf;

  return (0);
}



/**********************************************************/
/**
 * @brief      Combine the cell spectra across ranks and write them out
 *
 * @return     Always returns 0
 *
 * @details
 * Called at the end of every spectral cycle. The photon weights in the
 * spectral cycles are normalised so that all geo.pcycles cycles together
 * carry the luminosity, so the sum is scaled by pcycles / (cycles so far),
 * as for the detailed spectra. Each rank's photons carry the full
 * luminosity, so the ranks are averaged.
 *
 **********************************************************/

int
cell_spec_extract_write (void)
{
  double *jsum;
  double renorm, volume, lf, dfreq;
  size_t ntot;
  int n, i, nwind, ndom, ii, jj;
  char filename[LINELENGTH];
  FILE *fptr;

  if (cell_spec_j == NULL)
    return (0);

  cell_spec_ncycles++;
  ntot = (size_t) NPLASMA * cell_spec_nbins;

#ifdef MPI_ON
  jsum = NULL;
  if (rank_global == 0)
  {
    jsum = calloc (ntot, sizeof (double));
    if (jsum == NULL)
    {
      Error ("cell_spec_extract_write: unable to allocate reduction buffer\n");
      Exit (EXIT_FAILURE);
    }
  }
  /* reduce one cell at a time to keep the counts within the range of an int */
  for (n = 0; n < NPLASMA; n++)
  {
    MPI_Reduce (&cell_spec_j[(size_t) n * cell_spec_nbins], rank_global == 0 ? &jsum[(size_t) n * cell_spec_nbins] : NULL,
                cell_spec_nbins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  if (rank_global != 0)
    return (0);
  renorm = (double) geo.pcycles / cell_spec_ncycles / np_mpi_global;
#else
  jsum = cell_spec_j;
  renorm = (double) geo.pcycles / cell_spec_ncycles;
#endif

  sprintf (filename, "%.*s.cell_spec.txt", LINELENGTH - 20, files.root);
  if ((fptr = fopen (filename, "w")) == NULL)
  {
    Error ("cell_spec_extract_write: unable to open %s\n", filename);
#ifdef MPI_ON
    free (jsum);
#endif
    return (0);
  }

  fprintf (fptr, "# J_nu (erg/s/cm^2/Hz/sr) in the comoving frame of each wind cell, from the spectral cycles\n");
  fprintf (fptr, "# Spectral cycles accumulated: %d of %d, nbins %d\n", cell_spec_ncycles, geo.pcycles, cell_spec_nbins);
  fprintf (fptr, "# Freq and Lambda are bin centres; columns are named as in windsave2table .xspec files\n");
  fprintf (fptr, "Freq.           Lambda         ");
  for (n = 0; n < NPLASMA; n++)
  {
    nwind = plasmamain[n].nwind;
    ndom = wmain[nwind].ndom;
    if (zdom[ndom].coord_type == SPHERICAL)
    {
      fprintf (fptr, "F%03d       ", nwind - zdom[ndom].nstart);
    }
    else
    {
      wind_n_to_ij (ndom, nwind, &ii, &jj);
      fprintf (fptr, "F%03d_%03d   ", ii, jj);
    }
  }
  fprintf (fptr, "\n");

  for (i = 0; i < cell_spec_nbins; i++)
  {
    lf = cell_spec_lfmin + (i + 0.5) * cell_spec_dlf;
    dfreq = pow (10., cell_spec_lfmin + (i + 1) * cell_spec_dlf) - pow (10., cell_spec_lfmin + i * cell_spec_dlf);
    fprintf (fptr, "%.8e %.8e ", pow (10., lf), VLIGHT / pow (10., lf) / ANGSTROM);
    for (n = 0; n < NPLASMA; n++)
    {
      nwind = plasmamain[n].nwind;
      volume = wmain[nwind].vol / wmain[nwind].xgamma_cen;
      fprintf (fptr, "%10.3e ", jsum[(size_t) n * cell_spec_nbins + i] * renorm / (4. * PI * volume * dfreq));
    }
    fprintf (fptr, "\n");
  }

  fclose (fptr);
  Log ("Wrote cell spectra from %d spectral cycles to %s\n", cell_spec_ncycles, filename);

#ifdef MPI_ON
  free (jsum);
#endif

  return (0);
}
