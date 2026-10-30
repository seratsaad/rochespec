# rochespec

Roche-model metallicities for contact binaries from survey spectra (LAMOST LRS and MRS, GALAH DR4, DESI DR1, APOGEE DR19).

For each star the code builds the Roche surface for the adopted mass ratio, fill-out and inclination, sums payne-zero
specific intensities I(lambda, mu) over the visible surface at the orbital phase of each exposure, and fits Teff and [M/H]
on a grid. A photometric Teff prior (Casagrande et al. 2021 colour relation) breaks the Teff-[M/H] degeneracy. Zero points
come from APOGEE single dwarfs with The Payne (Astra DR19) labels, fitted with the same code.

Status: version 0.1.0, private. The code is the production code of the cb_feh_v0 catalogue with imports and data paths
made relative to the package; a fit of a LAMOST LRS star with the package reproduces the production fit to 1e-8.

## Install

    pip install -e .

Python 3.10 or later with numpy, scipy, pandas, astropy, requests and matplotlib.

## Data

Set `ROCHESPEC_DATA` to the data root (default `~/rochespec_data`). The modules read and write paths below it:

| path | content |
|---|---|
| `pz/grid/{opt1,opt2,opt3,H}/I_T*_g*_m*.npz` | payne-zero intensity grid (9 Teff x 4 log g x 5 [M/H] per window; made with `tools/pz_grid.py`) |
| `asassn/ii366_ecl.vot`, `qtest/liwang25.vot`, `qtest/li24.vot` | ASAS-SN eclipsing binaries and the light-curve geometries of Li & Wang (2025) and Li et al. (2024) |
| `cbscale/ew_lamost_dr10_obs.csv`, `cbscale/ew_teffphot.csv`, `cbscale/veb_all.csv` | LAMOST LRS spectra of the EWs, photometric Teff prior, Gaia DR3 VEB ephemerides |
| `lamost_dr10/ew/<obsid>.fits.gz`, `lamost_dr10/cal/` | LAMOST DR10 LRS spectra (contact binaries, calibrators) |
| `cbsynth/dr19_all_ew.csv`, `cbapo/visits/` | APOGEE DR19 EWs and mwmVisit files |
| `surveys/<survey>/` | MRS, GALAH and DESI targets, spectra and results |
| `cbscale/zp_*.json`, `surveys/calib_summary.csv` | zero points (copies in `data/`) |

`ROCHESPEC_A4B` points to a checkout of the agent4binary repository, whose DR19 census manifest lists the APOGEE single
dwarfs used as calibrators.

The fill-out factor uses the standard L2 outer potential (as in PHOEBE and Wilson-Devinney). `CB_OUTER=min` gives the
definition of the first catalogue release, which used the L3 potential.

## Running

Fits (all resumable, sharded):

    python -m rochespec.lrs <shard> <nshard>            # LAMOST LRS contact binaries
    python -m rochespec.apogee <shard> <nshard>         # APOGEE DR19 (APO_TAG=_lowsnr APO_SNR_MIN=10 for the second pass)
    python -m rochespec.surveys.mrs fit full_ids.csv full [first last]
    GALAH_MODE=comb python -m rochespec.surveys.galah fit full_ids.csv full [first last]
    python -m rochespec.surveys.desi fit full_ids.csv full
    python tools/run_lrs_pool.py <nworkers> <tag>       # LRS on one node, grid shared by forked workers

Calibration and catalogue:

    python -m rochespec.calib_lrs <shard> <nshard>      # LRS calibrators (single-star and spun-up fits)
    python -m rochespec.calib_apogee <shard> <nshard>   # APOGEE spun-up calibrators
    CAL_FIX=10,100 python -m rochespec.calib_fit lamost # zero point (kappa 10, prior scatter 100 K)
    python -m rochespec.calib_fit apogee
    python -m rochespec.apo_tphot                       # APOGEE [M/H] at T_phot - 100 K and its zero point
    python -m rochespec.surveys.mrs calib               # likewise galah, desi
    python -m rochespec.surveys.calib_summary
    python -m rochespec.surveys.compare
    python -m rochespec.catalog                         # catalog/cb_feh_v0.csv

## Catalogue

`catalog/cb_feh_v0.csv`: 9,525 ASAS-SN contact binaries, 7,111 with a calibrated [Fe/H] (`feh_all`), fitted with the standard
fill-out (2026-10-29).

## Layout

- `roche.py`: Roche mesh, surface gravity and temperature, visibility and velocities, intensity grids and disk integration
- `model_cfg.py`: grid nodes and loading
- `core.py`: joint (Teff, [M/H]) inference with the photometric prior, LAMOST DR10 reader, ephemerides, geometry
- `teffphot.py`, `teff_table.py`, `photmass.py`: photometric Teff prior
- `lrs.py`, `apogee.py`: LAMOST LRS and APOGEE fitters
- `wingrid.py`, `surveys/`: windowed FFT engine and the MRS, GALAH and DESI modules
- `calib_lrs.py`, `calib_apogee.py`, `calib_fit.py`, `apo_tphot.py`, `post.py`: calibration and zero points
- `catalog.py`: the combined catalogue
- `fine_joint.py`: optional cubic-spline interpolation of the chi^2 grid in Teff (`CB_FINE=1`)
- `tools/`: payne-zero intensity grid generation (needs a payne-zero checkout, `PZ_ROOT`)
