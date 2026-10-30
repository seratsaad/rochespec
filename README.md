# rochespec

Metallicities of contact binaries (W UMa stars) from their spectra.

A contact binary is not a single star: its two components share one envelope, the surface temperature and gravity change across
it, and at every orbital phase the spectrum is a sum of differently Doppler-shifted pieces. rochespec models this directly. It
builds the Roche surface for the mass ratio, fill-out and inclination, sums specific intensities I(lambda, mu) from the
[Payne Zero](https://github.com/tingyuansen/payne-zero) model grid over the visible surface at the phase of each exposure, and
fits Teff and [M/H]. A photometric Teff prior breaks the Teff-[M/H] degeneracy, and zero points measured on APOGEE single dwarfs
put the result on the [Fe/H] scale of The Payne (SDSS DR19) for LAMOST LRS and MRS, GALAH DR4, DESI DR1 and APOGEE spectra.

## Install

    pip install "rochespec @ git+https://github.com/seratsaad/rochespec"           # core: numpy, scipy, astropy
    pip install "rochespec[phoebe] @ git+https://github.com/seratsaad/rochespec"   # PHOEBE contact meshes (the default surface model)
    pip install "rochespec[all] @ git+https://github.com/seratsaad/rochespec"      # also SPICE/JAX, matplotlib and the batch-pipeline dependencies

Without PHOEBE, rochespec uses its own ray-traced Roche surface and says so in a warning; [Fe/H] then differs by about 0.002 dex.

### Model grid

The specific-intensity grid (9 Teff x 4 log g x 5 [M/H] nodes, 10 mu values, R = 200,000) is too large for the package and is
downloaded once into `$ROCHESPEC_DATA` (default `~/rochespec_data`):

    import rochespec
    rochespec.download_grid(["opt1", "opt2", "opt3"])        # optical surveys
    rochespec.download_grid(["H"])                           # APOGEE

| window | range (nm) | float32 | float16 |
|---|---|---|---|
| opt1 | 380-550 | 884 MB | 159 MB |
| opt2 | 550-720 | 587 MB | 91 MB |
| opt3 | 720-890 | 442 MB | 70 MB |
| H | 1510-1700 | 380 MB | 62 MB |

Sizes are the compressed downloads. `precision="float16"` changes [Fe/H] by up to 0.0013 dex (0.0006 typical), well below the
measurement errors; float32 is the reference. Each tarball is checked against its SHA-256 sum. The download location is a
placeholder until the Zenodo record is public; set `ROCHESPEC_GRID_URL` or pass `url=` meanwhile.

## Quick start

    import rochespec

    spec = rochespec.io.read_lamost_lrs("279212133.fits.gz", ra=119.94337, dec=41.07646)
    res = rochespec.fit(spec, period=0.41419978, t0=2457372.59501,       # period (d), time of primary minimum (BJD_TDB)
                        q=0.195, fillout=0.057, incl=88.409,               # light-curve geometry
                        bp_rp=0.7916, ebv=0.044)                           # Gaia BP-RP and E(B-V) for the Teff prior
    print(res.summary())
    res.plot()

`examples/fit_lamost_lrs.py` runs this on the spectrum in `examples/data/`.

Readers (`rochespec.io`): `read_lamost_lrs`, `read_lamost_mrs`, `read_galah` (the CCD files of one epoch), `read_desi` (DR1 coadd file
or URL and TARGETID), `read_apogee_visits` (DR19 mwmVisit). Several spectra of one star go into one call as a list; each carries
its own exposure times, and the model is averaged over each exposure.

Any other optical spectrum can be fitted from arrays, at its resolving power, with model [M/H] only (no zero point):

    res = rochespec.fit(wave=wave, flux=flux, err=err, times=bjd_samples, resolution=5000,
                        period=0.35, t0=2459000.123, teff_prior=(5900, 120))

`rochespec.model(teff=..., mh=..., period=..., phases=[...], q=..., fillout=..., incl=..., survey="lamost_lrs")` returns a model
spectrum (wavelength, normalised flux) averaged over the given phases.

### Inputs

- **Ephemeris.** `period` in days and `t0`, the time of primary minimum in BJD_TDB. Without `t0` the phase of each spectrum is fitted
  (LAMOST LRS, APOGEE and generic spectra; flag `free_phase`).
- **Geometry.** `q` (either component ratio), `fillout` (standard definition: 0 at the inner and 1 at the outer critical surface
  through L2) and `incl` in degrees, for instance from a light-curve solution. Without them the fit uses q = 0.35, f = 0.2,
  i = 78 deg (flag `default_geometry`).
- **Teff prior.** `bp_rp` and `ebv` (optionally `ebv_err`) for the Casagrande et al. (2021) colour-Teff relation, or
  `teff_prior=(Teff, sigma)`, or `(Teff at [Fe/H] = 0, sigma, dTeff/d[Fe/H])`.
- **Surface model.** `geometry="phoebe"` (PHOEBE contact meshes, default) or `"roche"`.

### Result

`fit` returns a `FitResult` with

| field | meaning |
|---|---|
| `feh`, `feh_err` | calibrated [Fe/H] (LAMOST LRS/MRS, GALAH, DESI, APOGEE; NaN for generic spectra) |
| `mh`, `mh_err` | model [M/H] (Payne Zero scale) |
| `teff`, `teff_err` | system Teff (joint fit for LAMOST LRS; at the photometric Teff for the other surveys) |
| `chi2_red`, `rv_sys` | reduced chi^2 and systemic velocity (km/s) |
| `flags` | `mh_edge`, `mh_extrap`, `teff_edge`, `poor_fit`, `free_phase`, `default_geometry`, `geometry_fallback`, `ebv`, `colour_range`, `teff_clipped` |
| `geometry`, `q`, `fillout`, `incl` | surface model and geometry used |
| `wave`, `flux`, `model` | the data and the best-fit model (with its continuum polynomial) for each spectrum |
| `chi2_grid`, `pipeline` | the chi^2 grid over (Teff, [M/H]) and the raw fitter output |

`res.as_dict()` gives the scalars for tables.

### Command line

    rochespec fit spec.fits.gz --survey lamost_lrs --period 0.4142 --t0 2457372.595 --q 0.195 --fillout 0.057 --incl 88.4 \
        --bp-rp 0.79 --ebv 0.044 --json result.json --plot fit.png
    rochespec fit mwmVisit-0.6.0-12345.fits --survey apogee --period 0.35 --t0 ... --teff 5900 --teff-err 150 --csv results.csv
    rochespec download-grid --windows opt1,opt2,opt3
    rochespec info

## How the fit works

For each trial Teff the surface temperatures follow gravity darkening (T^4 ~ g^0.32) scaled to the system Teff, the masses follow
M1 = (Teff/5772 K)^1.8 and the separation Kepler's law. The visible area of every surface element at each phase is spread with
linear weights onto the grid nodes (Teff, log g, [M/H], mu) and onto velocity bins; each occupied node's intensity spectrum is then
convolved with its kernel by FFT (with the instrumental profile), so the cost depends on the number of grid nodes touched, not on
the number of surface elements. Each spectrum gets a systemic velocity and a multiplicative continuum polynomial per arm; the
chi^2 over a (Teff, [M/H]) grid is combined with the Teff prior.

`spice_backend.py` applies the same integrator to SPICE (`pip install stellar-spice`) surface models (PHOEBE meshes or
SPICE meshes, with spots and abundance spots): it reproduces SPICE's own spectra to 1e-8 and is 150-400 times faster per spectrum.
`jax_kernel.py` is a JAX version (jit, gradients through SPICE meshes).

## Tests

    pytest tests/

`tests/test_quick.py` needs no data. `tests/test_catalogue.py` refits three LAMOST LRS stars from `tests/data/` and checks the
published values to 0.001 dex; it runs when the optical grid and PHOEBE are installed.

## Credits

The intensities come from Payne Zero (Y.-S. Ting; https://github.com/tingyuansen/payne-zero), the PHOEBE contact meshes from PHOEBE 2
(Prsa et al. 2016; Conroy et al. 2020), the SPICE interface from SPICE (Jablonska et al. 2025), the colour-Teff relation from
Casagrande et al. (2021) and the default light-curve geometries in the batch pipelines from Li et al. (2024) and Li & Wang (2025).
Please cite rochespec (`CITATION.cff`) together with Payne Zero and, if used, SPICE and PHOEBE.

## Batch pipelines

`rochespec/pipelines/` holds the scripts that fitted the survey samples of the paper (LAMOST LRS and the calibrators, APOGEE, the
MRS/GALAH/DESI modules with their download code, the zero-point fits and the merged table). They read and write below
`$ROCHESPEC_DATA` and need `rochespec[pipelines]`; `fit()` calls the same fitting functions. Run them as
`python -m rochespec.pipelines.<module>` with the arguments given in each module's docstring.

## License

MIT (see LICENSE).
