"""Public interface: fit the spectra of one contact binary with the Roche model, or compute a model spectrum.

fit() wraps the production fitters (rochespec.pipelines.lrs.fit_specs, apogee.fit_visits, wingrid.fit_ew) and the catalogue
calibration (post.calibrate_one, pipelines.apo_tphot.mh_at_tphot, the single-star zero points of MRS/GALAH/DESI), so for the same
inputs it returns the catalogue values.
"""
import os, json, copy, time, warnings, importlib.util
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
import numpy as np

from .io import Spectrum

SURVEYS = ("lamost_lrs", "lamost_mrs", "galah", "desi", "apogee")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_GEOMETRY = dict(q=0.35, fillout=0.2, incl=78.0)        # used when no light-curve solution is given (catalogue default)
_GRIDS = {}


# ------------------------------------------------------------------------------------------------------------------ settings
@contextmanager
def _settings(geometry):
    """production settings for one call: payne-zero model, standard fill-out, FFT kernel, and the requested surface model"""
    if geometry not in ("phoebe", "roche"):
        raise ValueError("geometry must be 'phoebe' or 'roche'")
    if geometry == "phoebe" and importlib.util.find_spec("phoebe") is None:
        warnings.warn("PHOEBE is not installed (pip install rochespec[phoebe]); using the ray-traced Roche surface instead. "
                      "[Fe/H] then differs from the PHOEBE-surface value by about 0.002 dex.", RuntimeWarning, stacklevel=3)
        geometry = "roche"
    keys = dict(CB_MODEL="pz", CB_OUTER="L2", CB_KERNEL="fft", CB_GEOM=geometry)
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update(keys)
    try:
        yield geometry
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def _zp(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def _survey_zp_single(survey):
    import csv
    with open(os.path.join(DATA_DIR, "calib_summary.csv")) as f:
        for row in csv.DictReader(f):
            if row["survey"] == survey:
                return float(row["zp_single"]), float(row["sd_single"])
    return None, None


def _optical_grid():
    if "opt" not in _GRIDS:
        from .config import DATA
        from .grid import check_grid
        check_grid(("opt1", "opt2", "opt3"))
        from .model_cfg import load
        _GRIDS["opt"] = load("opt", dv=5.0, rebin=True)
    return _GRIDS["opt"]


def _h_grid():
    if "H" not in _GRIDS:
        from .grid import check_grid
        check_grid(("H",))
        from .model_cfg import load
        _GRIDS["H"] = load("H", dv=1.0)
    return _GRIDS["H"]


def _win_grid(survey):
    if survey not in _GRIDS:
        from .grid import check_grid
        from .wingrid import WinGrid
        mod = _survey_module(survey)
        check_grid(("opt1", "opt2", "opt3"))
        _GRIDS[survey] = WinGrid(mod.SEGS, mod.DV)
    return _GRIDS[survey]


def _survey_module(survey):
    from .pipelines.surveys import mrs, galah, desi
    return {"lamost_mrs": mrs, "galah": galah, "desi": desi}[survey]


class _Row(dict):
    """a target row with attribute access, as the pipelines expect (pandas Series stand-in)"""
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)


# ------------------------------------------------------------------------------------------------------------------ inputs
def teff_prior_from_colour(bp_rp, ebv=0.0, ebv_err=None, logg=4.3):
    """Casagrande et al. (2021) Gaia (BP-RP)_0 Teff at [Fe/H] = 0, its [Fe/H] slope and error (the catalogue prior)"""
    from .teffphot import teff_phot
    T0, sT, x0 = teff_phot(np.atleast_1d(bp_rp), np.atleast_1d(ebv), 0.0, logg=logg, sig_ebv=None if ebv_err is None else np.atleast_1d(ebv_err))
    T1, _, _ = teff_phot(np.atleast_1d(bp_rp), np.atleast_1d(ebv), 1.0, logg=logg, sig_ebv=None if ebv_err is None else np.atleast_1d(ebv_err))
    return dict(Tphot0=float(T0[0]), dTdfeh=float(T1[0] - T0[0]), sig_Tphot=float(sT[0]), bp_rp0=float(x0[0]),
                flag_colour=int(not (0.2 <= x0[0] <= 2.0)), flag_ebv=int(float(np.nan_to_num(ebv)) > 0.3))


def _target_row(period, t0, q, fillout, incl, prior, ra, dec, ebv):
    geo_given = q is not None or fillout is not None or incl is not None
    if geo_given and (q is None or incl is None):
        raise ValueError("give q and incl together (fillout defaults to 0.2)")
    s = _Row(sid=0, Per=float(period), RAJ2000=np.nan if ra is None else float(ra), DEJ2000=np.nan if dec is None else float(dec),
             q=float(q) if geo_given else np.nan, i=float(incl) if geo_given else np.nan,
             f=float(fillout) if (geo_given and fillout is not None) else (0.2 if geo_given else np.nan), src="user" if geo_given else np.nan,
             **{"E_B-V_": np.nan if ebv is None else float(ebv)})
    if t0 is not None:          # ephemeris: phase 0 at t0 (primary minimum), as the Gaia-VEB phases of the catalogue
        s.update(Freq=1.0 / float(period), TimeRef=float(t0) - 2455197.5, PhaseGauss1=0.0, DepthGauss1=1.0, PhaseGauss2=np.nan,
                 DepthGauss2=0.0, PhaseCHP=np.nan)
    else:
        s.update(Freq=np.nan, TimeRef=np.nan, PhaseGauss1=np.nan, DepthGauss1=np.nan, PhaseGauss2=np.nan, DepthGauss2=np.nan, PhaseCHP=np.nan)
    s.update(Tphot0=prior["Tphot0"], dTdfeh=prior["dTdfeh"], sig_Tphot=prior["sig_Tphot"], flag_ebv=prior.get("flag_ebv", 0),
             flag_colour=prior.get("flag_colour", 0))
    return s


def _prior(teff_prior, bp_rp, ebv, ebv_err, logg):
    if teff_prior is not None:
        T, sig, slope = (tuple(teff_prior) + (0.0,))[:3]
        return dict(Tphot0=float(T), dTdfeh=float(slope), sig_Tphot=float(sig), flag_colour=0, flag_ebv=0)
    if bp_rp is not None:
        return teff_prior_from_colour(bp_rp, 0.0 if ebv is None else ebv, ebv_err, logg)
    warnings.warn("no Teff prior (teff_prior or bp_rp): the Teff-[M/H] degeneracy is then only broken by the spectrum; "
                  "using a flat prior centred on 5800 K", RuntimeWarning, stacklevel=3)
    return dict(Tphot0=5800.0, dTdfeh=0.0, sig_Tphot=1e4, flag_colour=0, flag_ebv=0)


def _as_spectra(spectra, wave, flux, err, ivar, times, survey, resolution):
    if spectra is None:
        if wave is None or flux is None or times is None:
            raise ValueError("give spectra=[Spectrum, ...] (see rochespec.io) or wave, flux, err (or ivar) and times")
        if ivar is None:
            if err is None: raise ValueError("give err or ivar")
            err = np.asarray(err, float); ivar = np.where(np.isfinite(err) & (err > 0), 1 / np.where(err > 0, err, 1) ** 2, 0.0)
        spectra = [Spectrum(np.asarray(wave, float), np.asarray(flux, float), np.asarray(ivar, float), np.atleast_1d(np.asarray(times, float)),
                            survey, resolution)]
    elif isinstance(spectra, Spectrum):
        spectra = [spectra]
    spectra = list(spectra)
    sv = {s.survey for s in spectra}
    if survey is None:
        survey = sv.pop() if len(sv) == 1 else None
    if survey is not None and survey not in SURVEYS:
        raise ValueError(f"survey must be one of {SURVEYS} or None (generic spectrum with a resolution)")
    if survey is None and resolution is None:
        resolution = spectra[0].resolution
        if resolution is None:
            raise ValueError("give survey= or resolution= (resolving power) for a generic spectrum")
    return spectra, survey, resolution


def _generic_segments(spectra, max_len=1800.0, lo=3950.0, hi=8850.0):
    """continuum segments for a generic optical spectrum: the data range inside the model grid, split at gaps > 50 A and into pieces
    of at most max_len A (one degree-8 Legendre polynomial each, like the LRS arms)"""
    w = np.unique(np.concatenate([s.wave[(s.ivar > 0) & (s.wave > lo + 20) & (s.wave < hi - 20)] for s in spectra]))
    if len(w) < 200: raise ValueError("no usable pixels between 3970 and 8830 A (the optical model grid)")
    cuts = np.where(np.diff(w) > 50)[0]; starts = np.r_[w[0], w[cuts + 1]]; ends = np.r_[w[cuts], w[-1]]
    segs = []
    for a, b in zip(starts, ends):
        n = max(1, int(np.ceil((b - a) / max_len))); e = np.linspace(a, b, n + 1)
        segs += [(float(e[k]), float(e[k + 1])) for k in range(n)]
    return [s for s in segs if s[1] - s[0] > 100]


# ------------------------------------------------------------------------------------------------------------------ result
@dataclass
class FitResult:
    survey: str
    mh: float                     # model [M/H] at the joint (or photometric) Teff, on the payne-zero scale
    mh_err: float
    feh: float                    # calibrated [Fe/H] (The Payne / APOGEE DR19 scale) when the survey has a zero point, else NaN
    feh_err: float
    teff: float                   # system Teff (K): joint inference (LRS, APOGEE read at T_phot - 100 K, MRS/GALAH/DESI at T_col)
    teff_err: float
    chi2_red: float
    rv_sys: float                 # systemic velocity (km/s) found by the fit, in the frame of the input spectra
    flags: list
    geometry: str                 # surface model used: 'phoebe' or 'roche'
    q: float
    fillout: float
    incl: float
    n_spec: int
    model: list = field(default_factory=list, repr=False)       # best-fit model on each spectrum's pixels (continuum included)
    wave: list = field(default_factory=list, repr=False)
    flux: list = field(default_factory=list, repr=False)
    weight: list = field(default_factory=list, repr=False)
    chi2_grid: dict = field(default_factory=dict, repr=False)
    pipeline: dict = field(default_factory=dict, repr=False)    # raw output of the wrapped production fitter

    def summary(self):
        f = f"[Fe/H] = {self.feh:+.3f} +- {self.feh_err:.3f}" if np.isfinite(self.feh) else "[Fe/H]: no zero point for this setup"
        return (f"{self.survey}: [M/H]_model = {self.mh:+.3f} +- {self.mh_err:.3f}, {f}, Teff = {self.teff:.0f} +- {self.teff_err:.0f} K, "
                f"chi2_red = {self.chi2_red:.2f}, RV_sys = {self.rv_sys:+.1f} km/s, geometry {self.geometry}, flags: {','.join(self.flags) or '-'}")

    def as_dict(self, arrays=False):
        d = {k: v for k, v in asdict(self).items() if k not in ("model", "wave", "flux", "weight", "chi2_grid", "pipeline")}
        if arrays:
            d.update(model=[np.asarray(x).tolist() for x in self.model], wave=[np.asarray(x).tolist() for x in self.wave],
                     flux=[np.asarray(x).tolist() for x in self.flux])
        return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in d.items()}

    def plot(self, ax=None, i=0):
        """data and best-fit model of spectrum i"""
        import matplotlib.pyplot as plt
        ax = ax or plt.gca(); w = np.asarray(self.weight[i]) > 0
        ax.plot(self.wave[i], np.where(w, self.flux[i], np.nan), c="k", lw=0.6, label="data")
        ax.plot(self.wave[i], self.model[i], c="#D55E00", lw=0.8, label="Roche model")
        ax.set_xlabel(r"Wavelength ($\AA$)"); ax.set_ylabel("Normalised flux"); ax.legend(frameon=False)
        return ax


def _flags(mh_edge, teff_edge, chi2_red, mh, free_phase, default_geo, fallback, prior):
    f = []
    if mh_edge: f.append("mh_edge")
    if mh > 0.3: f.append("mh_extrap")
    if teff_edge: f.append("teff_edge")
    if chi2_red > 5: f.append("poor_fit")
    if free_phase: f.append("free_phase")
    if default_geo: f.append("default_geometry")
    if fallback: f.append("geometry_fallback")
    if prior.get("flag_ebv"): f.append("ebv")
    if prior.get("flag_colour"): f.append("colour_range")
    return f


# ------------------------------------------------------------------------------------------------------------------ fit
def fit(spectra=None, *, wave=None, flux=None, err=None, ivar=None, times=None, survey=None, resolution=None,
        period, t0=None, q=None, fillout=None, incl=None, teff_prior=None, bp_rp=None, ebv=None, ebv_err=None, logg=4.3,
        geometry="phoebe", ra=None, dec=None, segments=None, models=True):
    """Fit the spectra of one contact binary.

    spectra      list of rochespec.io.Spectrum (from the readers in rochespec.io), or give wave/flux/err(or ivar)/times for one
                 spectrum (vacuum Angstrom; times = BJD_TDB sample times across the exposure).
    survey       'lamost_lrs', 'lamost_mrs', 'galah', 'desi', 'apogee' (calibrated [Fe/H] available), or None with resolution=R for a
                 generic optical spectrum (fitted like LAMOST LRS at resolving power R; model [M/H] only).
    period, t0   orbital period (d) and time of primary minimum (BJD_TDB). Without t0 the phase of each spectrum is a free parameter
                 (LAMOST LRS / generic / APOGEE only).
    q, fillout, incl  light-curve geometry (q <= 1 or its inverse, fill-out in the standard L2 definition, inclination in degrees);
                 missing: the catalogue default q 0.35, f 0.2, i 78 (flag default_geometry).
    teff_prior   (Teff, sigma) in K, or (Teff at [Fe/H] = 0, sigma, dTeff/d[Fe/H]) for a metallicity-dependent colour Teff; or give
                 bp_rp (+ ebv, ebv_err) for the Casagrande et al. (2021) colour Teff used in the catalogue.
    geometry     'phoebe' (PHOEBE contact meshes, as the catalogue; needs rochespec[phoebe]) or 'roche' (ray-traced Roche surface).
    Returns a FitResult (see .summary(), .as_dict(), .plot()).
    """
    spectra, survey, resolution = _as_spectra(spectra, wave, flux, err, ivar, times, survey, resolution)
    prior = _prior(teff_prior, bp_rp, ebv, ebv_err, logg)
    s = _target_row(period, t0, q, fillout, incl, prior, ra, dec, ebv)
    with _settings(geometry) as geom:
        from . import phoebe_geom
        phoebe_geom.STATUS.update(fallback=0, retries=0)
        if survey in (None, "lamost_lrs"):
            res = _fit_optical(s, spectra, survey, resolution, segments, prior, models)
        elif survey == "apogee":
            res = _fit_apogee(s, spectra, prior, models)
        else:
            if t0 is None:
                raise ValueError(f"{survey}: give t0 (the MRS/GALAH/DESI fits need fixed phases)")
            res = _fit_wingrid(s, spectra, survey, prior, models)
        fallback = bool(phoebe_geom.STATUS["fallback"]) if geom == "phoebe" else False
    res.geometry = "roche" if (geom == "roche" or fallback) else "phoebe"
    if fallback: res.flags.append("geometry_fallback")
    return res


def _fit_optical(s, spectra, survey, resolution, segments, prior, models):
    from .pipelines import lrs
    from .core import MH_FINE, joint
    from .post import calibrate_one
    from .teffphot import SIG_REL
    grid = _optical_grid()
    specs = []
    for k, sp in enumerate(spectra):
        fl, w, snr = lrs.prep(np.asarray(sp.wave, float), np.asarray(sp.flux, float), np.asarray(sp.ivar, float))
        specs.append(dict(oid=k, wl=np.asarray(sp.wave, float), fl=fl, w=w, bjd=np.atleast_1d(np.asarray(sp.times, float)), snr=snr))
    lsf_R = 1800.0 if survey == "lamost_lrs" else float(resolution)
    segs = segments if segments is not None else (None if survey == "lamost_lrs" else _generic_segments(spectra))
    r, x = lrs.fit_specs(s, specs, grid, lsf_R=lsf_R, segs=segs)
    nseg = len(segs or lrs.SEGS); npar = (9 * nseg + 1) * len(specs) + 2
    if survey == "lamost_lrs":
        zp = _zp("zp_lamost.json")
        j = calibrate_one(x["X2"], x["teff"], MH_FINE, x["npix"], s.Tphot0, s.dTdfeh, s.sig_Tphot, npar, zp)
        feh, feh_err = j["mh_cal"], j["mh_cal_err"]
    else:          # same inference settings as the catalogue (kappa 10, prior scatter 100 K), no zero point
        sT = min(float(s.sig_Tphot), 1e4); sigT = np.hypot(100.0, np.sqrt(max(sT ** 2 - SIG_REL ** 2, 0)))
        j = joint(x["X2"], np.asarray(x["teff"], float), MH_FINE, x["npix"], s.Tphot0, s.dTdfeh, sigT, npar=npar, kappa=10.0)
        feh, feh_err = np.nan, np.nan
    mh_err = float(np.hypot(j["mh_err_stat"], 0.07))
    teff_edge = bool(j.get("teff_edge", 0)) or j["teff_joint"] > 6700 or j["teff_joint"] < 4800
    out = FitResult(survey or "generic", float(j["mh_joint"]), mh_err, float(feh), float(feh_err), float(j["teff_joint"]), float(j.get("teff_err", np.nan)),
                    float(j["chi2_red"]), float(r["gamma"]), _flags(j.get("mh_edge", 0), teff_edge, j["chi2_red"], j["mh_joint"], r["eph"] != "VEB",
                    r["geo"] == "default", False, prior), "", float(r["q"]), float(r["f"]), float(r["i"]), len(specs),
                    wave=[sp["wl"] for sp in specs], flux=[sp["fl"] for sp in specs], weight=[sp["w"] for sp in specs],
                    chi2_grid=dict(teff=np.asarray(x["teff"]), mh=MH_FINE, X2=x["X2"]), pipeline=dict(fit=r, joint=j, segments=segs or lrs.SEGS))
    if models:
        out.model = x["best_model"](out.teff, out.mh)
    return out


def _fit_apogee(s, spectra, prior, models):
    from .pipelines import apogee as AP
    from .pipelines.apo_tphot import mh_at_tphot, DT
    from .post import invert
    grid = _h_grid()
    vis = []
    for sp in spectra:
        v = dict(sp.meta.get("visit", {}))
        if not v:       # a plain APOGEE-pixel spectrum: rest frame assumed (v_rad 0), already continuum normalised
            if len(sp.wave) != len(AP.WL) or not np.allclose(sp.wave, AP.WL, rtol=1e-6):
                sp = Spectrum(AP.WL.copy(), np.interp(AP.WL, sp.wave, sp.flux, left=np.nan, right=np.nan),
                              np.interp(AP.WL, sp.wave, sp.ivar, left=0, right=0), sp.times, "apogee", 22500.0)
            v = dict(f=np.asarray(sp.flux, float), iv=np.asarray(sp.ivar, float), v_rad=0.0, snr=float(np.nanmedian(np.sqrt(sp.ivar) * sp.flux)))
        v["bjd"] = np.atleast_1d(np.asarray(sp.times, float))
        vis.append(v)
    r, x = AP.fit_visits(s, vis, grid)
    zp = _zp("zp_apogee_tphot.json"); g = dict(X2=x["X2"], teff=np.asarray(x["teff"]), mh=AP.MH_FINE, npix=x["npix"])
    zs = 0.0; o = None
    for _ in range(2):
        o = mh_at_tphot(g, 9 * len(vis) + 2, s.Tphot0 + DT, s.dTdfeh, s.sig_Tphot, zp_shift=zs)
        if o is None: break
        F = float(invert(zp, o["T_used"], o["mh_tphot"])[0]); zs = o["mh_tphot"] - F
    if o is None:
        raise RuntimeError("APOGEE: no usable chi^2 grid (grid nodes undefined or too few pixels)")
    F = float(invert(zp, o["T_used"], o["mh_tphot"])[0]); dz = 1 + zp["coef"][zp["terms"].index("M")] if "M" in zp["terms"] else 1.0
    feh_err = float(np.sqrt((o["mh_tphot_stat"] / dz) ** 2 + (o["mh_tphot_err_T"] / dz) ** 2 + 0.07 ** 2))
    mh_err = float(np.sqrt(o["mh_tphot_stat"] ** 2 + o["mh_tphot_err_T"] ** 2 + 0.07 ** 2))
    out = FitResult("apogee", float(o["mh_tphot"]), mh_err, F, feh_err, float(o["T_used"]), float(s.sig_Tphot), float(o["chi2_red"]), float(r["gamma"]),
                    _flags(o["mh_tphot_edge"] or o["T_outside"], False, o["chi2_red"], o["mh_tphot"], r["eph"] != "VEB", r["geo"] == "default", False, prior),
                    "", float(r["q"]), float(r["f"]), float(r["i"]), len(vis), wave=[AP.WL.copy() for _ in vis], flux=[ob["o"] for ob in x["obs"]],
                    weight=[ob["w"] for ob in x["obs"]], chi2_grid=dict(teff=np.asarray(x["teff"]), mh=AP.MH_FINE, X2=x["X2"]), pipeline=dict(fit=r, tphot=o))
    if models:
        out.model = x["best_model"](out.teff, out.mh)
    return out


def _fit_wingrid(s, spectra, survey, prior, models):
    from .wingrid import RocheStar, fit_ew, mh_grid, model_on_pixels, legendre_basis
    from .core import geo_params
    mod = _survey_module(survey); grid = _win_grid(survey); mh = mh_grid(grid)
    specs = []
    for sp in spectra:
        if "spec" not in sp.meta:
            raise ValueError(f"{survey}: read the spectra with rochespec.io (the survey preprocessing and LSF are attached there)")
        x = copy.deepcopy(sp.meta["spec"]); x.ph = ((np.asarray(sp.times, float) - (s.TimeRef + 2455197.5)) * s.Freq) % 1; specs.append(x)
    qq, inc, fill, geo = geo_params(s)
    star = RocheStar(qq, fill, inc, s.Per); tc = float(np.clip(s.Tphot0 + s.dTdfeh * 0.0, 5000, 6500))
    kw = {"lamost_mrs": lambda: dict(sigma_kms=[mod.SIG] * len(mod.SEGS)),        # the production settings of each survey's fit()
          "galah": lambda: dict(sigma_kms=[mod.SIG] * len(mod.SEGS), gam_step=2.0, refine=6),
          "desi": lambda: dict(sigma_kms=[None] * len(mod.SEGS), lsf_apply=mod.make_lsf(grid), gam_range=200, gam_step=4.0, refine=0)}[survey]()
    sig = kw.pop("sigma_kms")
    res, sv = fit_ew(grid, star, specs, tc, sig, mh, **kw)
    zp, sd = _survey_zp_single(survey)
    feh = res["mh_at_Tcol"] - zp; hw = 0.5 * abs(res["mh_at_Tcol_hi"] - res["mh_at_Tcol_lo"])
    hw = 0.0 if not np.isfinite(hw) else hw
    tclip = not (4950 <= s.Tphot0 <= 6550)
    flags = _flags(res["edge"], False, res["chi2_red"], res["mh_at_Tcol"], False, geo == "default", False, prior)
    if tclip: flags.append("teff_clipped")
    if res["chi2_red"] >= 10: flags.append("poor_fit")
    out = FitResult(survey, float(res["mh_at_Tcol"]), float(np.hypot(hw, 0.07)), float(feh), float(np.sqrt(hw ** 2 + 0.07 ** 2 + sd ** 2)), tc,
                    float(s.sig_Tphot), float(res["chi2_red"]), float(res["gamma"]), sorted(set(flags), key=flags.index), "", float(star.q), fill, inc,
                    len(specs), wave=[x.wl for x in specs], flux=[x.fl for x in specs], weight=[x.w for x in specs],
                    chi2_grid=dict(teff=np.asarray(sv["teff"]), mh=np.asarray(sv["mh"]), X2=sv["X2"]), pipeline=dict(fit=res))
    if models:
        lsf = kw.get("lsf_apply")
        for x in specs:
            mods = grid.integrate(*star.elements(tc, x.ph, x.t_w), sigma_kms=sig)
            m = model_on_pixels(grid, mods, x, sv["gam"][x.group], np.array([out.mh]), lsf)[0]
            fm = np.full(len(x.wl), np.nan)
            for k, L in legendre_basis(x.wl, x.meta["segs"], x.meta["deg"]):
                kk = k & np.isfinite(m) & (x.w > 0)
                if kk.sum() < L.shape[1] + 5: continue
                Lk = L[kk[k]] * m[kk][:, None]; sw = np.sqrt(x.w[kk])
                c, *_ = np.linalg.lstsq(Lk * sw[:, None], x.fl[kk] * sw, rcond=None)
                ka = k & np.isfinite(m); fm[ka] = (L[ka[k]] * m[ka][:, None]) @ c
            out.model.append(fm)
    return out


# ------------------------------------------------------------------------------------------------------------------ model
def model(wave=None, *, teff, mh, period, phases=None, times=None, t0=None, q=None, fillout=None, incl=None, survey="lamost_lrs",
          resolution=None, rv_sys=0.0, geometry="phoebe"):
    """Normalised model spectrum F/Fc of a contact binary (optical or APOGEE H band), averaged over the given phases (or times with
    t0), at system Teff and [M/H], shifted by rv_sys (km/s), at the survey's resolution (or resolution=R). Returns (wave, flux):
    on the model grid, or interpolated onto wave (vacuum Angstrom) if given."""
    from .roche import ContactBinary, C
    from .model_cfg import MHS
    from .pipelines.lrs import norm_at
    if phases is None:
        if times is None or t0 is None: raise ValueError("give phases, or times and t0")
        phases = ((np.atleast_1d(np.asarray(times, float)) - t0) / period) % 1
    phases = np.atleast_1d(np.asarray(phases, float))
    g0 = DEFAULT_GEOMETRY
    qq = g0["q"] if q is None else (q if q <= 1 else 1 / q); f = g0["fillout"] if fillout is None else float(np.clip(fillout, 0.02, 0.9))
    inc = g0["incl"] if incl is None else incl
    band = "H" if survey == "apogee" else "opt"
    R = resolution or {"lamost_lrs": 1800.0, "lamost_mrs": 7500.0, "galah": 28000.0, "desi": 4000.0, "apogee": 22500.0}.get(survey, 1800.0)
    with _settings(geometry):
        grid = _h_grid() if band == "H" else _optical_grid()
        cb = ContactBinary(qq, f, ntheta=60, nphi=120)
        if hasattr(cb, "prepare"): cb.prepare(phases, inc, exact=len(phases) <= 40)
        M1 = (teff / 5772.0) ** 1.8; A = (M1 * (1 + qq) * (period / 365.25) ** 2) ** (1 / 3) * 215.03
        logg, Tl = cb.surface(M1, A, teff); el = [], [], [], [], []
        for p in phases:
            los, mu, vis = cb.view(p, inc); rv = cb.rv(los, A, period)
            for lst, val in zip(el, (Tl[vis], logg[vis], mu[vis], rv[vis], cb.dA[vis] / len(phases))): lst.append(val)
        F, Fc = grid.integrate_nodes(*[np.concatenate(x) for x in el], vmax=600, lsf_R=R)
    nm = norm_at(F, Fc, mh)
    wl = np.exp(grid.lnw + rv_sys / C)
    if wave is None:
        return wl, nm
    return np.asarray(wave, float), np.interp(np.log(np.asarray(wave, float)), grid.lnw + rv_sys / C, nm, left=np.nan, right=np.nan)
