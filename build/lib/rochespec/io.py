"""Readers for survey spectra. Each returns a list of Spectrum objects ready for rochespec.fit.

Times are BJD_TDB sample times across each exposure (three per exposure), computed from the file headers and the sky position;
the survey preprocessing (masks, normalisation, weights) is that of the production pipelines and is applied by fit().
"""
from dataclasses import dataclass, field
import numpy as np
from astropy.io import fits


@dataclass
class Spectrum:
    """one observed spectrum. wave: vacuum Angstrom (barycentric or heliocentric frame); flux; ivar (inverse variance, 0 = bad);
    times: BJD_TDB sample times covering the exposure (one value is fine for short exposures); survey: one of rochespec.SURVEYS or
    None for a generic spectrum (give resolution); meta: reader extras."""
    wave: np.ndarray
    flux: np.ndarray
    ivar: np.ndarray
    times: np.ndarray
    survey: str = None
    resolution: float = None
    meta: dict = field(default_factory=dict)


def _radec(h, ra, dec):
    if ra is None or dec is None:
        for kr, kd in (("RA", "DEC"), ("RA_OBJ", "DEC_OBJ"), ("RA_OBS", "DEC_OBS"), ("PLUG_RA", "PLUG_DEC")):
            if kr in h and kd in h:
                return float(h[kr]), float(h[kd])
        raise ValueError("no RA/Dec in the header: pass ra= and dec= (degrees) to compute barycentric times")
    return float(ra), float(dec)


def read_lamost_lrs(fn, ra=None, dec=None):
    """LAMOST DR10 (v2.0) LRS FITS file (COADD binary table). Exposure times come from the EXPID/LMJM keywords (Beijing time)."""
    from .core import read_dr10
    from .pipelines.lrs import lamost_bjd
    with fits.open(fn) as h:
        ra, dec = _radec(h[0].header, ra, dec)
    wl, fl, iv, starts, texp = read_dr10(fn)
    return [Spectrum(wl, fl, iv, lamost_bjd(starts, texp, ra, dec), "lamost_lrs", 1800.0, dict(file=str(fn), ra=ra, dec=dec))]


def read_lamost_mrs(fn, ra=None, dec=None, mode="exp"):
    """LAMOST DR10 (v2.0) MRS FITS file: one Spectrum per exposure (mode 'exp') or per coadd ('coadd')."""
    from .pipelines.surveys import mrs
    from .wingrid import bjd
    with fits.open(fn) as h:
        ra, dec = _radec(h[0].header, ra, dec)
    out = []
    for ep in mrs.read(fn, mode):
        sp = mrs.prep(ep, str(fn))
        if sp is None: continue
        out.append(Spectrum(sp.wl, sp.fl, sp.w, bjd(sp.t_mjd, ra, dec, "lamost"), "lamost_mrs", 7500.0, dict(spec=sp, file=str(fn), ra=ra, dec=dec)))
    return out


def read_galah(files, ra=None, dec=None, label="galah", exposures=None):
    """GALAH DR4 spectra of one epoch: the per-CCD FITS files (1-4); they are joined into one Spectrum. exposures: optional list of
    (mjd_beg, mjd_end) UTC of the individual exposures of a combined spectrum (as the catalogue fits used, from the DR4 datalink);
    otherwise the MJD-BEG/MJD-END span of the file is used."""
    from .pipelines.surveys import galah
    from .wingrid import bjd, substeps
    files = [files] if isinstance(files, str) else list(files)
    with fits.open(files[0]) as h:
        ra, dec = _radec(h[0].header, ra, dec)
    sp = galah.assemble(files, label, label)
    if sp is None:
        return []
    if exposures:
        sp.t_mjd = np.concatenate([substeps(b, e, 3) for b, e in exposures]); sp.t_w = np.ones(len(sp.t_mjd)) / len(sp.t_mjd)
        sp.meta["nexp"] = len(exposures)
    return [Spectrum(sp.wl, sp.fl, sp.w, bjd(sp.t_mjd, ra, dec, "aat"), "galah", 28000.0, dict(spec=sp, files=files, ra=ra, dec=dec))]


def read_desi(coadd, targetid, ra=None, dec=None):
    """DESI DR1 coadd file (local path or URL) and TARGETID: one Spectrum with the per-target resolution matrix."""
    from .pipelines.surveys import desi
    from .wingrid import bjd
    z = desi.extract(coadd, targetid)
    if ra is None or dec is None:
        with fits.open(coadd, lazy_load_hdus=True, **({"use_fsspec": True} if str(coadd).startswith("http") else {})) as h:
            fm = h["FIBERMAP"].data; j = np.where(np.asarray(fm["TARGETID"]) == int(targetid))[0][0]
            ra, dec = float(fm["TARGET_RA"][j]), float(fm["TARGET_DEC"][j])
    sp = desi.read(z)
    return [Spectrum(sp.wl, sp.fl, sp.w, bjd(sp.t_mjd, ra, dec, "kpno"), "desi", None, dict(spec=sp, file=str(coadd), targetid=int(targetid), ra=ra, dec=dec))]


def read_apogee_visits(fn, ra=None, dec=None, snr_min=20.0):
    """SDSS DR19 mwmVisit file: one Spectrum per APOGEE visit with S/N >= snr_min and a pipeline velocity. The flux is continuum
    normalised by the pipeline and in the stellar rest frame of the pipeline velocity, as in the production fits."""
    from .pipelines import apogee as AP
    old = AP.SNR_MIN; AP.SNR_MIN = snr_min
    try:
        vis = AP.read_visits(fn)
    finally:
        AP.SNR_MIN = old
    with fits.open(fn) as h:
        hdr = h[0].header if ("RA" in h[0].header or ra is not None) else h[1].header
        if ra is None and "RA" not in hdr and len(h) > 1 and h[1].data is not None and "ra" in [c.lower() for c in h[1].columns.names]:
            ra, dec = float(h[1].data["ra"][0]), float(h[1].data["dec"][0])
        ra, dec = _radec(hdr, ra, dec)
    return [Spectrum(AP.WL.copy(), v["f"], v["iv"], AP.visit_bjd(v, ra, dec), "apogee", 22500.0, dict(visit=v, file=str(fn), ra=ra, dec=dec)) for v in vis]


READERS = {"lamost_lrs": read_lamost_lrs, "lamost_mrs": read_lamost_mrs, "galah": read_galah, "desi": read_desi, "apogee": read_apogee_visits}


def read(path, survey, **kw):
    """read a survey file with the matching reader (see READERS)"""
    return READERS[survey](path, **kw)
