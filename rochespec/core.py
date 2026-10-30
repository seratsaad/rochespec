"""Shared pieces for the at-scale contact-binary fits: joint (Teff, [M/H]) inference with a photometric Teff prior,
LAMOST DR10 spectrum reader with correct exposure times, ephemerides and geometry."""
from .config import DATA
import sys, re
import numpy as np
try:
    import pandas as pd
except ImportError:      # pandas is needed only by the batch pipelines
    pd = None
from astropy.io import fits
from astropy.table import Table

D = DATA + ""
MH_FINE = np.round(np.arange(-1.0, 0.501, 0.05), 3)
MODEL_FLOOR = 0.07


def read_dr10(fn):
    """LAMOST DR10 v2.0 LRS spectrum (COADD binary table). Returns wl (vacuum A), flux, ivar, exposure start times
    (UTC MJD) and the single-exposure duration (days).
    LMJM values (in EXPIDnn and LMJMLIST) are Beijing local time (UTC+8) in minutes since MJD 0; LMJMLIST can be
    truncated by the 68-character header limit, so the EXPID keywords are used when present."""
    with fits.open(fn) as h:
        H = h[0].header; d = h[1].data
        wl = np.asarray(d["WAVELENGTH"][0], float); fl = np.asarray(d["FLUX"][0], float); iv = np.asarray(d["IVAR"][0], float)
    lm = set()
    for k in H:
        if k.startswith("EXPID"):
            m = re.search(r"-(\d{8})$", str(H[k]).strip())
            if m: lm.add(int(m.group(1)))
    if not lm:
        lm = {int(x) for x in str(H.get("LMJMLIST", "")).split("-") if len(x.strip()) == 8}
    nexp = int(H.get("NEXP", max(len(lm), 1)))
    starts = np.array(sorted(lm), float)
    if len(starts) == 0:
        raise ValueError("no exposure times")
    # merge blue/red ids that differ by <= 1 minute
    keep = np.r_[True, np.diff(starts) > 1.5]; starts = starts[keep]
    texp = float(H["EXPTIME"]) / max(nexp, 1) / 86400.0
    return wl, fl, iv, starts / 1440.0 - 8.0 / 24.0, texp


def geometry():
    t = Table.read(D + "asassn/ii366_ecl.vot"); a = t.to_pandas(); a["sid"] = np.asarray(t["GaiaDR3"].filled(0), dtype="int64")
    lw = Table.read(D + "qtest/liwang25.vot").to_pandas()[["ASASSN-V", "q", "i", "f", "T2_T1", "l3"]]
    li = Table.read(D + "qtest/li24.vot").to_pandas()[["ASASSN-V", "q", "Inc", "f", "Ts_Tp", "l3"]].rename(columns={"Inc": "i", "Ts_Tp": "T2_T1"})
    g = pd.concat([lw.assign(src="LW25"), li.assign(src="Li24")]).drop_duplicates("ASASSN-V")
    return a[a.sid > 0].drop_duplicates("sid")[["sid", "ASASSN-V"]].merge(g, on="ASASSN-V", how="left").drop(columns=["ASASSN-V"])


def geo_params(s):
    """q (<=1), inclination, fill-out and the geometry source for a target row"""
    ok = np.isfinite(s.get("q", np.nan)) and np.isfinite(s.get("i", np.nan))
    q = s.q if ok else 0.35; inc = s.i if ok else 78.0
    fill = float(np.clip(s.f if (ok and np.isfinite(s.f)) else 0.2, 0.02, 0.9)); qq = q if q <= 1 else 1 / q
    return qq, inc, fill, (s.src if ok and isinstance(s.src, str) else "default")


def veb_ok(s):
    return np.isfinite(s.get("Freq", np.nan)) and abs(s.Per * s.Freq - 1) < 0.01


def bjd_phase(bjd, s):
    pmin = s.PhaseGauss1 if (np.nan_to_num(s.DepthGauss1) >= np.nan_to_num(s.DepthGauss2)) else s.PhaseGauss2
    if not np.isfinite(pmin): pmin = s.PhaseCHP
    return ((bjd - (2455197.5 + s.TimeRef)) * s.Freq - pmin) % 1


MH_MAX = 0.3          # complete payne-zero grid ends at +0.3; columns above (extrapolated) are dropped in the inference


def joint(X2, teff, mh, npix, Tp0, dTdf, sigT, npar=30, kappa=1.0, floor=MODEL_FLOOR, mh_max=MH_MAX):
    """Joint (Teff, [M/H]) inference: chi^2 scaled by its reduced value plus the Gaussian photometric prior
    T ~ N(Tp0 + dTdf*[M/H], sigT). Profile likelihood over the other parameter gives each marginal."""
    keep = mh <= mh_max + 1e-9; X2 = X2[:, keep]; mh = mh[keep]
    X2 = np.where(np.isfinite(X2), X2, 1e30)
    if not (X2.min() > 0) or X2.min() >= 1e11:
        return dict(chi2_red=np.nan, mh_joint=np.nan, mh_lo=np.nan, mh_hi=np.nan, mh_err_stat=np.nan, teff_joint=np.nan, teff_lo=np.nan, teff_hi=np.nan,
                    teff_err=np.nan, mh_edge=1, teff_edge=1, mh_specbest=np.nan, teff_specbest=np.nan, slope_mh_per100K=np.nan, mh_err=np.nan)
    scale = X2.min() / max(npix - npar, 1)
    c2 = X2 / scale / kappa                      # kappa > 1 widens the spectral likelihood for correlated model errors
    Tp = Tp0 + dTdf * mh[None, :]
    tot = c2 + ((teff[:, None] - Tp) / sigT) ** 2 if np.isfinite(sigT) and np.isfinite(Tp0) else c2.copy()
    def prof_stats(p, x):
        j = int(np.argmin(p)); p = p - p[j]
        # parabola through the minimum for a sub-grid estimate
        if 0 < j < len(x) - 1:
            a = 0.5 * (p[j + 1] + p[j - 1] - 2 * p[j]) / (x[1] - x[0]) ** 2
            b = (p[j + 1] - p[j - 1]) / (2 * (x[1] - x[0]))
            best = x[j] - b / (2 * a) if a > 0 else x[j]
        else:
            best = x[j]
        xf = np.linspace(x[0], x[-1], 2001); pf = np.interp(xf, x, p); inside = xf[pf <= 1.0]
        lo, hi = (inside.min(), inside.max()) if len(inside) else (best, best)
        return float(best), float(lo), float(hi), j in (0, len(x) - 1)
    pm = tot.min(axis=0); pt = tot.min(axis=1)
    mh_b, mh_lo, mh_hi, mh_edge = prof_stats(pm, mh)
    T_b, T_lo, T_hi, T_edge = prof_stats(pt, teff)
    # spectrum-only optimum and the conditional [M/H] at the prior centre
    a_, b_ = np.unravel_index(np.argmin(c2), c2.shape)
    out = dict(chi2_red=scale, mh_joint=mh_b, mh_lo=mh_lo, mh_hi=mh_hi, mh_err_stat=0.5 * (mh_hi - mh_lo),
               teff_joint=T_b, teff_lo=T_lo, teff_hi=T_hi, teff_err=0.5 * (T_hi - T_lo),
               mh_edge=int(mh_edge or mh_b <= mh[0] + 0.02 or mh_b >= mh[-1] - 0.02), teff_edge=int(T_edge),
               mh_specbest=mh[b_], teff_specbest=teff[a_],
               slope_mh_per100K=np.polyfit(teff, mh[np.argmin(c2, axis=1)], 1)[0] * 100 if len(teff) > 1 else np.nan)
    out["mh_err"] = float(np.hypot(out["mh_err_stat"], floor))
    return out


def teff_grid_for(Tp, sig, lo=4800, hi=6700, step=100):
    w = float(np.clip(2.5 * sig, 300, 700)) + step
    g = np.arange(lo, hi + 1, step); g = g[np.abs(g - Tp) <= w]
    if len(g) < 5:
        j = int(np.argmin(np.abs(np.arange(lo, hi + 1, step) - Tp))); full = np.arange(lo, hi + 1, step)
        g = full[max(0, min(j - 2, len(full) - 5)): max(0, min(j - 2, len(full) - 5)) + 5]
    return g


# CB_FINE=1: interpolate each chi^2 grid in Teff with a cubic spline (10 K steps) before profiling (fine_joint.py).
# Variant only; the catalogue leaves it unset. Modules that do "from .core import joint" pick it up at import time.
import os as _os
if _os.environ.get("CB_FINE"):
    from .fine_joint import make as _make_fine
    joint = _make_fine(joint)
