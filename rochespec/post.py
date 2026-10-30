"""Re-run the joint (Teff, [M/H]) inference on the saved chi^2 grids with the calibration choices (kappa, prior
scatter) and apply the zero point. Used for the EW fits (LAMOST, APOGEE) and the calibrators alike.
"""
import json, os
import numpy as np, pandas as pd
from .core import D, joint
from .teffphot import SIG_REL


def zp_eval(zp, T, M):
    t = (np.asarray(T, float) - zp["T0"]) / zp["Tscale"]; M = np.asarray(M, float)
    cols = {"1": np.ones_like(t), "T": t, "M": M, "T2": t ** 2, "TM": t * M, "M2": M ** 2}
    return sum(c * cols[k] for k, c in zip(zp["terms"], zp["coef"]))


def invert(zp, T, mh, lo=-2.0, hi=1.0):
    """solve F + ZP(T, F) = mh for F on a fine grid (ZP is fitted against the true [Fe/H])"""
    T = np.atleast_1d(np.asarray(T, float)); mh = np.atleast_1d(np.asarray(mh, float)); Fg = np.linspace(lo, hi, 3001)
    out = np.full(len(T), np.nan)
    for i, (t, m) in enumerate(zip(T, mh)):
        if not (np.isfinite(t) and np.isfinite(m)): continue
        g = Fg + zp_eval(zp, np.full_like(Fg, t), Fg)
        if np.all(np.diff(g) > 0): out[i] = np.interp(m, g, Fg)
        else: out[i] = Fg[np.argmin(np.abs(g - m))]
    return out


def redo(rows, xdir, zp, npar_per_spec, nkey="n_spec", ebv_col=None, n_iter=2):
    """rows: DataFrame with sid, Tphot0, dTdfeh, sig_Tphot, n_spec/n_vis; returns calibrated columns"""
    out = []
    for _, r in rows.iterrows():
        fn = xdir + f"{int(r.sid)}.npz"
        if not os.path.exists(fn): out.append({}); continue
        g = np.load(fn); sT = min(float(r.sig_Tphot), 1e4) if np.isfinite(r.sig_Tphot) else 1e4; red = np.sqrt(max(sT ** 2 - SIG_REL ** 2, 0))
        sigT = np.hypot(zp["sig_rel"], red)
        npar = npar_per_spec * int(r[nkey]) + 2
        zp_now = 0.0; Tp0 = r.Tphot0
        for it in range(n_iter):
            # the C21 relation needs the true [Fe/H] = mh_model - ZP: shift the prior centre by -dTdfeh * ZP
            j = joint(g["X2"], g["teff"].astype(float), g["mh"], int(g["npix"]), Tp0 - r.dTdfeh * zp_now, r.dTdfeh, sigT, npar=npar, kappa=zp["kappa"])
            zp_now = float(j["mh_joint"] - invert(zp, j["teff_joint"], j["mh_joint"])[0])
        j["zp"] = zp_now; j["mh_cal"] = j["mh_joint"] - zp_now
        # the error propagates through the inverse: d F / d mh = 1 / (1 + dZP/dF)
        F = j["mh_cal"]; dz = (zp_eval(zp, j["teff_joint"], F + 0.01) - zp_eval(zp, j["teff_joint"], F - 0.01)) / 0.02
        j["mh_err_stat_cal"] = j["mh_err_stat"] / max(1 + float(dz), 0.2)
        j["mh_cal_err"] = float(np.hypot(j["mh_err_stat_cal"], 0.07))
        out.append(j)
    return pd.DataFrame(out, index=rows.index)
