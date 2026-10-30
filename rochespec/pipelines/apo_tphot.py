"""APOGEE [M/H] read at the photometric Teff (fix for the Teff grid-edge problem).

Diagnosis (catalog/PIPELINE_LOG.md section 14): the H-band payne-zero/Roche chi^2 prefers Teff far below the photometric value for
single stars as well as contact binaries (calibrators: joint Teff = ASPCAP - 300 K, the window edge, for 107 of 120), and with
kappa = 1 the spectrum overrides the prior, so every APOGEE joint Teff sits on the low edge of its window. The window half-width
differs between the calibrators (300 K) and the contact binaries (300-700 K, from sig_Tphot), so [M/H] was read at different Teff
offsets, which the zero point cannot absorb.
Fix: as for MRS/GALAH/DESI (T_col), read [M/H] at the photometric Teff T(m) = Tphot0 + dTdfeh [Fe/H] from the saved chi^2 grids
(linear interpolation in Teff), for the calibrators and the contact binaries alike, and refit the zero point on the calibrators.
Error: Delta chi^2 = 1 at that Teff (chi^2 scaled by its reduced value) plus the Teff error propagated through the star's
degeneracy slope k (best [M/H] against Teff), plus the 0.07 floor.
Outputs: cbscale/zp_apogee_tphot.json, cbscale/res_apo/tphot.csv, cbscale/calib_tphot_apogee.csv
"""
import sys, os, json, glob
import numpy as np
try:
    import pandas as pd
except ImportError:      # pandas is needed only by the batch pipelines
    pd = None
from ..core import D
from ..post import zp_eval, invert
rs = lambda x: 1.4826 * np.nanmedian(np.abs(np.asarray(x) - np.nanmedian(x)))
MH_MAX = 0.3
DT = -100.0          # read at T_phot + DT; chosen on the calibrators by apo_tfix_scan.py (scatter 0.090 at -100 K vs 0.130 at 0)


def mh_at_tphot(g, npar, Tp0, dTdf, sigT, zp_shift=0.0):
    X2, tg, mh = g["X2"], g["teff"].astype(float), g["mh"]; keep = mh <= MH_MAX + 1e-9; X2, mh = X2[:, keep], mh[keep]
    if not np.isfinite(X2).all() or X2.min() <= 0 or len(tg) < 2: return None
    scale = X2.min() / max(int(g["npix"]) - npar, 1)
    T_m = Tp0 + dTdf * (mh - zp_shift)                                   # C21 prior centre at the true [Fe/H] = mh - ZP
    outside = (T_m < tg[0] - 1) | (T_m > tg[-1] + 1)
    c = np.array([np.interp(t_, tg, X2[:, j]) for j, t_ in enumerate(np.clip(T_m, tg[0], tg[-1]))]) / scale
    j = int(np.argmin(c)); p = c - c[j]
    if 0 < j < len(mh) - 1:
        a = 0.5 * (p[j + 1] + p[j - 1]) / (mh[1] - mh[0]) ** 2; b = (p[j + 1] - p[j - 1]) / (2 * (mh[1] - mh[0]))
        best = mh[j] - b / (2 * a) if a > 0 else mh[j]
    else:
        best = mh[j]
    xf = np.linspace(mh[0], mh[-1], 3001); inside = xf[np.interp(xf, mh, p) <= 1.0]
    stat = 0.5 * (inside.max() - inside.min()) if len(inside) else 0.0
    k = np.polyfit(tg, mh[np.argmin(X2, axis=1)], 1)[0] if len(tg) > 2 else 0.0          # dex per K
    return dict(mh_tphot=float(best), mh_tphot_stat=float(stat), k_deg=float(k), mh_tphot_err_T=float(abs(k) * min(sigT, 1e3)),
                mh_tphot_edge=int(j in (0, len(mh) - 1)), T_used=float(np.interp(best, mh, T_m)), T_outside=int(outside[j]), chi2_red=float(scale))


def _main():
    # ---- calibrators (APOGEE spin-up test set, res_spin) -----------------------------------------------------------------
    pri = pd.read_csv(D + "cbscale/cal_teffprior.csv"); pri = pri[pri.set == "spin"]
    res = pd.concat([pd.read_csv(f) for f in glob.glob(D + "cbscale/res_spin/spin_*.csv")]).drop_duplicates("sdss_id")
    res = res.merge(pri[["sdss_id", "Tphot0", "dTdfeh", "sig_Tphot"]], on="sdss_id", how="left")
    rows = []
    for _, r in res.iterrows():
        fn = D + f"cbscale/res_spin/x2/{int(r.sdss_id)}.npz"
        if not os.path.exists(fn) or not np.isfinite(r.Tphot0): continue
        o = mh_at_tphot(np.load(fn), 9 + 2, r.Tphot0 + DT, r.dTdfeh, r.sig_Tphot)
        if o: rows.append(dict(sdss_id=r.sdss_id, teff=r.teff, feh_aspcap=r.feh_aspcap, Tphot0=r.Tphot0, **o))
    cal = pd.DataFrame(rows)
    cal["Tp"] = cal.Tphot0 + cal.dTdfeh * cal.feh_aspcap if "dTdfeh" in cal else cal.T_used
    ok = (cal.mh_tphot_edge == 0) & (cal.T_outside == 0)
    print(f"calibrators: {len(cal)}, usable {ok.sum()} (edge {(cal.mh_tphot_edge == 1).sum()}, T_phot outside the ASPCAP +-300 K grid {(cal.T_outside == 1).sum()})")
    # zero point: mh - F_true = ZP(t, F), BIC over term sets (same scheme as calib_analysis.py)
    c = cal[ok]; t = (c.T_used - 5800) / 1000; F = c.feh_aspcap; y = c.mh_tphot - F
    best = None
    for terms in [["1"], ["1", "T"], ["1", "M"], ["1", "T", "M"], ["1", "T", "M", "TM"]]:
        X = np.c_[[{"1": np.ones(len(t)), "T": t, "M": F, "TM": t * F}[k_] for k_ in terms]].T
        coef, *_ = np.linalg.lstsq(X, y, rcond=None); res_ = y - X @ coef
        bic = len(y) * np.log(np.mean(res_ ** 2)) + len(terms) * np.log(len(y))
        if best is None or bic < best[0]: best = (bic, terms, coef, res_)
    bic, terms, coef, res_ = best
    zp = dict(T0=5800, Tscale=1000, terms=terms, coef=[float(x) for x in coef], bic=float(bic), n_used=int(len(c)), mode="mh at T_phot + DT", DT=DT,
              note="mh_model(T_phot) - feh = sum coef*term(t=(T_phot-T0)/Tscale, M=feh); invert for feh")
    cal["mh_cal"] = invert(zp, cal.T_used.values, cal.mh_tphot.values)
    d = (cal.mh_cal - cal.feh_aspcap)[ok]
    zp.update(scatter=float(rs(d)), median_offset=float(np.median(d)), scatter_old_kappa1=0.0779)
    json.dump(zp, open(D + "cbscale/zp_apogee_tphot.json", "w"), indent=1); cal.to_csv(D + "cbscale/calib_tphot_apogee.csv", index=False)
    print(f"ZP terms {terms} coef {np.round(coef, 4).tolist()}; calibrated - ASPCAP median {np.median(d):+.3f}, robust sd {rs(d):.3f} (old joint/kappa=1: 0.078)")
    for lo, hi in [(-1, -0.4), (-0.4, -0.1), (-0.1, 0.2), (0.2, 0.6)]:
        k_ = ok & cal.feh_aspcap.between(lo, hi); print(f"   ASPCAP [Fe/H] {lo:+.1f}..{hi:+.1f}: N={k_.sum()} median {np.median((cal.mh_cal - cal.feh_aspcap)[k_]):+.3f}")
    print(f"   degeneracy slope k median {np.median(cal.k_deg) * 100:.3f} dex/100K; Teff error term median {np.median(cal.mh_tphot_err_T):.3f}")


def apply(rows, xdir, zp, nkey="n_vis", npp=9):
    """calibrated APOGEE [Fe/H] at T_phot + DT for result rows (sid, Tphot0, dTdfeh, sig_Tphot, n_vis); two iterations for the ZP shift"""
    out = []
    for _, r in rows.iterrows():
        fn = xdir + f"{int(r.sid)}.npz"
        if not os.path.exists(fn) or not np.isfinite(r.Tphot0): out.append({}); continue
        g = np.load(fn); zs = 0.0; o = None
        for _ in range(2):
            o = mh_at_tphot(g, npp * int(r[nkey]) + 2, r.Tphot0 + DT, r.dTdfeh, r.sig_Tphot, zp_shift=zs)
            if o is None: break
            F = float(invert(zp, o["T_used"], o["mh_tphot"])[0]); zs = o["mh_tphot"] - F
        if o is None: out.append({}); continue
        F = float(invert(zp, o["T_used"], o["mh_tphot"])[0]); dz = 1 + zp["coef"][zp["terms"].index("M")] if "M" in zp["terms"] else 1.0
        o.update(mh_cal=F, mh_cal_err=float(np.sqrt((o["mh_tphot_stat"] / dz) ** 2 + (o["mh_tphot_err_T"] / dz) ** 2 + 0.07 ** 2)), zp_used=o["mh_tphot"] - F)
        out.append(o)
    return pd.DataFrame(out, index=rows.index)


if __name__ == "__main__":
    _main()
    zp = json.load(open(D + "cbscale/zp_apogee_tphot.json"))
    A = pd.concat([pd.read_csv(f) for f in glob.glob(D + "cbscale/res_apo/results_*.csv")]).sort_values("n_vis", ascending=False).drop_duplicates("sid")
    A = A[A.n_vis > 0].reset_index(drop=True)
    R = A[["sid", "n_vis", "Tphot0", "sig_Tphot", "mh_cal" if "mh_cal" in A else "mh_joint"]].join(apply(A, D + "cbscale/res_apo/x2/", zp))
    R.to_csv(D + "cbscale/res_apo/tphot.csv", index=False)
    good = (R.mh_tphot_edge == 0) & (R.T_outside == 0)
    print(f"catalogue APOGEE: {len(R)} fitted, good at T_phot{DT:+.0f} K: {good.sum()} (edge {(R.mh_tphot_edge == 1).sum()}, outside grid {(R.T_outside == 1).sum()}); "
          f"median [Fe/H] {R.mh_cal[good].median():+.2f}, median error {R.mh_cal_err[good].median():.3f}")
