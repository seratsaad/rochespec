"""Zero point of the Roche/payne-zero [M/H] on LAMOST (spun-up single dwarfs) or APOGEE (spin-up test).

Forward model of the offset, fitted against the TRUE (ASPCAP) [Fe/H] so that the noise is only in the dependent
variable (a fit against the model [M/H] would shrink the calibrated scale):
    mh_model - feh = ZP(t, F),  t = (Teff_joint - 5800)/1000, F = feh (ASPCAP),  low-order polynomial chosen by BIC.
EWs are calibrated by solving F + ZP(t, F) = mh_model for F.
kappa (inflation of the spectral likelihood for correlated model errors) and the relation part of the prior scatter
are scanned; the adopted pair minimises the robust scatter of calibrated [M/H] - ASPCAP.
Usage: python -m rochespec.pipelines.calib_fit [lamost|apogee]
"""
import sys, os, json, glob, os
import numpy as np, pandas as pd
from ..core import D, joint
from ..teffphot import SIG_REL
from ..post import zp_eval, invert

which = sys.argv[1] if len(sys.argv) > 1 else "lamost"
rs = lambda x: 1.4826 * np.nanmedian(np.abs(np.asarray(x) - np.nanmedian(x)))
pri = pd.read_csv(D + "cbscale/cal_teffprior.csv")
if which == "lamost":
    res = pd.concat([pd.read_csv(f) for f in glob.glob(D + "cbscale/res_cal/calib_*.csv")])
    pri = pri[pri.set == "lamost_cal"]; xdir = D + "cbscale/res_cal/x2/"; nkey, npar, nph = "npix2", 19 + 2, 3
else:
    res = pd.concat([pd.read_csv(f) for f in glob.glob(D + "cbscale/res_spin/spin_*.csv")])
    pri = pri[pri.set == "spin"]; xdir = D + "cbscale/res_spin/x2/"; nkey, npar, nph = "npix", 9 + 2, 4
res = res.drop_duplicates("sdss_id").merge(pri[["sdss_id", "Tphot0", "dTdfeh", "sig_Tphot", "ebv"]], on="sdss_id", how="left")
grids = {s: np.load(xdir + f"{int(s)}.npz") for s in res.sdss_id if os.path.exists(xdir + f"{int(s)}.npz")}
res = res[res.sdss_id.isin(grids) & res.Tphot0.notna()].reset_index(drop=True)
F_true = res.feh_aspcap.values
print(f"{which}: calibrators with chi^2 grids: {len(res)}")


def run(kappa, sig_rel, use_prior=True, shift=None):
    out = []
    for i, r in res.iterrows():
        g = grids[r.sdss_id]
        sigT = np.hypot(sig_rel, np.sqrt(max(r.sig_Tphot ** 2 - SIG_REL ** 2, 0))) if use_prior else np.nan
        sh = 0.0 if shift is None else r.dTdfeh * shift[i]            # prior centre on the calibrated scale
        # the spun-up phases re-use one observed spectrum: count its information once
        out.append(joint(g["X2"] / nph, g["teff"].astype(float), g["mh"], int(g[nkey]) // nph, (r.Tphot0 - sh) if use_prior else np.nan,
                         r.dTdfeh, sigT, npar=npar, kappa=kappa))
    return pd.DataFrame(out)


def fit_zp(t, F, y, terms, nit=5):
    zp = dict(T0=5800, Tscale=1000, terms=terms, coef=[0.0] * len(terms))
    ok = np.isfinite(y) & np.isfinite(t)
    for _ in range(nit):
        A = np.stack([zp_eval(dict(zp, terms=[k], coef=[1.0]), t[ok], F[ok]) for k in terms], 1)
        c, *_ = np.linalg.lstsq(A, y[ok], rcond=None); zp["coef"] = c.tolist()
        r = y - zp_eval(zp, t, F); s = rs(r[ok]); ok = np.isfinite(y) & np.isfinite(t) & (np.abs(r) < 3 * s)
    n = ok.sum(); zp["bic"] = float(n * np.log(np.mean(r[ok] ** 2)) + len(terms) * np.log(n)); zp["n_used"] = int(n)
    return zp


TERMS = [["1"], ["1", "T"], ["1", "M"], ["1", "T", "M"], ["1", "T", "M", "TM"], ["1", "T", "M", "T2"], ["1", "T", "M", "T2", "TM"],
         ["1", "T", "M", "M2"], ["1", "T", "M", "T2", "M2"]]


def calibrate(jj, terms=None):
    T = jj.teff_joint.values; y = jj.mh_joint.values - F_true
    if terms is None:
        zps = [fit_zp(T, F_true, y, t) for t in TERMS]; zp = min(zps, key=lambda z: z["bic"])
    else:
        zp = fit_zp(T, F_true, y, terms); zps = [zp]
    mh_cal = invert(zp, T, jj.mh_joint.values)
    return zp, mh_cal, zps


# spectrum-only diagnostics
j0 = run(1.0, SIG_REL, use_prior=False)
dT = (j0.teff_joint - res.teff)[j0.teff_edge == 0]
print(f"spectrum-only Teff - ASPCAP: median {dT.median():+.0f} K, robust sigma {rs(dT):.0f} K, median formal error {j0.teff_err.median():.1f} K "
      f"(kappa matching the Teff scatter would be {(np.sqrt(max(rs(dT)**2 - 50**2, 100)) / max(j0.teff_err.median(), 0.5))**2:.0f})")
print(f"spectrum-only [M/H] - ASPCAP: median {np.median(j0.mh_joint - F_true):+.3f}, robust sigma {rs(j0.mh_joint - F_true):.3f}; "
      f"at the ASPCAP Teff: median {np.median(res.mh_roche - F_true):+.3f}, robust sigma {rs(res.mh_roche - F_true):.3f}")

scan = []
for kappa in [1, 10, 30, 100, 300, 1000, 3000]:
    for sig_rel in [60.0, 100.0, 150.0]:
        jj = run(kappa, sig_rel); zp, mh_cal, _ = calibrate(jj, ["1", "T", "M"])
        d = mh_cal - F_true; z = d / np.hypot(jj.mh_err_stat.values, 0.07)
        scan.append(dict(kappa=kappa, sig_rel=sig_rel, scatter=rs(d), z_sigma=rs(z), med_stat=float(np.median(jj.mh_err_stat)),
                         dT_prior=float(np.median(jj.teff_joint - (res.Tphot0 + res.dTdfeh * F_true)))))
scan = pd.DataFrame(scan); print(scan.round(3).to_string(index=False))
b = scan.loc[scan.scatter.idxmin()]; kappa, sig_rel = float(b.kappa), float(b.sig_rel)
if os.environ.get("CAL_FIX"):      # hold the production choice, e.g. CAL_FIX=10,100 (the scan picks among values 0.001 dex apart)
    kappa, sig_rel = (float(x) for x in os.environ["CAL_FIX"].split(","))
jj = run(kappa, sig_rel); zp, mh_cal, zps = calibrate(jj)
for z_ in zps: print(f"  ZP terms {'+'.join(z_['terms']):14s} BIC {z_['bic']:8.1f}")
for it in range(2):
    shift = jj.mh_joint.values - mh_cal                                  # = ZP at the solution
    jj = run(kappa, sig_rel, shift=shift); zp, mh_cal, _ = calibrate(jj, zp["terms"])
d = mh_cal - F_true; z = d / np.hypot(jj.mh_err_stat.values, 0.07)
floor = float(np.sqrt(max(rs(d) ** 2 - np.median(jj.mh_err_stat) ** 2 - 0.03 ** 2, 0.0)))
print(f"adopted: kappa {kappa:.0f}, sigma_rel {sig_rel:.0f} K, ZP terms {zp['terms']}, coef {np.round(zp['coef'], 4).tolist()}")
print(f"calibrated - ASPCAP: median {np.median(d):+.3f}, robust sigma {rs(d):.3f} (N={len(d)}); z-score robust sigma (stat + 0.07 floor) {rs(z):.2f}; "
      f"floor implied (ASPCAP 0.03 removed) {floor:.3f}; median stat error {np.median(jj.mh_err_stat):.3f}")
for nm, x, edges in [("ASPCAP Teff", res.teff.values, [4800, 5300, 5800, 6300, 6800]), ("ASPCAP [Fe/H]", F_true, [-1, -0.5, -0.2, 0.1, 0.5]),
                     ("log g", res.logg.values, [3.7, 4.1, 4.45, 4.8])]:
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x >= lo) & (x < hi)
        if k.sum(): print(f"   {nm} {lo}..{hi}: N={k.sum()} raw offset {np.median((jj.mh_joint.values - F_true)[k]):+.3f}, calibrated {np.median(d[k]):+.3f} +- {rs(d[k]):.3f}")
out = res.copy()
for col in ["mh_joint", "mh_err_stat", "teff_joint", "teff_err", "mh_edge", "teff_edge"]: out[col] = jj[col].values
out["mh_cal"] = mh_cal; out.to_csv(D + f"cbscale/calib_joint_{which}.csv", index=False); scan.to_csv(D + f"cbscale/calib_scan_{which}.csv", index=False)
zp.update(kappa=kappa, sig_rel=sig_rel, scatter=float(rs(d)), median_offset=float(np.median(d)), z_sigma=float(rs(z)), floor_implied=floor, n=int(len(res)),
          note="mh_model - feh = sum coef*term(t=(T_joint-T0)/Tscale, M=feh); invert for feh")
json.dump(zp, open(D + f"cbscale/zp_{which}.json", "w"), indent=1)
