"""Photometric Teff prior for every EW in the LAMOST DR10 and APOGEE DR19 samples; comparison with T_col (M&B 2020)."""
from ..config import DATA
import sys, numpy as np, pandas as pd
from astropy.table import Table
from ..teffphot import teff_phot
from ..photmass import teff_color
t = Table.read(DATA + "asassn/ii366_ecl.vot"); a = t.to_pandas(); a["sid"] = np.asarray(t["GaiaDR3"].filled(0), dtype="int64")
lam = pd.read_csv(DATA + "cbscale/ew_lamost_dr10_obs.csv")
ap = pd.read_csv(DATA + "cbsynth/dr19_all_ew.csv")
ids = set(lam.sid) | set(ap.sid)
e = a[a.sid.isin(ids)].drop_duplicates("sid")[["sid", "ASASSN-V", "RAJ2000", "DEJ2000", "Per", "Gmag", "BP-RP", "Plx", "E_B-V_", "Dist"]].copy()
fl = lam[lam.feh > -9].groupby("sid").apply(lambda g: np.average(g.feh, weights=g.snrg ** 2)).rename("feh_lam")
e = e.merge(fl, on="sid", how="left")
e["E_B-V_"] = e["E_B-V_"].fillna(0.0)
T0, s0, x0 = teff_phot(e["BP-RP"].values, e["E_B-V_"].values, 0.0)
T1, _, _ = teff_phot(e["BP-RP"].values, e["E_B-V_"].values, 1.0)
e["bprp0"] = x0; e["Tphot0"] = T0; e["dTdfeh"] = T1 - T0; e["sig_Tphot"] = s0
e["flag_ebv"] = (e["E_B-V_"] > 0.3).astype(int)
e["flag_colour"] = (~e.bprp0.between(0.2, 2.0)).astype(int)
# T_col of ii366 + LAMOST [Fe/H] (asassn/ew_lamost_colorT.csv) for comparison, at the same [Fe/H]
c = pd.read_csv(DATA + "asassn/ew_lamost_colorT.csv").drop_duplicates("source_id").rename(columns={"source_id": "sid"})
e = e.merge(c[["sid", "T_col", "feh_lam"]].rename(columns={"feh_lam": "feh_lam7"}), on="sid", how="left")
k = e.T_col.notna()
Tc21 = e.Tphot0 + e.dTdfeh * e.feh_lam7
d = (Tc21 - e.T_col)[k]
print(f"EWs: {len(e)}; with T_col: {k.sum()}; C21 - T_col: median {d.median():+.0f} K, MAD-sigma {1.4826*np.nanmedian(np.abs(d-d.median())):.0f} K, N finite {d.notna().sum()}")
for lo, hi in [(4800, 5300), (5300, 5800), (5800, 6300), (6300, 6900)]:
    kk = k & e.T_col.between(lo, hi); dd = (Tc21 - e.T_col)[kk]; print(f"  T_col {lo}-{hi}: N={kk.sum()} median {dd.median():+.0f}")
print("sigma_Tphot percentiles 10/50/90:", np.nanpercentile(e.sig_Tphot, [10, 50, 90]).round(0), " E(B-V)>0.3:", e.flag_ebv.sum(), " colour out of range:", e.flag_colour.sum())
# reddening check against Bayestar19 at the Gaia distance (where covered)
try:
    from dustmaps.bayestar import BayestarQuery
    from astropy.coordinates import SkyCoord; import astropy.units as u
    bq = BayestarQuery(max_samples=1); g = e[(e.Plx > 0.2)].copy()
    cc = SkyCoord(g.RAJ2000.values * u.deg, g.DEJ2000.values * u.deg, distance=(1000 / g.Plx.values) * u.pc)
    g["ebv_bayestar"] = 0.884 * bq(cc, mode="median")
    e = e.merge(g[["sid", "ebv_bayestar"]], on="sid", how="left")
    kb = e.ebv_bayestar.notna(); r = (e["E_B-V_"] - e.ebv_bayestar)[kb]
    print(f"ASAS-SN E(B-V) - Bayestar19 (at Gaia distance): N={kb.sum()} median {r.median():+.3f}, 16/84 {np.percentile(r,16):+.3f}/{np.percentile(r,84):+.3f}")
except Exception as ex:
    print("bayestar check failed", repr(ex)[:200])
e["sig_ebv"] = np.maximum(np.maximum(0.02, 0.15 * e["E_B-V_"]), (e["E_B-V_"] - e.ebv_bayestar).abs().fillna(0))
_, e["sig_Tphot"], _ = teff_phot(e["BP-RP"].values, e["E_B-V_"].values, 0.0, sig_ebv=e.sig_ebv.values)
e["flag_ebv"] = ((e["E_B-V_"] > 0.3) | ((e["E_B-V_"] - e.ebv_bayestar).abs() > 0.1)).astype(int)
print("final sigma_Tphot 10/50/90:", np.nanpercentile(e.sig_Tphot, [10, 50, 90]).round(0), " reddening flag:", e.flag_ebv.sum(), " no BP-RP:", e["BP-RP"].isna().sum())
e.to_csv(DATA + "cbscale/ew_teffphot.csv", index=False)
