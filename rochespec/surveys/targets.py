"""Target tables (EWs) per survey: Gaia DR3 VEB ephemeris, light-curve geometry, colour Teff, survey pipeline values, overlaps."""
from ..config import DATA
import sys, glob, numpy as np, pandas as pd
from astropy.table import Table
from ..photmass import teff_color
D = DATA + ""


def geometry():
    t = Table.read(D + "asassn/ii366_ecl.vot"); a = t.to_pandas(); a["sid"] = np.asarray(t["GaiaDR3"].filled(0), dtype="int64")
    lw = Table.read(D + "qtest/liwang25.vot").to_pandas()[["ASASSN-V", "q", "i", "f", "T2_T1", "l3"]]
    li = Table.read(D + "qtest/li24.vot").to_pandas()[["ASASSN-V", "q", "Inc", "f", "Ts_Tp", "l3"]].rename(columns={"Inc": "i", "Ts_Tp": "T2_T1"})
    g = pd.concat([lw.assign(src="LW25"), li.assign(src="Li24")]).drop_duplicates("ASASSN-V")
    a = a[a.sid > 0].drop_duplicates("sid")
    return a[["sid", "ASASSN-V", "RAJ2000", "DEJ2000", "Per", "Vmag", "E_B-V_", "Prob", "Type"]].merge(g, on="ASASSN-V", how="left")


def base():
    geo = geometry()
    veb = pd.read_csv("cats/veb_raw.csv").rename(columns={"Source": "sid"})
    gaia = pd.read_csv(D + "asassn/asassn_gaia.csv", usecols=["source_id", "bp_rp", "gmag"]).rename(columns={"source_id": "sid"}).drop_duplicates("sid")
    lam = pd.read_csv(D + "asassn/ew_lamost_colorT.csv").drop_duplicates("source_id").rename(columns={"source_id": "sid"})[["sid", "T_col", "feh_lam"]]
    apo = pd.concat([pd.read_csv(f) for f in glob.glob(D + "cbapo/results_pz_*.csv")]).drop_duplicates("sid")[["sid", "mh_at_Tcol", "chi2_red"]].rename(columns={"mh_at_Tcol": "mh_apo", "chi2_red": "chi2_apo"})
    lrs = pd.concat([pd.read_csv(f) for f in glob.glob(D + "cblam/results_pz_*.csv")]).drop_duplicates("sid")[["sid", "mh_at_Tcol", "chi2_red"]].rename(columns={"mh_at_Tcol": "mh_lrs", "chi2_red": "chi2_lrs"})
    return geo, veb, gaia, lam, apo, lrs


def build(survey, ids, feh_pipe):
    geo, veb, gaia, lam, apo, lrs = base()
    t = pd.DataFrame({"sid": ids, "feh_pipe": feh_pipe}).merge(geo, on="sid", how="left").merge(veb, on="sid", how="left").merge(gaia, on="sid", how="left")
    t = t.merge(lam, on="sid", how="left").merge(apo, on="sid", how="left").merge(lrs, on="sid", how="left")
    tc = teff_color(t.bp_rp - 1.34 * t["E_B-V_"].fillna(0), np.clip(np.nan_to_num(t.feh_pipe, nan=0.0), -1, 0.3))
    t["T_col"] = np.where(np.isfinite(t.T_col), t.T_col, tc)
    t["has_veb"] = np.isfinite(t.Freq); t["P"] = np.where(t.has_veb, 1 / t.Freq, t.Per)
    t.to_csv(f"{survey}/targets.csv", index=False)
    print(survey, "EWs", len(t), "VEB", t.has_veb.sum(), "geometry", t.src.notna().sum(), "APOGEE overlap", t.mh_apo.notna().sum(), "LRS overlap", t.mh_lrs.notna().sum(),
          "both", (t.mh_apo.notna() & t.mh_lrs.notna()).sum(), "VEB & (APO|LRS)", (t.has_veb & (t.mh_apo.notna() | t.mh_lrs.notna())).sum())
    return t


if __name__ == "__main__":
    import os
    for s in ["lamost_mrs", "galah", "desi"]: os.makedirs(s, exist_ok=True)
    m = pd.read_csv("cats/mrs_dr10.csv").query("is_ew"); m["feh_lasp"] = m.feh_lasp.where(m.feh_lasp > -9)
    a = m.groupby("sid").agg(feh=("feh_lasp", "median"), n_obs=("obsid", "size"), snr=("snr", "max")).reset_index()
    t = build("lamost_mrs", a.sid.values, a.feh.values); t.merge(a[["sid", "n_obs", "snr"]], on="sid").to_csv("lamost_mrs/targets.csv", index=False)
    g = pd.read_csv("cats/galah_allspec.csv").query("is_ew")
    a = g.groupby("sid").agg(feh=("fe_h", "median"), n_obs=("sobject_id", "size"), snr=("snr_px_ccd2", "max")).reset_index()
    t = build("galah", a.sid.values, a.feh.values); t.merge(a[["sid", "n_obs", "snr"]], on="sid").to_csv("galah/targets.csv", index=False)
    d = pd.read_csv("cats/desi_mws.csv").query("is_ew")
    a = d.groupby("sid").agg(feh=("FEH", "median"), n_obs=("TARGETID", "size"), snr=("SN_R", "max")).reset_index()
    t = build("desi", a.sid.values, a.feh.values); t.merge(a[["sid", "n_obs", "snr"]], on="sid").to_csv("desi/targets.csv", index=False)
