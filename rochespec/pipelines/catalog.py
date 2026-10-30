"""Combined calibrated contact-binary metallicity catalogue: $ROCHESPEC_DATA/catalog/cb_feh_v0.csv

Re-runs the joint inference on every saved chi^2 grid with the calibration choices (zp_lamost.json, zp_apogee.json),
applies the zero points, and merges the surveys (inverse-variance mean where both exist).
"""
from ..config import DATA
import sys, os, glob, json
import numpy as np, pandas as pd
from ..core import D
from ..post import redo

zl = json.load(open(D + "cbscale/zp_lamost.json")); za = json.load(open(D + "cbscale/zp_apogee.json"))
L = pd.concat([pd.read_csv(f) for f in glob.glob(D + "cbscale/res_lam/results_*.csv")]).sort_values("n_spec", ascending=False).drop_duplicates("sid")
A = pd.concat([pd.read_csv(f) for f in glob.glob(D + "cbscale/res_apo/results_*.csv")]).sort_values("n_vis", ascending=False).drop_duplicates("sid")
nL0, nA0 = len(L), len(A)
L = L[L.n_spec > 0].reset_index(drop=True); A = A[A.n_vis > 0].reset_index(drop=True)
print(f"LAMOST: {nL0} stars attempted, {len(L)} fitted; APOGEE: {nA0} attempted, {len(A)} fitted")
L = L.join(redo(L, D + "cbscale/res_lam/x2/", zl, 19, "n_spec"), rsuffix="_c")
A = A.join(redo(A, D + "cbscale/res_apo/x2/", za, 9, "n_vis"), rsuffix="_c")
# APOGEE: [M/H] read at T_phot - 100 K with its own zero point (apo_tfix.py; PIPELINE_LOG section 14). APO_MODE=joint keeps the old
# joint inference, whose Teff always ends on the low edge of its window.
if os.environ.get("APO_MODE", "tphot") == "tphot":
    from .apo_tphot import apply as apo_apply
    za = json.load(open(D + "cbscale/zp_apogee_tphot.json"))
    T = apo_apply(A, D + "cbscale/res_apo/x2/", za)
    A["mh_joint_c"], A["mh_cal"], A["mh_cal_err"], A["zp"] = T.mh_tphot, T.mh_cal, T.mh_cal_err, T.zp_used
    A["mh_err_stat_c"], A["chi2_red_c"], A["teff_joint_c"], A["teff_err_c"] = T.mh_tphot_stat, T.chi2_red, T.T_used, A.sig_Tphot
    A["mh_edge_c"] = ((T.mh_tphot_edge == 1) | (T.T_outside == 1)).astype(int); A["teff_edge_c"] = 0
    print(f"APOGEE at T_phot{za['DT']:+.0f} K: {T.mh_cal.notna().sum()} values, edge/outside {A.mh_edge_c.sum()}")


def flags(d):
    f = []
    for _, r in d.iterrows():
        s = []
        if r.mh_edge_c: s.append("mh_edge")
        if r.mh_joint_c > 0.3: s.append("mh_extrap")
        if r.teff_edge_c or r.teff_joint_c > 6700 or r.teff_joint_c < 4800: s.append("teff_edge")
        if r.chi2_red_c > 5: s.append("poor_fit")
        if r.flag_ebv: s.append("ebv")
        if r.flag_colour: s.append("colour_range")
        if r.eph == "nuisance": s.append("no_veb")
        if r.geo == "default": s.append("default_geometry")
        f.append(";".join(s))
    return f


L["flags"] = flags(L); A["flags"] = flags(A)
keep = ["sid", "P", "teff_joint_c", "teff_err_c", "mh_joint_c", "zp", "mh_cal", "mh_cal_err", "mh_err_stat_c", "chi2_red_c", "geo", "eph", "flags",
        "Tphot0", "dTdfeh", "sig_Tphot", "ebv", "q", "i", "f"]
Lk = L[keep + ["n_spec", "feh_lam", "teff_lam", "logg_lam", "snr_tot"]].rename(columns=lambda c: c.replace("_c", "") if c.endswith("_c") else c)
Ak = A[keep + ["n_vis", "sdss_id", "snr_tot"]].rename(columns=lambda c: c.replace("_c", "") if c.endswith("_c") else c)
Lk = Lk.add_suffix("_lam").rename(columns={"sid_lam": "source_id"}); Ak = Ak.add_suffix("_apo").rename(columns={"sid_apo": "source_id"})
cat = Lk.merge(Ak, on="source_id", how="outer")
cat["period"] = cat.P_lam.fillna(cat.P_apo)
cat["survey"] = np.where(cat.mh_cal_lam.notna() & cat.mh_cal_apo.notna(), "LAMOST+APOGEE", np.where(cat.mh_cal_lam.notna(), "LAMOST", "APOGEE"))
cat["n_spectra"] = cat.n_spec_lam.fillna(0).astype(int) + cat.n_vis_apo.fillna(0).astype(int)
wl = 1 / cat.mh_cal_err_lam ** 2; wa = 1 / cat.mh_cal_err_apo ** 2
both = cat.survey == "LAMOST+APOGEE"
cat["feh"] = np.where(both, (cat.mh_cal_lam * wl + cat.mh_cal_apo * wa) / (wl + wa), cat.mh_cal_lam.fillna(cat.mh_cal_apo))
cat["feh_err"] = np.where(both, 1 / np.sqrt(wl + wa), cat.mh_cal_err_lam.fillna(cat.mh_cal_err_apo))
tl = 1 / cat.teff_err_lam.clip(lower=10) ** 2; ta = 1 / cat.teff_err_apo.clip(lower=10) ** 2
cat["teff"] = np.where(both, (cat.teff_joint_lam * tl + cat.teff_joint_apo * ta) / (tl + ta), cat.teff_joint_lam.fillna(cat.teff_joint_apo))
cat["teff_err"] = np.where(both, 1 / np.sqrt(tl + ta), cat.teff_err_lam.fillna(cat.teff_err_apo))
cat["geometry_source"] = cat.geo_lam.fillna(cat.geo_apo)
cat["flags"] = [";".join(sorted(set([x for x in (str(a) if isinstance(a, str) else "").split(";") + (str(b) if isinstance(b, str) else "").split(";") if x])))
                for a, b in zip(cat.flags_lam, cat.flags_apo)]
cat["good"] = ~cat["flags"].str.contains("mh_edge|teff_edge|poor_fit")
# per-survey flags (a flag from one survey no longer removes the other survey's value)
bad = "mh_edge|teff_edge|poor_fit"
cat["good_lam"] = cat.mh_cal_lam.notna() & ~cat.flags_lam.fillna("").str.contains(bad)
cat["good_apo"] = cat.mh_cal_apo.notna() & ~cat.flags_apo.fillna("").str.contains(bad)
cat["good"] = cat.good_lam | cat.good_apo
cols = ["source_id", "period", "survey", "teff", "teff_err", "feh", "feh_err", "n_spectra", "geometry_source", "flags", "good", "good_lam", "good_apo",
        "mh_cal_lam", "mh_cal_err_lam", "teff_joint_lam", "teff_err_lam", "chi2_red_lam", "n_spec_lam", "feh_lam_lam", "teff_lam_lam", "zp_lam", "mh_joint_lam",
        "mh_cal_apo", "mh_cal_err_apo", "teff_joint_apo", "teff_err_apo", "chi2_red_apo", "n_vis_apo", "sdss_id_apo", "zp_apo", "mh_joint_apo",
        "Tphot0_lam", "Tphot0_apo", "dTdfeh_lam", "dTdfeh_apo", "sig_Tphot_lam", "sig_Tphot_apo", "ebv_lam", "ebv_apo", "eph_lam", "eph_apo", "q_lam", "i_lam", "f_lam"]
cat = cat[cols].rename(columns={"feh_lam_lam": "feh_lamost_pipeline", "teff_lam_lam": "teff_lamost_pipeline", "chi2_red_lam": "chi2_red_lam", "sdss_id_apo": "sdss_id"})
cat["chi2_red"] = cat.chi2_red_lam.fillna(cat.chi2_red_apo)
# other surveys (separate agent; T_col-based fits with single-star zero points, surveys/<s>/results_cal.csv), adopted mh_at_Tcol_cal
import os
S = DATA + "surveys/"
summ = pd.read_csv(S + "calib_summary.csv").set_index("survey") if os.path.exists(S + "calib_summary.csv") else None
extra = []
for sv, tag in [("lamost_mrs", "mrs"), ("galah", "galah"), ("desi", "desi")]:
    fn = S + f"{sv}/results_cal.csv"
    if not os.path.exists(fn): continue
    r = pd.read_csv(fn).drop_duplicates("sid")
    sd = float(summ.loc[sv, "sd_single"]) if summ is not None and sv in summ.index else 0.1
    hw = 0.5 * (r.mh_at_Tcol_hi_cal - r.mh_at_Tcol_lo_cal).abs().fillna(0)
    ok = (r.chi2_red < 10) & (~r.edge.astype(bool)) & (~r.tcol_clipped.astype(bool))
    e = pd.DataFrame({"source_id": r.sid.astype("int64"), f"mh_cal_{tag}": r.mh_at_Tcol_cal, f"mh_cal_err_{tag}": np.sqrt(hw ** 2 + 0.07 ** 2 + sd ** 2),
                      f"good_{tag}": ok.values, f"n_spec_{tag}": r.n_spec, f"P_{tag}": r.P})
    extra.append((tag, e)); print(f"merged {sv}: {len(e)} stars, good {ok.sum()}")
for tag, e in extra:
    cat = cat.merge(e, on="source_id", how="outer")
    cat["period"] = cat.period.fillna(cat[f"P_{tag}"]); cat = cat.drop(columns=[f"P_{tag}"])
cat["survey"] = cat.survey.where(cat.mh_cal_lam.notna() | cat.mh_cal_apo.notna(), "").fillna("")
for tag, e in extra:
    has = cat[f"mh_cal_{tag}"].notna()
    cat.loc[has, "survey"] = [(s + "+" if s else "") + {"mrs": "LAMOST-MRS", "galah": "GALAH", "desi": "DESI"}[tag] for s in cat.loc[has, "survey"]]
cat["good"] = cat["good"].fillna(False); cat["flags"] = cat["flags"].fillna("")
# feh_all: inverse-variance mean over every calibrated survey value from a good fit (LRS/APOGEE good = not edge/poor fit)
num = np.zeros(len(cat)); den = np.zeros(len(cat))
for m, e, g in [("mh_cal_lam", "mh_cal_err_lam", "good_lam"), ("mh_cal_apo", "mh_cal_err_apo", "good_apo")] + [(f"mh_cal_{t}", f"mh_cal_err_{t}", f"good_{t}") for t, _ in extra]:
    ok = cat[m].notna() & (cat[g].fillna(False).astype(bool) if g else True)
    if g is None: ok = ok & cat["good"].fillna(False).astype(bool)
    w = np.where(ok, 1 / cat[e].fillna(1) ** 2, 0.0); num += w * cat[m].fillna(0).values; den += w
cat["feh_all"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan); cat["feh_all_err"] = np.where(den > 0, 1 / np.sqrt(np.where(den > 0, den, 1)), np.nan)
cat["good"] = cat["good"].fillna(False)
cat["n_spectra"] = cat.n_spectra.fillna(0).astype(int) + sum(cat[f"n_spec_{t}"].fillna(0).astype(int) for t, _ in extra)
cat.to_csv(D + "catalog/cb_feh_v0.csv", index=False)
print(f"catalogue rows {len(cat)}: " + ", ".join(f"{k} {v}" for k, v in cat.survey.value_counts().items()) + f"; good {cat.good.sum()}")
print("flag counts:", pd.Series([x for f in cat["flags"] for x in f.split(";") if x]).value_counts().to_dict())
