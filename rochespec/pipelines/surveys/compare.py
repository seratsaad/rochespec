"""Calibrated Roche [M/H] per survey and cross-survey comparison figures.

mh_cal = mh_at_Tcol - zp_single (single-star zero point vs ASPCAP, calib_summary.csv); mh_at_Tcol_cal_spin uses the spin-up zero point.
References: APOGEE Roche (cbapo/results_pz*) + 0.1033 and LAMOST LRS Roche (cblam/results_pz*) + 0.1147, the spin-up
zero points used in cblam/cross_survey_cal_pz.csv; plus the other new surveys' calibrated values.
Writes <survey>/results_cal.csv and compare_<survey>.png/pdf.
"""
from ...config import DATA
import sys, glob, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 9, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True})
S = DATA + "surveys/"; D = DATA + ""
NAMES = {"lamost_mrs": "LAMOST MRS", "galah": "GALAH DR4", "desi": "DESI DR1"}
PIPE = {"lamost_mrs": "LASP", "galah": "GALAH DR4", "desi": "DESI RVS"}
ZP_APO, ZP_LRS = -0.1033, -0.1147
COLS = ["sid", "n_spec", "q", "i", "f", "geo", "teff_best", "mh_best", "T_col", "mh_at_Tcol", "mh_at_Tcol_lo", "mh_at_Tcol_hi", "slope", "chi2_red",
        "gamma", "feh_pipe", "feh_lam", "P", "snr_med", "edge", "dt", "dt_per_spec"]


def rsd(x):
    x = np.asarray(x); x = x[np.isfinite(x)]; return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def calibrated(sv, z):
    zp = z.zp_single          # adopted: single-star zero point (see LOG.md: the spin-up shift at R>=7500 is not confirmed by cross-survey overlaps)
    fs = sorted(glob.glob(S + f"{sv}/results_*_*.csv"))
    full = [f for f in fs if "results_full_" in f]
    fs = full if full else [f for f in fs if "results_pilot_" in f]
    if not fs: return None
    r = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("sid", keep="last")
    r = r[[c for c in COLS if c in r.columns]].copy()
    for c in ["mh_best", "mh_at_Tcol", "mh_at_Tcol_lo", "mh_at_Tcol_hi"]: r[c + "_cal"] = r[c] - zp
    r["mh_at_Tcol_cal_spin"] = r.mh_at_Tcol - z.zp_roche; r["mh_at_Tcol_cal_g"] = r.mh_at_Tcol - z.zp_roche_g
    t = pd.read_csv(S + f"{sv}/targets.csv")[["sid", "T_col"]].rename(columns={"T_col": "T_col_orig"}).drop_duplicates("sid")
    r = r.merge(t, on="sid", how="left"); r["tcol_clipped"] = (r.T_col_orig < 4950) | (r.T_col_orig > 6550)
    r["zp"] = zp; r["zp_roche"] = z.zp_roche; r["zp_roche_g"] = z.zp_roche_g; r["survey"] = sv
    r.to_csv(S + f"{sv}/results_cal.csv", index=False); return r


def references():
    apo = pd.concat([pd.read_csv(f) for f in glob.glob(D + "cbapo/results_pz_*.csv")]).drop_duplicates("sid")
    lrs = pd.concat([pd.read_csv(f) for f in glob.glob(D + "cblam/results_pz_*.csv")]).drop_duplicates("sid")
    return (apo.assign(ref_cal=apo.mh_at_Tcol - ZP_APO)[["sid", "ref_cal", "chi2_red"]],
            lrs.assign(ref_cal=lrs.mh_at_Tcol - ZP_LRS)[["sid", "ref_cal", "chi2_red"]])


def main():
    zs = pd.read_csv(S + "calib_summary.csv").set_index("survey")
    R = {sv: calibrated(sv, zs.loc[sv]) for sv in NAMES if sv in zs.index}
    R = {k: v for k, v in R.items() if v is not None}
    apo, lrs = references(); stats = []
    for sv, r in R.items():
        good = r[(r.chi2_red < 10) & ~r.edge.astype(bool) & ~r.tcol_clipped]
        refs = [("APOGEE Roche", apo, "o", "#0072B2"), ("LAMOST LRS Roche", lrs, "s", "#E69F00")]
        refs += [(NAMES[o] + " Roche", R[o].rename(columns={"mh_at_Tcol_cal": "ref_cal", "mh_at_Tcol_cal_spin": "ref_cal_spin"})[["sid", "ref_cal", "ref_cal_spin", "chi2_red"]], "^" if o != "desi" else "v", {"galah": "#009E73", "desi": "#CC79A7", "lamost_mrs": "#56B4E9"}[o])
                 for o in R if o != sv]
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8)); lim = (-1.05, 0.45)
        for lab, ref, mk, col in refs:
            m = good.merge(ref, on="sid", suffixes=("", "_ref"))
            m = m[m.chi2_red_ref < 10]
            if not len(m): continue
            d = m.mh_at_Tcol_cal - m.ref_cal
            ds = m.mh_at_Tcol_cal_spin - (m.ref_cal_spin if "ref_cal_spin" in m else m.ref_cal)
            stats.append(dict(survey=sv, reference=lab, N=len(m), median_diff=np.median(d), robust_sd=rsd(d), median_diff_spinZP=np.median(ds)))
            ax[0].errorbar(m.ref_cal, m.mh_at_Tcol_cal, yerr=[m.mh_at_Tcol_cal - m.mh_at_Tcol_lo_cal, m.mh_at_Tcol_hi_cal - m.mh_at_Tcol_cal],
                           fmt=mk, ms=3.2, lw=0.5, color=col, alpha=0.85, label=f"{lab} ({len(m)})")
        ax[0].plot(lim, lim, color="0.4", ls="--", lw=0.9); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
        ax[0].set_xlabel("Reference Roche-model [M/H] (calibrated)"); ax[0].set_ylabel(f"{NAMES[sv]} Roche-model [M/H]")
        ax[0].legend(frameon=False, loc="lower right", fontsize=6.5, handletextpad=0.2, borderaxespad=0.3)
        g = good[np.isfinite(good.feh_pipe) & (good.feh_pipe > -5)]
        bias = g.feh_pipe - g.mh_at_Tcol_cal
        stats.append(dict(survey=sv, reference=f"{PIPE[sv]} pipeline", N=len(g), median_diff=np.median(bias), robust_sd=rsd(bias)))
        if g.feh_lam.notna().any():
            gl = good[good.feh_lam.notna()]; bl = gl.feh_lam - gl.mh_at_Tcol_cal
            stats.append(dict(survey=sv, reference="LAMOST LRS pipeline", N=len(gl), median_diff=np.median(bl), robust_sd=rsd(bl)))
        ax[1].plot(g.T_col_orig, bias, "o", ms=3.0, color="0.45", alpha=0.8)
        ax[1].axhline(0, color="0.4", ls="--", lw=0.9); ax[1].axhline(np.median(bias), color="#D55E00", ls=":", lw=1.0)
        ax[1].set_xlabel(r"Color $T_{\rm eff}$ (K)"); ax[1].set_ylabel(f"{PIPE[sv]} [Fe/H] $-$ Roche model")
        fig.tight_layout(); fig.savefig(S + f"compare_{sv}.png", dpi=200); fig.savefig(S + f"compare_{sv}.pdf"); plt.close(fig)
    st = pd.DataFrame(stats); st.to_csv(S + "compare_stats.csv", index=False); print(st.round(3).to_string())


if __name__ == "__main__":
    main()
