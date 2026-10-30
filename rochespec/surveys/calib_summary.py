"""Zero points and scatter of the payne-zero Roche model against ASPCAP per survey (single-star mode and spin-up test).
Writes calib_summary.csv and calib_<survey>.png/pdf. Stars with [Fe/H]_ASPCAP < -0.9 are excluded (grid floor -1.0) and fits
that hit a grid edge are dropped."""
from ..config import DATA
import sys, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 9, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True})
S = DATA + "surveys/"
NAMES = {"lamost_mrs": "LAMOST MRS", "galah": "GALAH DR4", "desi": "DESI DR1"}


def rsd(x):
    x = np.asarray(x); x = x[np.isfinite(x)]; return 1.4826 * np.median(np.abs(x - np.median(x)))


def load(sv):
    c = pd.read_csv(S + f"{sv}/calib_results.csv")
    top = c.mh_single.max() if len(c) else 0.3
    c = c[(c.feh_aspcap > -0.9)]
    return c, c[(c.mh_single > -0.99) & (c.mh_roche > -0.99) & (c.mh_single < top) & (c.mh_roche < top)]


def summary():
    rows = []
    for sv in NAMES:
        try: call, c = load(sv)
        except FileNotFoundError: continue
        if not len(c): continue
        d1, d2, dp = c.mh_single - c.feh_aspcap, c.mh_roche - c.feh_aspcap, c.feh_pipe - c.feh_aspcap
        d3 = (c.mh_roche_g - c.feh_aspcap) if "mh_roche_g" in c else d2 * np.nan
        sl = np.polyfit(c.teff, d2, 1)[0] * 1000
        rows.append(dict(survey=sv, n_fit=len(call), n_used=len(c), zp_single=np.median(d1), sd_single=rsd(d1), zp_roche=np.median(d2), sd_roche=rsd(d2),
                         err_zp_roche=1.2533 * rsd(d2) / np.sqrt(len(c)), slope_roche_per_kK=sl, zp_roche_g=np.median(d3), sd_roche_g=rsd(d3),
                         err_zp_roche_g=1.2533 * rsd(d3) / np.sqrt(len(c)), slope_roche_g_per_kK=np.polyfit(c.teff, d3, 1)[0] * 1000 if np.isfinite(d3).all() else np.nan, pipe_minus_aspcap=np.nanmedian(dp), sd_pipe=rsd(dp),
                         snr_med=c.snr.median(), dt_med=c.dt.median()))
    s = pd.DataFrame(rows); s.to_csv(S + "calib_summary.csv", index=False); print(s.round(3).to_string()); return s


def figure(sv, zp):
    call, c = load(sv)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    lim = (-1.0, 0.5)
    ax[0].plot(c.feh_aspcap, c.mh_single - zp.zp_single, "o", ms=2.8, color="#0072B2", alpha=0.75, label="Single-star model")
    ax[0].plot(c.feh_aspcap, c.mh_roche - zp.zp_roche, "s", ms=2.6, mfc="none", mew=0.7, color="#D55E00", alpha=0.85, label="Spun-up, Roche model")
    if "mh_roche_g" in c:
        ax[0].plot(c.feh_aspcap, c.mh_roche_g - zp.zp_roche_g, "^", ms=2.6, mfc="none", mew=0.7, color="#009E73", alpha=0.85, label="Spun-up, Roche, star's log g")
        ax[1].plot(c.teff, c.mh_roche_g - c.feh_aspcap, "^", ms=2.6, mfc="none", mew=0.7, color="#009E73", alpha=0.85)
    ax[0].plot(lim, lim, color="0.4", ls="--", lw=0.9); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("ASPCAP [Fe/H]"); ax[0].set_ylabel(f"{NAMES[sv]} [M/H] $-$ zero point")
    ax[0].legend(frameon=False, loc="lower right", fontsize=6.5, handletextpad=0.3, borderaxespad=0.2)
    ax[1].plot(c.teff, c.mh_roche - c.feh_aspcap, "s", ms=2.6, mfc="none", mew=0.7, color="#D55E00", alpha=0.85)
    ax[1].plot(c.teff, c.mh_single - c.feh_aspcap, "o", ms=2.8, color="#0072B2", alpha=0.6)
    ax[1].axhline(zp.zp_roche, color="#D55E00", lw=0.9, ls=":"); ax[1].axhline(zp.zp_single, color="#0072B2", lw=0.9, ls=":"); ax[1].axhline(0, color="0.4", lw=0.9, ls="--")
    ax[1].set_xlabel(r"ASPCAP $T_{\rm eff}$ (K)"); ax[1].set_ylabel(r"Model $-$ ASPCAP")
    ax[1].set_ylim(-0.6, 0.4)
    fig.tight_layout(); fig.savefig(S + f"calib_{sv}.png", dpi=200); fig.savefig(S + f"calib_{sv}.pdf"); plt.close(fig)


if __name__ == "__main__":
    s = summary()
    for _, z in s.iterrows(): figure(z.survey, z)
