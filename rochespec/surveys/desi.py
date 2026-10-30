"""DESI DR1 MWS module: per-target cutouts from the public healpix coadds by HTTP range reads, reader, calibration, Roche fits.

Access: https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix/{survey}/{program}/{hpx//100}/{hpx}/coadd-{survey}-{program}-{hpx}.fits
is uncompressed and served with byte ranges; astropy (use_fsspec) reads only the target's B/R/Z FLUX/IVAR/MASK/RESOLUTION rows and its
EXP_FIBERMAP rows (MJD = exposure start, EXPTIME). Wavelengths vacuum, barycentric. The coadd is modelled as the exposure-time
weighted mean over exposures (3 sub-steps each). LSF: the coadd resolution matrix applied to the model binned onto DESI pixels.
Fit windows: B 4000-5750, R 5800-7550, Z 7600-8850 A; H-beta/H-alpha cores, telluric bands and Ca II triplet cores masked;
multiplicative Legendre (deg 6) per arm.

Usage: python -m rochespec.surveys.desi download_cal | calib | download_ew <ids.csv> | fit <ids.csv> <tag> [first last]
"""
import sys, os, time
import numpy as np, pandas as pd
from astropy.io import fits
from ..wingrid import *   # noqa

H = S + "desi/"
BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix/{s}/{p}/{h100}/{h}/coadd-{s}-{p}-{h}.fits"
SEGS = [(4000, 5750), (5800, 7550), (7600, 8850)]
ARMS = ["B", "R", "Z"]
MASK = [(4855, 4871), (5885, 5902), (6556, 6574), (6862, 6940), (7160, 7340), (7586, 7712), (8100, 8360),
        (8494.4, 8506.4), (8538.4, 8550.4), (8658.5, 8670.5)]
DEG, FLOOR = 6, 0.01
DV = 8.0
EDGE = 6


def cut_path(tid, s, p):
    return H + f"cut/{int(tid)}_{s}_{p}.npz"


def fetch(tid, s, p, hpx):
    fn = cut_path(tid, s, p)
    if os.path.exists(fn): return fn
    url = BASE.format(s=s, p=p, h100=int(hpx) // 100, h=int(hpx))
    with fits.open(url, use_fsspec=True, fsspec_kwargs={"block_size": 2 ** 18, "cache_type": "bytes"}, lazy_load_hdus=True) as h:
        fm = h["FIBERMAP"].data; j = np.where(np.asarray(fm["TARGETID"]) == int(tid))[0]
        if not len(j): raise RuntimeError("target not in coadd")
        j = int(j[0]); ef = h["EXP_FIBERMAP"].data; k = np.asarray(ef["TARGETID"]) == int(tid)
        out = dict(mjd=np.asarray(ef["MJD"])[k], exptime=np.asarray(ef["EXPTIME"])[k], night=np.asarray(ef["NIGHT"])[k],
                   expid=np.asarray(ef["EXPID"])[k], fiberstatus=np.asarray(ef["FIBERSTATUS"])[k])
        for b in ARMS:
            out[b + "_wl"] = np.asarray(h[f"{b}_WAVELENGTH"].data, float)
            for c in ("FLUX", "IVAR", "MASK"): out[f"{b}_{c}"] = np.asarray(h[f"{b}_{c}"].section[j, :])
            out[b + "_RES"] = np.asarray(h[f"{b}_RESOLUTION"].section[j, :, :], np.float32)
    os.makedirs(H + "cut", exist_ok=True); np.savez_compressed(fn + ".tmp.npz", **out); os.replace(fn + ".tmp.npz", fn)
    return fn


def download(rows, sleep=0.5):
    n = 0
    for _, r in rows.iterrows():
        if os.path.exists(cut_path(r.TARGETID, r.SURVEY, r.PROGRAM)): continue
        for k in range(3):
            try:
                fetch(r.TARGETID, r.SURVEY, r.PROGRAM, r.HEALPIX); n += 1; break
            except Exception as e:
                print("retry", r.TARGETID, repr(e)[:100], flush=True); time.sleep(5)
        time.sleep(sleep)
    print("fetched", n, flush=True)


def read(fn):
    """Spec with arms cropped to SEGS (resolution-matrix columns kept per arm), sample times for the exposures"""
    z = np.load(fn); W, F, IV, RES, segs = [], [], [], [], []
    for b, (lo, hi) in zip(ARMS, SEGS):
        wl = z[b + "_wl"]; k = (wl >= lo) & (wl <= hi)
        fl = z[b + "_FLUX"][k].astype(float); iv = np.where(z[b + "_MASK"][k] == 0, z[b + "_IVAR"][k].astype(float), 0.0)
        iv[:EDGE] = 0; iv[-EDGE:] = 0
        W.append(wl[k]); F.append(fl); IV.append(iv); RES.append(z[b + "_RES"][:, k])
    wl = np.concatenate(W); fl = np.concatenate(F); iv = np.concatenate(IV)
    good = (iv > 0) & np.isfinite(fl)
    for lo, hi in MASK: good &= ~((wl > lo) & (wl < hi))
    # per-arm median normalisation
    off = 0
    for a in range(3):
        n = len(W[a]); sl = slice(off, off + n); g = good[sl]
        if g.sum() > 50:
            med = np.median(fl[sl][g]); fl[sl] /= med; iv[sl] *= med ** 2
        off += n
    good &= fl > 0
    snr = np.median(np.sqrt(iv[good]) * fl[good]) if good.any() else 0.0
    w = np.where(good, 1 / (1 / np.maximum(iv, 1e-30) + (FLOOR * np.abs(fl)) ** 2), 0.0)
    ok = z["fiberstatus"] == 0
    if not ok.any(): ok = np.ones(len(z["mjd"]), bool)
    mjd, te = z["mjd"][ok], z["exptime"][ok]
    tt = np.concatenate([substeps(m, m + e / 86400.0, 3) for m, e in zip(mjd, te)]); tw = np.repeat(te, 3)
    return Spec(wl, fl, w, tt, tw, os.path.basename(fn), os.path.basename(fn), dict(segs=SEGS, deg=DEG, snr=snr, res=RES, nexp=int(ok.sum()), arm_len=[len(x) for x in W]))


def make_lsf(grid):
    """returns lsf_apply(k_w, nm, x, spec): model rows nm on grid.lnw[k_w] -> binned onto the DESI pixels (rest-frame log
    wavelengths x) and multiplied by the coadd resolution matrix of that arm"""
    def apply(k_w, nm, x, spec):
        ln = grid.lnw[k_w]; dl = ln[1] - ln[0]
        e = np.r_[x[0] - 0.5 * (x[1] - x[0]), 0.5 * (x[1:] + x[:-1]), x[-1] + 0.5 * (x[-1] - x[-2])]
        Rd = spec.meta["res"][k_w]; nd = Rd.shape[0]; h = nd // 2; n = len(x)
        if Rd.shape[1] != n: raise RuntimeError("resolution/pixel mismatch")
        out = np.empty((len(nm), n))
        lnE = np.r_[ln[0] - 0.5 * dl, ln + 0.5 * dl]
        rows = [np.ones(n)] + [np.diff(np.interp(e, lnE, np.r_[0.0, np.cumsum(nm[r]) * dl], left=np.nan, right=np.nan)) / np.diff(e) for r in range(len(nm))]
        for r, b in enumerate(rows):
            c = np.zeros(n)
            for k in range(nd):
                o = h - k
                if o >= 0:
                    c[:n - o] += Rd[k, o:] * b[o:]
                else:
                    c[-o:] += Rd[k, :n + o] * b[:n + o]
            if r == 0:
                rs = c; bad = np.abs(rs - 1) > 0.1       # rows touching zeroed (masked) resolution columns
            else:
                out[r - 1] = np.where(bad, np.nan, c / rs)
        return out
    return apply


def cal_select(n_per=10):
    d = pd.read_csv(S + "cats/desi_mws.csv").query("~is_ew"); a = pd.read_csv(S + "apogee_singles.csv").rename(columns={"gaia_dr3_source_id": "sid"})
    d = d[d.SUCCESS & (d.RVS_WARN == 0)].sort_values("SN_R", ascending=False).drop_duplicates("sid").merge(a, on="sid")
    d["tb"] = pd.cut(d.teff, [4800, 5300, 5800, 6300, 6700]); d["fb"] = pd.cut(d.fe_h, [-2, -0.5, -0.2, 0.1, 1])
    sel = d.groupby(["tb", "fb"], observed=True).head(n_per)
    sel.to_csv(H + "cal_sample.csv", index=False); print("calibrators", len(sel)); print(sel.groupby(["tb", "fb"], observed=True).size())
    return sel


def calib():
    grid = WinGrid(SEGS, DV); mh = mh_grid(grid); lsf = make_lsf(grid); cs = pd.read_csv(H + "cal_sample.csv"); rows = []
    for _, st in cs.iterrows():
        fn = cut_path(st.TARGETID, st.SURVEY, st.PROGRAM)
        if not os.path.exists(fn): continue
        t0 = time.time(); sp = read(fn)
        if sp.meta["snr"] < 20: continue
        x1, g0, vm = fit_single(grid, sp, st.teff, st.logg, [None] * 3, mh, vmac=(0.0,), lsf_apply=lsf, gam_step=4.0)
        lo1, hi1, sc1 = interval(x1, mh, int((sp.w > 0).sum()), 25)
        x2 = spinup_fit(grid, sp, st.teff, g0, mh, [None] * 3, lsf_apply=lsf)
        x3 = spinup_fit(grid, sp, st.teff, g0, mh, [None] * 3, lsf_apply=lsf, logg=st.logg, ld=True)
        x4 = spinup_fit(grid, sp, st.teff, g0, mh, [None] * 3, lsf_apply=lsf, ld=True)
        rows.append(dict(sid=st.sid, sdss_id=st.sdss_id, targetid=st.TARGETID, teff=st.teff, logg=st.logg, feh_aspcap=st.fe_h, v_macro=st.v_macro,
                         snr=sp.meta["snr"], feh_pipe=st.FEH, feh_sp=st.SP_FEH, gamma=g0, rv_pipe=st.VRAD, mh_single=mh[np.argmin(x1)],
                         mh_single_lo=lo1, mh_single_hi=hi1, chi2r_single=sc1, mh_roche=mh[np.argmin(x2)], mh_roche_g=mh[np.argmin(x3)], mh_roche_ld=mh[np.argmin(x4)], dt=time.time() - t0))
        print(f"{st.sid} T={st.teff:.0f} g={st.logg:.2f} ASPCAP {st.fe_h:+.2f} RVS {st.FEH:+.2f} SP {st.SP_FEH:+.2f} | single {mh[np.argmin(x1)]:+.2f} (gam {g0:+.0f} vs {st.VRAD:+.0f}) | spun-up {mh[np.argmin(x2)]:+.2f} | {time.time()-t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(H + "calib_results.csv", index=False)


def fit(ids_file, tag, i0=0, i1=10 ** 6):
    tg = pd.read_csv(H + "targets.csv"); ids = pd.read_csv(ids_file).sid.values
    tg = tg[tg.sid.isin(ids) & tg.has_veb].drop_duplicates("sid").iloc[i0:i1]
    obs = pd.read_csv(S + "cats/desi_mws.csv").query("is_ew")
    grid = WinGrid(SEGS, DV); mh = mh_grid(grid); lsf = make_lsf(grid); rows = []; os.makedirs(H + "x2", exist_ok=True)
    out = H + f"results_{tag}_{i0}_{min(i1, 10 ** 5)}.csv"
    for _, s in tg.iterrows():
        t0 = time.time(); specs = []
        for _, r in obs[obs.sid == s.sid].iterrows():
            fn = cut_path(r.TARGETID, r.SURVEY, r.PROGRAM)
            if not os.path.exists(fn): continue
            sp = read(fn)
            if sp.meta["snr"] < 8: continue
            sp.ph = phase_of(bjd(sp.t_mjd, s.RAJ2000, s.DEJ2000, "kpno"), s); specs.append(sp)
        if not specs: continue
        q = s.q if np.isfinite(s.q) else 0.35; inc = s.i if np.isfinite(s.i) else 78.0
        fill = float(np.clip(s.f if np.isfinite(s.f) else 0.2, 0.02, 0.9))
        star = RocheStar(q, fill, inc, s.P); tc = float(np.clip(s.T_col, 5000, 6500))
        res, sv = fit_ew(grid, star, specs, tc, [None] * 3, mh, lsf_apply=lsf, gam_range=200, gam_step=4.0, refine=0)
        dt = time.time() - t0
        res.update(sid=s.sid, q=star.q, i=inc, f=fill, geo=s.src if isinstance(s.src, str) else "default", feh_pipe=s.feh_pipe, feh_lam=s.feh_lam,
                   mh_apo=s.mh_apo, mh_lrs=s.mh_lrs, P=s.P, snr_med=np.median([x.meta["snr"] for x in specs]), nexp=sum(x.meta["nexp"] for x in specs),
                   dt=dt, dt_per_spec=dt / len(specs))
        rows.append(res); np.save(H + f"x2/{s.sid}.npy", sv)
        print(f"{s.sid} nspec={len(specs)} nexp={res['nexp']} best {res['teff_best']:.0f} {res['mh_best']:+.2f} | T_col {tc:.0f}: {res['mh_at_Tcol']:+.2f} | DESI {s.feh_pipe:+.2f} APO {s.mh_apo:+.2f} LRS {s.mh_lrs:+.2f} | chi2r {res['chi2_red']:.2f} ({dt:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "download_cal":
        download(cal_select())
    elif cmd == "calib":
        calib()
    elif cmd == "download_ew":
        ids = pd.read_csv(sys.argv[2]).sid.values; obs = pd.read_csv(S + "cats/desi_mws.csv").query("is_ew")
        download(obs[obs.sid.isin(ids)])
    elif cmd == "fit":
        a = sys.argv[4:6]
        fit(sys.argv[2], sys.argv[3], *(int(x) for x in a))
