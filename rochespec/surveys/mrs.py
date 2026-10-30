"""LAMOST MRS DR10 v2.0 module: download, reader, calibration on APOGEE single dwarfs, Roche fits of contact binaries.

Spectra: https://www.lamost.org/dr10/v2.0/medspectrum/fits/{obsid}; per-exposure HDUs B-<LMJM>, R-<LMJM> (DATE-OBS = mid-exposure
UTC, EXPTIME), vacuum wavelengths, heliocentric frame. R ~ 7500. Fit windows: blue 4960-5340 A, red 6330-6800 A (O2 gamma band
below 6330 and the H-alpha core masked), multiplicative Legendre (deg 5) per arm, 3 sub-steps per 20-min exposure.

Usage:  python -m rochespec.surveys.mrs download_cal | calib | download_ew <ids.csv> | fit <ids.csv> <tag> [first last]
"""
import sys, os, time, glob
import numpy as np, pandas as pd, requests
from astropy.io import fits
from ..wingrid import *   # noqa

H = S + "lamost_mrs/"
URL = "https://www.lamost.org/dr10/v2.0/medspectrum/fits/{}"
SEGS = [(4960, 5340), (6330, 6800)]
MASK = [(6554, 6575), (5888, 5898)]                 # H-alpha core (+-9 A), Na D (not in range; harmless)
DEG, FLOOR, R = 5, 0.01, 7500.0
SIG = C / R / 2.3548
DV = 4.0


def path(obsid):
    return H + f"fits/{int(obsid)}.fits.gz"


def valid(fn, obsid=None):
    try:
        with fits.open(fn) as h:
            ok = h[0].header.get("OBSID") is not None and len(h) >= 3
            return ok and (obsid is None or int(h[0].header["OBSID"]) == int(obsid))
    except Exception:
        return False


def download(obsids, sleep=0.6):
    os.makedirs(H + "fits", exist_ok=True); n_new = n_bad = 0
    for o in obsids:
        fn = path(o)
        if os.path.exists(fn) and valid(fn, o): continue
        for k in range(3):
            try:
                r = requests.get(URL.format(int(o)), timeout=120)
                if r.status_code == 200 and r.content[:2] == b"\x1f\x8b":
                    open(fn + ".part", "wb").write(r.content); os.replace(fn + ".part", fn)
                    if valid(fn, o): n_new += 1; break
                time.sleep(3)
            except Exception as e:
                print("retry", o, repr(e)[:80], flush=True); time.sleep(5)
        else:
            n_bad += 1; print("failed", o, flush=True)
        time.sleep(sleep)
    print("downloaded", n_new, "failed", n_bad, flush=True)


def _arm(x, mask_col):
    d = x.data[0]; wl = np.asarray(d["WAVELENGTH"], float); fl = np.asarray(d["FLUX"], float); iv = np.asarray(d["IVAR"], float)
    if mask_col in x.columns.names: iv = np.where(np.asarray(d[mask_col]) > 0, 0.0, iv)
    return wl, fl, iv


def read(fn, mode="exp"):
    """list of epochs: dict(wl vacuum heliocentric, fl, iv, mjd_beg, mjd_end (UTC), label). mode 'exp' = each exposure, 'coadd' = COADD_B/R"""
    out = []
    with fits.open(fn) as h:
        names = [x.name for x in h]
        if mode == "coadd":
            pairs = [("COADD_B", "COADD_R")]
        else:
            lm = sorted(set(n.split("-", 1)[1] for n in names if n.startswith(("B-", "R-"))))
            pairs = [(f"B-{m}", f"R-{m}") for m in lm]
        for b, r in pairs:
            parts, t = [], None
            for nm in (b, r):
                if nm not in names: continue
                x = h[nm]; parts.append(_arm(x, "ANDMASK" if nm.startswith("COADD") else "PIXMASK"))
                if t is None:
                    if nm.startswith("COADD"):
                        lml = [int(v) for v in str(x.header.get("LMJMLIST", "")).split("-") if v.strip()]
                        texp = 1200.0 / 86400
                        t = (np.array(lml) / 1440.0 - 8 / 24.0, np.array(lml) / 1440.0 - 8 / 24.0 + texp)   # LMJM is Beijing time (UTC+8), exposure start
                        if not lml:
                            mid = Time(x.header.get("DATE-OBS", h[0].header["DATE-OBS"]), scale="utc").mjd
                            t = (np.array([mid - texp / 2]), np.array([mid + texp / 2]))
                    else:
                        mid = Time(x.header["DATE-OBS"], scale="utc").mjd; te = float(x.header.get("EXPTIME", 1200)) / 86400
                        t = (np.array([mid - te / 2]), np.array([mid + te / 2]))
            if not parts or t is None: continue
            wl = np.concatenate([p[0] for p in parts]); fl = np.concatenate([p[1] for p in parts]); iv = np.concatenate([p[2] for p in parts])
            out.append(dict(wl=wl, fl=fl, iv=iv, mjd_beg=t[0], mjd_end=t[1], label=b))
    return out


def prep(ep, obsid, min_snr=5.0):
    """normalise each arm by its median, mask, weights with a fractional floor; returns Spec or None"""
    wl, fl, iv = ep["wl"], ep["fl"].copy(), ep["iv"].copy()
    good = (iv > 0) & np.isfinite(fl) & (fl > 0)
    inseg = np.zeros(len(wl), bool)
    for lo, hi in SEGS: inseg |= (wl >= lo) & (wl <= hi)
    good &= inseg
    for lo, hi in MASK: good &= ~((wl > lo) & (wl < hi))
    for lo, hi in SEGS:
        k = (wl >= lo) & (wl <= hi) & good
        if k.sum() < 200: good &= ~((wl >= lo) & (wl <= hi)); continue
        med = np.median(fl[k]); sel = (wl >= lo) & (wl <= hi); fl[sel] /= med; iv[sel] *= med ** 2
    if good.sum() < 300: return None
    snr = np.median(np.sqrt(iv[good]) * fl[good])
    if snr < min_snr: return None
    w = np.where(good, 1 / (1 / np.maximum(iv, 1e-30) + (FLOOR * np.abs(fl)) ** 2), 0.0)
    tt = np.concatenate([substeps(b, e, 3) for b, e in zip(ep["mjd_beg"], ep["mjd_end"])])
    return Spec(wl, fl, w, tt, np.ones(len(tt)), f"{obsid}:{ep['label']}", str(obsid), dict(segs=SEGS, deg=DEG, snr=snr))


def cal_select(n_per=10):
    m = pd.read_csv(S + "cats/mrs_dr10.csv").query("~is_ew"); a = pd.read_csv(S + "apogee_singles.csv").rename(columns={"gaia_dr3_source_id": "sid"})
    m = m.sort_values("snr", ascending=False).drop_duplicates("sid").merge(a, on="sid")
    m["tb"] = pd.cut(m.teff, [4800, 5300, 5800, 6300, 6700]); m["fb"] = pd.cut(m.fe_h, [-2, -0.5, -0.2, 0.1, 1])
    sel = m.groupby(["tb", "fb"], observed=True).head(n_per)
    sel.to_csv(H + "cal_sample.csv", index=False); print("calibrators", len(sel)); print(sel.groupby(["tb", "fb"], observed=True).size())
    return sel


def calib():
    grid = WinGrid(SEGS, DV); mh = mh_grid(grid); cs = pd.read_csv(H + "cal_sample.csv"); rows = []
    out = H + "calib_results.csv"
    for _, st in cs.iterrows():
        fn = path(st.obsid)
        if not valid(fn): continue
        t0 = time.time()
        eps = read(fn, "coadd")
        if not eps: continue
        sp = prep(eps[0], st.obsid, min_snr=20)
        if sp is None: continue
        x1, g0, vm = fit_single(grid, sp, st.teff, st.logg, [SIG, SIG], mh)
        lo1, hi1, sc1 = interval(x1, mh, int((sp.w > 0).sum()), 15)
        x2 = spinup_fit(grid, sp, st.teff, g0, mh, [SIG, SIG])
        x3 = spinup_fit(grid, sp, st.teff, g0, mh, [SIG, SIG], logg=st.logg, ld=True)
        x4 = spinup_fit(grid, sp, st.teff, g0, mh, [SIG, SIG], ld=True)
        rows.append(dict(sid=st.sid, sdss_id=st.sdss_id, obsid=st.obsid, teff=st.teff, logg=st.logg, feh_aspcap=st.fe_h, v_macro=st.v_macro,
                         snr=sp.meta["snr"], feh_pipe=st.feh_lasp, gamma=g0, vmac_fit=vm, mh_single=mh[np.argmin(x1)], mh_single_lo=lo1, mh_single_hi=hi1,
                         chi2r_single=sc1, mh_roche=mh[np.argmin(x2)], mh_roche_g=mh[np.argmin(x3)], mh_roche_ld=mh[np.argmin(x4)], dt=time.time() - t0))
        print(f"{st.sid} T={st.teff:.0f} g={st.logg:.2f} ASPCAP {st.fe_h:+.2f} LASP {st.feh_lasp:+.2f} | single {mh[np.argmin(x1)]:+.2f} (vm {vm}) | spun-up {mh[np.argmin(x2)]:+.2f} | {time.time()-t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)


def ew_specs(sid, obs, ra, dec, s):
    specs = []
    for oid in obs[obs.sid == sid].obsid.astype(int).unique():
        fn = path(oid)
        if not valid(fn): continue
        for ep in read(fn, "exp"):
            sp = prep(ep, oid)
            if sp is None: continue
            sp.ph = phase_of(bjd(sp.t_mjd, ra, dec, "lamost"), s); specs.append(sp)
    return specs


def fit(ids_file, tag, i0=0, i1=10 ** 6):
    tg = pd.read_csv(H + "targets.csv"); ids = pd.read_csv(ids_file).sid.values
    tg = tg[tg.sid.isin(ids) & tg.has_veb].drop_duplicates("sid").iloc[i0:i1]
    obs = pd.read_csv(S + "cats/mrs_dr10.csv").query("is_ew")
    grid = WinGrid(SEGS, DV); mh = mh_grid(grid); rows = []; os.makedirs(H + "x2", exist_ok=True)
    out = H + f"results_{tag}_{i0}_{min(i1, 10 ** 5)}.csv"
    for _, s in tg.iterrows():
        t0 = time.time()
        specs = ew_specs(s.sid, obs, s.RAJ2000, s.DEJ2000, s)
        if not specs: continue
        q = s.q if np.isfinite(s.q) else 0.35; inc = s.i if np.isfinite(s.i) else 78.0
        fill = float(np.clip(s.f if np.isfinite(s.f) else 0.2, 0.02, 0.9))
        star = RocheStar(q, fill, inc, s.P)
        tc = float(np.clip(s.T_col, 5000, 6500))
        res, sv = fit_ew(grid, star, specs, tc, [SIG, SIG], mh)
        dt = time.time() - t0
        res.update(sid=s.sid, q=star.q, i=inc, f=fill, geo=s.src if isinstance(s.src, str) else "default", n_obsid=len(set(x.group for x in specs)),
                   feh_pipe=s.feh_pipe, feh_lam=s.feh_lam, mh_apo=s.mh_apo, mh_lrs=s.mh_lrs, P=s.P, snr_med=np.median([x.meta["snr"] for x in specs]),
                   dt=dt, dt_per_spec=dt / len(specs))
        rows.append(res); np.save(H + f"x2/{s.sid}.npy", sv)
        print(f"{s.sid} nspec={len(specs)} best {res['teff_best']:.0f} {res['mh_best']:+.2f} | T_col {tc:.0f}: {res['mh_at_Tcol']:+.2f} | LASP {s.feh_pipe:+.2f} APO {s.mh_apo:+.2f} LRS {s.mh_lrs:+.2f} | chi2r {res['chi2_red']:.2f} ({dt:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "download_cal":
        download(cal_select().obsid.values)
    elif cmd == "calib":
        calib()
    elif cmd == "download_ew":
        ids = pd.read_csv(sys.argv[2]).sid.values; obs = pd.read_csv(S + "cats/mrs_dr10.csv").query("is_ew")
        download(obs[obs.sid.isin(ids)].obsid.unique())
    elif cmd == "fit":
        a = sys.argv[4:6]
        fit(sys.argv[2], sys.argv[3], *(int(x) for x in a))
