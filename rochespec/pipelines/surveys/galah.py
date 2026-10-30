"""GALAH DR4 module: download (Data Central datalink, anonymous SSA products), reader, calibration, Roche fits of contact binaries.

Access: https://datacentral.org.au/vo/slink/links?ID={sobject_id}&DR=galah_dr4&IDX=1&FILT={B,V,R,I} lists the combined spectrum and
each individual exposure (unnormalised counts, 4096 px, linear WCS, MJD-BEG/MJD-END per exposure). No error array -> noise from the
second-difference scatter scaled as sqrt(counts). Wavelengths are air and barycentric (converted to vacuum here). R ~ 28000.
Fit windows (vacuum): the four HERMES CCDs, H-beta/H-alpha cores and the O2 A band masked, multiplicative Legendre (deg 4) per CCD.

Usage: python -m rochespec.pipelines.surveys.galah download_cal | calib | download_ew <ids.csv> | fit <ids.csv> <tag> [first last]
"""
import sys, os, re, time
import numpy as np
try:
    import pandas as pd
except ImportError:      # pandas is needed only by the batch pipelines
    pd = None
try:
    import requests
except ImportError:      # only for downloads
    requests = None
from astropy.io import fits
from ...wingrid import *   # noqa

H = S + "galah/"
LINK = "https://datacentral.org.au/vo/slink/links?ID={}&DR=galah_dr4&IDX=1&FILT={}"
FILTS = ["B", "V", "R", "I"]
SEGS = [(4716, 4902), (5652, 5872), (6482, 6738), (7706, 7886)]
MASK = [(4855, 4871), (6556, 6574), (7590, 7706)]
DEG, FLOOR, R = 4, 0.01, 28000.0
SIG = C / R / 2.3548
DV = 2.0


def sdir(sob):
    return H + f"spec/{int(sob)}/"


def valid(fn):
    try:
        with fits.open(fn) as h:
            return h[0].data is not None and h[0].data.size == h[0].header["NAXIS1"] and "MJD-BEG" in h[0].header
    except Exception:
        return False


def links(sob, filt):
    r = requests.get(LINK.format(int(sob), filt), timeout=120); r.raise_for_status()
    rows = re.findall(r"CDATA\[([^\]]*)\]\]></TD>.*?<TD>([\d.]+)</TD><TD>([\d.]+)</TD><TD>(\w*)</TD>", r.text)
    return [dict(url=u, mjd_beg=float(a), mjd_end=float(b), kind=k) for u, a, b, k in rows]


def download(sobs, exposures=True, sleep=0.5):
    n_new = 0
    for sob in sobs:
        d = sdir(sob); os.makedirs(d, exist_ok=True)
        if os.path.exists(d + "done" + ("_exp" if exposures else "")): continue
        ok_all = True
        for f in FILTS:
            try:
                L = links(sob, f); time.sleep(sleep)
                pd.DataFrame(L).to_csv(d + f"links_{f}.csv", index=False)
            except Exception as e:
                print("links fail", sob, f, repr(e)[:80], flush=True); ok_all = False; time.sleep(5); continue
            for l in L:
                if l["kind"] == "science" and not exposures: continue
                fn = d + l["url"].rsplit("/", 1)[1]
                if os.path.exists(fn) and valid(fn): continue
                for k in range(3):
                    try:
                        r = requests.get(l["url"], timeout=120)
                        if r.status_code == 200 and r.content[:6] == b"SIMPLE":
                            open(fn, "wb").write(r.content)
                            if valid(fn): n_new += 1; break
                    except Exception as e:
                        print("retry", fn, repr(e)[:60], flush=True)
                    time.sleep(3)
                else:
                    ok_all = False
                time.sleep(sleep)
        if ok_all: open(d + "done" + ("_exp" if exposures else ""), "w").write("ok")
    print("downloaded files", n_new, flush=True)


def read_file(fn):
    with fits.open(fn) as h:
        Hd = h[0].header; fl = np.asarray(h[0].data, float)
        wl_air = Hd["CRVAL1"] + Hd["PC1_1"] * Hd.get("CDELT1", 1.0) * (np.arange(Hd["NAXIS1"]) + 1 - Hd["CRPIX1"])
        return air_to_vac(wl_air), fl, Hd["MJD-BEG"], Hd["MJD-END"], Hd.get("TXEL", 1)


def noise(fl):
    """per-pixel sigma: second-difference MAD at the median level, scaled as sqrt(counts)"""
    ok = np.isfinite(fl) & (fl > 0)
    d2 = fl[1:-1] - 0.5 * (fl[:-2] + fl[2:])
    okd = ok[1:-1] & ok[:-2] & ok[2:]
    if okd.sum() < 100: return np.full_like(fl, np.inf)
    rel = 1.4826 * np.median(np.abs(d2[okd] / fl[1:-1][okd])) / np.sqrt(1.5)
    med = np.median(fl[ok])
    return np.where(ok, rel * med * np.sqrt(np.maximum(fl, 1e-3 * med) / med), np.inf)


def assemble(files, label, group, min_snr=8.0):
    """join the CCD files of one epoch into a Spec (each CCD normalised by its median)"""
    W, F, IV, tb, te = [], [], [], [], []
    for fn in files:
        try: wl, fl, b, e, n = read_file(fn)
        except Exception: continue
        sg = noise(fl); good = np.isfinite(fl) & (fl > 0) & np.isfinite(sg)
        if good.sum() < 500: continue
        med = np.median(fl[good]); W.append(wl); F.append(fl / med); IV.append(np.where(good, (med / sg) ** 2, 0.0)); tb.append(b); te.append(e)
    if not W: return None
    wl = np.concatenate(W); fl = np.concatenate(F); iv = np.concatenate(IV); o = np.argsort(wl); wl, fl, iv = wl[o], fl[o], iv[o]
    good = iv > 0
    inseg = np.zeros(len(wl), bool)
    for lo, hi in SEGS: inseg |= (wl >= lo) & (wl <= hi)
    good &= inseg
    for lo, hi in MASK: good &= ~((wl > lo) & (wl < hi))
    if good.sum() < 500: return None
    snr = np.median(np.sqrt(iv[good]) * fl[good])
    if snr < min_snr: return None
    w = np.where(good, 1 / (1 / np.maximum(iv, 1e-30) + (FLOOR * np.abs(fl)) ** 2), 0.0)
    b, e = np.median(tb), np.median(te)
    return Spec(wl, fl, w, substeps(b, e, 3 if e - b < 0.02 else 5), np.ones(3 if e - b < 0.02 else 5), label, group, dict(segs=SEGS, deg=DEG, snr=snr, mjd_beg=b, mjd_end=e))


def epochs(sob, mode="exp"):
    """mode 'exp': one Spec per individual exposure; 'comb': the combined spectrum"""
    d = sdir(sob)
    if not os.path.isdir(d): return []
    fs = sorted(f for f in os.listdir(d) if f.endswith("-1.fits"))
    s = str(int(sob)); comb = [d + f for f in fs if f.startswith(s)]
    if mode == "comb":
        sp = assemble(comb, s, s)
        if sp is None: return []
        # exposure intervals from the saved datalink listing (individual exposures need not be downloaded)
        L = []
        for f in sorted(os.listdir(d)):
            if not f.startswith("links_"): continue
            try: L.append(pd.read_csv(d + f))
            except Exception: pass
        L = [x for x in L if len(x) and "kind" in x]
        if L:
            L = pd.concat(L); L = L[L.kind == "science"].drop_duplicates("mjd_beg")
            if len(L):
                sp.t_mjd = np.concatenate([substeps(b, e, 3) for b, e in zip(L.mjd_beg, L.mjd_end)]); sp.t_w = np.ones(len(sp.t_mjd)) / len(sp.t_mjd)
                sp.meta["nexp"] = len(L)
        return [sp]
    runs = sorted(set(f[:-8] for f in fs if not f.startswith(s)))       # e.g. 210125003800370
    out = []
    for r in runs:
        sp = assemble([d + f for f in fs if f.startswith(r)], r, s, min_snr=5.0)
        if sp is not None: out.append(sp)
    if not out:
        sp = assemble(comb, s, s); out = [sp] if sp is not None else []
    return out


def cal_select(n_per=10):
    g = pd.read_csv(S + "cats/galah_allspec.csv").query("~is_ew"); a = pd.read_csv(S + "apogee_singles.csv").rename(columns={"gaia_dr3_source_id": "sid"})
    g = g[g.flag_sp == 0].sort_values("snr_px_ccd2", ascending=False).drop_duplicates("sid").merge(a, on="sid")
    g["tb"] = pd.cut(g.teff_y, [4800, 5300, 5800, 6300, 6700]); g["fb"] = pd.cut(g.fe_h_y, [-2, -0.5, -0.2, 0.1, 1])
    sel = g.groupby(["tb", "fb"], observed=True).head(n_per)
    sel.to_csv(H + "cal_sample.csv", index=False); print("calibrators", len(sel)); print(sel.groupby(["tb", "fb"], observed=True).size())
    return sel


def calib():
    grid = WinGrid(SEGS, DV); mh = mh_grid(grid); cs = pd.read_csv(H + "cal_sample.csv"); rows = []
    sig = [SIG] * len(SEGS)
    for _, st in cs.iterrows():
        t0 = time.time(); eps = epochs(st.sobject_id, "comb")
        if not eps: continue
        sp = eps[0]
        x1, g0, vm = fit_single(grid, sp, st.teff_y, st.logg_y, sig, mh, vmac=(0.0, 2.0, 4.0, 6.0))
        lo1, hi1, sc1 = interval(x1, mh, int((sp.w > 0).sum()), 25)
        x2 = spinup_fit(grid, sp, st.teff_y, g0, mh, sig)
        x3 = spinup_fit(grid, sp, st.teff_y, g0, mh, sig, logg=st.logg_y, ld=True)
        x4 = spinup_fit(grid, sp, st.teff_y, g0, mh, sig, ld=True)
        rows.append(dict(sid=st.sid, sdss_id=st.sdss_id, sobject_id=st.sobject_id, teff=st.teff_y, logg=st.logg_y, feh_aspcap=st.fe_h_y, v_macro=st.v_macro,
                         snr=sp.meta["snr"], feh_pipe=st.fe_h_x, teff_pipe=st.teff_x, gamma=g0, rv_pipe=st.rv_comp_1, vmac_fit=vm, mh_single=mh[np.argmin(x1)],
                         mh_single_lo=lo1, mh_single_hi=hi1, chi2r_single=sc1, mh_roche=mh[np.argmin(x2)], mh_roche_g=mh[np.argmin(x3)], mh_roche_ld=mh[np.argmin(x4)], dt=time.time() - t0))
        print(f"{st.sid} T={st.teff_y:.0f} g={st.logg_y:.2f} ASPCAP {st.fe_h_y:+.2f} GALAH {st.fe_h_x:+.2f} | single {mh[np.argmin(x1)]:+.2f} (vm {vm}, gam {g0:+.1f} vs {st.rv_comp_1:+.1f}) | spun-up {mh[np.argmin(x2)]:+.2f} | {time.time()-t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(H + "calib_results.csv", index=False)


def fit(ids_file, tag, i0=0, i1=10 ** 6, mode="exp"):
    mode = os.environ.get("GALAH_MODE", mode)
    tg = pd.read_csv(H + "targets.csv"); ids = pd.read_csv(ids_file).sid.values
    tg = tg[tg.sid.isin(ids) & tg.has_veb].drop_duplicates("sid").iloc[i0:i1]
    obs = pd.read_csv(S + "cats/galah_allspec.csv").query("is_ew")
    grid = WinGrid(SEGS, DV); mh = mh_grid(grid); rows = []; os.makedirs(H + "x2", exist_ok=True)
    sig = [SIG] * len(SEGS); out = H + f"results_{tag}_{i0}_{min(i1, 10 ** 5)}.csv"
    if os.path.exists(out):                      # resume
        rows = pd.read_csv(out).to_dict("records"); tg = tg[~tg.sid.isin([r["sid"] for r in rows])]
    for _, s in tg.iterrows():
        t0 = time.time(); specs = []
        for sob in obs[obs.sid == s.sid].sobject_id.unique():
            for sp in epochs(sob, mode):
                sp.ph = phase_of(bjd(sp.t_mjd, s.RAJ2000, s.DEJ2000, "aat"), s); specs.append(sp)
        if not specs: continue
        q = s.q if np.isfinite(s.q) else 0.35; inc = s.i if np.isfinite(s.i) else 78.0
        fill = float(np.clip(s.f if np.isfinite(s.f) else 0.2, 0.02, 0.9))
        star = RocheStar(q, fill, inc, s.P); tc = float(np.clip(s.T_col, 5000, 6500))
        res, sv = fit_ew(grid, star, specs, tc, sig, mh, gam_step=2.0, refine=6)
        dt = time.time() - t0
        res.update(sid=s.sid, q=star.q, i=inc, f=fill, geo=s.src if isinstance(s.src, str) else "default", feh_pipe=s.feh_pipe, feh_lam=s.feh_lam,
                   mh_apo=s.mh_apo, mh_lrs=s.mh_lrs, P=s.P, snr_med=np.median([x.meta["snr"] for x in specs]), dt=dt, dt_per_spec=dt / len(specs))
        rows.append(res); np.save(H + f"x2/{s.sid}.npy", sv)
        print(f"{s.sid} nspec={len(specs)} best {res['teff_best']:.0f} {res['mh_best']:+.2f} | T_col {tc:.0f}: {res['mh_at_Tcol']:+.2f} | GALAH {s.feh_pipe:+.2f} APO {s.mh_apo:+.2f} LRS {s.mh_lrs:+.2f} | chi2r {res['chi2_red']:.2f} ({dt:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "download_cal":
        download(cal_select().sobject_id.values, exposures=False)
    elif cmd == "calib":
        calib()
    elif cmd == "download_ew":
        ids = pd.read_csv(sys.argv[2]).sid.values; obs = pd.read_csv(S + "cats/galah_allspec.csv").query("is_ew")
        download(obs[obs.sid.isin(ids)].sobject_id.unique(), exposures=os.environ.get("GALAH_EXP", "1") == "1")
    elif cmd == "fit":
        a = sys.argv[4:6]
        fit(sys.argv[2], sys.argv[3], *(int(x) for x in a))
