"""Roche-mesh + payne-zero forward model fitted to LAMOST DR10 LRS spectra of ASAS-SN EWs, at scale.

Per star: geometry from LW25/Li24 light curves (else default q=0.35, i=78, f=0.2), ephemeris from Gaia DR3 VEB;
without VEB the phase of the first exposure is a nuisance parameter (6-point grid, chosen with the velocity).
Per spectrum: model integrated over the exposures, systemic velocity by shifting, multiplicative Legendre
polynomial per arm; chi^2 summed over the star's spectra on a (Teff, [M/H]) grid; joint inference with the
Casagrande+21 photometric Teff prior (see common.joint).
Usage: CB_MODEL=pz python -m rochespec.lrs <shard> <nshard> [max_stars]
Writes cbscale/res_lam/results_<shard>.csv incrementally (resumable) and per-star chi^2 grids.
"""
import sys, os, time
import numpy as np, pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from .roche import ContactBinary, C
from .model_cfg import MHS, load, MODEL
from .core import D, MH_FINE, read_dr10, geometry, geo_params, veb_ok, bjd_phase, joint, teff_grid_for

assert MODEL == "pz", "run with CB_MODEL=pz"
DV = 5.0
SEGS = [(4000, 5700), (6000, 8800)]
MASK = [(6855, 6970), (7150, 7340), (7580, 7710), (8100, 8360), (6558, 6568), (4855, 4867)]
FLOOR = 0.01
SNR_MIN = 15
XL = EarthLocation.from_geodetic(117.58 * u.deg, 40.39 * u.deg, 960 * u.m)      # Xinglong
SPEC_DIR = D + "lamost_dr10/ew/"
OUT = D + "cbscale/res_lam/"
PH0 = np.arange(6) / 12.0                                                         # nuisance phases 0 .. 5/12


def design(wl, seg, deg=8):
    x = 2 * (wl - seg[0]) / (seg[1] - seg[0]) - 1
    return np.polynomial.legendre.legvander(x, deg)


def chi2_spec(wl, fo, w, mod):
    tot = 0.0
    for seg in SEGS:
        k = (wl > seg[0]) & (wl < seg[1]) & (w > 0) & np.isfinite(mod)
        kd = (wl > seg[0]) & (wl < seg[1]) & (w > 0)
        if kd.sum() < 100: continue                                              # arm without usable data
        if k.sum() < 0.8 * kd.sum(): tot += 1e12; continue                       # model undefined (bad grid node)
        A = design(wl[k], seg) * mod[k][:, None]; sw = np.sqrt(w[k])
        c, *_ = np.linalg.lstsq(A * sw[:, None], fo[k] * sw, rcond=None)
        tot += np.sum(w[k] * (fo[k] - A @ c) ** 2)
    return tot


def norm_at(F, Fc, m):
    j = min(np.searchsorted(MHS, m, side="right") - 1, len(MHS) - 2); f_ = (m - MHS[j]) / (MHS[j + 1] - MHS[j])
    return ((1 - f_) * F[j] + f_ * F[j + 1]) / ((1 - f_) * Fc[j] + f_ * Fc[j + 1])


def load_spec(oid, ra, dec):
    wl, fl, iv, starts, texp = read_dr10(SPEC_DIR + f"{oid}.fits.gz")
    good = (iv > 0) & np.isfinite(fl) & (wl > 3900)
    for lo, hi in MASK: good &= ~((wl > lo) & (wl < hi))
    med = np.nanmedian(fl[good]); fl = fl / med; iv = iv * med ** 2
    snr = np.median(np.sqrt(iv[good]) * fl[good])
    w = np.where(good, 1 / (1 / np.maximum(iv, 1e-30) + (FLOOR * np.abs(fl)) ** 2), 0.0)
    sub = np.concatenate([s0 + texp * np.array([1, 3, 5]) / 6.0 for s0 in starts])
    t = Time(sub, format="mjd", scale="utc", location=XL); c = SkyCoord(ra * u.deg, dec * u.deg)
    bjd = (t.tdb + t.light_travel_time(c, kind="barycentric")).jd
    return wl, fl, w, bjd, snr


def fit_star(s, obs_s, grid, cb_cache={}):
    t0 = time.time()
    qq, inc, fill, geo = geo_params(s)
    cb = ContactBinary(qq, fill, ntheta=60, nphi=120)
    P = s.Per if not veb_ok(s) else 1 / s.Freq
    specs = []
    for oid in obs_s.obsid.astype(int).unique():
        try: wl, fl, w, bjd, snr = load_spec(oid, s.RAJ2000, s.DEJ2000)
        except Exception: continue
        if snr < SNR_MIN: continue
        specs.append(dict(oid=oid, wl=wl, fl=fl, w=w, bjd=bjd, snr=snr))
    if not specs: return None
    eph = "VEB" if veb_ok(s) else "nuisance"
    for sp in specs:
        sp["ph"] = bjd_phase(sp["bjd"], s) if eph == "VEB" else ((sp["bjd"] - sp["bjd"][0]) / P) % 1
    if hasattr(cb, "prepare"):     # PHOEBE geometry (CB_GEOM=phoebe): exact phases with a VEB ephemeris, else the shared phase library
        cb.prepare(np.concatenate([sp["ph"] for sp in specs]), inc, exact=(eph == "VEB"))
    Tp_m02 = s.Tphot0 - 0.2 * s.dTdfeh
    sigT = s.sig_Tphot if np.isfinite(s.sig_Tphot) else 1e4
    tg = teff_grid_for(Tp_m02 if np.isfinite(Tp_m02) else 5800, sigT)

    def models(T, sp_list, ph_shift=None):
        M1 = (T / 5772.0) ** 1.8; A = (M1 * (1 + qq) * (P / 365.25) ** 2) ** (1 / 3) * 215.03
        logg, Tl = cb.surface(M1, A, T); out = []
        for k, sp in enumerate(sp_list):
            ph = sp["ph"] if ph_shift is None else (sp["ph"] + ph_shift[k]) % 1
            el = [], [], [], [], []
            for p in ph:
                los, mu, vis = cb.view(p, inc); rv = cb.rv(los, A, P)
                for lst, val in zip(el, (Tl[vis], logg[vis], mu[vis], rv[vis], cb.dA[vis] / len(ph))): lst.append(val)
            out.append(grid.integrate_nodes(*[np.concatenate(x) for x in el], lsf_R=1800.0))
        return out

    nsh = 30; lnw = grid.lnw
    T0 = tg[np.argmin(np.abs(tg - Tp_m02))] if np.isfinite(Tp_m02) else tg[len(tg) // 2]
    def vel_search(F, Fc, sp):
        nm = norm_at(F, Fc, -0.2)
        c2 = [chi2_spec(sp["wl"], sp["fl"], sp["w"], np.interp(np.log(sp["wl"]), lnw + k * DV / C, nm, left=np.nan, right=np.nan)) for k in range(-nsh, nsh + 1)]
        return int(np.argmin(c2)) - nsh, min(c2)
    if eph == "VEB":
        for (F, Fc), sp in zip(models(T0, specs), specs): sp["k"], _ = vel_search(F, Fc, sp)
        shifts = [0.0] * len(specs)
    else:
        best = [(np.inf, 0, 0.0)] * len(specs)
        for p0 in PH0:
            for j, ((F, Fc), sp) in enumerate(zip(models(T0, specs, [p0] * len(specs)), specs)):
                k, c = vel_search(F, Fc, sp)
                if c < best[j][0]: best[j] = (c, k, p0)
        for sp, b in zip(specs, best): sp["k"] = b[1]
        shifts = [b[2] for b in best]
    X2 = np.zeros((len(tg), len(MH_FINE)))
    for a_, T in enumerate(tg):
        for (F, Fc), sp in zip(models(T, specs, shifts), specs):
            lw = np.log(sp["wl"])
            for b_, m in enumerate(MH_FINE):
                X2[a_, b_] += chi2_spec(sp["wl"], sp["fl"], sp["w"], np.interp(lw, lnw + sp["k"] * DV / C, norm_at(F, Fc, m), left=np.nan, right=np.nan))
    npix = sum(int(np.sum(sp["w"] > 0)) for sp in specs)
    r = joint(X2, tg.astype(float), MH_FINE, npix, s.Tphot0, s.dTdfeh, sigT, npar=19 * len(specs) + 2)
    o = obs_s.set_index("obsid").loc[[sp["oid"] for sp in specs]]
    wts = np.array([sp["snr"] for sp in specs]) ** 2
    good_feh = o.feh.values > -9
    r.update(sid=s.sid, n_spec=len(specs), obsids=";".join(str(sp["oid"]) for sp in specs), snr_tot=float(np.sqrt(wts.sum())),
             P=P, q=qq, i=inc, f=fill, geo=geo, eph=eph, Tphot0=s.Tphot0, dTdfeh=s.dTdfeh, sig_Tphot=sigT, ebv=s["E_B-V_"],
             flag_ebv=s.flag_ebv, flag_colour=s.flag_colour, gamma=float(np.median([sp["k"] for sp in specs]) * DV),
             feh_lam=float(np.average(o.feh.values[good_feh], weights=wts[good_feh])) if good_feh.any() else np.nan,
             teff_lam=float(np.average(o.teff.values[good_feh], weights=wts[good_feh])) if good_feh.any() else np.nan,
             logg_lam=float(np.average(o.logg.values[good_feh], weights=wts[good_feh])) if good_feh.any() else np.nan,
             tg_lo=tg[0], tg_hi=tg[-1], secs=time.time() - t0)
    np.savez_compressed(OUT + f"x2/{s.sid}.npz", X2=X2, teff=tg, mh=MH_FINE, npix=npix)
    return r


def targets():
    obs = pd.read_csv(D + "cbscale/ew_lamost_dr10_obs.csv"); obs = obs[obs.snrg > 10]
    tp = pd.read_csv(D + "cbscale/ew_teffphot.csv")
    veb = pd.read_csv(D + "cbscale/veb_all.csv")
    st = obs.drop_duplicates("sid")[["sid"]].merge(tp, on="sid").merge(veb, left_on="sid", right_on="Source", how="left").merge(geometry(), on="sid", how="left")
    st["snr_best"] = st.sid.map(obs.groupby("sid").snrg.max())
    Tp = st.Tphot0 - 0.2 * st.dTdfeh
    st = st[~(((Tp > 7000) | (Tp < 4500)) & (st.sig_Tphot < 300))]           # confidently outside the model grid: skipped
    return st.sort_values("snr_best", ascending=False).reset_index(drop=True), obs


if __name__ == "__main__":
    shard, nshard = int(sys.argv[1]), int(sys.argv[2]); maxn = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    os.makedirs(OUT + "x2", exist_ok=True)
    fn = OUT + f"results_{shard}.csv"
    rows = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []
    done = {r["sid"] for r in rows}
    st, obs = targets(); st = st.iloc[shard::nshard]
    grid = load("opt", dv=DV, rebin=True)
    failed = set(); n_new = 0
    while True:
        todo = st[~st.sid.isin(done | failed)]
        if len(todo) == 0 or n_new >= maxn: break
        progressed = False
        for _, s in todo.iterrows():
            if n_new >= maxn: break
            o = obs[obs.sid == s.sid]
            have = [os.path.exists(SPEC_DIR + f"{x}.fits.gz") for x in o.obsid]
            bad = set(open(SPEC_DIR + "failed.txt").read().split()) if os.path.exists(SPEC_DIR + "failed.txt") else set()
            if not all(h or str(x) in bad for h, x in zip(have, o.obsid)): continue      # wait for downloads
            progressed = True
            try:
                r = fit_star(s, o, grid)
            except Exception as e:
                print(f"{s.sid} ERROR {repr(e)[:200]}", flush=True); failed.add(s.sid); continue
            if r is None:
                failed.add(s.sid); rows.append(dict(sid=s.sid, n_spec=0)); done.add(s.sid)
            else:
                rows.append(r); done.add(s.sid); n_new += 1
                print(f"{s.sid} n={r['n_spec']} {r['eph']} T={r['teff_joint']:.0f}+-{r['teff_err']:.0f} (phot {s.Tphot0 - 0.2*s.dTdfeh:.0f}+-{r['sig_Tphot']:.0f}) "
                      f"[M/H]={r['mh_joint']:+.2f}+-{r['mh_err']:.2f} LAMOST {r['feh_lam']:+.2f} chi2r {r['chi2_red']:.2f} ({r['secs']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)
        if not progressed: time.sleep(120)
    print("finished", len(done), "failed", len(failed), flush=True)
