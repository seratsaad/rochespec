"""Roche-mesh + payne-zero H-band model fitted to APOGEE DR19 mwmVisit spectra of all 802 ASAS-SN EWs in DR19.

Same joint (Teff, [M/H]) inference with the photometric Teff prior as fit_lamost_dr10.py.
mwmVisit flux is resampled to the stellar rest frame with the per-visit pipeline velocity (checked by cross-correlating
visits of single control stars: stellar lines align, telluric residuals sit at -dv_rel), so each visit is shifted back by
+v_rad to the barycentric frame, where the Roche model (orbital velocity field + one systemic velocity per star) applies;
visits without a finite v_rad are dropped; EWs without a usable VEB ephemeris get one
nuisance phase per visit.
Usage: CB_MODEL=pz python -m rochespec.pipelines.apogee <shard> <nshard>
"""
import sys, os, time, glob
import numpy as np
try:
    import pandas as pd
except ImportError:      # pandas is needed only by the batch pipelines
    pd = None
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from ..roche import ContactBinary, C
from ..model_cfg import MHS, load, MODEL
from ..core import D, MH_FINE, geometry, geo_params, veb_ok, bjd_phase, joint, teff_grid_for

assert MODEL == "pz"
CHIPS = [(15140, 15810), (15860, 16430), (16480, 16950)]
FLOOR = 0.005
WL = (10 ** (4.179 + 6e-6 * np.arange(8575))).astype(np.float32).astype(float)     # APOGEE apStar/mwmStar log-lambda grid (as stored, float32)
SITES = {"apo25m": EarthLocation.of_site("apo"), "lco25m": EarthLocation.of_site("lco")}
OUT = D + "cbscale/res_apo/"
PH0 = np.arange(6) / 12.0
SNR_MIN = float(os.environ.get("APO_SNR_MIN", 20))


def read_visits(fn):
    out = []
    with fits.open(fn) as h:
        for hdu in (3, 4):
            if hdu >= len(h) or h[hdu].data is None: continue
            v = h[hdu].data
            for i in range(len(v)):
                cont = np.asarray(v["continuum"][i], float)
                f = np.asarray(v["flux"][i], float) / cont; iv = np.asarray(v["ivar"][i], float) * cont ** 2
                if not np.any(iv > 0) or not np.isfinite(v["snr"][i]) or v["snr"][i] < SNR_MIN: continue
                bc = float(v["bc"][i])
                if not np.isfinite(v["v_rad"][i]): continue
                out.append(dict(f=f, iv=iv, date=str(v["date_obs"][i]), exp=float(v["exptime"][i]), bc=bc, snr=float(v["snr"][i]),
                                v_rad=float(v["v_rad"][i]), site="lco25m" if hdu == 4 else "apo25m"))
    return out


def visit_bjd(vis, ra, dec):
    c = SkyCoord(ra * u.deg, dec * u.deg)
    t = Time(vis["date"], format="isot", scale="utc", location=SITES[vis["site"]]) + (vis["exp"] * np.array([1, 3, 5]) / 6.0) * u.s
    return (t.tdb + t.light_travel_time(c, kind="barycentric")).jd


def chi2_chips(wl, o, w, m, deg=2):
    tot = 0.0
    for lo, hi in CHIPS:
        k = (wl > lo) & (wl < hi) & (w > 0) & np.isfinite(m) & np.isfinite(o)
        kd = (wl > lo) & (wl < hi) & (w > 0) & np.isfinite(o)
        if kd.sum() < 100: continue                                   # chip without usable data
        if k.sum() < 0.8 * kd.sum(): tot += 1e12; continue           # model undefined
        x = (wl[k] - 0.5 * (lo + hi)) / (hi - lo); A = np.vander(x, deg + 1) * m[k][:, None]; sw = np.sqrt(w[k])
        c, *_ = np.linalg.lstsq(A * sw[:, None], o[k] * sw, rcond=None); tot += np.sum(w[k] * (o[k] - A @ c) ** 2)
    return tot


def norm_at(F, Fc, m):
    j = min(np.searchsorted(MHS, m, side="right") - 1, len(MHS) - 2); f_ = (m - MHS[j]) / (MHS[j + 1] - MHS[j])
    return ((1 - f_) * F[j] + f_ * F[j + 1]) / ((1 - f_) * Fc[j] + f_ * Fc[j + 1])


def fit_star(s, grid):
    fn = D + f"cbapo/visits/mwmVisit-0.6.0-{int(s.sdss_id)}.fits"
    if not os.path.exists(fn) or open(fn, "rb").read(6) != b"SIMPLE": return dict(sid=s.sid, sdss_id=int(s.sdss_id), n_vis=0, note="no mwmVisit")
    t0 = time.time(); vis = read_visits(fn)
    if not vis: return dict(sid=s.sid, sdss_id=int(s.sdss_id), n_vis=0, note=f"no visit with S/N>={SNR_MIN:.0f}")
    r, x = fit_visits(s, vis, grid, t_start=t0)
    r.update(sdss_id=int(s.sdss_id))
    np.savez_compressed(OUT + f"x2/{s.sid}.npz", X2=x["X2"], teff=x["teff"], mh=MH_FINE, npix=x["npix"])
    return r


def fit_visits(s, vis, grid, t_start=None):
    """the APOGEE fit of one star from visits already in memory (read_visits format; a visit may carry 'bjd' sample times instead
    of date/exp/site). Returns (result dict, extras with the chi^2 grid and a best-fit model function)."""
    t0 = t_start or time.time()
    qq, inc, fill, geo = geo_params(s)
    cb = ContactBinary(qq, fill, ntheta=60, nphi=120)
    eph = "VEB" if veb_ok(s) else "nuisance"; P = 1 / s.Freq if eph == "VEB" else s.Per
    lw = np.log(WL); obs = []
    for v in vis:
        sh = v["v_rad"] / C; good = np.isfinite(v["f"]) & (v["iv"] > 0) & (v["f"] > 0.3) & (v["f"] < 1.3)
        o = np.interp(lw, lw + sh, np.where(good, v["f"], np.nan), left=np.nan, right=np.nan)
        iv = np.interp(lw, lw + sh, np.where(good, v["iv"], 0.0), left=0, right=0)
        w = np.where(np.isfinite(o) & (iv > 0), 1 / (1 / np.maximum(iv, 1e-30) + FLOOR ** 2), 0.0)
        bjd = np.asarray(v["bjd"], float) if "bjd" in v else visit_bjd(v, s.RAJ2000, s.DEJ2000)
        obs.append(dict(o=o, w=w, ph=bjd_phase(bjd, s) if eph == "VEB" else ((bjd - bjd[0]) / P) % 1, snr=v["snr"]))
    if hasattr(cb, "prepare"):     # PHOEBE geometry (CB_GEOM=phoebe): exact phases with a VEB ephemeris, else the shared phase library
        cb.prepare(np.concatenate([ob["ph"] for ob in obs]), inc, exact=(eph == "VEB"))
    Tp_m02 = s.Tphot0 - 0.2 * s.dTdfeh; sigT = s.sig_Tphot if np.isfinite(s.sig_Tphot) else 1e4
    tg = teff_grid_for(Tp_m02 if np.isfinite(Tp_m02) else 5800, sigT)
    lnw = grid.lnw

    def models(T, shifts):
        M1 = (T / 5772.0) ** 1.8; A = (M1 * (1 + qq) * (P / 365.25) ** 2) ** (1 / 3) * 215.03
        logg, Tl = cb.surface(M1, A, T); out = []
        for ob, dp in zip(obs, shifts):
            el = [], [], [], [], []
            for p in (ob["ph"] + dp) % 1:
                los, mu, vv = cb.view(p, inc); rv = cb.rv(los, A, P)
                for lst, val in zip(el, (Tl[vv], logg[vv], mu[vv], rv[vv], cb.dA[vv] / len(ob["ph"]))): lst.append(val)
            out.append(grid.integrate_nodes(*[np.concatenate(x) for x in el], vmax=600, lsf_R=22500.0))
        return out

    T0 = tg[np.argmin(np.abs(tg - Tp_m02))] if np.isfinite(Tp_m02) else tg[len(tg) // 2]
    gsh = np.arange(-150, 151, 2)
    if eph == "VEB":
        c2 = np.zeros(len(gsh))
        for (F, Fc), ob in zip(models(T0, [0.0] * len(obs)), obs):
            nm = norm_at(F, Fc, -0.2); c2 += [chi2_chips(WL, ob["o"], ob["w"], np.interp(lw, lnw + g / C, nm, left=np.nan, right=np.nan)) for g in gsh]
        shifts = [0.0] * len(obs)
    else:
        # per-visit phase chosen by the minimum over velocity, then a common systemic velocity
        cmat = np.full((len(PH0), len(obs), len(gsh)), np.inf)
        for a, p0 in enumerate(PH0):
            for j, ((F, Fc), ob) in enumerate(zip(models(T0, [p0] * len(obs)), obs)):
                nm = norm_at(F, Fc, -0.2); cmat[a, j] = [chi2_chips(WL, ob["o"], ob["w"], np.interp(lw, lnw + g / C, nm, left=np.nan, right=np.nan)) for g in gsh]
        best_a = np.argmin(cmat.min(axis=2), axis=0); shifts = [PH0[a] for a in best_a]
        c2 = sum(cmat[best_a[j], j] for j in range(len(obs)))
    gam = gsh[int(np.argmin(c2))]
    X2 = np.zeros((len(tg), len(MH_FINE)))
    for a_, T in enumerate(tg):
        for (F, Fc), ob in zip(models(T, shifts), obs):
            for b_, m in enumerate(MH_FINE):
                X2[a_, b_] += chi2_chips(WL, ob["o"], ob["w"], np.interp(lw, lnw + gam / C, norm_at(F, Fc, m), left=np.nan, right=np.nan))
    npix = sum(int(np.sum(ob["w"] > 0)) for ob in obs)
    r = joint(X2, tg.astype(float), MH_FINE, npix, s.Tphot0, s.dTdfeh, sigT, npar=9 * len(obs) + 2)
    r.update(sid=s.sid, n_vis=len(obs), snr_tot=float(np.sqrt(sum(ob["snr"] ** 2 for ob in obs))), P=P, q=qq, i=inc, f=fill,
             geo=geo, eph=eph, gamma=gam, Tphot0=s.Tphot0, dTdfeh=s.dTdfeh, sig_Tphot=sigT, ebv=s["E_B-V_"], flag_ebv=s.flag_ebv,
             flag_colour=s.flag_colour, tg_lo=tg[0], tg_hi=tg[-1], secs=time.time() - t0)

    def best_model(T=None, m=None):
        """best-fit model on the APOGEE pixels per visit (rest frame of the pipeline velocity, as fitted), with the chip polynomials"""
        T = r["teff_joint"] if T is None else T; m = r["mh_joint"] if m is None else m; out = []
        for (F, Fc), ob in zip(models(T, shifts), obs):
            mod = np.interp(lw, lnw + gam / C, norm_at(F, Fc, m), left=np.nan, right=np.nan); fm = np.full(len(WL), np.nan)
            for lo, hi in CHIPS:
                k = (WL > lo) & (WL < hi) & (ob["w"] > 0) & np.isfinite(mod) & np.isfinite(ob["o"]); ka = (WL > lo) & (WL < hi) & np.isfinite(mod)
                if k.sum() < 100: continue
                xx = (WL[k] - 0.5 * (lo + hi)) / (hi - lo); A = np.vander(xx, 3) * mod[k][:, None]; sw = np.sqrt(ob["w"][k])
                c, *_ = np.linalg.lstsq(A * sw[:, None], ob["o"][k] * sw, rcond=None)
                fm[ka] = (np.vander((WL[ka] - 0.5 * (lo + hi)) / (hi - lo), 3) * mod[ka][:, None]) @ c
            out.append(fm)
        return out
    return r, dict(X2=X2, teff=tg, npix=npix, best_model=best_model, obs=obs, phase_shifts=shifts)


if __name__ == "__main__":
    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(OUT + "x2", exist_ok=True)
    tag = os.environ.get("APO_TAG", "")
    fn = OUT + f"results{tag}_{shard}.csv"
    rows = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []
    done = {r["sid"] for r in rows}
    ap = pd.read_csv(D + "cbsynth/dr19_all_ew.csv").drop(columns=["Per"])
    tp = pd.read_csv(D + "cbscale/ew_teffphot.csv"); veb = pd.read_csv(D + "cbscale/veb_all.csv")
    st = ap.merge(tp, on="sid").merge(veb, left_on="sid", right_on="Source", how="left").merge(geometry(), on="sid", how="left")
    st = st.drop_duplicates("sid").sort_values("snr_tot", ascending=False).iloc[shard::nshard]
    if tag:   # second pass: only stars that had no usable visit in the first pass
        prev = pd.concat([pd.read_csv(f) for f in glob.glob(OUT + "results_*.csv")])
        st = st[st.sid.isin(prev.sid[prev.note.astype(str).str.startswith("no visit")])]
    grid = load("H", dv=1.0)
    for _, s in st.iterrows():
        if s.sid in done: continue
        try:
            r = fit_star(s, grid)
        except Exception as e:
            print(f"{s.sid} ERROR {repr(e)[:200]}", flush=True); r = dict(sid=s.sid, sdss_id=int(s.sdss_id), n_vis=0, note="error " + repr(e)[:80])
        rows.append(r); done.add(s.sid)
        if r.get("n_vis", 0) > 0:
            print(f"{s.sid} nvis={r['n_vis']} {r['eph']} gamma={r['gamma']} T={r['teff_joint']:.0f}+-{r['teff_err']:.0f} (phot {s.Tphot0-0.2*s.dTdfeh:.0f}) "
                  f"[M/H]={r['mh_joint']:+.2f}+-{r['mh_err']:.2f} chi2r {r['chi2_red']:.2f} ({r['secs']:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)
    print("finished", len(rows), flush=True)
