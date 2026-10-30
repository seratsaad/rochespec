"""Zero point of the payne-zero optical model on LAMOST DR10 LRS from ~1000 APOGEE DR19 single dwarfs.

For every calibrator (ASPCAP Teff, log g, [Fe/H] known):
 (1) single-star model, chi^2 on a (Teff, [M/H]) grid centred on the ASPCAP Teff (offsets -300..+300 K)
 (2) the observed spectrum spun up by Roche kernels (q=0.4, f=0.25, i=80, P=0.37 d; phases 0.1, 0.25, 0.4) and
     fitted with the Roche model on the same grid
The chi^2 grids are saved, so [M/H] can be read at the ASPCAP Teff or with the same photometric-prior joint
inference used for the contact binaries.
Usage: CB_MODEL=pz python -m rochespec.pipelines.calib_lrs <shard> <nshard>
"""
import sys, os, time
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from ..roche import ContactBinary, C
from ..model_cfg import MHS, load, MODEL
from ..core import D, MH_FINE, read_dr10
from .lrs import DV, MASK, FLOOR, chi2_spec, norm_at

assert MODEL == "pz"
OUT = D + "cbscale/res_cal/"
DT = np.arange(-300, 301, 100)
PH = [0.1, 0.25, 0.4]
grid = load("opt", dv=DV, rebin=True)
lnw, mu, wmu = grid.lnw, grid.mu, grid.wmu
sig = C / 1800.0 / 2.3548 / DV


def lin(n, x):
    x = np.clip(x, n[0], n[-1]); j = min(np.searchsorted(n, x, side="right") - 1, len(n) - 2); return j, (x - n[j]) / (n[j + 1] - n[j])


def single_nodes(T, g):
    """disk-integrated F, Fc of a non-rotating star at every [M/H] node, LSF-convolved"""
    jt, ft = lin(grid.teffs, T); jg, fg = lin(grid.loggs, g); F = Fc = 0
    for a, wa in ((0, 1 - ft), (1, ft)):
        for b, wb in ((0, 1 - fg), (1, fg)):
            w = wa * wb
            F = F + w * np.sum((wmu * mu)[None, :, None] * grid.I[:, jt + a, jg + b], 1)
            Fc = Fc + w * np.sum((wmu * mu)[None, :, None] * grid.Ic[:, jt + a, jg + b], 1)
    return gaussian_filter1d(F, sig, axis=1), gaussian_filter1d(Fc, sig, axis=1)


if __name__ == "__main__":
    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(OUT + "x2", exist_ok=True)
    fn = OUT + f"calib_{shard}.csv"
    rows = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []
    done = {r["sdss_id"] for r in rows}
    cs = pd.read_csv(D + "cbscale/cal_targets.csv").iloc[shard::nshard]
    cb = ContactBinary(0.4, 0.25, ntheta=60, nphi=120); INC, P = 80.0, 0.37
    while True:
        todo = cs[~cs.sdss_id.isin(done)]
        if len(todo) == 0: break
        progressed = False
        for _, st in todo.iterrows():
            f = D + f"lamost_dr10/cal/{int(st.obsid)}.fits.gz"
            if not os.path.exists(f):
                bad = open(D + "lamost_dr10/cal/failed.txt").read().split() if os.path.exists(D + "lamost_dr10/cal/failed.txt") else []
                if str(int(st.obsid)) in bad: done.add(st.sdss_id)
                continue
            progressed = True; t0 = time.time()
            try:
                wl, fl, iv, _, _ = read_dr10(f)
            except Exception as e:
                done.add(st.sdss_id); print("read error", st.sdss_id, e, flush=True); continue
            good = (iv > 0) & np.isfinite(fl) & (wl > 3900)
            for lo, hi in MASK: good &= ~((wl > lo) & (wl < hi))
            med = np.nanmedian(fl[good]); fl = fl / med; iv = iv * med ** 2
            w = np.where(good, 1 / (1 / np.maximum(iv, 1e-30) + (FLOOR * np.abs(fl)) ** 2), 0.0); lw = np.log(wl)
            Ta, g = float(st.teff), float(st.logg); tg = Ta + DT
            F0, Fc0 = single_nodes(Ta, g); base = norm_at(F0, Fc0, -0.2)
            c2 = [chi2_spec(wl, fl, w, np.interp(lw, lnw + k * DV / C, base, left=np.nan, right=np.nan)) for k in range(-30, 31)]
            k0 = int(np.argmin(c2)) - 30
            X1 = np.zeros((len(tg), len(MH_FINE)))
            for a, T in enumerate(tg):
                F, Fc = single_nodes(T, g)
                for b, m in enumerate(MH_FINE):
                    X1[a, b] = chi2_spec(wl, fl, w, np.interp(lw, lnw + k0 * DV / C, norm_at(F, Fc, m), left=np.nan, right=np.nan))
            # spin-up
            o_rest = np.interp(lnw, lw - k0 * DV / C, np.where(good, fl, np.nan)); okm = np.isfinite(o_rest); o_fill = np.where(okm, o_rest, 1.0)
            X2 = np.zeros((len(tg), len(MH_FINE))); npix2 = 0
            fakes = []
            for p in PH:
                M1 = (Ta / 5772.0) ** 1.8; A = (M1 * 1.4 * (P / 365.25) ** 2) ** (1 / 3) * 215.03
                logg_e, T_e = cb.surface(M1, A, Ta)
                los, m_, vis = cb.view(p, INC); rv = cb.rv(los, A, P)
                K = 120; ker = np.zeros(2 * K + 1)
                np.add.at(ker, np.clip(np.round(rv[vis] / DV).astype(int) + K, 0, 2 * K), cb.dA[vis] * m_[vis]); ker /= ker.sum()
                fake = np.convolve(o_fill, ker, mode="same"); fake[~okm] = np.nan
                fk = np.interp(lw, lnw, fake); wk = np.where(np.isfinite(fk) & good, w, 0.0); fakes.append((p, fk, wk)); npix2 += int((wk > 0).sum())
            for a, T in enumerate(tg):
                M1 = (T / 5772.0) ** 1.8; A = (M1 * 1.4 * (P / 365.25) ** 2) ** (1 / 3) * 215.03
                logg_e, T_e = cb.surface(M1, A, T)
                for p, fk, wk in fakes:
                    los, m_, vis = cb.view(p, INC); rv = cb.rv(los, A, P)
                    F, Fc = grid.integrate_nodes(T_e[vis], logg_e[vis], m_[vis], rv[vis], cb.dA[vis], vmax=600, lsf_R=1800.0)
                    for b, m in enumerate(MH_FINE):
                        X2[a, b] += chi2_spec(wl, fk, wk, np.interp(lw, lnw, norm_at(F, Fc, m), left=np.nan, right=np.nan))
            j0 = int(np.argmin(np.abs(DT)))
            np.savez_compressed(OUT + f"x2/{int(st.sdss_id)}.npz", X1=X1, X2=X2, teff=tg, mh=MH_FINE, npix1=int((w > 0).sum()), npix2=npix2)
            rows.append(dict(sdss_id=int(st.sdss_id), obsid=int(st.obsid), sid=int(st.gaia_dr3_source_id), teff=Ta, logg=g, feh_aspcap=st.fe_h,
                             v_macro=st.v_macro, snrg=st.snrg, feh_lamost=st.feh, teff_lamost=st.teff_lam if "teff_lam" in st else np.nan,
                             bp_rp=st.bp_mag - st.rp_mag, plx=st.plx, ra=st.ra, dec=st.dec,
                             mh_single=MH_FINE[np.argmin(X1[j0])], mh_roche=MH_FINE[np.argmin(X2[j0])],
                             chi2r_single=X1.min() / max((w > 0).sum() - 20, 1), secs=time.time() - t0))
            done.add(st.sdss_id)
            print(f"{int(st.sdss_id)} T={Ta:.0f} g={g:.2f} ASPCAP {st.fe_h:+.2f} LAMOST {st.feh:+.2f} | single {rows[-1]['mh_single']:+.2f} | Roche {rows[-1]['mh_roche']:+.2f} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)
        if not progressed: time.sleep(60)
    print("finished", len(rows), flush=True)
