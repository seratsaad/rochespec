"""APOGEE zero point with the extended payne-zero grid (log g to 4.8): spin-up injection test on slow single dwarfs.

Real DR19 mwmStar spectra of APOGEE single dwarfs (census verdict SINGLE) are convolved with Roche kernels
(q=0.4, f=0.25, i=80, P=0.37 d; 4 phases; S/N 100 noise) and fitted with the Roche model on a (Teff, [M/H]) grid
centred on the ASPCAP Teff (offsets -300..+300 K). chi^2 grids are saved for the joint (photometric-prior) inference.
Usage: CB_MODEL=pz python -m rochespec.pipelines.calib_apogee <shard> <nshard>
"""
from ..config import A4B
import sys, os, time
import numpy as np, pandas as pd
from ..roche import ContactBinary, C
from ..model_cfg import MHS, load, MODEL
from ..core import D, MH_FINE
from .apogee import chi2_chips, norm_at, CHIPS

assert MODEL == "pz"
R = A4B
OUT = D + "cbscale/res_spin/"
PH = [0.1, 0.25, 0.4, 0.75]
DT = np.arange(-300, 301, 100)


def select(n=120):
    man = pd.read_csv(R + "resources/dr19_census272k_manifest.csv", low_memory=False)
    cat = pd.read_csv(R + "resources/census/dr19_sb2_catalog_full.csv", usecols=["sdss_id", "verdict"])
    s = man.merge(cat, on="sdss_id")
    have = {int(f[:-4]) for f in os.listdir(R + "data/dr19_raw") if f[:-4].isdigit()}
    s = s[s.sdss_id.isin(have) & (s.verdict == "SINGLE") & (s.v_macro < 6) & (s.snr > 150) & s.teff.between(4850, 6600)
          & s.logg.between(3.8, 4.75) & s.fe_h.between(-0.95, 0.3)].copy()
    s["cell"] = (s.teff // 300).astype(int).astype(str) + "_" + (s.fe_h // 0.25).astype(int).astype(str)
    s = s.sample(frac=1, random_state=2); s["rank"] = s.groupby("cell").cumcount()
    return s.sort_values(["rank", "teff"]).head(n)


if __name__ == "__main__":
    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(OUT + "x2", exist_ok=True)
    sel = select(); sel.to_csv(OUT + "spin_targets.csv", index=False); sel = sel.iloc[shard::nshard]
    fn = OUT + f"spin_{shard}.csv"; rows = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []
    done = {r["sdss_id"] for r in rows}
    grid = load("H", dv=1.0); lnw = grid.lnw
    cb = ContactBinary(0.4, 0.25, ntheta=60, nphi=120); INC, P = 80.0, 0.37
    rng = np.random.default_rng(3)
    for _, st in sel.iterrows():
        if st.sdss_id in done: continue
        t0 = time.time()
        d = np.load(R + f"data/dr19_raw/{int(st.sdss_id)}.npz"); wl = d["wl"]; lw = np.log(wl)
        f = d["flux_raw"]; iv = d["ivar"]; good = (iv > 0) & np.isfinite(f) & (f > 0)
        obs = np.full_like(f, np.nan)
        for lo, hi in CHIPS:
            k = good & (wl > lo) & (wl < hi); obs[k] = f[k] / np.median(f[k])
        Ta, g = float(st.teff), float(st.logg); tg = Ta + DT
        obs_m = np.interp(lnw, lw, np.where(np.isfinite(obs), obs, 1.0))
        M1 = (Ta / 5772.0) ** 1.8; A = (M1 * 1.4 * (P / 365.25) ** 2) ** (1 / 3) * 215.03
        fakes = []
        for p in PH:
            logg_e, T_e = cb.surface(M1, A, Ta); los, m_, vis = cb.view(p, INC); rv = cb.rv(los, A, P)
            K = int(600 / grid.dv); ker = np.zeros(2 * K + 1)
            np.add.at(ker, np.clip(np.round(rv[vis] / grid.dv).astype(int) + K, 0, 2 * K), cb.dA[vis] * m_[vis]); ker /= ker.sum()
            fake = np.interp(lw, lnw, np.convolve(obs_m, ker, mode="same")); fake[~np.isfinite(obs)] = np.nan
            fake = fake + rng.normal(0, 0.01, fake.size) * np.nanmedian(fake)
            fakes.append((p, fake, np.where(np.isfinite(fake), 1 / 0.01 ** 2, 0.0)))
        X2 = np.zeros((len(tg), len(MH_FINE)))
        for a, T in enumerate(tg):
            M1 = (T / 5772.0) ** 1.8; A = (M1 * 1.4 * (P / 365.25) ** 2) ** (1 / 3) * 215.03
            logg_e, T_e = cb.surface(M1, A, T)
            for p, fk, wk in fakes:
                los, m_, vis = cb.view(p, INC); rv = cb.rv(los, A, P)
                F, Fc = grid.integrate_nodes(T_e[vis], logg_e[vis], m_[vis], rv[vis], cb.dA[vis], vmax=600, lsf_R=22500.0)
                for b, m in enumerate(MH_FINE):
                    X2[a, b] += chi2_chips(wl, fk, wk, np.interp(lw, lnw, norm_at(F, Fc, m)))
        npix = sum(int((wk > 0).sum()) for _, _, wk in fakes)
        np.savez_compressed(OUT + f"x2/{int(st.sdss_id)}.npz", X2=X2, teff=tg, mh=MH_FINE, npix=npix)
        j0 = int(np.argmin(np.abs(DT)))
        rows.append(dict(sdss_id=int(st.sdss_id), sid=int(st.gaia_dr3_source_id), teff=Ta, logg=g, feh_aspcap=st.fe_h, v_macro=st.v_macro,
                         bp_rp=st.bp_mag - st.rp_mag, plx=st.plx, mh_roche=MH_FINE[np.argmin(X2[j0])], secs=time.time() - t0))
        done.add(st.sdss_id)
        print(f"{int(st.sdss_id)} T={Ta:.0f} g={g:.2f} ASPCAP {st.fe_h:+.2f} | Roche {rows[-1]['mh_roche']:+.2f} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)
    print("finished", len(rows), flush=True)
