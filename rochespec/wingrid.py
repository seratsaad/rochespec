"""Shared engine for the survey modules (LAMOST MRS, GALAH DR4, DESI DR1).

WinGrid: payne-zero I(lambda, mu) restricted to wavelength windows, resampled (flux conserving) onto uniform ln(lambda)
grids, with the FFT of every node precomputed so a Roche disk integral is one einsum + one inverse FFT per window.
The continuum Fc is the same weighted sum of Ic without the velocity convolution (Ic is smooth).
Roche geometry comes from ../cbsynth/roche.py (read-only import); node lists from ../cbsynth/model_cfg.py at run time.
"""
from .config import DATA
import os, sys, glob, time
import numpy as np, pandas as pd
from scipy.fft import rfft, irfft, next_fast_len
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

os.environ.setdefault("CB_MODEL", "pz")
from .roche import ContactBinary, C           # noqa: E402
from . import model_cfg                              # noqa: E402

D = DATA + ""
S = D + "surveys/"
PZ = D + "pz/grid"
SITES = {"lamost": EarthLocation.from_geodetic(117.58 * u.deg, 40.39 * u.deg, 960 * u.m),
         "aat": EarthLocation.from_geodetic(149.0661 * u.deg, -31.2770 * u.deg, 1164 * u.m),
         "kpno": EarthLocation.from_geodetic(-111.5997 * u.deg, 31.9634 * u.deg, 2120 * u.m)}
MH_STEP = 0.05


def air_to_vac(wl):
    """Ciddor/Morton (IAU) air -> vacuum, Angstrom"""
    s2 = (1e4 / wl) ** 2
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s2) + 0.0001599740894897 / (38.92568793293 - s2)
    return wl * n


def nodes():
    """node lists from model_cfg at run time, keeping only nodes whose opt1-3 files all exist"""
    T, G, M = list(model_cfg.TEFFS), list(model_cfg.LOGGS), list(model_cfg.MHS)
    have = lambda t, g, m: all(os.path.exists(f"{PZ}/{w}/I_T{int(t)}_g{g}_m{m}.npz") for w in ("opt1", "opt2", "opt3"))
    G = [g for g in G if all(have(t, g, m) for t in T for m in M)]
    M = [m for m in M if all(have(t, g, m) for t in T for g in G)]
    T = [t for t in T if all(have(t, g, m) for g in G for m in M)]
    return np.array(T, float), np.array(G, float), np.array(M, float)


class WinGrid:
    def __init__(self, windows, dv, vmax=600.0, teffs=None):
        self.TEFFS, self.LOGGS, self.MHS = nodes()
        if teffs is not None:
            self.TEFFS = self.TEFFS[(self.TEFFS >= teffs[0]) & (self.TEFFS <= teffs[1])]
        self.dv, self.K = dv, int(vmax / dv)
        self.windows = windows
        f0 = np.load(f"{PZ}/opt1/I_T{int(self.TEFFS[0])}_g{self.LOGGS[0]}_m{self.MHS[0]}.npz")
        self.mu, self.wmu = f0["mu"], f0["wmu"]
        nT, ng, nm, nM = len(self.TEFFS), len(self.LOGGS), len(self.mu), len(self.MHS)
        self.lnw, self.nfft, self.Ihat, self.Ic, self.edges = [], [], [], [], []
        marg = (vmax + 200.0) / C
        for lo, hi in windows:
            ln = np.arange(np.log(lo) - marg, np.log(hi) + marg, dv / C)
            self.lnw.append(ln); self.edges.append(np.exp(np.r_[ln - 0.5 * dv / C, ln[-1] + 0.5 * dv / C]))
            nf = next_fast_len(len(ln) + self.K + 8); self.nfft.append(nf)
            self.Ihat.append(np.zeros((nM, nT * ng * nm, nf // 2 + 1), np.complex64))
            self.Ic.append(np.zeros((nM, nT * ng * nm, len(ln)), np.float32))
        chunks = {"opt1": (3800, 5500), "opt2": (5500, 7200), "opt3": (7200, 8900)}
        for a, m in enumerate(self.MHS):
            for b, t in enumerate(self.TEFFS):
                for c, g in enumerate(self.LOGGS):
                    data = {}
                    for w, (clo, chi) in chunks.items():
                        if any(lo * np.exp(-marg) - 5 < chi and hi * np.exp(marg) + 5 > clo for lo, hi in windows):
                            f = np.load(f"{PZ}/{w}/I_T{int(t)}_g{g}_m{m}.npz"); data[w] = (f["wl_nm"] * 10, f["I"], f["Ic"])
                    wl = np.concatenate([v[0] for v in data.values()]); o = np.argsort(wl); wl = wl[o]
                    keep = np.r_[True, np.diff(wl) > 0]; wl = wl[keep]
                    I = np.concatenate([v[1] for v in data.values()], axis=1)[:, o][:, keep]
                    Ic = np.concatenate([v[2] for v in data.values()], axis=1)[:, o][:, keep]
                    dwl = np.diff(wl)
                    for k_w in range(len(windows)):
                        ed = self.edges[k_w]; j0 = max(np.searchsorted(wl, ed[0]) - 2, 0); j1 = min(np.searchsorted(wl, ed[-1]) + 2, len(wl))
                        sl = slice(j0, j1)
                        for k in range(nm):
                            row = (b * ng + c) * nm + k
                            for arr, out, fft in ((I, None, True), (Ic, None, False)):
                                y = arr[k, sl]; x = wl[sl]
                                cum = np.r_[0.0, np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))]
                                r = np.diff(np.interp(ed, x, cum)) / np.diff(ed)
                                if fft:
                                    self.Ihat[k_w][a, row] = rfft(r, n=self.nfft[k_w])
                                else:
                                    self.Ic[k_w][a, row] = r
        self.wl = [np.exp(x) for x in self.lnw]
        self._patch_nan()

    def _patch_nan(self):
        """fill NaN (Teff, logg, [M/H]) nodes by linear inter/extrapolation in logg (as cbsynth/model_cfg._patch_nan_nodes)"""
        nT, ng, nm = len(self.TEFFS), len(self.LOGGS), len(self.mu); self.patched = []
        for k_w in range(len(self.windows)):
            Ih = self.Ihat[k_w].reshape(len(self.MHS), nT, ng, nm, -1); Ic = self.Ic[k_w].reshape(len(self.MHS), nT, ng, nm, -1)
            for a in range(len(self.MHS)):
                for b in range(nT):
                    fin = [np.isfinite(Ih[a, b, c]).all() and np.isfinite(Ic[a, b, c]).all() for c in range(ng)]
                    for c in range(ng):
                        if fin[c]: continue
                        ok = [k for k in range(ng) if fin[k]]
                        k1, k2 = sorted(ok, key=lambda k: abs(k - c))[:2]
                        w = (self.LOGGS[c] - self.LOGGS[k1]) / (self.LOGGS[k2] - self.LOGGS[k1])
                        for arr in (Ih, Ic): arr[a, b, c] = (1 - w) * arr[a, b, k1] + w * arr[a, b, k2]
                        self.patched.append((k_w, self.MHS[a], self.TEFFS[b], self.LOGGS[c]))

    @staticmethod
    def _lin(nodes_, x):
        x = np.clip(x, nodes_[0], nodes_[-1]); j = np.clip(np.searchsorted(nodes_, x) - 1, 0, len(nodes_) - 2)
        return j, (x - nodes_[j]) / (nodes_[j + 1] - nodes_[j])

    def weights(self, T, logg, mu, v, dA):
        """W[node, vbin]: projected-area weights per (Teff, logg, mu) node and velocity bin"""
        K = self.K; nT, ng, nm = len(self.TEFFS), len(self.LOGGS), len(self.mu)
        W = np.zeros((nT, ng, nm, 2 * K + 1))
        jt, wt = self._lin(self.TEFFS, T); jg, wg = self._lin(self.LOGGS, logg); jm, wm = self._lin(self.mu, mu)
        x = np.clip(v / self.dv, -K + 1, K - 2); jv = np.floor(x).astype(int); wv = x - jv
        base = dA * mu
        for a_, fa in ((0, 1 - wt), (1, wt)):
            for b_, fb in ((0, 1 - wg), (1, wg)):
                for c_, fc in ((0, 1 - wm), (1, wm)):
                    for d_, fd in ((0, 1 - wv), (1, wv)):
                        np.add.at(W, (jt + a_, jg + b_, jm + c_, jv + d_ + K), base * fa * fb * fc * fd)
        return W.reshape(nT * ng * nm, 2 * K + 1)

    def integrate(self, T, logg, mu, v, dA, sigma_kms=None):
        """F, Fc per window for every [M/H] node: lists of (n_mh, n_pix). sigma_kms: Gaussian LSF sigma per window (None = no LSF)."""
        W = self.weights(T, logg, mu, v, dA); wsum = W.sum(1); nz = np.where(wsum > 0)[0]; W = W[nz]; K = self.K
        out = []
        for k_w in range(len(self.windows)):
            nf = self.nfft[k_w]; kk = np.zeros((len(nz), nf))
            kk[:, :K + 1] = W[:, K:]; kk[:, nf - K:] = W[:, :K]
            Kh = rfft(kk, axis=1).astype(np.complex64)
            Fh = np.einsum("mnf,nf->mf", self.Ihat[k_w][:, nz], Kh)
            sg = None if sigma_kms is None else (sigma_kms[k_w] if np.ndim(sigma_kms) else sigma_kms)
            if sg:
                f = np.fft.rfftfreq(nf); Fh = Fh * np.exp(-2 * (np.pi * f * sg / self.dv) ** 2)[None, :]
            F = irfft(Fh, n=nf, axis=1)[:, :len(self.lnw[k_w])]
            Fc = np.einsum("mnx,n->mx", self.Ic[k_w][:, nz], wsum[nz])
            out.append((F, Fc))
        return out

    def cont_intensity(self, T, logg, mu, k_w):
        """continuum intensity Ic(T, logg, mu) of each element at the centre of window k_w ([M/H] node nearest 0)"""
        nT, ng, nm = len(self.TEFFS), len(self.LOGGS), len(self.mu); a = int(np.argmin(np.abs(self.MHS)))
        Ic = self.Ic[k_w][a, :, len(self.lnw[k_w]) // 2].reshape(nT, ng, nm)
        jt, wt = self._lin(self.TEFFS, T); jg, wg = self._lin(self.LOGGS, logg); jm, wm = self._lin(self.mu, mu); out = 0.0
        for a_, fa in ((0, 1 - wt), (1, wt)):
            for b_, fb in ((0, 1 - wg), (1, wg)):
                for c_, fc in ((0, 1 - wm), (1, wm)):
                    out = out + fa * fb * fc * Ic[jt + a_, jg + b_, jm + c_]
        return out

    def single(self, T, logg, sigma_kms):
        """disk-integrated single-star F, Fc (flux = sum w_mu mu I) with Gaussian broadening"""
        n = len(self.mu)
        return self.integrate(np.full(n, T), np.full(n, logg), self.mu.copy(), np.zeros(n), self.wmu.copy(), sigma_kms)


def mh_interp(MHS, F, Fc, mh_fine):
    """normalised model F/Fc at each fine [M/H] (linear in the node fluxes): (n_fine, n_pix)"""
    j = np.clip(np.searchsorted(MHS, mh_fine, side="right") - 1, 0, len(MHS) - 2)
    f = ((mh_fine - MHS[j]) / (MHS[j + 1] - MHS[j]))[:, None]
    return ((1 - f) * F[j] + f * F[j + 1]) / ((1 - f) * Fc[j] + f * Fc[j + 1])


def legendre_basis(wl, segs, deg):
    out = []
    for lo, hi in segs:
        k = (wl >= lo) & (wl <= hi)
        x = 2 * (wl[k] - lo) / (hi - lo) - 1
        out.append((k, np.polynomial.legendre.legvander(x, deg)))
    return out


def chi2_multi(o, w, M, basis):
    """chi^2 of observed o (weights w) against each model row of M (n_mod, n_pix) after a multiplicative
    Legendre polynomial per segment, solved by normal equations; vectorised over models. NaN model pixels are dropped."""
    M = np.atleast_2d(M); tot = np.zeros(len(M))
    good = np.all(np.isfinite(M), axis=0) & (w > 0) & np.isfinite(o)
    for k, L in basis:
        kk = k & good
        if kk.sum() < L.shape[1] + 5: continue
        Lk = L[good[k]]; ww = w[kk]; oo = o[kk]; Mk = M[:, kk]
        G = np.einsum("kn,np,nq->kpq", Mk * Mk * ww, Lk, Lk)
        b = np.einsum("kn,np->kp", Mk * ww * oo, Lk)
        c = np.linalg.solve(G + 1e-12 * np.eye(G.shape[1]), b[..., None])[..., 0]
        tot += np.sum(ww * oo * oo) - np.einsum("kp,kp->k", c, b)
    return tot


def bjd(mjd_utc, ra, dec, site):
    t = Time(np.atleast_1d(mjd_utc), format="mjd", scale="utc", location=SITES[site]); c = SkyCoord(ra * u.deg, dec * u.deg)
    return (t.tdb + t.light_travel_time(c, kind="barycentric")).jd


def phase_of(bjd_, s):
    """phase with the deeper (Gaia derived primary) minimum at 0, as in cblam/fit_lamost.py"""
    p1, p2 = s.PhaseGauss1, s.PhaseGauss2
    d1, d2 = np.nan_to_num(s.DepthGauss1), np.nan_to_num(s.DepthGauss2)
    pmin = p1 if (np.isfinite(p1) and (d1 >= d2 or not np.isfinite(p2))) else p2
    if not np.isfinite(pmin): pmin = 0.0
    return ((bjd_ - (2455197.5 + s.TimeRef)) * s.Freq - pmin) % 1


def substeps(mjd_beg, mjd_end, n=3):
    """sub-exposure sample times (UTC MJD) for smearing: centres of n equal slices"""
    return mjd_beg + (mjd_end - mjd_beg) * (np.arange(n) + 0.5) / n


class Spec:
    """one observed spectrum: vacuum wavelength (Angstrom, barycentric/heliocentric frame), flux, weights,
    sample times (UTC MJD, sub-steps across the exposure(s)) and their weights, group label for velocity offsets"""
    def __init__(self, wl, fl, w, t_mjd, t_w, label, group, meta=None):
        self.wl, self.fl, self.w = wl, fl, w
        self.lw = np.log(wl)
        self.t_mjd, self.t_w = np.atleast_1d(t_mjd), np.atleast_1d(t_w) / np.sum(t_w)
        self.label, self.group, self.meta = label, group, meta or {}


def model_on_pixels(grid, mods, spec, gamma, mh_fine, lsf_apply=None):
    """mods: integrate() output for this spectrum; returns (n_fine, n_pix) normalised model at the observed pixels"""
    out = np.full((len(mh_fine), len(spec.wl)), np.nan)
    for k_w, (F, Fc) in enumerate(mods):
        lo, hi = grid.windows[k_w]
        k = (spec.wl >= lo) & (spec.wl <= hi)
        if not k.any(): continue
        nm = mh_interp(grid.MHS, F, Fc, mh_fine)
        x = spec.lw[k] - gamma / C
        if lsf_apply is not None:
            out[:, k] = lsf_apply(k_w, nm, x, spec)
        else:
            out[:, k] = np.array([np.interp(x, grid.lnw[k_w], r, left=np.nan, right=np.nan) for r in nm])
    return out


class RocheStar:
    """Roche model for one contact binary; views cached per sample phase (independent of Teff)"""
    def __init__(self, q, fill, inc, P, ntheta=60, nphi=120):
        self.q = q if q <= 1 else 1 / q; self.inc, self.P = inc, P
        self.cb = ContactBinary(self.q, fill, ntheta=ntheta, nphi=nphi); self.cache = {}

    def view(self, ph):
        key = round(float(ph), 4)
        if key not in self.cache:
            los, mu, vis = self.cb.view(key, self.inc)
            # rv for a=1 Rsun, P=1 d (scales as a/P); dA stored per phase (PHOEBE meshes: area x visible fraction of that phase)
            self.cache[key] = (vis, mu[vis], self.cb.rv(los, 1.0, 1.0)[vis], self.cb.dA[vis].copy())
        return self.cache[key]

    def elements(self, T, phases, wts):
        M1 = (T / 5772.0) ** 1.8; A = (M1 * (1 + self.q) * (self.P / 365.25) ** 2) ** (1 / 3) * 215.03
        logg, Tl = self.cb.surface(M1, A, T); el = [], [], [], [], []
        for p, wt in zip(phases, wts):
            vis, mu, rv1, dA = self.view(p)
            for lst, val in zip(el, (Tl[vis], logg[vis], mu, rv1 * A / self.P, dA * wt)): lst.append(val)
        return [np.concatenate(x) for x in el]


def clip(spec, m, basis, hi=5.0, lo=8.0):
    """zero the weights of outlier pixels (cosmics, emission) relative to model row m after the polynomial scaling"""
    good = np.isfinite(m) & (spec.w > 0); w = spec.w.copy()
    for k, L in basis:
        kk = k & good
        if kk.sum() < L.shape[1] + 5: continue
        Lk = L[good[k]]; A = Lk * m[kk][:, None]; sw = np.sqrt(w[kk])
        c, *_ = np.linalg.lstsq(A * sw[:, None], spec.fl[kk] * sw, rcond=None)
        z = (spec.fl[kk] - A @ c) * sw
        sc = 1.4826 * np.median(np.abs(z))
        bad = (z > hi * max(sc, 1)) | (z < -lo * max(sc, 1))
        idx = np.where(kk)[0][bad]; w[idx] = 0.0
        w[np.clip(idx - 1, 0, len(w) - 1)] = 0.0; w[np.clip(idx + 1, 0, len(w) - 1)] = 0.0
    spec.w = w
    return int(np.sum(spec.w > 0))


def fit_ew(grid, star, specs, tc, sigma_kms, mh_fine, lsf_apply=None, gam_range=150, gam_step=2.0, refine=10, teff_half=700):
    """chi^2 over (Teff_sys, [M/H]) for one contact binary; returns dict of results and the chi^2 surface"""
    tg = np.arange(4900, 6601, 100.0); tg = tg[np.abs(tg - tc) <= teff_half]
    tg = tg[(tg >= grid.TEFFS[0]) & (tg <= grid.TEFFS[-1])]
    basis = [legendre_basis(s.wl, s.meta["segs"], s.meta["deg"]) for s in specs]
    phs = [(s.ph, s.t_w) for s in specs]
    if hasattr(star.cb, "prepare"):     # PHOEBE geometry: exact phases (as RocheStar.view rounds them), library if there are many
        star.cb.prepare(np.round(np.concatenate([np.atleast_1d(p) for p, _ in phs]), 4), star.inc)

    def mods_at(T):
        return [grid.integrate(*star.elements(T, ph, wt), sigma_kms=sigma_kms) for ph, wt in phs]

    T0 = tg[np.argmin(np.abs(tg - tc))]; M0 = mods_at(T0); m0 = np.array([0.0])
    shifts = np.arange(-gam_range, gam_range + 1e-6, gam_step)
    c2 = np.zeros(len(shifts))
    for mods, s, B in zip(M0, specs, basis):
        c2 += np.array([chi2_multi(s.fl, s.w, model_on_pixels(grid, mods, s, g, m0, lsf_apply), B)[0] for g in shifts])
    gam0 = shifts[np.argmin(c2)]
    for mods, s, B in zip(M0, specs, basis):
        clip(s, model_on_pixels(grid, mods, s, gam0, m0, lsf_apply)[0], B)
    # per-group (night / plate) refinement of the systemic velocity
    gam = {}
    for gname in sorted(set(s.group for s in specs)):
        idx = [i for i, s in enumerate(specs) if s.group == gname]
        sh = np.arange(gam0 - refine, gam0 + refine + 1e-6, gam_step) if refine else np.array([gam0])
        cc = np.zeros(len(sh))
        for i in idx:
            cc += np.array([chi2_multi(specs[i].fl, specs[i].w, model_on_pixels(grid, M0[i], specs[i], g, m0, lsf_apply), basis[i])[0] for g in sh])
        gam[gname] = sh[np.argmin(cc)]
    X2 = np.zeros((len(tg), len(mh_fine)))
    for a_, T in enumerate(tg):
        Ms = M0 if T == T0 else mods_at(T)
        for mods, s, B in zip(Ms, specs, basis):
            X2[a_] += chi2_multi(s.fl, s.w, model_on_pixels(grid, mods, s, gam[s.group], mh_fine, lsf_apply), B)
    npix = sum(int(np.sum(s.w > 0)) for s in specs)
    a_, b_ = np.unravel_index(np.argmin(X2), X2.shape); jT = np.argmin(np.abs(tg - tc)); b_col = np.argmin(X2[jT])
    npar = sum(len(s.meta["segs"]) * (s.meta["deg"] + 1) for s in specs) + 3
    scale = X2.min() / max(npix - npar, 1); prof = (X2[jT] - X2[jT].min()) / scale; ok = mh_fine[prof < 1]
    res = dict(n_spec=len(specs), teff_best=tg[a_], mh_best=mh_fine[b_], T_col=tc, mh_at_Tcol=mh_fine[b_col],
               mh_at_Tcol_lo=ok.min() if len(ok) else np.nan, mh_at_Tcol_hi=ok.max() if len(ok) else np.nan,
               slope=np.polyfit(tg, mh_fine[np.argmin(X2, axis=1)], 1)[0] * 100 if len(tg) > 1 else np.nan,
               chi2_red=scale, gamma=gam0, npix=npix, edge=bool(b_col in (0, len(mh_fine) - 1)))
    return res, dict(X2=X2, teff=tg, mh=mh_fine, gam=gam)


def fit_single(grid, spec, T, logg, sigma_lsf, mh_fine, vmac=(0.0, 3.0, 6.0), lsf_apply=None, gam_range=200, gam_step=2.0):
    """single-star mode: Teff, logg fixed (ASPCAP); velocity search, then [M/H] (min over Gaussian macro broadening)"""
    B = legendre_basis(spec.wl, spec.meta["segs"], spec.meta["deg"]); m0 = np.array([0.0])
    sg = lambda vm: [np.hypot(s, vm) if s is not None else (vm if vm > 0 else None) for s in sigma_lsf]
    base = grid.single(T, logg, sg(vmac[0]))
    shifts = np.arange(-gam_range, gam_range + 1e-6, gam_step)
    c2 = np.array([chi2_multi(spec.fl, spec.w, model_on_pixels(grid, base, spec, g, m0, lsf_apply), B)[0] for g in shifts])
    g0 = shifts[np.argmin(c2)]
    fine = np.arange(g0 - 3, g0 + 3.01, 0.5)
    c2 = np.array([chi2_multi(spec.fl, spec.w, model_on_pixels(grid, base, spec, g, m0, lsf_apply), B)[0] for g in fine]); g0 = fine[np.argmin(c2)]
    clip(spec, model_on_pixels(grid, base, spec, g0, m0, lsf_apply)[0], B)
    best = None
    for vm in vmac:
        x = chi2_multi(spec.fl, spec.w, model_on_pixels(grid, grid.single(T, logg, sg(vm)), spec, g0, mh_fine, lsf_apply), B)
        if best is None or x.min() < best[0].min(): best = (x, vm)
    return best[0], g0, best[1]


def spinup_fit(grid, spec, T, gamma, mh_fine, sigma_lsf, lsf_apply=None, phases=(0.1, 0.25, 0.4, 0.75), q=0.4, fill=0.25, inc=80.0, P=0.37, logg=None, ld=False):
    """spin-up test: the observed (rest-frame) spectrum convolved with Roche kernels, fitted with the Roche model at the same Teff"""
    star = RocheStar(q, fill, inc, P); B = legendre_basis(spec.wl, spec.meta["segs"], spec.meta["deg"])
    x2 = np.zeros(len(mh_fine)); rest = spec.lw - gamma / C
    good = spec.w > 0
    for p in phases:
        Tl, lg, mu, rv, dA = star.elements(T, [p], [1.0])
        if logg is not None:          # shift the Roche log g distribution so its projected-area mean equals the star's log g
            lg = lg + (logg - np.sum(lg * dA * mu) / np.sum(dA * mu))
        # rotation kernel on the model velocity grid; ld=True weights each element by its continuum intensity
        # (limb and gravity darkening from the model), otherwise by projected area only (as cblam/calib_lamost.py)
        K = grid.K; vb = np.clip(np.round(rv / grid.dv).astype(int) + K, 0, 2 * K)
        fake = np.full(len(spec.wl), np.nan)
        for k_w, (lo, hi) in enumerate(grid.windows):
            k = (spec.wl >= lo) & (spec.wl <= hi)
            if not k.any(): continue
            ker = np.zeros(2 * K + 1)
            np.add.at(ker, vb, dA * mu * (grid.cont_intensity(Tl, lg, mu, k_w) if ld else 1.0)); ker /= ker.sum()
            ln = grid.lnw[k_w]
            o = np.interp(ln, rest[good & k], spec.fl[good & k], left=np.nan, right=np.nan)
            okm = np.isfinite(o); fill_ = np.where(okm, o, np.nanmedian(o))
            conv = np.convolve(fill_, ker, mode="same")
            bad = np.convolve((~okm).astype(float), np.ones(2 * K + 1), mode="same") > 0.5 * (2 * K + 1)
            conv[bad] = np.nan
            fake[k] = np.interp(rest[k], ln, conv, left=np.nan, right=np.nan)
        wk = np.where(np.isfinite(fake) & good, spec.w, 0.0)
        sp = Spec(spec.wl, np.nan_to_num(fake), wk, [0], [1], "spin", "spin", spec.meta)
        mods = grid.integrate(Tl, lg, mu, rv, dA, sigma_kms=sigma_lsf)
        x2 += chi2_multi(sp.fl, sp.w, model_on_pixels(grid, mods, sp, gamma, mh_fine, lsf_apply), B)
    return x2


def mh_grid(grid):
    return np.round(np.arange(-1.0, grid.MHS[-1] + 1e-6, MH_STEP), 3)


def interval(x2, mh, npix, npar):
    scale = x2.min() / max(npix - npar, 1); prof = (x2 - x2.min()) / scale; ok = mh[prof < 1]
    return (ok.min() if len(ok) else np.nan), (ok.max() if len(ok) else np.nan), scale
