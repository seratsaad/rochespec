"""Roche-geometry contact binary: surface mesh, visibility and disk integration of I(lambda, mu).

Units: separation a = 1, primary (mass M1) at the origin, secondary at (1, 0, 0), q = M2/M1.
Modified Roche potential (Kopal): Omega = 1/r1 + q (1/r2 - x) + (1+q)/2 (x^2 + y^2).
"""
import os
import numpy as np
from scipy.optimize import brentq
from scipy.signal import fftconvolve

C = 299792.458
GSUN = 2.7398e4          # G Msun / Rsun^2 in cgs


def omega(x, y, z, q):
    r1 = np.sqrt(x * x + y * y + z * z); r2 = np.sqrt((x - 1) ** 2 + y * y + z * z)
    return 1 / r1 + q * (1 / r2 - x) + 0.5 * (1 + q) * (x * x + y * y)


def grad_omega(x, y, z, q):
    r1 = np.sqrt(x * x + y * y + z * z); r2 = np.sqrt((x - 1) ** 2 + y * y + z * z)
    gx = -x / r1 ** 3 - q * (x - 1) / r2 ** 3 - q + (1 + q) * x
    gy = -y / r1 ** 3 - q * y / r2 ** 3 + (1 + q) * y
    gz = -z / r1 ** 3 - q * z / r2 ** 3
    return gx, gy, gz


def lagrange(q):
    dx = lambda x: grad_omega(np.array(x), 0.0, 0.0, q)[0]
    xl1 = brentq(dx, 0.05, 0.95); xl2 = brentq(dx, 1.01, 2.5); xl3 = brentq(dx, -2.5, -0.01)
    return xl1, xl2, xl3


class ContactBinary:
    def __init__(self, q, fillout, ntheta=90, nphi=180, outer=None):
        self.q, self.f = q, fillout
        self.xl1, self.xl2, self.xl3 = lagrange(q)
        o_in = omega(self.xl1, 0, 0, q)
        # Outer critical potential for the fill-out factor. The standard definition (PHOEBE, Wilson-Devinney) uses the first outer
        # critical surface reached, max(Omega_L2, Omega_L3) (L2, behind the less massive star). The original code used the min
        # (L3), which makes the surface larger than the nominal fill-out and can open it past L2 (catalog PIPELINE_LOG, SPICE check).
        # Default is now "L2" (production refit 2026-10-29); CB_OUTER=min (or outer="min") reproduces the old catalogue.
        outer = outer or os.environ.get("CB_OUTER", "L2")
        o2, o3 = omega(self.xl2, 0, 0, q), omega(self.xl3, 0, 0, q)
        o_out = max(o2, o3) if outer == "L2" else min(o2, o3)
        self.omega0 = o_in - fillout * (o_in - o_out)
        self._mesh(ntheta, nphi)

    def _mesh(self, nth, nph):
        q, O0 = self.q, self.omega0
        th = (np.arange(nth) + 0.5) * np.pi / nth; ph = (np.arange(nph) + 0.5) * 2 * np.pi / nph
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        dw = (np.sin(TH) * (np.pi / nth) * (2 * np.pi / nph)).ravel()
        d = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1).reshape(-1, 3)
        pts, area, comp = [], [], []
        for k, cx in enumerate([0.0, 1.0]):
            dd = d if k == 0 else d * np.array([-1, 1, 1])      # secondary rays mirrored so +x faces outward
            t = np.linspace(0.02, 1.0, 400)
            X = cx + t[None, :] * dd[:, :1]; Y = t[None, :] * dd[:, 1:2]; Z = t[None, :] * dd[:, 2:3]
            O = omega(X, Y, Z, q)
            inside = O > O0
            side = (X < self.xl1) if k == 0 else (X > self.xl1)
            ok_lobe = inside | ~side
            j = np.argmax(~inside, axis=1)                         # first sample outside the surface
            has = (~inside).any(axis=1) & (j > 0)
            # reject rays that leave the star's own half before reaching the surface (that part is the neck)
            cross = np.argmax(~side, axis=1); crossed = (~side).any(axis=1) & (cross <= j)
            use = has & ~crossed
            tt = np.empty(use.sum())
            for n, i in enumerate(np.where(use)[0]):
                a_, b_ = t[j[i] - 1], t[j[i]]
                g = lambda s: omega(cx + s * dd[i, 0], s * dd[i, 1], s * dd[i, 2], q) - O0
                tt[n] = brentq(g, a_, b_, xtol=1e-7)
            p = np.array([cx, 0, 0]) + tt[:, None] * dd[use]
            gx, gy, gz = grad_omega(p[:, 0], p[:, 1], p[:, 2], q)
            nrm = -np.stack([gx, gy, gz], -1); nrm /= np.linalg.norm(nrm, axis=1)[:, None]
            cosang = np.abs(np.sum(dd[use] * nrm, axis=1))
            pts.append(p); area.append(tt ** 2 * dw[use] / np.maximum(cosang, 0.05)); comp.append(np.full(use.sum(), k))
        self.p = np.concatenate(pts); self.dA = np.concatenate(area); self.comp = np.concatenate(comp)
        gx, gy, gz = grad_omega(self.p[:, 0], self.p[:, 1], self.p[:, 2], q)
        g = np.stack([gx, gy, gz], -1); self.gmag = np.linalg.norm(g, axis=1); self.n = -g / self.gmag[:, None]

    def surface(self, M1, a_rsun, Teff, beta=0.08):
        """local log g (cgs) and T with gravity darkening T ~ g^beta, scaled so the area-weighted T^4 gives Teff"""
        self.logg = np.log10(GSUN * M1 / a_rsun ** 2 * self.gmag)
        Tl = (self.gmag / np.median(self.gmag)) ** beta
        self.T = Teff * Tl / (np.sum(self.dA * Tl ** 4) / self.dA.sum()) ** 0.25
        return self.logg, self.T

    def view(self, phase, incl_deg, nstep=150):
        """line-of-sight unit vector, mu, and visibility (self-eclipse by ray marching through the envelope)"""
        i = np.radians(incl_deg); w = 2 * np.pi * phase
        los = np.array([np.sin(i) * np.cos(w), -np.sin(i) * np.sin(w), np.cos(i)])   # toward the observer
        mu = self.n @ los
        vis = mu > 0
        idx = np.where(vis)[0]
        s = np.linspace(0.01, 2.2, nstep)
        P = self.p[idx, None, :] + s[None, :, None] * los[None, None, :]
        box = (P[..., 0] > self.xl3) & (P[..., 0] < self.xl2) & (np.abs(P[..., 1]) < 0.75) & (np.abs(P[..., 2]) < 0.75)
        blocked = ((omega(P[..., 0], P[..., 1], P[..., 2], self.q) > self.omega0 + 1e-6) & box).any(axis=1)
        vis[idx[blocked]] = False
        return los, mu, vis

    def rv(self, los, a_rsun, P_day):
        """radial velocity (km/s, positive = receding) of each surface element in synchronous rotation"""
        wa = 2 * np.pi * a_rsun * 695700.0 / (P_day * 86400.0)
        xcm = self.q / (1 + self.q)
        vx = -self.p[:, 1]; vy = self.p[:, 0] - xcm
        return -wa * (vx * los[0] + vy * los[1])


class IntensityGrid:
    """Korg I(lambda, mu) on (Teff, logg, [M/H]) nodes, resampled onto a uniform ln(lambda) grid with step dv"""
    def __init__(self, path, teffs, loggs, mhs, dv=1.0, rebin=False):
        wl = np.fromfile(f"{path}/wl.bin", dtype=np.float64)  # Korg wavelengths in vacuum Angstrom
        mu = np.loadtxt(f"{path}/mu.txt"); self.mu = mu[:, 0]; self.wmu = mu[:, 1]
        nmu, nwl = len(self.mu), len(wl)
        self.lnw = np.arange(np.log(wl[0]) + 0.002, np.log(wl[-1]) - 0.002, dv / C); self.wl = np.exp(self.lnw)
        if rebin:
            # flux-conserving bin averages on the coarse grid (keeps equivalent widths when dv >> raw step)
            edges = np.exp(np.r_[self.lnw - 0.5 * dv / C, self.lnw[-1] + 0.5 * dv / C])
            def resample(y):
                cum = np.r_[0.0, np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(wl))]
                return np.diff(np.interp(edges, wl, cum)) / np.diff(edges)
        else:
            resample = lambda y: np.interp(self.lnw, np.log(wl), y)
        self.teffs, self.loggs, self.mhs, self.dv = np.array(teffs), np.array(loggs), np.array(mhs), dv
        self.I = np.zeros((len(mhs), len(teffs), len(loggs), nmu, len(self.lnw)), np.float32); self.Ic = np.zeros_like(self.I)
        for a, m in enumerate(mhs):
            for b, t in enumerate(teffs):
                for c, g in enumerate(loggs):
                    raw = np.fromfile(f"{path}/I_T{int(t)}_g{g}_m{m}.bin", dtype=np.float32)
                    # Julia writes column-major (nmu, nwl) -> numpy (nwl, nmu)
                    I = raw[: nmu * nwl].reshape(nwl, nmu).T; Ic = raw[nmu * nwl:].reshape(nwl, nmu).T
                    for k in range(nmu):
                        self.I[a, b, c, k] = resample(I[k])
                        self.Ic[a, b, c, k] = resample(Ic[k])

    @staticmethod
    def _lin(nodes, x):
        x = np.clip(x, nodes[0], nodes[-1]); j = np.clip(np.searchsorted(nodes, x) - 1, 0, len(nodes) - 2)
        w = (x - nodes[j]) / (nodes[j + 1] - nodes[j]); return j, w

    def integrate_nodes(self, T, logg, mu, v, dA, vmax=600, lsf_R=1800.0):
        """F, Fc for every [M/H] node at once: arrays (n_mh, n_lnw); elements from several views may be stacked"""
        K = int(vmax / self.dv); nT, ng, nm = len(self.teffs), len(self.loggs), len(self.mu)
        W = np.zeros((nT, ng, nm, 2 * K + 1))
        jt, wt = self._lin(self.teffs, T); jg, wg = self._lin(self.loggs, logg); jm, wm = self._lin(self.mu, mu)
        x = np.clip(v / self.dv, -K + 1, K - 2); jv = np.floor(x).astype(int); wv = x - jv
        base = dA * mu
        for a_, fa in ((0, 1 - wt), (1, wt)):
            for b_, fb in ((0, 1 - wg), (1, wg)):
                for c_, fc in ((0, 1 - wm), (1, wm)):
                    for d_, fd in ((0, 1 - wv), (1, wv)):
                        np.add.at(W, (jt + a_, jg + b_, jm + c_, jv + d_ + K), base * fa * fb * fc * fd)
        sig = C / lsf_R / 2.3548 / self.dv
        xk = np.arange(-int(5 * sig), int(5 * sig) + 1); g = np.exp(-0.5 * (xk / sig) ** 2); g /= g.sum()
        F = np.zeros((len(self.mhs), len(self.lnw))); Fc = np.zeros_like(F)
        for a_ in range(nT):
            for b_ in range(ng):
                for c_ in range(nm):
                    k = W[a_, b_, c_]
                    if k.sum() == 0: continue
                    k = np.convolve(k, g)                    # rotation kernel convolved with the LSF
                    for m in range(len(self.mhs)):
                        F[m] += fftconvolve(self.I[m, a_, b_, c_], k, mode="same")
                        Fc[m] += fftconvolve(self.Ic[m, a_, b_, c_], k, mode="same")
        return F, Fc

    def integrate(self, T, logg, mu, v, dA, mh, vmax=500, lsf_R=22500.0):
        """disk-integrated F(lambda) and continuum Fc(lambda) for visible elements"""
        K = int(vmax / self.dv); nT, ng, nm = len(self.teffs), len(self.loggs), len(self.mu)
        W = np.zeros((nT, ng, nm, 2 * K + 1))
        jt, wt = self._lin(self.teffs, T); jg, wg = self._lin(self.loggs, logg); jm, wm = self._lin(self.mu, mu)
        x = v / self.dv; jv = np.floor(x).astype(int); wv = x - jv
        base = dA * mu
        for a_, fa in ((0, 1 - wt), (1, wt)):
            for b_, fb in ((0, 1 - wg), (1, wg)):
                for c_, fc in ((0, 1 - wm), (1, wm)):
                    for d_, fd in ((0, 1 - wv), (1, wv)):
                        np.add.at(W, (jt + a_, jg + b_, jm + c_, jv + d_ + K), base * fa * fb * fc * fd)
        jmh, wmh = self._lin(self.mhs, np.array(mh))
        I = (1 - wmh) * self.I[jmh] + wmh * self.I[jmh + 1]; Ic = (1 - wmh) * self.Ic[jmh] + wmh * self.Ic[jmh + 1]
        F = np.zeros(len(self.lnw)); Fc = np.zeros(len(self.lnw))
        for a_ in range(nT):
            for b_ in range(ng):
                for c_ in range(nm):
                    k = W[a_, b_, c_]
                    if k.sum() == 0: continue
                    F += fftconvolve(I[a_, b_, c_], k, mode="same"); Fc += fftconvolve(Ic[a_, b_, c_], k, mode="same")
        sig = C / lsf_R / 2.3548 / self.dv
        xk = np.arange(-int(5 * sig), int(5 * sig) + 1); g = np.exp(-0.5 * (xk / sig) ** 2); g /= g.sum()
        return fftconvolve(F, g, mode="same"), fftconvolve(Fc, g, mode="same")


class PZGrid(IntensityGrid):
    """payne-zero I(lambda, mu) grid (../pz/grid/<window>/I_T*_g*_m*.npz), same interface as IntensityGrid.
    windows: list of chunk names concatenated in wavelength (e.g. ["opt1", "opt2", "opt3"] or ["H"])."""
    def __init__(self, path, windows, teffs, loggs, mhs, dv=1.0, rebin=False):
        self.teffs, self.loggs, self.mhs, self.dv = np.array(teffs), np.array(loggs), np.array(mhs), dv
        first = [np.load(f"{path}/{w}/I_T{int(teffs[0])}_g{loggs[0]}_m{mhs[0]}.npz") for w in windows]
        wl = np.concatenate([f["wl_nm"] * 10 for f in first]); order = np.argsort(wl); wl = wl[order]
        keep = np.r_[True, np.diff(wl) > 0]; wl = wl[keep]
        self.mu = first[0]["mu"]; self.wmu = first[0]["wmu"]; nmu = len(self.mu)
        self.lnw = np.arange(np.log(wl[0]) + 0.002, np.log(wl[-1]) - 0.002, dv / C); self.wl = np.exp(self.lnw)
        if rebin:
            edges = np.exp(np.r_[self.lnw - 0.5 * dv / C, self.lnw[-1] + 0.5 * dv / C])
            def resample(y):
                cum = np.r_[0.0, np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(wl))]
                return np.diff(np.interp(edges, wl, cum)) / np.diff(edges)
        else:
            resample = lambda y: np.interp(self.lnw, np.log(wl), y)
        self.I = np.zeros((len(mhs), len(teffs), len(loggs), nmu, len(self.lnw)), np.float32); self.Ic = np.zeros_like(self.I)
        for a, m in enumerate(mhs):
            for b, t in enumerate(teffs):
                for c, g in enumerate(loggs):
                    fs = [np.load(f"{path}/{w}/I_T{int(t)}_g{g}_m{m}.npz") for w in windows]
                    I = np.concatenate([f["I"] for f in fs], axis=1)[:, order][:, keep]
                    Ic = np.concatenate([f["Ic"] for f in fs], axis=1)[:, order][:, keep]
                    for k in range(nmu):
                        self.I[a, b, c, k] = resample(I[k]); self.Ic[a, b, c, k] = resample(Ic[k])
