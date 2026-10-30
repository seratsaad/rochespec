"""SPICE surface models with the rochespec kernel integrator.

SPICE (Jablonska et al. 2025) supplies the surface: PHOEBE contact meshes (spice.models.phoebe_model.PhoebeModel) or its own meshes
(spice.models.mesh_model.MeshModel), with spots, abundance spots and partial visibility. rochespec supplies the disk integral.
Instead of evaluating I(lambda, mu) for every visible element at every wavelength, each element's projected area is spread with
linear weights onto the intensity-grid nodes ([M/H], Teff, log g, mu) and onto velocity bins of width dv, giving a kernel
W[m, t, g, k, v]. The spectrum is then sum over occupied nodes of I_node convolved with W_node, done in Fourier space with the FFT of
every node cached, so the cost scales with the number of occupied nodes, not with elements x wavelengths.

The weights reproduce SPICE's simulate_observed_flux with a linear intensity interpolator on the same grid exactly: the intensity at
a shifted wavelength is linear in ln(lambda) between pixels, which is what the two-bin velocity split does, and Teff, log g, [M/H]
and mu are interpolated linearly in both. Conventions taken from SPICE:
- Doppler: SPICE evaluates I at log10(lambda) + log10(1 + v/c), so an element with SPICE line-of-sight velocity v is placed at the
  kernel velocity -c ln(1 + v/c).
- Areas: the weight is visible_cast_areas (area x mu x visibility). PHOEBE areas are physical already; SPICE's own meshes are in
  units of the stellar radius, so they get radius^2 (simulate_observed_flux multiplies every model by radius^2, which double counts
  for PHOEBE meshes; see spice/spice_cb.spectrum in the analysis scripts).
"""
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len

C = 299792.458


def _lin(nodes, x):
    x = np.clip(x, nodes[0], nodes[-1]); j = np.clip(np.searchsorted(nodes, x) - 1, 0, len(nodes) - 2)
    return j, (x - nodes[j]) / (nodes[j + 1] - nodes[j])


def lsf_kernel(R, dv):
    """Gaussian line-spread function sampled on the velocity grid (same truncation as the fitters)"""
    sig = C / R / 2.3548 / dv
    x = np.arange(-int(5 * sig), int(5 * sig) + 1); g = np.exp(-0.5 * (x / sig) ** 2)
    return g / g.sum()


def elements(models, index=None, doppler="spice"):
    """visible elements of a list of SPICE models -> dict of 1-D arrays T, logg, mh, mu, v (kernel velocity, km/s), w (area weight).
    index: positions of Teff, log g and [M/H] in the parameter vector (default 0, 1, 2)."""
    ix = dict(teff=0, logg=1, mh=2, **(index or {}))
    out = {k: [] for k in ["T", "logg", "mh", "mu", "v", "w"]}
    for m in models:
        w = np.asarray(m.visible_cast_areas, float)
        if type(m).__name__ != "PhoebeModel":
            w = w * float(m.radius) ** 2
        mu = np.asarray(m.mus, float); keep = (w > 0) & (mu > 0)
        p = np.asarray(m.parameters, float)
        v = np.asarray(m.los_velocities, float)
        vk = -C * np.log1p(v / C) if doppler == "spice" else v
        for k, x in [("T", p[:, ix["teff"]]), ("logg", p[:, ix["logg"]]), ("mh", p[:, ix["mh"]]), ("mu", mu), ("v", vk), ("w", w)]:
            out[k].append(x[keep])
    return {k: np.concatenate(x) for k, x in out.items()}


class KernelIntegrator:
    """grid: rochespec.roche.PZGrid / IntensityGrid (I, Ic of shape (n_mh, n_teff, n_logg, n_mu, n_lnw) on a uniform ln(lambda) grid
    with step dv). vmax: half-width of the velocity kernel (km/s). lsf_R: resolving power of a Gaussian LSF folded into the kernel
    (None for none)."""

    def __init__(self, grid, vmax=600.0, lsf_R=None):
        self.g = grid; self.dv = grid.dv; self.K = int(vmax / grid.dv)
        self.nodes = (grid.mhs, grid.teffs, grid.loggs, grid.mu)
        self.shape = tuple(len(n) for n in self.nodes) + (2 * self.K + 1,)
        self.lsf = lsf_kernel(lsf_R, grid.dv) if lsf_R else np.ones(1)
        self.L = 2 * self.K + len(self.lsf)                     # length of (velocity kernel conv LSF)
        self.n = grid.I.shape[-1]; self.N = next_fast_len(self.n + self.L - 1)
        self.off = self.K + (len(self.lsf) - 1) // 2            # 'same' alignment
        self.lsf_f = rfft(self.lsf, self.N)
        self._cache = {}

    # --- kernel weights -------------------------------------------------------------------------------------------------------
    def weights(self, T, logg, mh, mu, v, w):
        """W[m, t, g, k, v] (flattened 5-D histogram, 32 interpolation corners per element)"""
        nM, nT, nG, nK, nV = self.shape
        jm, fm = _lin(self.nodes[0], mh); jt, ft = _lin(self.nodes[1], T); jg, fg = _lin(self.nodes[2], logg); jk, fk = _lin(self.nodes[3], mu)
        x = np.clip(v / self.dv, -self.K + 1, self.K - 2); jv = np.floor(x).astype(int); fv = x - jv; jv = jv + self.K
        idx, val = [], []
        for a, wa in ((0, 1 - fm), (1, fm)):
            for b, wb in ((0, 1 - ft), (1, ft)):
                for c, wc in ((0, 1 - fg), (1, fg)):
                    for d, wd in ((0, 1 - fk), (1, fk)):
                        for e, we in ((0, 1 - fv), (1, fv)):
                            idx.append(((((jm + a) * nT + jt + b) * nG + jg + c) * nK + jk + d) * nV + jv + e)
                            val.append(w * wa * wb * wc * wd * we)
        return np.bincount(np.concatenate(idx), np.concatenate(val), minlength=int(np.prod(self.shape))).reshape(self.shape)

    # --- node FFTs (cached) -----------------------------------------------------------------------------------------------------
    def _node_fft(self, node):
        f = self._cache.get(node)
        if f is None:
            I = np.stack([self.g.I[node], self.g.Ic[node]]).astype(np.float64)
            f = rfft(I, self.N, axis=-1).astype(np.complex64); self._cache[node] = f
        return f

    def spectrum(self, W):
        """F, Fc (n_lnw,) from a 5-D kernel"""
        occ = np.argwhere(W.reshape(-1, W.shape[-1]).sum(-1) != 0)[:, 0]
        nodes = [tuple(int(i) for i in np.unravel_index(o, W.shape[:-1])) for o in occ]
        Kf = rfft(W.reshape(-1, W.shape[-1])[occ], self.N, axis=-1) * self.lsf_f
        acc = np.zeros((2, self.N // 2 + 1), np.complex128)
        for node, kf in zip(nodes, Kf):
            acc += self._node_fft(node) * kf
        out = irfft(acc, self.N, axis=-1)[:, self.off:self.off + self.n]
        return out[0], out[1]

    def spectrum_all_mh(self, T, logg, mu, v, w):
        """fitter path: one surface, F and Fc at every [M/H] node at once, arrays (n_mh, n_lnw); same as roche.integrate_nodes"""
        W = self.weights(T, logg, np.full(len(T), self.nodes[0][0]), mu, v, w)[0]       # [M/H] weight all on node 0
        flat = W.reshape(-1, W.shape[-1]); occ = np.argwhere(flat.sum(-1) != 0)[:, 0]
        Kf = rfft(flat[occ], self.N, axis=-1) * self.lsf_f
        acc = np.zeros((len(self.nodes[0]), 2, self.N // 2 + 1), np.complex128)
        for o, kf in zip(occ, Kf):
            t, g, k = np.unravel_index(o, W.shape[:-1])
            for m in range(len(self.nodes[0])):
                acc[m] += self._node_fft((m, int(t), int(g), int(k))) * kf
        out = irfft(acc, self.N, axis=-1)[..., self.off:self.off + self.n]
        return out[:, 0], out[:, 1]

    def warm_all(self):
        """fill the node-FFT cache for every grid node (do this before forking workers so they share it)"""
        for node in np.ndindex(*self.shape[:-1]):
            self._node_fft(node)

    def from_spice(self, models, index=None):
        e = elements(models, index)
        return self.spectrum(self.weights(e["T"], e["logg"], e["mh"], e["mu"], e["v"], e["w"]))
