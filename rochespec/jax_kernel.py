"""JAX version of the kernel integrator (spice_backend.KernelIntegrator): scatter-add of the element weights onto the
([M/H], Teff, log g, mu, velocity) kernel with .at[].add, then one batched FFT product over all grid nodes. jit-compilable and
differentiable with respect to the element arrays (Teff, log g, [M/H], mu, velocity, area), hence through any JAX surface model
(SPICE's own meshes) to its parameters. PHOEBE contact meshes are built outside JAX, so gradients stop at the mesh there.

The node FFTs are precomputed for the whole grid (or a wavelength window of it): memory 2 x n_nodes x (N/2 + 1) complex64.
"""
import numpy as np
import jax, jax.numpy as jnp
from scipy.fft import next_fast_len

C = 299792.458


def _lin(nodes, x):
    x = jnp.clip(x, nodes[0], nodes[-1]); j = jnp.clip(jnp.searchsorted(nodes, x) - 1, 0, nodes.shape[0] - 2)
    return j, (x - nodes[j]) / (nodes[j + 1] - nodes[j])


class JaxKernel:
    def __init__(self, grid, vmax=600.0, lsf_R=None, wl_range=None):
        k = np.ones(grid.I.shape[-1], bool) if wl_range is None else (grid.wl > wl_range[0]) & (grid.wl < wl_range[1])
        self.wl = grid.wl[k]; self.dv = grid.dv; self.K = int(vmax / grid.dv)
        self.nodes = tuple(jnp.asarray(n, jnp.float64) for n in (grid.mhs, grid.teffs, grid.loggs, grid.mu))
        self.shape = tuple(len(n) for n in self.nodes) + (2 * self.K + 1,)
        if lsf_R:
            sig = C / lsf_R / 2.3548 / grid.dv; x = np.arange(-int(5 * sig), int(5 * sig) + 1); g = np.exp(-0.5 * (x / sig) ** 2); g /= g.sum()
        else:
            g = np.ones(1)
        self.n = int(k.sum()); L = 2 * self.K + len(g); self.N = next_fast_len(self.n + L - 1); self.off = self.K + (len(g) - 1) // 2
        self.lsf_f = jnp.asarray(np.fft.rfft(g, self.N))
        nn = int(np.prod(self.shape[:-1]))
        I = np.stack([grid.I[..., k].reshape(nn, -1), grid.Ic[..., k].reshape(nn, -1)], 1)          # (nodes, 2, n)
        self.If = jnp.asarray(np.fft.rfft(I.astype(np.float64), self.N, axis=-1).astype(np.complex64))
        self.spectrum = jax.jit(self._spectrum)

    def weights(self, T, logg, mh, mu, v, w):
        nM, nT, nG, nK, nV = self.shape
        jm, fm = _lin(self.nodes[0], mh); jt, ft = _lin(self.nodes[1], T); jg, fg = _lin(self.nodes[2], logg); jk, fk = _lin(self.nodes[3], mu)
        x = jnp.clip(v / self.dv, -self.K + 1, self.K - 2); jv = jnp.floor(x); fv = x - jv; jv = jv.astype(jnp.int32) + self.K
        W = jnp.zeros(int(np.prod(self.shape)))
        for a, wa in ((0, 1 - fm), (1, fm)):
            for b, wb in ((0, 1 - ft), (1, ft)):
                for c, wc in ((0, 1 - fg), (1, fg)):
                    for d, wd in ((0, 1 - fk), (1, fk)):
                        for e, we in ((0, 1 - fv), (1, fv)):
                            idx = ((((jm + a) * nT + jt + b) * nG + jg + c) * nK + jk + d) * nV + jv + e
                            W = W.at[idx].add(w * wa * wb * wc * wd * we)
        return W.reshape(self.shape)

    def _spectrum(self, T, logg, mh, mu, v, w):
        """F, Fc on self.wl (LSF folded in when lsf_R was given)"""
        W = self.weights(T, logg, mh, mu, v, w).reshape(-1, self.shape[-1])
        Kf = jnp.fft.rfft(W, self.N, axis=-1) * self.lsf_f
        acc = jnp.einsum("nk,nck->ck", Kf, self.If)
        out = jnp.fft.irfft(acc, self.N, axis=-1)[:, self.off:self.off + self.n]
        return out[0], out[1]


def elements_jax(m, index=None, doppler="spice"):
    """element arrays of one SPICE model as JAX arrays (differentiable); invisible elements keep zero weight"""
    ix = dict(teff=0, logg=1, mh=2, **(index or {}))
    w = m.visible_cast_areas * (1.0 if type(m).__name__ == "PhoebeModel" else m.radius ** 2)
    mu = jnp.where(m.mus > 0, m.mus, 0.0); w = jnp.where(mu > 0, w, 0.0)
    v = m.los_velocities; vk = -C * jnp.log1p(v / C) if doppler == "spice" else v
    p = m.parameters
    return p[:, ix["teff"]], p[:, ix["logg"]], p[:, ix["mh"]], mu, vk, w
