"""Joint (Teff, [M/H]) inference on a chi^2 grid interpolated in Teff (cubic spline onto 10 K steps) before profiling.
The production joint() profiles [M/H] over the 100 K Teff rows, so a true Teff between rows gives a [M/H] error of up to
k x 50 K (about 0.04-0.06 dex) through the Teff-[M/H] degeneracy. No extra model evaluations are needed."""
import numpy as np
from scipy.interpolate import CubicSpline


def make(J0, step=10.0):
    def joint_fine(X2, teff, mh, npix, Tp0, dTdf, sigT, npar=30, kappa=1.0, **kw):
        teff = np.asarray(teff, float)
        if len(teff) < 4 or not np.isfinite(X2).all(): return J0(X2, teff, mh, npix, Tp0, dTdf, sigT, npar=npar, kappa=kappa, **kw)
        Tf = np.arange(teff[0], teff[-1] + 1e-6, step); X2f = np.maximum(CubicSpline(teff, X2, axis=0)(Tf), X2.min() * 0.5)
        return J0(X2f, Tf, mh, npix, Tp0, dTdf, sigT, npar=npar, kappa=kappa, **kw)
    return joint_fine


def finer_teff_grid(FL, step=50):
    """use a finer Teff grid of model evaluations in the fitter (production: 100 K)"""
    tg0 = FL.teff_grid_for
    FL.teff_grid_for = lambda Tp, sig, lo=4800, hi=6700, step_=step: tg0(Tp, sig, lo=lo, hi=hi, step=step_)
