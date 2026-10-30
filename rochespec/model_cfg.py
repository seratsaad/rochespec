"""Spectral-model switch for the contact-binary fitters: CB_MODEL=pz (payne-zero, default in rochespec) or CB_MODEL=korg (Korg grids, not distributed).

payne-zero grid (2026-10-24 extension, cbpz-ext): log g 4.8 is complete; the [M/H] = +0.5 layer is missing 17 of 36
models (the Pitzer job was cancelled after 31 min), so the largest complete regular grid is
9 Teff x 4 log g x 5 [M/H] (-1.0 .. +0.3) = 180 models per window. One node (Teff 6750, log g 3.75, [M/H] +0.3) is NaN
in every window and is filled by linear extrapolation in log g from the 4.1 and 4.45 nodes.
"""
from .config import DATA
import os
import numpy as np
from .roche import IntensityGrid, PZGrid

MODEL = os.environ.get("CB_MODEL", "pz")
D = DATA + ""
TEFFS = [4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750]
MHS = sorted([-1.0, -0.6, -0.3, 0.0, 0.3])
LOGGS = [3.75, 4.1, 4.45, 4.8] if MODEL == "pz" else [3.6, 3.9, 4.2, 4.5]
SUFFIX = "_pz" if MODEL == "pz" else ""


def _patch_nan_nodes(g):
    """fill NaN (Teff, log g, [M/H]) nodes by linear extrapolation/interpolation in log g"""
    for a in range(len(g.mhs)):
        for b in range(len(g.teffs)):
            for c in range(len(g.loggs)):
                if np.isfinite(g.I[a, b, c]).all() and np.isfinite(g.Ic[a, b, c]).all(): continue
                ok = [k for k in range(len(g.loggs)) if k != c and np.isfinite(g.I[a, b, k]).all()]
                k1, k2 = sorted(ok, key=lambda k: abs(k - c))[:2]
                w = (g.loggs[c] - g.loggs[k1]) / (g.loggs[k2] - g.loggs[k1])
                for arr in (g.I, g.Ic): arr[a, b, c] = (1 - w) * arr[a, b, k1] + w * arr[a, b, k2]
    return g


def load(band, dv, rebin=False):
    """band: 'H' (APOGEE) or 'opt' (LAMOST)"""
    if MODEL == "pz":
        win = ["H"] if band == "H" else ["opt1", "opt2", "opt3"]
        with np.errstate(invalid="ignore"):
            return _patch_nan_nodes(PZGrid(D + "pz/grid", win, TEFFS, LOGGS, MHS, dv=dv, rebin=rebin))
    return IntensityGrid(D + ("cbsynth/grid" if band == "H" else "cbsynth/grid_opt"), TEFFS, LOGGS, MHS, dv=dv, rebin=rebin)
