"""PHOEBE contact-binary meshes as the geometry of the fitters (CB_GEOM=phoebe), a drop-in for roche.ContactBinary.

The mesh is built once per (q, f, i) in units of the separation a, at a reference Teff (5800 K) and period (0.35 d), with the fitters'
physical scale (M1 = (Teff/5772)^1.8, Kepler separation, T^4 ~ g^0.32 per component, no irradiation; standard L2 fill-out). Every
Teff row and every star reuses it: log g is shifted by log10(M1/M1_ref) - 2 log10(A/A_ref), the temperature pattern is scaled so the
area-weighted T^4 over the whole surface equals Teff (as roche.ContactBinary.surface), and line-of-sight velocities scale as a/P.
Per phase the mesh gives mu, the visible fraction of each triangle (horizon and eclipses, from PHOEBE) and the velocity.

Two modes per (q, f, i):
- exact: PHOEBE run at the requested phases (prepare(phases, incl)), used when the ephemeris fixes the phases;
- library: N phases (CB_PHOEBE_NPH, default 60) on a uniform grid, linear interpolation between neighbours (velocity error
  < 0.002 of its amplitude), cached on disk in $ROCHESPEC_DATA/phoebe_lib/, shared by all stars with that geometry.
Interface as roche.ContactBinary: surface(M1, a, Teff) -> (logg, T) for all triangles; view(phase, incl) -> (key, mu, visible mask)
and sets self.dA to area x visible fraction for that phase; rv(key, a, P) -> velocity of every triangle (km/s, positive receding).
"""
import os, time
import numpy as np

from .config import DATA
LIB = DATA + "phoebe_lib/"
TREF, PREF = 5800.0, 0.35
NLIB = int(os.environ.get("CB_PHOEBE_NPH", 60))
_phoebe = None
STATUS = {"fallback": 0, "retries": 0}      # per-process counters; the drivers read and reset them per star


def _ph():
    global _phoebe
    if _phoebe is None:
        import phoebe
        phoebe.logger(clevel="ERROR"); _phoebe = phoebe
    return _phoebe


def ntri_for(q):
    return 8000 if min(q, 1 / q) < 0.2 else 3000


def kepler_a(M1, q, P):
    return (M1 * (1 + q) * (P / 365.25) ** 2) ** (1 / 3) * 215.03


def compute(q, f, incl, phases, ntri=None):
    """PHOEBE meshes at the given phases: static (area, T_rel, logg_ref) and per-phase (mu, vis, v) arrays, both halves concatenated"""
    phoebe = _ph(); ntri = ntri or ntri_for(q)
    b = phoebe.default_contact_binary()
    b.flip_constraint("pot", component="contact_envelope", solve_for="requiv@primary")
    b.flip_constraint("fillout_factor", component="contact_envelope", solve_for="pot")
    M1 = (TREF / 5772.0) ** 1.8; A = kepler_a(M1, q, PREF)
    b.set_value("q", context="component", value=q); b.set_value("period", component="binary", value=PREF)
    b.set_value("sma", component="binary", value=A); b.set_value("incl", component="binary", value=incl)
    b.set_value("fillout_factor", context="component", component="contact_envelope", value=f)
    for c in ["primary", "secondary"]:
        b.set_value("teff", component=c, value=TREF); b.set_value("gravb_bol", component=c, value=0.32)
    b.set_value_all("irrad_method", "none"); b.set_value("ntriangles", component="contact_envelope", value=ntri)
    times = np.asarray(phases, float) * PREF
    b.add_dataset("mesh", compute_times=times, dataset="cbmesh", columns=["teffs", "loggs", "mus", "visibilities", "areas", "vws"])
    b.run_compute(irrad_method="none", progressbar=False)
    out = {k: [] for k in ["mu", "vis", "v"]}
    M = b.filter(dataset="cbmesh", context="model")
    def arr(qual, t):
        return np.concatenate([M.get_value(qualifier=qual, component=c, time=t) for c in ("primary", "secondary")])
    area = arr("areas", times[0]); teff = arr("teffs", times[0]); logg = arr("loggs", times[0])
    for t in times:
        out["mu"].append(arr("mus", t)); out["vis"].append(arr("visibilities", t)); out["v"].append(-arr("vws", t))     # PHOEBE w points toward the observer: roche/fitter convention is positive = receding
    T_rel = teff / (np.sum(area * teff ** 4) / area.sum()) ** 0.25
    return dict(area=area / A ** 2, T_rel=T_rel, logg_ref=logg, M1_ref=M1, A_ref=A,
                mu=np.array(out["mu"]), vis=np.array(out["vis"]), v=np.array(out["v"]), phases=np.asarray(phases, float))


def compute_robust(q, f, incl, phases):
    """PHOEBE occasionally fails to mesh or project ('Mesh failed or incomplete', 'Projections are failing'); retry with 2x and 3x
    the triangles before giving up"""
    n0 = ntri_for(q); err = None
    for k, m in enumerate((1, 2, 3)):
        try:
            d = compute(q, f, incl, phases, ntri=n0 * m); STATUS["retries"] += k; return d
        except Exception as e:
            err = e
    raise RuntimeError(f"PHOEBE mesh failed for q={q} f={f} i={incl}: {err!r}")


def library(q, f, incl, n=NLIB):
    fn = LIB + f"q{q:.5f}_f{f:.5f}_i{incl:.4f}_n{n}.npz"
    if os.path.exists(fn):
        return dict(np.load(fn))
    d = compute_robust(q, f, incl, np.arange(n) / n)
    os.makedirs(LIB, exist_ok=True); tmp = fn + f".{os.getpid()}.tmp.npz"; np.savez(tmp, **d); os.replace(tmp, fn)
    return d


class PhoebeContact:
    def __init__(self, q, fillout, ntheta=None, nphi=None, outer=None, incl=None):
        self.q, self.f = q, fillout; self.incl0 = incl; self._lib = {}; self._exact = {}; self._static = None; self.n_phoebe_runs = 0

    def _set_static(self, d):
        if self._static is None:
            self._static = d; self.area = d["area"]; self.dA = d["area"].copy()

    def _fallback(self, err):
        """if PHOEBE cannot mesh this geometry, use the ray-traced roche.py surface for the rest of this star (flagged in STATUS)"""
        from . import roche
        fb = object.__new__(roche.ContactBinary); roche.ContactBinary.__init__(fb, self.q, self.f, ntheta=60, nphi=120)
        self._fb = fb; STATUS["fallback"] = 1; self.dA = fb.dA

    def prepare(self, phases, incl, exact=True):
        """build the meshes needed for these phases: exact phases (a few dozen at most) or the shared library"""
        if getattr(self, "_fb", None) is not None: return
        ph = np.unique(np.round(np.mod(np.asarray(phases, float), 1.0), 9))
        try:
            self._prepare(ph, incl, exact)
        except RuntimeError as e:
            self._fallback(e)

    def _prepare(self, ph, incl, exact):
        if exact and len(ph) <= 40:
            d = compute_robust(self.q, self.f, incl, ph); self.n_phoebe_runs += 1; self._set_static(d)
            for k, p in enumerate(ph):
                self._exact[(round(float(p), 9), float(incl))] = (d["mu"][k], d["vis"][k], d["v"][k])
        else:
            self._get_lib(incl)

    def _get_lib(self, incl):
        if incl not in self._lib:
            self._lib[incl] = library(self.q, self.f, incl); self._set_static(self._lib[incl])
        return self._lib[incl]

    def surface(self, M1, a_rsun, Teff, beta=None):
        if getattr(self, "_fb", None) is not None: return self._fb.surface(M1, a_rsun, Teff)
        if self._static is None:
            self._get_lib(self.incl0 if self.incl0 is not None else 90.0)
        s = self._static
        self.logg = s["logg_ref"] + np.log10(M1 / s["M1_ref"]) - 2 * np.log10(a_rsun / s["A_ref"])
        self.T = Teff * s["T_rel"]
        return self.logg, self.T

    def view(self, phase, incl, nstep=None):
        if getattr(self, "_fb", None) is not None:
            r = self._fb.view(phase, incl); self.dA = self._fb.dA; return r
        incl = float(incl); key = (round(float(np.mod(phase, 1.0)), 9), incl)
        if key in self._exact:
            mu, vis, v = self._exact[key]
        else:
            d = self._get_lib(incl); n = len(d["phases"]); x = np.mod(phase, 1.0) * n; j = int(np.floor(x)) % n; w = x - np.floor(x); j2 = (j + 1) % n
            mu = (1 - w) * d["mu"][j] + w * d["mu"][j2]; vis = (1 - w) * d["vis"][j] + w * d["vis"][j2]; v = (1 - w) * d["v"][j] + w * d["v"][j2]
        self._v = v; self.dA = self.area * vis
        return key, mu, (vis > 0) & (mu > 0)

    def rv(self, key, a_rsun, P_day):
        if getattr(self, "_fb", None) is not None: return self._fb.rv(key, a_rsun, P_day)
        s = self._static
        return self._v * (a_rsun / s["A_ref"]) * (PREF / P_day)
