"""payne-zero I(lambda, mu) grid for the contact-binary model: solve atmosphere, synthesize intensities per window.

Usage: python pz_grid.py <task_index> <n_tasks> <outdir> [--windows H,opt] [--only teff,logg,mh]
Each task handles every n_tasks-th grid point. Atmospheres are kept so windows can be added later.
"""
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pz_intensity import synth_intensity
from payne_zero_atmosphere import solve_structured_atmosphere

TEFFS = [4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750]; LOGGS = [3.75, 4.1, 4.45, 4.8]
MHS = [0.0, -0.6, -0.3, 0.3, -1.0, 0.5]          # priority order: if the budget runs out, the last layer is missing
WINDOWS = {"H": (1510.0, 1700.0, 300000.0), "opt1": (380.0, 550.0, 200000.0), "opt2": (550.0, 720.0, 200000.0),
           "opt3": (720.0, 890.0, 200000.0), "optA": (500.0, 600.0, 300000.0)}
VMIC = 1.2

ap = argparse.ArgumentParser(); ap.add_argument("task", type=int); ap.add_argument("ntask", type=int); ap.add_argument("out")
ap.add_argument("--windows", default="H,opt1,opt2,opt3"); ap.add_argument("--only", default=None); a = ap.parse_args()
pts = [(t, g, m) for m in MHS for t in TEFFS for g in LOGGS]
if a.only: pts = [tuple(float(x) for x in a.only.split(","))]
wins = a.windows.split(",")
pts = [p for p in pts if not all(os.path.exists(os.path.join(a.out, w, f"I_T{int(p[0])}_g{p[1]}_m{p[2]}.npz")) for w in wins)]   # only missing models
print("models to compute:", len(pts), flush=True)
for k, (t, g, m) in enumerate(pts):
    if k % a.ntask != a.task: continue
    tag = f"T{int(t)}_g{g}_m{m}"; adir = os.path.join(a.out, "atm", tag); os.makedirs(adir, exist_ok=True)
    npz = os.path.join(adir, "payne_zero_structured_atmosphere.npz"); rec = dict(tag=tag, teff=t, logg=g, mh=m, vmic=VMIC)
    t0 = time.time()
    if not os.path.exists(npz):
        solve_structured_atmosphere(teff=t, logg=g, out_dir=adir, metallicity=m, microturbulence_km_s=VMIC)
    rec["atm_s"] = round(time.time() - t0, 1)
    for wname in wins:
        fn = os.path.join(a.out, wname, f"I_{tag}.npz")
        if os.path.exists(fn): continue
        os.makedirs(os.path.dirname(fn), exist_ok=True); t1 = time.time()
        w0, w1, Rw = WINDOWS[wname]; r = synth_intensity(npz, w0, w1, R=Rw, device="cpu", dtype="float32")
        np.savez(fn, I=r["I"].astype(np.float32), Ic=r["Ic"].astype(np.float32), H=r["H"].astype(np.float32), Hc=r["Hc"].astype(np.float32),
                 wl_nm=r["wl_nm"], mu=r["mu"], wmu=r["wmu"])
        rec[f"{wname}_s"] = round(time.time() - t1, 1); rec[f"{wname}_npix"] = int(len(r["wl_nm"]))
    print(json.dumps(rec), flush=True)
