"""Command line: rochespec fit FILE ... | rochespec download-grid | rochespec info"""
import argparse, json, sys
import numpy as np


def _fit(a):
    from . import io, fit
    kw = dict(ra=a.ra, dec=a.dec)
    spectra = []
    if a.survey == "galah":
        spectra = io.read_galah(a.files, **kw)
    elif a.survey == "desi":
        if a.targetid is None: sys.exit("--targetid is needed for DESI coadd files")
        for f in a.files: spectra += io.read_desi(f, a.targetid, **kw)
    elif a.survey in io.READERS:
        for f in a.files: spectra += io.READERS[a.survey](f, **kw)
    else:
        sys.exit("--survey must be one of " + ", ".join(io.READERS))
    if not spectra: sys.exit("no usable spectra in the input files")
    prior = dict(teff_prior=(a.teff, a.teff_err)) if a.teff is not None else dict(bp_rp=a.bp_rp, ebv=a.ebv, ebv_err=a.ebv_err)
    res = fit(spectra, survey=a.survey, period=a.period, t0=a.t0, q=a.q, fillout=a.fillout, incl=a.incl, geometry=a.geometry, models=bool(a.plot), **prior)
    print(res.summary())
    d = res.as_dict()
    if a.json:
        with open(a.json, "w") as f: json.dump(d, f, indent=1, default=lambda x: x if not isinstance(x, np.generic) else x.item())
    if a.csv:
        import csv
        new = not __import__("os").path.exists(a.csv)
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file"] + list(d)); w.writeheader() if new else None
            w.writerow({"file": ";".join(a.files), **{k: (";".join(v) if isinstance(v, list) else v) for k, v in d.items()}})
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(res.model), 1, figsize=(8, 2.6 * len(res.model)), squeeze=False)
        for i, ax in enumerate(axes[:, 0]): res.plot(ax, i)
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser(prog="rochespec", description="Roche-model metallicities for contact binaries")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit the spectra of one star")
    f.add_argument("files", nargs="+", help="spectrum file(s) of one star (GALAH: the CCD files of one epoch)")
    f.add_argument("--survey", required=True, choices=["lamost_lrs", "lamost_mrs", "galah", "desi", "apogee"])
    f.add_argument("--period", type=float, required=True, help="orbital period (d)")
    f.add_argument("--t0", type=float, help="time of primary minimum (BJD_TDB); omit for a free phase (LRS, APOGEE)")
    f.add_argument("--q", type=float); f.add_argument("--fillout", type=float); f.add_argument("--incl", type=float)
    f.add_argument("--bp-rp", type=float, help="Gaia BP-RP for the colour Teff prior"); f.add_argument("--ebv", type=float, default=0.0)
    f.add_argument("--ebv-err", type=float); f.add_argument("--teff", type=float, help="Teff prior centre (K) instead of --bp-rp")
    f.add_argument("--teff-err", type=float, default=150.0); f.add_argument("--ra", type=float); f.add_argument("--dec", type=float)
    f.add_argument("--targetid", type=int, help="DESI TARGETID"); f.add_argument("--geometry", default="phoebe", choices=["phoebe", "roche"])
    f.add_argument("--json", help="write the result as JSON"); f.add_argument("--csv", help="append the result to a CSV table")
    f.add_argument("--plot", help="save a plot of data and model (PNG/PDF)")
    d = sub.add_parser("download-grid", help="download the payne-zero intensity grid into $ROCHESPEC_DATA")
    d.add_argument("--windows", default="opt1,opt2,opt3", help="comma list of opt1,opt2,opt3,H"); d.add_argument("--precision", default="float32", choices=["float32", "float16"])
    d.add_argument("--url", help="URL template with {name}")
    sub.add_parser("info", help="data directory and grid status")
    a = p.parse_args(argv)
    if a.cmd == "fit":
        if a.teff is None and a.bp_rp is None: p.error("give --bp-rp (and --ebv) or --teff for the Teff prior")
        _fit(a)
    elif a.cmd == "download-grid":
        from .grid import download_grid
        download_grid(a.windows.split(","), precision=a.precision, url=a.url)
    else:
        from .config import DATA
        from .grid import available, grid_dir
        from . import __version__
        import importlib.util as iu
        print(f"rochespec {__version__}; data {DATA}; grid {grid_dir()}")
        for w in ("opt1", "opt2", "opt3", "H"): print(f"  {w}: {'ok' if available((w,)) else 'missing'}")
        print("  phoebe:", "installed" if iu.find_spec("phoebe") else "not installed (Roche surface fallback)")


if __name__ == "__main__":
    main()
