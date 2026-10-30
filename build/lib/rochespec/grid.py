"""The payne-zero specific-intensity grid: location, check and download.

Nodes: Teff 4750-6750 K (250 K), log g 3.75/4.10/4.45/4.80, [M/H] -1.0/-0.6/-0.3/0.0/+0.3, 10 mu values; windows opt1 (380-550 nm),
opt2 (550-720 nm), opt3 (720-890 nm) for the optical surveys and H (1510-1700 nm) for APOGEE. The fitters read
$ROCHESPEC_DATA/pz/grid/<window>/I_T<teff>_g<logg>_m<mh>.npz.
"""
import os, hashlib, tarfile, shutil, urllib.request
from .config import DATA

# >>> PLACEHOLDER: the Zenodo record of the grid is not public yet. Set ROCHESPEC_GRID_URL or pass url= until it is. <<<
GRID_URL = os.environ.get("ROCHESPEC_GRID_URL", "https://zenodo.org/records/PLACEHOLDER/files/{name}")
WINDOWS = ("opt1", "opt2", "opt3", "H")
SURVEY_WINDOWS = {"lamost_lrs": ("opt1", "opt2", "opt3"), "lamost_mrs": ("opt1", "opt2", "opt3"), "galah": ("opt1", "opt2", "opt3"),
                  "desi": ("opt1", "opt2", "opt3"), "generic": ("opt1", "opt2", "opt3"), "apogee": ("H",)}
TEFFS = (4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750)
LOGGS = (3.75, 4.1, 4.45, 4.8)
MHS = (-1.0, -0.6, -0.3, 0.0, 0.3)
CHECKSUMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "grid_sha256.txt")


def grid_dir():
    return os.path.join(DATA, "pz", "grid")


def missing(windows=("opt1", "opt2", "opt3")):
    """list of grid files that are not on disk"""
    out = []
    for w in windows:
        for t in TEFFS:
            for g in LOGGS:
                for m in MHS:
                    fn = os.path.join(grid_dir(), w, f"I_T{t}_g{g}_m{m}.npz")
                    if not os.path.exists(fn): out.append(fn)
    return out


def available(windows=("opt1", "opt2", "opt3")):
    return not missing(windows)


def check_grid(windows):
    m = missing(windows)
    if m:
        raise FileNotFoundError(f"payne-zero intensity grid incomplete in {grid_dir()} ({len(m)} files missing, e.g. {m[0]}). "
                                f"Run rochespec.download_grid({list(windows)}) or set ROCHESPEC_DATA.")


def _sha256(fn, block=2 ** 22):
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(block), b""): h.update(b)
    return h.hexdigest()


def _checksums():
    out = {}
    if os.path.exists(CHECKSUMS):
        for line in open(CHECKSUMS):
            if line.strip() and not line.startswith("#"):
                h, name = line.split()[:2]; out[name] = h
    return out


def download_grid(windows=("opt1", "opt2", "opt3"), precision="float32", url=None, keep_tarballs=False, force=False):
    """download and unpack the grid windows into $ROCHESPEC_DATA/pz/grid, checking each tarball's SHA-256.
    precision: 'float32' (as used for the catalogue) or 'float16' (a sixth of the download; changes [Fe/H] by up to 0.0013 dex)."""
    if precision not in ("float32", "float16"):
        raise ValueError("precision must be 'float32' or 'float16'")
    url = url or GRID_URL
    if "PLACEHOLDER" in url:
        raise RuntimeError("the grid download URL is not set yet (Zenodo record pending): pass url= or set ROCHESPEC_GRID_URL")
    sums = _checksums(); os.makedirs(grid_dir(), exist_ok=True); tmp = os.path.join(DATA, "downloads"); os.makedirs(tmp, exist_ok=True)
    for w in windows:
        if w not in WINDOWS: raise ValueError(f"unknown window {w}; choose from {WINDOWS}")
        if available((w,)) and not force:
            print(f"{w}: already present"); continue
        name = f"rochespec_grid_{w}_{precision}.tar.gz"; fn = os.path.join(tmp, name)
        if not os.path.exists(fn):
            print(f"downloading {name} ..."); urllib.request.urlretrieve(url.format(name=name), fn + ".part"); os.replace(fn + ".part", fn)
        if name in sums and _sha256(fn) != sums[name]:
            os.remove(fn); raise IOError(f"checksum mismatch for {name}; the file was removed, try again")
        with tarfile.open(fn) as tf:
            tf.extractall(grid_dir(), filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(grid_dir())
        if not keep_tarballs: os.remove(fn)
        print(f"{w}: installed in {os.path.join(grid_dir(), w)}")
    shutil.rmtree(tmp, ignore_errors=True) if not keep_tarballs and not os.listdir(tmp) else None
