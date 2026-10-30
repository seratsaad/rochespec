"""rochespec: Roche-model metallicities for contact binaries from survey spectra.

    import rochespec
    spec = rochespec.io.read_lamost_lrs("spec-55859-F5902_sp01-001.fits.gz")
    res = rochespec.fit(spec, period=0.3713, t0=2457000.123, bp_rp=0.78, ebv=0.03)
    print(res.summary())
"""
__version__ = "0.2.0"

from . import io
from .io import Spectrum
from .api import fit, model, FitResult, SURVEYS, teff_prior_from_colour
from .grid import download_grid, available as grid_available

__all__ = ["fit", "model", "FitResult", "Spectrum", "SURVEYS", "teff_prior_from_colour", "download_grid", "grid_available", "io", "__version__"]
