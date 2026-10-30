"""The public API against the catalogue values for LAMOST LRS stars (spectra in tests/data). Needs the optical intensity grid
(rochespec.download_grid) and PHOEBE for the exact comparison (the catalogue geometry); skipped otherwise."""
import os, json, importlib.util
import numpy as np
import pytest
import rochespec

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = json.load(open(os.path.join(HERE, "data", "reference_stars.json")))
need = pytest.mark.skipif(not rochespec.grid_available(("opt1", "opt2", "opt3")), reason="intensity grid not installed")
need_phoebe = pytest.mark.skipif(importlib.util.find_spec("phoebe") is None, reason="PHOEBE not installed (catalogue geometry)")


@need
@need_phoebe
@pytest.mark.parametrize("ref", REFS, ids=[str(r["source_id"]) for r in REFS])
def test_catalogue_values(ref):
    spec = sum([rochespec.io.read_lamost_lrs(os.path.join(HERE, "data", f), ra=ref["ra"], dec=ref["dec"]) for f in ref["files"]], [])
    res = rochespec.fit(spec, period=ref["period"], t0=ref["t0"], teff_prior=(ref["Tphot0"], ref["sig_Tphot"], ref["dTdfeh"]),
                        geometry="phoebe", models=False, **ref["geometry"])
    assert res.geometry == "phoebe"
    assert abs(res.feh - ref["expected"]["feh"]) < 1e-3
    assert abs(res.mh - ref["expected"]["mh"]) < 1e-3
    assert abs(res.teff - ref["expected"]["teff"]) < 5
    assert abs(res.feh_err - ref["expected"]["feh_err"]) < 2e-3


@need
def test_model_spectrum():
    w, f = rochespec.model(teff=5800, mh=-0.2, period=0.35, phases=[0.25], geometry="roche")
    k = (w > 5000) & (w < 5500)
    assert np.all(np.isfinite(f[k])) and 0.2 < f[k].min() < 0.8 and f[k].max() <= 1.05
