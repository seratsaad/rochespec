"""Quick checks that need neither the intensity grid nor PHOEBE."""
import os, json
import numpy as np
import pytest
import rochespec
from rochespec import api, grid, io
from rochespec.teffphot import teff_phot
from rochespec.roche import ContactBinary, lagrange, omega

HERE = os.path.dirname(os.path.abspath(__file__))


def test_version():
    assert rochespec.__version__ == "0.2.0"


def test_c21_teff():
    T, sT, x0 = teff_phot(np.array([0.82]), np.array([0.0]), 0.0, logg=4.44)
    assert abs(T[0] - 5762) < 5 and x0[0] == pytest.approx(0.82)


def test_colour_prior_slope():
    p = rochespec.teff_prior_from_colour(0.82, 0.0)
    assert 5700 < p["Tphot0"] < 5850 and 30 < p["dTdfeh"] < 90 and p["flag_colour"] == 0


def test_fillout_standard_definition():
    q = 0.3; xl1, xl2, xl3 = lagrange(q)
    o1, o2, o3 = omega(xl1, 0, 0, q), omega(xl2, 0, 0, q), omega(xl3, 0, 0, q)
    os.environ.pop("CB_GEOM", None)
    cb = ContactBinary(q, 1.0, ntheta=20, nphi=40)
    assert cb.omega0 == pytest.approx(max(o2, o3))          # f = 1 is the L2 surface
    assert ContactBinary(q, 0.0, ntheta=20, nphi=40).omega0 == pytest.approx(o1)


def test_surface_and_view():
    os.environ.pop("CB_GEOM", None)
    cb = ContactBinary(0.35, 0.2, ntheta=30, nphi=60)
    logg, T = cb.surface(1.0, 2.5, 5800.0)
    assert np.all(cb.dA > 0) and 3.5 < np.median(logg) < 4.8
    assert (np.sum(cb.dA * T ** 4) / cb.dA.sum()) ** 0.25 == pytest.approx(5800.0, rel=1e-6)
    los, mu, vis = cb.view(0.25, 78.0); rv = cb.rv(los, 2.5, 0.35)
    assert vis.sum() > 0 and np.all(mu[vis] > 0) and np.abs(rv).max() < 400


def test_target_row_phase():
    from rochespec.core import bjd_phase, veb_ok
    prior = dict(Tphot0=5800.0, dTdfeh=50.0, sig_Tphot=100.0)
    s = api._target_row(0.4, 2457000.1, None, None, None, prior, None, None, 0.02)
    assert veb_ok(s)
    t = 2457000.1 + 0.4 * np.array([0.0, 0.25, 10.6])
    assert np.allclose(bjd_phase(t, s), [0.0, 0.25, 0.6], atol=1e-9)
    s2 = api._target_row(0.4, None, None, None, None, prior, None, None, 0.02)
    assert not veb_ok(s2)


def test_geometry_arguments():
    prior = dict(Tphot0=5800.0, dTdfeh=0.0, sig_Tphot=100.0)
    with pytest.raises(ValueError):
        api._target_row(0.4, None, 0.3, None, None, prior, None, None, None)
    s = api._target_row(0.4, None, 3.0, 0.2, 80.0, prior, None, None, None)
    from rochespec.core import geo_params
    q, i, f, src = geo_params(s)
    assert q == pytest.approx(1 / 3) and i == 80.0 and f == 0.2 and src == "user"


def test_generic_segments():
    w = np.arange(3700.0, 9200.0, 1.0); sp = io.Spectrum(w, np.ones_like(w), np.ones_like(w), np.array([0.0]))
    segs = api._generic_segments([sp])
    assert segs[0][0] > 3950 and segs[-1][1] < 8850 and all(b - a <= 1800.01 for a, b in segs)


def test_inputs():
    with pytest.raises(ValueError):
        rochespec.fit(wave=np.arange(10.0), flux=np.ones(10), err=np.ones(10), times=[0.0], period=0.3)   # no survey/resolution
    with pytest.raises(ValueError):
        rochespec.fit(wave=np.arange(10.0), flux=np.ones(10), err=np.ones(10), times=[0.0], period=0.3, survey="sdss")


def test_grid_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(grid, "DATA", str(tmp_path) + "/")
    assert len(grid.missing(("H",))) == 180 and not grid.available(("H",))
    with pytest.raises(FileNotFoundError):
        grid.check_grid(("H",))
    with pytest.raises(RuntimeError):
        grid.download_grid(("H",), url="https://zenodo.org/records/PLACEHOLDER/files/{name}")


def test_checksums_listed():
    sums = grid._checksums()
    assert {f"rochespec_grid_{w}_{p}.tar.gz" for w in grid.WINDOWS for p in ("float32", "float16")} <= set(sums)


def test_kernel_weights_conserve_area():
    from rochespec.spice_backend import KernelIntegrator

    class G:                                         # minimal grid stand-in
        mhs = np.array([-0.5, 0.0]); teffs = np.array([5000.0, 6000.0]); loggs = np.array([4.0, 4.5]); mu = np.array([0.1, 0.5, 1.0])
        dv = 5.0; I = np.ones((2, 2, 2, 3, 200), np.float32); Ic = I.copy()
    ki = KernelIntegrator(G(), vmax=100.0)
    rng = np.random.default_rng(1); n = 50
    w = rng.uniform(0, 1, n)
    W = ki.weights(rng.uniform(5000, 6000, n), rng.uniform(4, 4.5, n), rng.uniform(-0.5, 0, n), rng.uniform(0.1, 1, n), rng.uniform(-50, 50, n), w)
    assert W.sum() == pytest.approx(w.sum()) and W.min() >= 0          # linear weights redistribute each element's area exactly


def test_reader_lamost_lrs():
    fn = os.path.join(HERE, "data", "48002204.fits.gz")
    sp = io.read_lamost_lrs(fn, ra=228.19959, dec=28.67041)
    assert len(sp) == 1 and sp[0].survey == "lamost_lrs"
    s = sp[0]
    assert 3600 < s.wave.min() < 4000 and s.wave.max() > 8800 and np.sum(s.ivar > 0) > 3000
    assert len(s.times) % 3 == 0 and 2455000 < s.times.min() < 2462000


def test_cli_help(capsys):
    from rochespec.cli import main
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "fit" in capsys.readouterr().out
