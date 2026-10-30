"""Photometric Teff from dereddened Gaia (BP-RP) with the Casagrande et al. (2021, MNRAS 507, 2684; arXiv:2011.02517)
EDR3 relation (their Table A1, eq. 1), including the [Fe/H] and log g terms. Extinction coefficients are their
colour-dependent COD-law EDR3 values (Table B1): E(BP-RP) = (R_BP - R_RP) E(B-V), iterated on the intrinsic colour.
"""
import numpy as np

A = dict(a0=7981, a1=-4138.3457, a2=1264.9366, a3=-130.4388, a5=285.8393, a6=-324.2196, a7=106.8511,
         a8=-4.9825, a10=4.5138, a11=-203.7774, a12=126.6981, a13=-14.7442, a14=40.7376)
SIG_REL = 60.0                      # K, quoted precision 55-64 K


def r_bp(x): return 3.526 - 0.168 * x - 0.170 * x ** 2 + 0.060 * x ** 3
def r_rp(x): return 2.062 - 0.072 * x
def r_g(x): return 3.071 - 0.511 * x + 0.058 * x ** 2


def bprp0(bprp, ebv, n=5):
    x = np.asarray(bprp, float) - np.asarray(ebv, float)
    for _ in range(n): x = bprp - (r_bp(x) - r_rp(x)) * ebv
    return x


def teff_c21(x, feh, logg=4.3):
    """x = (BP-RP)_0; valid for dwarfs 0.2 <= x <= 2.0"""
    x = np.asarray(x, float); f = np.asarray(feh, float); g = np.asarray(logg, float)
    return (A["a0"] + A["a1"] * x + A["a2"] * x ** 2 + A["a3"] * x ** 3 + A["a5"] * g + A["a6"] * g * x + A["a7"] * g * x ** 2
            + A["a8"] * g * x ** 3 + A["a10"] * f + A["a11"] * f * x + A["a12"] * f * x ** 2 + A["a13"] * f * x ** 3 + A["a14"] * f * g * x)


def teff_phot(bprp, ebv, feh, logg=4.3, sig_ebv=None):
    """Teff and its 1-sigma (relation scatter + reddening error). sig_ebv default: max(0.02, 0.15 E(B-V))."""
    ebv = np.nan_to_num(np.asarray(ebv, float), nan=0.0)
    if sig_ebv is None: sig_ebv = np.maximum(0.02, 0.15 * ebv)
    x = bprp0(bprp, ebv); T = teff_c21(x, feh, logg)
    dTdE = (teff_c21(bprp0(bprp, ebv + 0.01), feh, logg) - T) / 0.01
    return T, np.hypot(SIG_REL, dTdE * sig_ebv), x


if __name__ == "__main__":
    # sanity: Sun-like (BP-RP)_0 = 0.82 at [Fe/H] = 0, log g = 4.44 should give ~5770 K
    print(teff_c21(0.82, 0.0, 4.44), teff_c21(0.82, -0.5, 4.44), teff_phot(0.9, 0.05, 0.0))
