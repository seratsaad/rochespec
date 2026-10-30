"""Fit one LAMOST DR10 LRS spectrum of a contact binary and plot the model.

Star: ASASSN-V J075946.41+410435.3 (Gaia DR3 922162429440295424), P = 0.4142 d. Geometry from the light-curve solution of Li & Wang
(2025): q = 0.195, f = 0.057, i = 88.4 deg; ephemeris (time of primary minimum) from the Gaia DR3 eclipsing-binary model.
Needs the optical intensity grid: rochespec.download_grid(["opt1", "opt2", "opt3"]).
"""
import os
import matplotlib.pyplot as plt
import rochespec

here = os.path.dirname(os.path.abspath(__file__))
spec = rochespec.io.read_lamost_lrs(os.path.join(here, "data", "279212133.fits.gz"), ra=119.94337, dec=41.07646)

res = rochespec.fit(spec, period=0.41419978, t0=2457372.59501,
                    q=0.195, fillout=0.057, incl=88.409,
                    bp_rp=0.7916, ebv=0.044)          # Gaia BP-RP and E(B-V): Casagrande et al. (2021) Teff prior
print(res.summary())
# with the catalogue's own prior, teff_prior=(6020.8, 106.4, 45.8), the result is the catalogue value, [Fe/H] = -0.278 +- 0.084

fig, ax = plt.subplots(figsize=(9, 3.2))
res.plot(ax)
ax.set_xlim(4950, 5450); ax.set_ylim(0.8, 1.25)
fig.tight_layout(); fig.savefig(os.path.join(here, "fit_lamost_lrs.png"), dpi=150)
