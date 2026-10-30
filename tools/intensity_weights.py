"""Angle-dependent intensity weights on payne-zero's transfer optical-depth grid.

payne-zero gives the emergent Eddington flux as  H = S(tau_grid) . surface_flux_weights.
The specific intensity is  I(mu) = int_0^inf S(tau) exp(-tau/mu) dtau/mu ; with S piecewise linear on the
grid (constant beyond the last point) this is  I(mu) = S . W[:, mu], with W computed analytically here.
Check: H = 1/2 int_0^1 I(mu) mu dmu should reproduce surface_flux_weights.
"""
import os
import numpy as np


def intensity_weights(tau, mus):
    W = np.zeros((len(tau), len(mus)))
    for k, mu in enumerate(mus):
        for j in range(len(tau) - 1):
            a, b = tau[j], tau[j + 1]; h = b - a
            ea, eb = np.exp(-a / mu), np.exp(-b / mu)
            # int_a^b exp(-t/mu)/mu * (b-t)/h dt  and  * (t-a)/h dt
            i0 = ea - eb                                      # int exp(-t/mu)/mu
            i1 = (a + mu) * ea - (b + mu) * eb                # int t exp(-t/mu)/mu
            W[j, k] += (b * i0 - i1) / h
            W[j + 1, k] += (i1 - a * i0) / h
        W[-1, k] += np.exp(-tau[-1] / mu)                      # constant source below the grid
    return W


if __name__ == "__main__":
    d = np.load(os.path.expanduser(os.environ.get("PZ_ROOT", "~/payne-zero")) + "/source_data_files/synthesis_tables/transfer_tables.npz")
    tau, fw = d["optical_depth_grid"], d["surface_flux_weights"]
    x, w = np.polynomial.legendre.leggauss(40); mus = 0.5 * (x + 1); wm = 0.5 * w
    W = intensity_weights(tau, mus)
    H = 0.5 * W @ (wm * mus)
    print("sum flux weights payne-zero:", fw.sum(), " from intensities:", H.sum())
    print("ratio per depth (first, middle, last):", (H / fw)[[1, 10, 25, 40, 50]].round(4))
    S = 1 + 1.5 * tau                                          # Eddington-Barbier test source
    print("linear source: H_pz", S @ fw, " H_int", S @ H, " analytic 0.5*(1/2+1.5/3)=", 0.5 * (0.5 + 0.5))
