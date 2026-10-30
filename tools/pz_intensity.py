"""payne-zero synthesis with angle-resolved emergent intensities (scratch layer; payne-zero code untouched).

The transfer tables' surface_flux_weights vector is replaced by a matrix [flux_weights | W_mu] so the solver's
single matmul returns the exact payne-zero Eddington flux (column 0) and I(mu) at Gauss-Legendre nodes
(columns 1..n). I(mu) is then rescaled per wavelength so that 1/2 int I mu dmu equals the payne-zero flux.
"""
import sys, numpy as np, torch
import os; sys.path.insert(0, os.path.expanduser(os.environ.get("PZ_ROOT", "~/payne-zero")))
import payne_zero_synthesis.radiative_transfer as rt
from payne_zero_synthesis import synthesize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intensity_weights import intensity_weights

NMU = 10
_x, _w = np.polynomial.legendre.leggauss(NMU); MU = 0.5 * (_x + 1); WMU = 0.5 * _w

_orig_from_npz = rt.TransferTables.from_npz.__func__
_orig_sat = rt._saturated_core_flux


def _from_npz(cls, *a, **k):
    t = _orig_from_npz(cls, *a, **k)
    tau = t.optical_depth_grid.detach().cpu().numpy().astype(float)
    W = intensity_weights(tau, MU)
    fw = t.surface_flux_weights.detach().cpu().numpy().astype(float)
    t.surface_flux_weights = torch.as_tensor(np.column_stack([fw, W]), dtype=t.surface_flux_weights.dtype, device=t.surface_flux_weights.device)
    return t


def _sat(*a, **k):
    f = _orig_sat(*a, **k)
    return f[:, None].expand(-1, NMU + 1)            # saturated cores: same value in every column (negligible)


rt.TransferTables.from_npz = classmethod(_from_npz)
rt._saturated_core_flux = _sat


def synth_intensity(atm_npz, wl0_nm, wl1_nm, R=300000.0, device="mps", dtype="float32"):
    s = synthesize(atm_npz, wavelength_start_nm=wl0_nm, wavelength_end_nm=wl1_nm, resolution=R,
                   molecular_lines=True, device=device, dtype=dtype)
    F, Fc = np.asarray(s.flux_total), np.asarray(s.flux_continuum)       # (n_wl, 1 + NMU)
    out = {}
    for name, A in (("I", F), ("Ic", Fc)):
        H, I = A[:, 0], A[:, 1:]
        Hi = 0.5 * I @ (WMU * MU)
        out[name] = (I * (H / Hi)[:, None]).T                              # (NMU, n_wl), flux-consistent
        out["H" if name == "I" else "Hc"] = H
    out["wl_nm"] = np.asarray(s.wavelength_nm); out["mu"] = MU; out["wmu"] = WMU; out["seconds"] = s.seconds
    return out


if __name__ == "__main__":
    import time
    atm = os.path.expanduser(os.environ.get("PZ_ROOT", "~/payne-zero")) + "/reference/reference_atmospheres_from_pyk_20260706/sun_structured_atmosphere.npz"
    t0 = time.time(); r = synth_intensity(atm, 1510.0, 1700.0, R=300000.0)
    print(f"{len(r['wl_nm'])} wavelengths in {time.time()-t0:.0f}s (synthesis {r['seconds']:.0f}s)")
    lim = r["Ic"][:, len(r["wl_nm"]) // 2]; print("continuum limb darkening I(mu)/I(1):", (lim / lim[-1]).round(3))
    print("normalized flux min/median:", (r["H"] / r["Hc"]).min().round(3), np.median(r["H"] / r["Hc"]).round(4))
    np.savez("test_sun_H.npz", **r)
