import numpy as np
h, c, kB = 6.626e-34, 2.998e8, 1.381e-23
def teff_color(bprp0, feh=0.0):
    feh = np.nan_to_num(feh, nan=0.0)
    th = 0.4929 + 0.5092 * bprp0 - 0.0353 * bprp0 ** 2 + 0.0192 * feh - 0.0020 * feh ** 2 - 0.0395 * bprp0 * feh   # Mucciarelli & Bellazzini 2020, dwarfs
    return 5040. / th
def bc_g(T):
    x = T - 5772.
    return 6.000e-02 + 6.731e-05 * x - 6.647e-08 * x ** 2 + 2.859e-11 * x ** 3 - 7.197e-15 * x ** 4   # Andrae+2018
def photmass(G, plx_mas, bprp, AG, EBR, feh, P_d, rsum, Tratio, Gratio, lam=0.64e-6):
    T = teff_color(bprp - EBR, feh)
    Mbol = G + 5 * np.log10(plx_mas / 100.) - AG + bc_g(T)
    L = 10 ** (-0.4 * (Mbol - 4.74))
    x1 = h * c / (lam * kB * T); S = (np.exp(x1) - 1) / (np.exp(x1 / Tratio) - 1)    # G-band surface-brightness ratio (secondary/primary), T1~T first
    k = np.sqrt(Gratio / S)
    T1 = T * ((1 + k ** 2) / (1 + k ** 2 * Tratio ** 4)) ** 0.25
    S = (np.exp(h * c / (lam * kB * T1)) - 1) / (np.exp(h * c / (lam * kB * T1 * Tratio)) - 1); k = np.sqrt(Gratio / S)   # one iteration
    R1 = np.sqrt(L / (1 + k ** 2 * Tratio ** 4)) * (5772. / T1) ** 2
    R2 = k * R1
    a_rsun = (R1 + R2) / rsum
    Mtot = (a_rsun / 215.032) ** 3 / (P_d / 365.25) ** 2
    return dict(Teff=T, L=L, k=k, R1=R1, R2=R2, a=a_rsun, Mtot=Mtot)
