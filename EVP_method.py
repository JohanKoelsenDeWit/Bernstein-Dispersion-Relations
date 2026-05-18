"""
Author: J. K. de Wit

This code computes Bernstein wave dispersion relations in a homogeneous plasma
(electrostatic, perpendicular limit) using a linear eigenvalue problem in ω².
For each species ξ and harmonic n=1..Nξ, one auxiliary variable u_{ξn} is
introduced, yielding the EVP:

(W + A) X = ω² X

whose eigenvalues approximate ω² for a given wavenumber.

"""

import numpy as np
import scipy.constants as sc
from scipy.special import ive
from numpy.linalg import eig
import matplotlib.pyplot as plt


# Helper functions for characteristic frequencies and velocities.
def wp2_f(n_m3, q_C, m_kg):
    return n_m3 * q_C * q_C / (sc.epsilon_0 * m_kg)


def wc_abs_f(B_T, q_C, m_kg):
    return abs(q_C * B_T / m_kg)


def vth_f(T_eV, m_kg):
    return np.sqrt(2.0 * T_eV * sc.e / m_kg)


# Computes the EVP in ω²
def ebw_evp_multispecies(B_T, species, k_perp, harmonics):
    if isinstance(harmonics, int):
        Nmap = {sp["name"]: int(harmonics) for sp in species}
    elif isinstance(harmonics, dict):
        Nmap = {sp["name"]: int(harmonics[sp["name"]]) for sp in species}
    else:
        raise TypeError("harmonics must be int or dict")

    pairs = []
    for sp in species:
        for n in range(1, Nmap[sp["name"]] + 1):
            pairs.append((sp, n))

    M = len(pairs)

    Wdiag = np.zeros(M, dtype=np.float64)
    a_vec = np.zeros(M, dtype=np.float64)

    k = float(k_perp)
    for j, (sp, n) in enumerate(pairs):
        n_m3 = float(sp["n_m3"])
        T_eV = float(sp["T_eV"])
        q_C = float(sp["q_C"])
        m_kg = float(sp["m_kg"])

        wp2 = wp2_f(n_m3, q_C, m_kg)
        wc = wc_abs_f(B_T, q_C, m_kg)
        vth = vth_f(T_eV, m_kg)

        rhoL = vth / wc if wc > 0.0 else np.inf
        lam = 0.5 * (k * rhoL) ** 2

        Wdiag[j] = (float(n) * wc) ** 2

        if lam > 0.0 and np.isfinite(lam):
            a_vec[j] = 2.0 * wp2 * (n ** 2) * ive(n, lam) / lam
        else:
            a_vec[j] = 0.0

    W = np.diag(Wdiag)
    A = np.outer(a_vec, np.ones(M, dtype=np.float64))

    Mmat = W + A
    w2, V = eig(Mmat)

    meta = {
        "pairs": [(sp["name"], int(n)) for (sp, n) in pairs],
        "Wdiag": Wdiag,
        "a_vec": a_vec,
    }
    return w2, V, meta


# The main function where the plasma parameters are defined.
def main():
    B_T = 0.05
    Te_eV = 5.0
    Ti_eV = 5.0
    ne_m3 = 5e16
    ni_m3 = 5e16

    species = [
        {"name": "e", "n_m3": ne_m3, "T_eV": Te_eV, "q_C": -sc.e, "m_kg": sc.m_e},
        {"name": "i", "n_m3": ni_m3, "T_eV": Ti_eV, "q_C": +sc.e, "m_kg": sc.m_p},
    ]

    harmonics = {"e": 6, "i": 25}

    k_min = 10
    k_max = 2.0e4
    Nk = 100
    k_list = np.linspace(k_max, k_min, Nk)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))

    for k in k_list:
        w2, V, meta = ebw_evp_multispecies(B_T, species, k, harmonics)

        # Keep only real and positive roots
        w2_real = np.real(w2)
        m = np.isfinite(w2_real) & (w2_real > 0.0)

        f = np.sqrt(w2_real[m]) / (2.0 * np.pi)
        ax.plot(np.full(np.count_nonzero(m), k), f / 1e9, ".", ms=2.5, color="k")

    ax.grid(True)
    ax.set_xlabel("k [1/m]")
    ax.set_ylabel("f [GHz]")
    ax.set_xlim(k_min, k_max)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()