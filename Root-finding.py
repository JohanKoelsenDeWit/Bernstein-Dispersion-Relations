
"""
Author: J. K. de Wit

This code computes Bernstein-wave dispersion branches in a homogeneous plasma
(electrostatic, perpendicular limit) by solving D(k,ω)=0
with SciPy's Newton-Raphon's method. Each branch is initialized near ω ≈ n|ω_c|
at the largest k and continued to smaller k using the previous root
as the next initial guess.
"""

import numpy as np
import scipy.constants as sc
from scipy.special import ive
from scipy import optimize
import matplotlib.pyplot as plt


# Helper functions for characteristic frequencies and velocities.
def wp_f(n, q, m):
    return np.sqrt(n*q*q/(m*sc.epsilon_0))

def wc_f(B, q, m):
    return q*B/m

def vth_f(T_eV, m):
    return np.sqrt(2*T_eV*sc.e/m)

# Pre-computes values for faster computations.
def precompute_tables(B_T, species, harmonics, k_list):
    k_list = np.asarray(k_list, dtype=np.float64)
    tables = []
    for sp in species:
        name = sp['name']
        ns = float(sp['n_m3'])
        Ts = float(sp['T_eV'])
        qs = float(sp['q_C'])
        ms = float(sp['m_kg'])

        wps = wp_f(ns, qs, ms)
        wcs = wc_f(B_T, qs, ms)
        wc_mag = abs(wcs)
        vths = vth_f(Ts, ms)

        Nh = int(harmonics.get(name, 5))
        n_vec = np.arange(-Nh, Nh + 1, dtype=np.float64)

        kappa = 0.5*(vths*k_list/wc_mag)**2
        Ivals = ive(n_vec[None, :], kappa[:, None]).astype(np.float64)
        nI = (n_vec[None, :]*Ivals).astype(np.float64)

        pref = 2.0*wps*wps/(vths*vths)

        tables.append({
            'name': name,
            'wc': wc_mag,
            'n_vec': n_vec,
            'nI': nI,
            'pref': pref
        })
    return tables

# This function computes the dispersion function.
def D_dispersion(w, i, k2, tables):
    val = k2
    for tab in tables:
        wc = tab['wc']
        a = w / wc
        denom = tab['n_vec'] - a
        nI_row = tab['nI'][i, :]
        pref = tab['pref']
        val += pref * np.sum(nI_row / denom)
    return val

# This function computes the derivative of the dispersion funktion
# with respect to the angular frequency. Used for Newton-Rahpon's
# method.
def dDdw_dispersion(w, i, k2, tables):
    val = 0.0
    for tab in tables:
        wc = tab['wc']
        a = w / wc
        denom = tab['n_vec'] - a
        nI_row = tab['nI'][i, :]
        pref = tab['pref']
        val += (pref / wc) * np.sum(nI_row / (denom * denom))
    return val

# The main function that solves the roots for the k array.
def solve_dispersion_branches(B_T, species, harmonics, k_list, w0_list, rtol=1e-5, maxiter=60):
    k_list = np.asarray(k_list, dtype=np.float64)
    Nk = k_list.size
    tables = precompute_tables(B_T, species, harmonics, k_list)

    branches = []
    for w0 in w0_list:
        w_list = np.zeros(Nk, dtype=np.float64)
        w = float(w0)
        for i in range(Nk):
            k2 = k_list[i] * k_list[i]
            sol = optimize.root_scalar(
                D_dispersion,
                x0=w,
                fprime=dDdw_dispersion,
                args=(i, k2, tables),
                method="newton",
                rtol=rtol,
                maxiter=maxiter
            )
            if sol.converged and np.isfinite(sol.root):
                w = float(sol.root)
            w_list[i] = w
        branches.append(w_list)
    return np.array(branches)


# Main function of the script. Plasma parameters are defined here.
# The example includes electrons and a single ion species, modeled
# as protons.
def main():
    B_T = 0.05
    Te_eV = 5.0
    Ti_eV = 5.0
    ne_m3 = 5e16
    ni_m3 = 5e16

    species = [
        {'name': 'e', 'n_m3': ne_m3, 'T_eV': Te_eV, 'q_C': -sc.e, 'm_kg': sc.m_e},
        {'name': 'i', 'n_m3': ni_m3, 'T_eV': Ti_eV, 'q_C': +sc.e, 'm_kg': sc.m_p},
    ]

    harmonics = {'e': 6, 'i': 6}

    # The k values computed. These parameters highlight the
    # electron Bernstein branches for the chosen parameters.
    k_min = 1e3
    k_max = 40e3
    Nk = 1000
    k_list = np.linspace(k_max, k_min, Nk)

    rtol = 1e-5
    maxiter = 60

    # The initial guess at large k, chosen here to be
    # electron Bernstein branches. In this example, 
    # set species[1]['q_C'] and species[1]['m_kg']
    # to see the ion branches. Set also k_min = 10 and k_max = 1e3.
    wce_mag = abs(wc_f(B_T, species[0]['q_C'], species[0]['m_kg']))
    # Compute first 4 branches.
    harmonic_numbers = [1, 2, 3, 4]
    w0_list = [wce_mag * n * 1.005 for n in harmonic_numbers]

    w_branches = solve_dispersion_branches(
        B_T=B_T,
        species=species,
        harmonics=harmonics,
        k_list=k_list,
        w0_list=w0_list,
        rtol=rtol,
        maxiter=maxiter
    )

    # Plotting dispersion branches.
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for idx, n in enumerate(harmonic_numbers):
        ax.plot(k_list, w_branches[idx] / (2*np.pi*1e9), lw=1.5, label=f"n={n}")

    ax.grid(True)
    ax.set_xlim(k_min, k_max)
    ax.set_xlabel("k [1/m]")
    ax.set_ylabel("f [GHz]")
    ax.legend(title="Cyclotron harmonic")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()