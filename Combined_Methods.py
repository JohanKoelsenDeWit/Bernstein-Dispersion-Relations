"""
Author: J. K. de Wit

This script combines the three methods into a single comparison of 
the fundamental electron Bernstein branch.
"""

import numpy as np
import scipy.constants as sc
from scipy.special import ive
from scipy import optimize
from numpy.linalg import eig
import matplotlib.pyplot as plt

# Helper functions for the plasma parameters
def wp_f(n_m3, q_C, m_kg):
    return np.sqrt(n_m3 * q_C * q_C / (m_kg * sc.epsilon_0))


def wp2_f(n_m3, q_C, m_kg):
    return n_m3 * q_C * q_C / (m_kg * sc.epsilon_0)


def wc_f(B_T, q_C, m_kg):
    return q_C * B_T / m_kg


def vth_f(T_eV, m_kg):
    return np.sqrt(2.0 * T_eV * sc.e / m_kg)

# Function to selected the wanted species
def _get_species(species, name):
    for sp in species:
        if sp["name"] == name:
            return sp
    raise ValueError(f"Species '{name}' not found in species list.")



# Helper function to precompute values for numerical efficiency.
def precompute_root_tables(B_T, species, harmonics, k_list):
    k_list = np.asarray(k_list, dtype=np.float64)
    tables = []
    for sp in species:
        name = sp["name"]
        ns = float(sp["n_m3"])
        Ts = float(sp["T_eV"])
        qs = float(sp["q_C"])
        ms = float(sp["m_kg"])

        wps = wp_f(ns, qs, ms)
        wc = abs(wc_f(B_T, qs, ms))
        vth = vth_f(Ts, ms)

        Nh = int(harmonics.get(name, 5))
        n_vec = np.arange(-Nh, Nh + 1, dtype=np.float64)

        kappa = 0.5 * (vth * k_list / wc) ** 2
        Ivals = ive(n_vec[None, :], kappa[:, None]).astype(np.float64)
        nI = (n_vec[None, :] * Ivals).astype(np.float64)

        pref = 2.0 * wps * wps / (vth * vth)

        tables.append({"wc": wc, "n_vec": n_vec, "nI": nI, "pref": pref})
    return tables


# The dispersion function.
def D_root(omega, i, k2, tables):
    val = k2
    for tab in tables:
        wc = tab["wc"]
        a = omega / wc
        denom = tab["n_vec"] - a
        val += tab["pref"] * np.sum(tab["nI"][i, :] / denom)
    return val


# The derivative of the dispersion function with respect to omega.
def dDdw_root(omega, i, k2, tables):
    val = 0.0
    for tab in tables:
        wc = tab["wc"]
        a = omega / wc
        denom = tab["n_vec"] - a
        val += (tab["pref"] / wc) * np.sum(tab["nI"][i, :] / (denom * denom))
    return val


# The root-finding method to compute the fundamental branch.
def solve_one_branch_omega_of_k(B_T, species, harmonics, k_list, w0, rtol=1e-5, maxiter=60):
    k_list = np.asarray(k_list, dtype=np.float64)
    tables = precompute_root_tables(B_T, species, harmonics, k_list)

    omega_out = np.zeros_like(k_list)
    omega = float(w0)

    for i in range(k_list.size):
        k2 = k_list[i] * k_list[i]
        sol = optimize.root_scalar(
            D_root,
            x0=omega,
            fprime=dDdw_root,
            args=(i, k2, tables),
            method="newton",
            rtol=rtol,
            maxiter=maxiter,
        )
        if sol.converged and np.isfinite(sol.root):
            omega = float(sol.root)
        omega_out[i] = omega

    return omega_out


# Stix S coefficient for the expanded expression.
def stix_S(omega_rad_s, B_T, species):
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    S = np.ones_like(omega)
    for sp in species:
        wpe = wp_f(float(sp["n_m3"]), float(sp["q_C"]), float(sp["m_kg"]))
        wc = abs(wc_f(B_T, float(sp["q_C"]), float(sp["m_kg"])))
        S -= (wpe * wpe) / (omega * omega - wc * wc)
    return S

# The hot correct to the cold limit.
def ell_T2(omega_rad_s, B_T, species):
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    val = np.zeros_like(omega)
    for sp in species:
        wpe = wp_f(float(sp["n_m3"]), float(sp["q_C"]), float(sp["m_kg"]))
        wc = abs(wc_f(B_T, float(sp["q_C"]), float(sp["m_kg"])))
        vth = vth_f(float(sp["T_eV"]), float(sp["m_kg"]))
        rhoL = vth / wc
        denom = (4.0 * wc * wc - omega * omega) * (omega * omega - wc * wc)
        val += 1.5 * (wpe * wpe) * (wc * wc) * (rhoL * rhoL) / denom
    return val


# Function that computes k_expans
def k_expansion(S, ellT2):
    S = np.asarray(S, dtype=np.float64)
    ellT2 = np.asarray(ellT2, dtype=np.float64)
    k2 = -S / ellT2
    return np.where(k2 > 0.0, np.sqrt(k2), np.nan)

# Function that computes the asymptotic expression.
def k_asymptotic_sum(omega_rad_s, B_T, species):
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    n = 1

    ksum = np.zeros_like(omega)
    for sp in species:
        wpe = wp_f(float(sp["n_m3"]), float(sp["q_C"]), float(sp["m_kg"]))
        wc = abs(wc_f(B_T, float(sp["q_C"]), float(sp["m_kg"])))
        vth = vth_f(float(sp["T_eV"]), float(sp["m_kg"]))
        rhoL = vth / wc

        delta = omega - n * wc
        C = (np.sqrt(np.pi) * wc) / (2.0 * n * (wpe * wpe))
        base = C * delta
        term = np.where(base > 0.0, (1.0 / rhoL) * np.power(base, -1.0 / 3.0), 0.0)
        ksum = ksum + term

    return np.where(ksum > 0.0, ksum, np.nan)

# The sigmoid weight function.
def sigmoid_weight(k_exp, rhoL_target, Delta, k_rho0=1.0):
    arg = (k_exp * rhoL_target - k_rho0) / Delta
    return 1.0 / (1.0 + np.exp(-arg))

# The stitched approximative expression for the dispersion function
def complete_dispersion_relation(
    omega_rad_s,
    B_T,
    species,
    target_species_name="e",
    Delta=0.175,
    k_rho0=1.0,
):
    omega = np.asarray(omega_rad_s, dtype=np.float64)

    S = stix_S(omega, B_T, species)
    ellT2 = ell_T2(omega, B_T, species)
    k_exp = k_expansion(S, ellT2)

    sp_tgt = _get_species(species, target_species_name)
    wc_tgt = abs(wc_f(B_T, float(sp_tgt["q_C"]), float(sp_tgt["m_kg"])))
    vth_tgt = vth_f(float(sp_tgt["T_eV"]), float(sp_tgt["m_kg"]))
    rhoL_tgt = vth_tgt / wc_tgt

    k_asym = k_asymptotic_sum(omega, B_T, species)

    sigma = sigmoid_weight(k_exp, rhoL_tgt, Delta, k_rho0=k_rho0)
    k_tot = (1.0 - sigma) * np.abs(k_exp) + sigma * np.abs(k_asym)

    return {"k_tot": k_tot, "k_exp": k_exp, "k_asym": k_asym, "sigma": sigma}


# Computes the EVP
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
    dim = 2 * M

    Ddiag = np.zeros(dim, dtype=np.float64)
    g = np.zeros(dim, dtype=np.float64)

    k = float(k_perp)
    lam_floor = 1e-30

    for j, (sp, n) in enumerate(pairs):
        n_m3 = float(sp["n_m3"])
        T_eV = float(sp["T_eV"])
        q_C = float(sp["q_C"])
        m_kg = float(sp["m_kg"])

        wp2 = wp2_f(n_m3, q_C, m_kg)
        Omega = abs(wc_f(B_T, q_C, m_kg))
        vth = vth_f(T_eV, m_kg)
        rhoL = vth / Omega if Omega > 0.0 else np.inf
        lam = 0.5 * (k * rhoL) ** 2
        lam = max(float(lam), lam_floor)

        a = (wp2 / Omega) * int(n) * ive(int(n), lam) / lam

        ip = 2 * j
        im = 2 * j + 1

        Ddiag[ip] = +float(n) * Omega
        Ddiag[im] = -float(n) * Omega
        g[ip] = +a
        g[im] = -a

    Mmat = np.diag(Ddiag) + np.outer(g, np.ones(dim, dtype=np.float64))
    w, V = eig(Mmat)
    return w

# Function to only select the fundamental branch of the EVP roots.
def select_evp_branch_near_root(B_T, species, harmonics, k_list, omega_root, wmin, wmax, imag_rel_tol=1e-6):
    k_list = np.asarray(k_list, dtype=np.float64)
    omega_root = np.asarray(omega_root, dtype=np.float64)

    omega_sel = np.full_like(k_list, np.nan, dtype=np.float64)

    for i, k in enumerate(k_list):
        w = ebw_evp_multispecies(B_T, species, k, harmonics)
        wr = np.real(w)
        wi = np.imag(w)

        mask = np.isfinite(wr) & np.isfinite(wi) & (wr > 0.0)
        mask = mask & (wr >= wmin) & (wr <= wmax)
        mask = mask & (np.abs(wi) <= imag_rel_tol * np.maximum(1.0, np.abs(wr)))

        wr = wr[mask]
        if wr.size == 0:
            continue

        omega_sel[i] = float(wr[np.argmin(np.abs(wr - omega_root[i]))])

    return omega_sel

# Main function that computes the three solutions and plots them.
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

    harmonics = {"e": 6, "i": 6}

    target_species_name = "e"
    harmonic_n = 1

    k_min = 10.0
    k_max = 40e3
    Nk = 6000
    k_list = np.linspace(k_max, k_min, Nk)

    rtol = 1e-5
    maxiter = 60

    sp_tgt = _get_species(species, target_species_name)
    wc_tgt = abs(wc_f(B_T, sp_tgt["q_C"], sp_tgt["m_kg"]))
    w0 = (harmonic_n * wc_tgt) * 1.005

    omega_root = solve_one_branch_omega_of_k(
        B_T=B_T,
        species=species,
        harmonics=harmonics,
        k_list=k_list,
        w0=w0,
        rtol=rtol,
        maxiter=maxiter,
    )

    wmin = float(np.nanmin(omega_root))
    wmax = float(np.nanmax(omega_root))

    Delta = 0.175
    k_rho0 = 1.2

    omega_plot = np.linspace(harmonic_n * wc_tgt * 1.001, 2.0 * harmonic_n * wc_tgt * 0.999, 4000)

    out = complete_dispersion_relation(
        omega_rad_s=omega_plot,
        B_T=B_T,
        species=species,
        target_species_name=target_species_name,
        Delta=Delta,
        k_rho0=k_rho0,
    )

    omega_evp = select_evp_branch_near_root(
        B_T=B_T,
        species=species,
        harmonics=harmonics,
        k_list=k_list,
        omega_root=omega_root,
        wmin=wmin,
        wmax=wmax,
    )

    f_root = omega_root/(2 * np.pi)
    f_plot = omega_plot/(2 * np.pi)
    f_evp = omega_evp/(2 * np.pi)

    fig, ax = plt.subplots(figsize=(6.8, 3.3))

    ax.plot(k_list, f_root, lw=1.7, label="root (Newton)")
    ax.plot(out["k_tot"], f_plot, lw=1.5, label="expansion (sigmoid mix)")
    ax.plot(k_list[::100], f_evp[::100], ".", color='k', ms=2.0, label="EVP (selected)")

    ax.grid(True)
    ax.set_xlabel(r"$k\ \mathrm{[1/m]}$")
    ax.set_ylabel(r'$f\ \mathrm{[GHz]}$')
    ax.set_xlim(0.0, k_max)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
