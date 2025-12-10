"""
This code computes the k(f) roots based on the expansion method.

"""
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt


# Helper functions for characteristic frequencies and velocities.
def wp_f(n_m3, q_C, m_kg):
    return np.sqrt(n_m3 * q_C * q_C / (m_kg * sc.epsilon_0))

def wc_f(B_T, q_C, m_kg):
    return q_C * B_T / m_kg

def vth_f(T_eV, m_kg):
    return np.sqrt(2.0 * T_eV * sc.e / m_kg)

# Helper function to select the dispersion branch for the desired species.
def _get_species(species, name):
    for sp in species:
        if sp["name"] == name:
            return sp
    raise ValueError(f"Species '{name}' not found in species list.")


# Helper function to compute the Stix S coefficient.
def stix_S(omega_rad_s, B_T, species):
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    S = np.ones_like(omega)
    for sp in species:
        wpe = wp_f(float(sp["n_m3"]), float(sp["q_C"]), float(sp["m_kg"]))
        wc = abs(wc_f(B_T, float(sp["q_C"]), float(sp["m_kg"])))
        S -= (wpe * wpe) / (omega * omega - wc * wc)
    return S

# Helper function to compute the hot correction factor.
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

# This function computes the expanded expression valid for small k.
def k_expansion_from_S_ellT2(S, ellT2):
    S = np.asarray(S, dtype=np.float64)
    ellT2 = np.asarray(ellT2, dtype=np.float64)
    k2 = -S / ellT2
    k = np.where(k2 > 0.0, np.sqrt(k2), np.nan)
    return k

# This function computes the asymptotic expression valid for large k.
def k_asymptotic(omega_rad_s, B_T, sp):
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    # Always set the harmonic to the fundamental
    n = 1

    wpe = wp_f(float(sp["n_m3"]), float(sp["q_C"]), float(sp["m_kg"]))
    wc = abs(wc_f(B_T, float(sp["q_C"]), float(sp["m_kg"])))
    vth = vth_f(float(sp["T_eV"]), float(sp["m_kg"]))
    rhoL = vth / wc

    delta = omega - n * wc
    C = (np.sqrt(np.pi) * wc) / (2.0 * n * (wpe * wpe))
    base = C * delta
    k = (1.0 / rhoL) * np.power(base, -1.0 / 3.0)
    k = np.where(base > 0.0, k, np.nan)
    return k

# The weight function to stitch the 
def sigmoid_weight(k_exp, rhoL_target, Delta, k_rho0=1.0):
    arg = (k_exp * rhoL_target - k_rho0) / Delta
    return 1.0 / (1.0 + np.exp(-arg))


# This function computes the three dispersion branches:
# The expanded one, the asymptotic one, and the stitched one.
def complete_dispersion_relation(
    omega_rad_s,
    B_T,
    species,
    target_species_name="e",
    Delta=0.175,
    k_rho0=1.0
):
    omega = np.asarray(omega_rad_s, dtype=np.float64)

    S = stix_S(omega, B_T, species)
    ellT2 = ell_T2(omega, B_T, species)
    k_exp = k_expansion_from_S_ellT2(S, ellT2)

    sp_tgt = _get_species(species, target_species_name)
    wc_tgt = abs(wc_f(B_T, float(sp_tgt["q_C"]), float(sp_tgt["m_kg"])))
    vth_tgt = vth_f(float(sp_tgt["T_eV"]), float(sp_tgt["m_kg"]))
    rhoL_tgt = vth_tgt / wc_tgt

    k_asym = k_asymptotic(omega, B_T, sp_tgt)

    sigma = sigmoid_weight(k_exp, rhoL_tgt, Delta, k_rho0=k_rho0)
    k_tot = (1.0 - sigma) * np.abs(k_exp) + sigma * np.abs(k_asym)

    return {
        "k_tot": k_tot,
        "k_exp": k_exp,
        "k_asym": k_asym,
        "sigma": sigma,
        "S": S,
        "ellT2": ellT2
    }

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

    target_species_name = "e"
    Delta = 0.175
    k_rho0 = 1

    sp_tgt = _get_species(species, target_species_name)
    wc_tgt = abs(wc_f(B_T, float(sp_tgt["q_C"]), float(sp_tgt["m_kg"])))

    # The frequency range is selected to compute the k values
    # between the fundamental EC frequency and the second harmonic.
    f_min_GHz = wc_tgt / (2 * np.pi * 1e9)
    f_max_GHz = 2 * wc_tgt / (2.0 * np.pi * 1e9)
    Nw = 2000
    omega = 2.0 * np.pi * 1e9 * np.linspace(f_min_GHz, f_max_GHz, Nw)

    out = complete_dispersion_relation(
        omega_rad_s=omega,
        B_T=B_T,
        species=species,
        target_species_name=target_species_name,
        Delta=Delta,
        k_rho0=k_rho0
    )

    # Collecting all three dispersion branches and ploting them.
    k_tot = out["k_tot"]
    k_exp = out["k_exp"]
    k_asym = out["k_asym"]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(k_exp, omega / (2.0 * np.pi * 1e9), lw=1.2, ls="--", label="k_expans")
    ax.plot(k_asym, omega / (2.0 * np.pi * 1e9), lw=1.2, ls=":", label="k_asym")
    ax.plot(k_tot, omega / (2.0 * np.pi * 1e9), lw=1.6, label="k_tot")

    ax.grid(True)
    ax.set_xlabel("k [1/m]")
    ax.set_ylabel("f [GHz]")
    ax.legend()
    ax.set_xlim(0, 40e3)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
