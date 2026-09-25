r"""Emission coefficients for nonequilibrium 5-species air.

Computes the monochromatic emission coefficient j(λ) in W/m³/sr/nm at
every wavelength on a user-supplied spectral grid, given the local
thermochemical state (T_tr, T_ve, species number densities, pressure).

Molecular bands use the smeared rotational band (SRB) model: each
vibronic (v', v'') transition is represented as a Gaussian profile whose
centre is the band origin and whose width is set by the rotational
temperature, with the total power fixed by the Franck–Condon factor and
the electronic transition moment.

Atomic lines use a Voigt profile with Doppler broadening (thermal) and
Lorentzian broadening (collisional/Stark).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .spectral_data import (
    ALL_ATOMIC_LINES,
    ALL_BAND_SYSTEMS,
    MOLAR_MASS,
    AtomicLine,
    BandSystem,
    _G,
    _EV,
    _H,
    _C,
    _KB,
    _NA,
    _AMU,
    band_origin_cm,
    franck_condon,
    partition_N,
    partition_O,
)

__all__ = ["emission_spectrum", "total_emission"]

# Debye² → C²·m² conversion:  1 D = 3.33564e-30 C·m
_DEBYE_SI = 3.33564e-30      # C·m per Debye
_EPS0 = 8.8541878128e-12     # F/m, vacuum permittivity


def _molecular_vibrational_partition(state, T_ve: float) -> float:
    """Vibrational partition function Z_vib at T_ve (sum over bound levels)."""
    Z = 0.0
    De_cm = state.omega_e**2 / (4.0 * state.omega_xe)
    for v in range(200):
        Gv = _G(state, v)
        if Gv >= De_cm:
            break
        arg = -Gv * _H * _C * 100.0 / (_KB * max(T_ve, 100.0))
        if arg > -500.0:
            Z += np.exp(arg)
    return max(Z, 1e-300)


def _rotational_band_width_cm(state, T_tr: float) -> float:
    r"""Approximate FWHM of the rotational envelope in cm⁻¹.

    For a Σ state the rotational lines span roughly
    Δν ≈ 2√(2 B_v k_B T / h c) in the P/R branch envelope.
    This gives the Gaussian σ for the SRB model.
    """
    Bv = state.Be  # could subtract α_e*(v+½), but Be is good enough
    return 2.0 * np.sqrt(2.0 * Bv * _KB * T_tr / (_H * _C * 100.0))


def _band_emission(
    system: BandSystem,
    wavelength_nm: NDArray[np.float64],
    T_tr: float,
    T_ve: float,
    n_species: float,
) -> NDArray[np.float64]:
    r"""Spectral emission coefficient j(λ) from one molecular band system.

    Uses the SRB model: each (v',v'') band is a Gaussian centred on the
    band origin with width from rotational structure, and total integrated
    emission from the Einstein A coefficient derived via

        A(v',v'') = (64 π⁴ ν³)/(3 h c³) · |Re|² · q(v',v'')

    The upper-state population follows Boltzmann at T_ve for both
    electronic and vibrational degrees of freedom, consistent with
    Park's two-temperature model.
    """
    j = np.zeros_like(wavelength_nm)
    if n_species <= 0 or T_ve < 300:
        return j

    Z_vib_upper = _molecular_vibrational_partition(system.upper, T_ve)
    Te_upper_J = system.upper.Te * _H * _C * 100.0
    Te_lower_J = system.lower.Te * _H * _C * 100.0

    g_ratio = system.upper.g_e / max(system.lower.g_e, 1)

    Z_el = system.lower.g_e * np.exp(-Te_lower_J / (_KB * T_ve))
    Z_el += system.upper.g_e * np.exp(-Te_upper_J / (_KB * T_ve))
    Z_el += 1e-300

    n_electronic_upper = (
        n_species * system.upper.g_e
        * np.exp(-Te_upper_J / (_KB * T_ve)) / Z_el
    )
    if n_electronic_upper <= 0:
        return j

    sigma_rot_cm = max(_rotational_band_width_cm(system.upper, T_tr), 1.0)
    sigma_rot_nm_factor = 1e7  # cm⁻¹ → nm conversion depends on λ

    Re_sq_SI = system.Re_sq * _DEBYE_SI**2

    De_upper_cm = system.upper.omega_e**2 / (4.0 * system.upper.omega_xe)

    for vp in range(system.v_max_upper + 1):
        Gp = _G(system.upper, vp)
        if Gp >= De_upper_cm:
            break

        pop_vp = np.exp(-Gp * _H * _C * 100.0 / (_KB * T_ve)) / Z_vib_upper
        if pop_vp < 1e-15:
            continue

        for vpp in range(system.v_max_lower + 1):
            nu_cm = band_origin_cm(system, vp, vpp)
            if nu_cm <= 0:
                continue

            lam_nm = 1e7 / nu_cm
            if lam_nm < wavelength_nm[0] - 50 or lam_nm > wavelength_nm[-1] + 50:
                continue

            qvv = franck_condon(system.name, vp, vpp)
            if qvv < 1e-6:
                continue

            nu_hz = nu_cm * _C * 100.0
            A_vv = (16.0 * np.pi**3 * nu_hz**3) / (3.0 * _EPS0 * _H * _C**3) * Re_sq_SI * qvv

            sigma_nm = sigma_rot_cm * lam_nm**2 / 1e7
            sigma_nm = max(sigma_nm, 0.5)

            power_per_mol = n_electronic_upper * pop_vp * A_vv * _H * nu_hz / (4.0 * np.pi)

            profile = np.exp(-0.5 * ((wavelength_nm - lam_nm) / sigma_nm)**2)
            profile /= max(np.sum(profile) * (wavelength_nm[1] - wavelength_nm[0]), 1e-300)

            j += power_per_mol * profile

    return j


def _atomic_emission(
    line: AtomicLine,
    wavelength_nm: NDArray[np.float64],
    T_excitation: float,
    T_kinetic: float,
    n_atom: float,
    pressure: float,
) -> NDArray[np.float64]:
    """Spectral emission from one atomic line with Voigt profile.

    T_excitation sets the upper-level population (T_ve in Park's model).
    T_kinetic sets Doppler broadening width (T_tr).
    """
    j = np.zeros_like(wavelength_nm)
    if n_atom <= 0 or T_excitation < 300:
        return j

    if line.species == "N":
        Z = float(np.squeeze(partition_N(T_excitation)))
        mass_kg = 14.007e-3 / _NA
    else:
        Z = float(np.squeeze(partition_O(T_excitation)))
        mass_kg = 15.999e-3 / _NA

    n_upper = n_atom * line.g_upper * np.exp(-line.E_upper_eV * _EV / (_KB * T_excitation)) / Z

    if n_upper <= 0:
        return j

    nu_hz = _C / (line.wavelength_nm * 1e-9)
    power = n_upper * line.A_ul * _H * nu_hz / (4.0 * np.pi)

    sigma_D_hz = nu_hz * np.sqrt(_KB * T_kinetic / (mass_kg * _C**2))
    sigma_D_nm = line.wavelength_nm * np.sqrt(_KB * T_kinetic / (mass_kg * _C**2))

    n_total = pressure / (_KB * T_kinetic) if T_kinetic > 0 else 0
    gamma_L_hz = n_total * 1e-19 * np.sqrt(8 * _KB * T_kinetic / (np.pi * mass_kg))
    gamma_L_nm = gamma_L_hz * line.wavelength_nm / nu_hz

    sigma_V = np.sqrt(sigma_D_nm**2 + (gamma_L_nm / 2.355)**2)
    sigma_V = max(sigma_V, 0.01)

    profile = np.exp(-0.5 * ((wavelength_nm - line.wavelength_nm) / sigma_V)**2)
    area = np.sum(profile) * (wavelength_nm[1] - wavelength_nm[0])
    if area > 0:
        profile /= area

    j += power * profile
    return j


def emission_spectrum(
    wavelength_nm: NDArray[np.float64],
    T_tr: float,
    T_ve: float,
    n_N2: float,
    n_O2: float,
    n_NO: float,
    n_N: float,
    n_O: float,
    pressure: float,
) -> NDArray[np.float64]:
    r"""Total spectral emission coefficient j(λ) in W/m³/sr/nm.

    Parameters
    ----------
    wavelength_nm : (N,) grid of wavelengths in nm
    T_tr : translational-rotational temperature, K
    T_ve : vibrational-electronic temperature, K
    n_N2, n_O2, n_NO, n_N, n_O : number densities, m⁻³
    pressure : Pa

    Returns
    -------
    j : (N,) emission coefficient at each wavelength
    """
    j = np.zeros_like(wavelength_nm)

    species_map = {"N2": n_N2, "O2": n_O2, "NO": n_NO}
    for system in ALL_BAND_SYSTEMS:
        n_s = species_map.get(system.species, 0.0)
        if n_s > 0:
            j += _band_emission(system, wavelength_nm, T_tr, T_ve, n_s)

    atom_map = {"N": n_N, "O": n_O}
    for line in ALL_ATOMIC_LINES:
        n_a = atom_map.get(line.species, 0.0)
        if n_a > 0:
            j += _atomic_emission(line, wavelength_nm, T_ve, T_tr, n_a, pressure)

    return j


def total_emission(
    wavelength_nm: NDArray[np.float64],
    T_tr: float,
    T_ve: float,
    rho_species: NDArray[np.float64],
    pressure: float,
) -> NDArray[np.float64]:
    r"""Total emission from partial densities (kg/m³) in SU2 NEMO order.

    Parameters
    ----------
    rho_species : (5,) partial densities [N2, O2, NO, N, O] in kg/m³
    """
    names = ["N2", "O2", "NO", "N", "O"]
    n = np.array([rho_species[i] / (MOLAR_MASS[names[i]] / _NA)
                  for i in range(5)])
    return emission_spectrum(wavelength_nm, T_tr, T_ve,
                             n[0], n[1], n[2], n[3], n[4], pressure)
