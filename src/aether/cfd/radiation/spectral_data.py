r"""Spectroscopic constants and transition data for 5-species air.

Molecular band systems
----------------------

Each band system is defined by two electronic states with Morse-potential
spectroscopic constants.  Franck–Condon factors are computed from numerical
overlap of Morse wavefunctions rather than tabulated, so any (v', v'')
band can be evaluated on demand.

The four systems that matter for visible/near-UV emission at 8 000–15 000 K:

* **N₂ first positive** B³Πg → A³Σu⁺ : 500–1100 nm, orange/red
* **N₂ second positive** C³Πu → B³Πg : 280–550 nm, violet/blue
* **NO β** B²Π → X²Π : 190–370 nm, UV
* **NO γ** A²Σ⁺ → X²Π : 215–340 nm, UV/violet edge

Atomic lines
------------

NIST Atomic Spectra Database values for N I and O I transitions in the
visible and near-IR.  Each line carries wavelength, Einstein A coefficient,
upper-level energy, and statistical weight.

CIE colour matching
-------------------

The 1931 2° observer at 5 nm spacing, enough for the broadband emission
we integrate.  Converts spectral radiance I(λ) to XYZ, then to sRGB.

References
----------

Laux, C.O. (1993). Optical diagnostics and radiative emission of air
    plasmas. PhD thesis, Stanford University.
Babou, Y. et al. (2009). JQSRT 110, 89–108.
Park, C. (1990). Nonequilibrium Hypersonic Aerothermodynamics. Wiley.
Nicholls, R.W. (1961). JQSRT 1, 98–116.
Kramida, A. et al. NIST Atomic Spectra Database (ver. 5.11).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AtomicLine",
    "BandSystem",
    "ElectronicState",
    "N2_FIRST_POS",
    "N2_SECOND_POS",
    "NO_BETA",
    "NO_GAMMA",
    "N_LINES",
    "O_LINES",
    "ALL_BAND_SYSTEMS",
    "franck_condon",
    "band_origin_cm",
    "cie_xyz",
    "xyz_to_srgb",
]

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
_H = 6.62607015e-34       # J·s
_C = 2.99792458e8          # m/s
_KB = 1.380649e-23         # J/K
_NA = 6.02214076e23        # mol⁻¹
_EV = 1.602176634e-19      # J/eV
_AMU = 1.66053906660e-27   # kg

# ---------------------------------------------------------------------------
# Molecular spectroscopic constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElectronicState:
    """Morse-potential spectroscopic constants for one electronic state."""
    name: str
    Te: float          # cm⁻¹, electronic term energy
    omega_e: float     # cm⁻¹, harmonic vibrational frequency
    omega_xe: float    # cm⁻¹, first anharmonicity
    Be: float          # cm⁻¹, rotational constant
    re: float          # Å, equilibrium internuclear distance
    g_e: int           # electronic degeneracy (2S+1)*(2-delta_{Lambda,0})


# ---- N2 electronic states ----
N2_X = ElectronicState("N2 X¹Σg⁺", Te=0.0, omega_e=2358.57, omega_xe=14.324,
                        Be=1.99824, re=1.0977, g_e=1)
N2_A = ElectronicState("N2 A³Σu⁺", Te=50203.6, omega_e=1460.64, omega_xe=13.872,
                        Be=1.4546, re=1.2866, g_e=3)
N2_B = ElectronicState("N2 B³Πg", Te=59306.8, omega_e=1733.39, omega_xe=14.122,
                        Be=1.6375, re=1.2126, g_e=6)
N2_C = ElectronicState("N2 C³Πu", Te=89136.9, omega_e=2035.1, omega_xe=17.08,
                        Be=1.8247, re=1.1487, g_e=6)

# ---- NO electronic states ----
NO_X = ElectronicState("NO X²Π", Te=0.0, omega_e=1904.20, omega_xe=14.075,
                        Be=1.6720, re=1.1508, g_e=4)
NO_A = ElectronicState("NO A²Σ⁺", Te=43965.7, omega_e=2374.31, omega_xe=16.106,
                        Be=1.9965, re=1.0637, g_e=2)
NO_B = ElectronicState("NO B²Π", Te=45492.4, omega_e=1037.18, omega_xe=7.596,
                        Be=1.0707, re=1.4516, g_e=4)


@dataclass(frozen=True)
class BandSystem:
    """A molecular electronic transition (band system)."""
    name: str
    species: str       # "N2" or "NO"
    upper: ElectronicState
    lower: ElectronicState
    Re_sq: float       # |Re|² electronic transition moment squared (D²)
    v_max_upper: int   # max vibrational quantum number to include
    v_max_lower: int


N2_FIRST_POS = BandSystem(
    name="N2 1+", species="N2",
    upper=N2_B, lower=N2_A,
    Re_sq=3.4,   # Debye², Laux (1993) Table 4.1
    v_max_upper=13, v_max_lower=18,
)

N2_SECOND_POS = BandSystem(
    name="N2 2+", species="N2",
    upper=N2_C, lower=N2_B,
    Re_sq=0.52,  # Debye²
    v_max_upper=5, v_max_lower=8,
)

NO_BETA = BandSystem(
    name="NO β", species="NO",
    upper=NO_B, lower=NO_X,
    Re_sq=1.9,  # Debye²
    v_max_upper=8, v_max_lower=15,
)

NO_GAMMA = BandSystem(
    name="NO γ", species="NO",
    upper=NO_A, lower=NO_X,
    Re_sq=0.53,  # Debye²
    v_max_upper=5, v_max_lower=10,
)

ALL_BAND_SYSTEMS = [N2_FIRST_POS, N2_SECOND_POS, NO_BETA, NO_GAMMA]

# ---------------------------------------------------------------------------
# Molar masses for number density conversion (kg/mol)
# ---------------------------------------------------------------------------
MOLAR_MASS = {"N2": 28.014e-3, "O2": 31.998e-3, "NO": 30.006e-3,
              "N": 14.007e-3, "O": 15.999e-3}

# ---------------------------------------------------------------------------
# Vibrational term values G(v) = ωe(v+½) - ωeχe(v+½)²
# ---------------------------------------------------------------------------

def _G(state: ElectronicState, v: int) -> float:
    """Vibrational term value G(v) in cm⁻¹."""
    vph = v + 0.5
    return state.omega_e * vph - state.omega_xe * vph * vph


def band_origin_cm(system: BandSystem, v_upper: int, v_lower: int) -> float:
    """Band origin ν₀ in cm⁻¹ for the (v', v'') vibronic transition."""
    dTe = system.upper.Te - system.lower.Te
    return dTe + _G(system.upper, v_upper) - _G(system.lower, v_lower)


def band_origin_nm(system: BandSystem, v_upper: int, v_lower: int) -> float:
    """Band origin wavelength in nm."""
    nu_cm = band_origin_cm(system, v_upper, v_lower)
    if nu_cm <= 0:
        return np.inf
    return 1e7 / nu_cm


# ---------------------------------------------------------------------------
# Franck–Condon factors from Morse wavefunctions
# ---------------------------------------------------------------------------

def _morse_dvr(
    state: ElectronicState,
    v_max: int,
    mu_kg: float,
    n_grid: int = 500,
    r_range: tuple[float, float] = (0.5, 5.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Solve the Morse oscillator via finite-difference DVR.

    Returns (eigenvalues_J, eigenvectors[:, v], r_m) for v = 0..v_max.
    Numerically stable — no Numerov turning-point issues.
    """
    from scipy.linalg import eigh_tridiagonal

    De_cm = state.omega_e**2 / (4.0 * state.omega_xe)
    De_J = De_cm * _H * _C * 100.0
    beta = state.omega_e * 2.0 * np.pi * _C * 100.0 * np.sqrt(mu_kg / (2.0 * De_J))
    re_m = state.re * 1e-10

    r_m = np.linspace(r_range[0] * 1e-10, r_range[1] * 1e-10, n_grid)
    dr = r_m[1] - r_m[0]

    exponent = np.clip(-beta * (r_m - re_m), -500.0, 500.0)
    V = De_J * (1.0 - np.exp(exponent))**2
    np.minimum(V, 10.0 * De_J, out=V)

    hbar = _H / (2.0 * np.pi)
    T_coeff = hbar**2 / (2.0 * mu_kg * dr**2)

    diag = 2.0 * T_coeff + V
    off_diag = np.full(n_grid - 1, -T_coeff)

    n_states = min(v_max + 1, n_grid - 2)
    eigenvalues, eigenvectors = eigh_tridiagonal(
        diag, off_diag, select="i", select_range=(0, n_states - 1),
    )

    for i in range(eigenvectors.shape[1]):
        norm = np.sqrt(np.trapezoid(eigenvectors[:, i]**2, r_m))
        if norm > 0:
            eigenvectors[:, i] /= norm

    return eigenvalues, eigenvectors, r_m


_DVR_CACHE: dict[tuple[str, str], tuple[NDArray, NDArray, NDArray]] = {}


def _get_dvr(state: ElectronicState, species: str, v_max: int):
    """Cached DVR solve for an electronic state."""
    key = (state.name, species)
    if key not in _DVR_CACHE:
        if species == "N2":
            mu_kg = 14.007e-3 / _NA / 2.0
        else:
            mu_kg = (14.007e-3 * 15.999e-3) / ((14.007e-3 + 15.999e-3) * _NA)
        _DVR_CACHE[key] = _morse_dvr(state, v_max, mu_kg)
    return _DVR_CACHE[key]


@lru_cache(maxsize=512)
def franck_condon(
    system_name: str,
    v_upper: int,
    v_lower: int,
) -> float:
    """Franck–Condon factor q(v', v'') for a vibronic transition.

    Computed from the squared overlap integral of Morse-DVR wavefunctions.
    """
    systems = {s.name: s for s in ALL_BAND_SYSTEMS}
    system = systems[system_name]

    _, psi_up, r_up = _get_dvr(system.upper, system.species,
                                max(v_upper, system.v_max_upper))
    _, psi_lo, r_lo = _get_dvr(system.lower, system.species,
                                max(v_lower, system.v_max_lower))

    if v_upper >= psi_up.shape[1] or v_lower >= psi_lo.shape[1]:
        return 0.0

    r_common = np.linspace(
        max(r_up[0], r_lo[0]), min(r_up[-1], r_lo[-1]), 2000,
    )
    psi_u_interp = np.interp(r_common, r_up, psi_up[:, v_upper])
    psi_l_interp = np.interp(r_common, r_lo, psi_lo[:, v_lower])

    overlap = np.trapezoid(psi_u_interp * psi_l_interp, r_common)
    return float(overlap**2)


# ---------------------------------------------------------------------------
# Atomic line data (NIST ASD)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomicLine:
    """One atomic emission line."""
    species: str
    wavelength_nm: float
    A_ul: float          # Einstein A coefficient, s⁻¹
    E_upper_eV: float    # upper level energy
    g_upper: int         # statistical weight of upper level
    g_lower: int         # statistical weight of lower level

N_LINES: list[AtomicLine] = [
    # 4s ⁴P → 3p ⁴S multiplet (red/NIR)
    AtomicLine("N", 742.364, 2.76e7, 11.758, 6, 4),
    AtomicLine("N", 744.229, 1.93e7, 11.760, 4, 4),
    AtomicLine("N", 746.831, 6.29e6, 11.764, 2, 4),
    # 4s ⁴P → 3p ⁴D (NIR)
    AtomicLine("N", 818.487, 2.26e7, 12.000, 6, 8),
    AtomicLine("N", 821.634, 3.38e7, 12.000, 4, 6),
    AtomicLine("N", 824.239, 2.21e7, 11.996, 6, 4),
    # 3p ⁴D → 3s ⁴P (NIR)
    AtomicLine("N", 862.924, 1.74e7, 11.760, 4, 6),
    AtomicLine("N", 868.028, 2.60e7, 11.764, 2, 4),
    # visible
    AtomicLine("N", 493.503, 1.38e7, 13.653, 4, 6),
    AtomicLine("N", 500.515, 6.77e6, 13.627, 6, 8),
    AtomicLine("N", 567.956, 5.55e6, 12.976, 6, 4),
    AtomicLine("N", 575.250, 4.18e6, 12.971, 4, 4),
]

O_LINES: list[AtomicLine] = [
    # 3p ⁵P → 3s ⁵S (777 nm triplet — the strongest visible O line)
    AtomicLine("O", 777.194, 3.69e7, 10.741, 7, 5),
    AtomicLine("O", 777.417, 3.69e7, 10.741, 5, 3),
    AtomicLine("O", 777.539, 3.69e7, 10.741, 3, 1),
    # 3p ³P → 3s ³S
    AtomicLine("O", 844.636, 3.22e7, 10.989, 3, 5),
    # visible
    AtomicLine("O", 615.597, 1.07e7, 12.754, 5, 7),
    AtomicLine("O", 615.818, 8.91e6, 12.754, 3, 5),
    AtomicLine("O", 436.826, 2.79e6, 12.661, 5, 3),
]

ALL_ATOMIC_LINES: list[AtomicLine] = N_LINES + O_LINES


# ---------------------------------------------------------------------------
# Partition functions for N and O atoms (truncated, internal)
# ---------------------------------------------------------------------------

# Ground and low-lying electronic levels: (g_i, E_i in eV)
_N_LEVELS: list[tuple[int, float]] = [
    (4, 0.000),    # ⁴S₃/₂
    (10, 2.384),   # ²D
    (6, 3.576),    # ²P
    (12, 10.330),  # ⁴P
    (6, 10.689),   # ²P
    (10, 10.926),  # ²D
    (28, 11.600),  # ⁴D, ⁴P, ²D, etc. (merged high-lying)
    (20, 11.758),  # 4s ⁴P
    (20, 12.000),  # 4s levels
    (12, 12.976),  # higher
    (10, 13.627),  # higher
]

_O_LEVELS: list[tuple[int, float]] = [
    (9, 0.000),    # ³P₂,₁,₀ (J=2,1,0 → 5+3+1)
    (5, 1.967),    # ¹D₂
    (1, 4.190),    # ¹S₀
    (15, 9.146),   # ³S° + ⁵S°
    (15, 10.741),  # 3p ⁵P
    (3, 10.989),   # 3p ³P
    (20, 12.100),  # merged high-lying
    (12, 12.661),  # higher
    (8, 12.754),   # higher
]


def _partition_function(
    levels: list[tuple[int, float]], T: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Electronic partition function from explicit level list."""
    T = np.atleast_1d(np.asarray(T, dtype=np.float64))
    Z = np.zeros_like(T)
    for g, E_eV in levels:
        Z += g * np.exp(-E_eV * _EV / (_KB * T))
    return Z


def partition_N(T: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Electronic partition function for atomic nitrogen."""
    return _partition_function(_N_LEVELS, T)


def partition_O(T: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Electronic partition function for atomic oxygen."""
    return _partition_function(_O_LEVELS, T)


# ---------------------------------------------------------------------------
# CIE 1931 2° observer colour matching functions (5 nm spacing, 380–780 nm)
# ---------------------------------------------------------------------------

_CIE_WL = np.arange(380.0, 785.0, 5.0)  # 81 points

_CIE_X = np.array([
    0.001368, 0.002236, 0.004243, 0.00765, 0.01431,
    0.02319, 0.04351, 0.07763, 0.13438, 0.21477,
    0.2839, 0.3285, 0.34828, 0.34806, 0.3362,
    0.3187, 0.2908, 0.2511, 0.19536, 0.1421,
    0.09564, 0.05795, 0.03201, 0.0147, 0.0049,
    0.0024, 0.0093, 0.0291, 0.06327, 0.1096,
    0.1655, 0.22575, 0.2904, 0.3597, 0.43345,
    0.51205, 0.5945, 0.6784, 0.7621, 0.8425,
    0.9163, 0.9786, 1.0263, 1.0567, 1.0622,
    1.0456, 1.0026, 0.9384, 0.85445, 0.7514,
    0.6424, 0.5419, 0.4479, 0.3608, 0.2835,
    0.2187, 0.1649, 0.1212, 0.0874, 0.0636,
    0.04677, 0.0329, 0.0227, 0.01584, 0.01136,
    0.00811, 0.00579, 0.004109, 0.002899, 0.002049,
    0.001440, 0.001000, 0.000690, 0.000476, 0.000332,
    0.000235, 0.000166, 0.000117, 0.0000830, 0.0000590,
    0.0000420,
])

_CIE_Y = np.array([
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396,
    0.000640, 0.001210, 0.002180, 0.004000, 0.007300,
    0.011600, 0.016840, 0.023000, 0.029800, 0.038000,
    0.048000, 0.060000, 0.073900, 0.090980, 0.112600,
    0.139020, 0.169300, 0.208020, 0.258600, 0.323000,
    0.407300, 0.503000, 0.608200, 0.710000, 0.793200,
    0.862000, 0.914850, 0.954000, 0.980300, 0.994950,
    1.000100, 0.995000, 0.978600, 0.952000, 0.915400,
    0.870000, 0.816300, 0.757000, 0.694900, 0.631000,
    0.566800, 0.503000, 0.441200, 0.381000, 0.321000,
    0.265000, 0.217000, 0.175000, 0.138200, 0.107000,
    0.081600, 0.061000, 0.044580, 0.032000, 0.023200,
    0.017000, 0.011920, 0.008210, 0.005723, 0.004102,
    0.002929, 0.002091, 0.001484, 0.001047, 0.000740,
    0.000520, 0.000361, 0.000249, 0.000172, 0.000120,
    0.000085, 0.000060, 0.000042, 0.000030, 0.000021,
    0.000015,
])

_CIE_Z = np.array([
    0.006450, 0.010550, 0.020050, 0.036210, 0.067850,
    0.110200, 0.207400, 0.371300, 0.645600, 1.039050,
    1.385600, 1.622960, 1.747060, 1.782600, 1.772110,
    1.744100, 1.669200, 1.528100, 1.287640, 1.041900,
    0.812950, 0.616200, 0.465180, 0.353300, 0.272000,
    0.212300, 0.158200, 0.111700, 0.078250, 0.057250,
    0.042160, 0.029840, 0.020300, 0.013400, 0.008750,
    0.005750, 0.003900, 0.002750, 0.002100, 0.001800,
    0.001650, 0.001400, 0.001100, 0.001000, 0.000800,
    0.000600, 0.000340, 0.000240, 0.000190, 0.000100,
    0.000050, 0.000030, 0.000020, 0.000010, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000,
])


def cie_xyz(
    wavelength_nm: NDArray[np.float64],
    spectral_radiance: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Integrate spectral radiance against CIE 1931 colour matching functions.

    Parameters
    ----------
    wavelength_nm : (N,) array, nm
    spectral_radiance : (N,) array, W/m²/sr/nm or proportional

    Returns
    -------
    (X, Y, Z) tristimulus values (unnormalised)
    """
    x_bar = np.interp(wavelength_nm, _CIE_WL, _CIE_X, left=0.0, right=0.0)
    y_bar = np.interp(wavelength_nm, _CIE_WL, _CIE_Y, left=0.0, right=0.0)
    z_bar = np.interp(wavelength_nm, _CIE_WL, _CIE_Z, left=0.0, right=0.0)
    dw = np.gradient(wavelength_nm)
    X = float(np.sum(spectral_radiance * x_bar * dw))
    Y = float(np.sum(spectral_radiance * y_bar * dw))
    Z = float(np.sum(spectral_radiance * z_bar * dw))
    return X, Y, Z


def camera_rgb(
    wavelength_nm: NDArray[np.float64],
    spectral_radiance: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Integrate spectral radiance against a camera sensor model.

    Uses Gaussian approximations to typical CMOS sensor spectral response,
    which extends further into the NIR than the CIE 1931 observer.  This
    captures the deep-red emission from atomic N (742 nm) and O (777 nm)
    that dominates reentry plasma but is nearly invisible to the CIE
    photopic standard.

    Returns unnormalised linear (R, G, B).
    """
    wl = wavelength_nm
    r_sens = (np.exp(-0.5 * ((wl - 610) / 60)**2)
              + 0.3 * np.exp(-0.5 * ((wl - 740) / 40)**2))
    g_sens = np.exp(-0.5 * ((wl - 540) / 50)**2)
    b_sens = np.exp(-0.5 * ((wl - 460) / 40)**2)

    dw = np.gradient(wavelength_nm)
    R = float(np.sum(spectral_radiance * r_sens * dw))
    G = float(np.sum(spectral_radiance * g_sens * dw))
    B = float(np.sum(spectral_radiance * b_sens * dw))
    return R, G, B


def xyz_to_srgb(X: float, Y: float, Z: float) -> tuple[float, float, float]:
    """CIE XYZ to linear sRGB (D65), values in [0, ∞)."""
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    return r, g, b


def srgb_gamma(linear: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply sRGB gamma curve (linear → display)."""
    out = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )
    return np.clip(out, 0.0, 1.0)
