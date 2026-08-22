"""Stagnation-point convective heating (Paper II, §4.1).

Fay–Riddell (Eq. 4.1) with the terms Paper II insists on specifying
because the equation "is frequently quoted without them": the Lewis
exponent is an explicit argument with the two physically distinct values
named (0.52 equilibrium boundary layer, 0.63 frozen with fully catalytic
wall), the dissociation enthalpy enters through the bracket, and the
stagnation velocity gradient comes from the modified Newtonian estimate
(Eq. 4.2) — which is where recession feeds back: growing
:math:`R_{\\mathrm{eff}}` reduces convective heating as
:math:`R_{\\mathrm{eff}}^{-1/2}`.

The Sutton–Graves correlation is provided as the screening fallback the
Remark in §4.1 describes, as a *separate function*: it is a correlation,
not a theory, and results obtained with it must not be reported as
Fay–Riddell results.

Wall catalycity is a bracket, not an exponent
---------------------------------------------

Fay & Riddell give **three** correlations, not two — Anderson
Eqs. (17.89)–(17.91) sets them out side by side. The first two differ only
in the Lewis exponent (0.52 equilibrium, 0.63 frozen with an equilibrium
catalytic wall) and share the bracket
:math:`1 + (Le^{\\beta}-1)h_D/h_{0e}`. The third — a frozen boundary layer
over a **noncatalytic** wall — has a structurally different bracket,

.. math:: 1 - h_D/h_{0e}

which no choice of :math:`\\beta` can reach: matching it would need
:math:`Le^{\\beta} = 0`. Carrying catalycity as an exponent therefore
cannot express the noncatalytic case at all, which is why
:class:`WallCatalycity` exists and why ``lewis_exponent`` is now derived
from it rather than passed alongside it.

The distinction is worth this much machinery because it is large. Anderson
(Fig. 17.5, curve 2) reports heat transfer dropping **by more than a factor
of two** between a catalytic and a noncatalytic wall as the boundary layer
goes from equilibrium to frozen, and PICA's charred carbon surface is not
fully catalytic. Assuming a catalytic wall is conservative for sizing, but
conservative by a factor that should be stated rather than absorbed.

The two brackets meet where the physics says they should: with no
dissociation (:math:`h_D = 0`) both reduce to 1, because catalycity can
only matter if there is something to recombine.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "FAY_RIDDELL_COEFFICIENT_EXACT_094",
    "FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT",
    "FAY_RIDDELL_COEFFICIENT_LITERATURE",
    "FAY_RIDDELL_COEFFICIENT_SOURCE",
    "FAY_RIDDELL_FACTOR_PR071",
    "FAY_RIDDELL_NUSSELT_COEFFICIENT",
    "LEWIS_EXPONENT_EQUILIBRIUM",
    "LEWIS_EXPONENT_FROZEN_CATALYTIC",
    "SUTTON_GRAVES_EARTH",
    "WallCatalycity",
    "catalycity_bracket",
    "fay_riddell",
    "newtonian_velocity_gradient",
    "sutton_graves",
]

_FloatArray = NDArray[np.float64]


#: Lewis exponent for an equilibrium boundary layer — Fay & Riddell Eq. (60)
#: and (62), verified against the source PDF in ``reference/``.
LEWIS_EXPONENT_EQUILIBRIUM = 0.52
#: Lewis exponent for a frozen boundary layer with a fully catalytic wall —
#: Fay & Riddell Eq. (65) and Fig. 4.
LEWIS_EXPONENT_FROZEN_CATALYTIC = 0.63


class WallCatalycity(Enum):
    """Which of Fay & Riddell's three correlations applies.

    Selecting the correlation by name rather than by exponent keeps the
    noncatalytic case reachable — its bracket is not of the same form, so
    an exponent alone cannot express it (see the module docstring).

    Members
    -------
    EQUILIBRIUM:
        Equilibrium boundary layer, :math:`Le^{0.52}` — Anderson
        Eq. (17.89), Fay & Riddell Eqs. (60) and (62).
    FROZEN_CATALYTIC:
        Frozen boundary layer over an equilibrium catalytic wall,
        :math:`Le^{0.63}` — Anderson Eq. (17.90), Fay & Riddell Eq. (65).
    FROZEN_NONCATALYTIC:
        Frozen boundary layer over a noncatalytic wall — Anderson
        Eq. (17.91). Bracket :math:`1 - h_D/h_{0e}`, with no Lewis
        dependence at all: if the wall will not recombine atoms and the
        boundary layer will not either, the dissociation enthalpy simply
        never reaches the surface, and how fast species *would* diffuse
        stops mattering.
    """

    EQUILIBRIUM = "equilibrium"
    FROZEN_CATALYTIC = "frozen-catalytic"
    FROZEN_NONCATALYTIC = "frozen-noncatalytic"

    @property
    def lewis_exponent(self) -> float | None:
        """:math:`\\beta_{Le}`, or ``None`` where the Lewis number drops out."""
        if self is WallCatalycity.EQUILIBRIUM:
            return LEWIS_EXPONENT_EQUILIBRIUM
        if self is WallCatalycity.FROZEN_CATALYTIC:
            return LEWIS_EXPONENT_FROZEN_CATALYTIC
        return None

#: Leading coefficient of Fay & Riddell Eq. (63).
#:
#: Reconstructing the modern form from the source settles the exponents but
#: **not** the constant to the precision it is usually quoted at.
#:
#: The source's Eq. (45) defines the flux as
#: :math:`q = (Nu/\sqrt{Re})\sqrt{\rho_w \mu_w (du_e/dx)_s}\,(h_s-h_w)/\sigma`
#: and its Eq. (62) correlates
#: :math:`Nu/\sqrt{Re} = 0.67(\rho_s\mu_s/\rho_w\mu_w)^{0.4}
#: \{1 + (Le^{0.52}-1)h_D/h_s\}`. Substituting one into the other *derives*
#: the familiar split exponents — the ratio contributes :math:`0.4` and the
#: :math:`\sqrt{\rho_w\mu_w}` contributes :math:`0.5`, leaving
#: :math:`(\rho_e\mu_e)^{0.4}(\rho_w\mu_w)^{0.1}`. That assignment is
#: derived here, not assumed; the archived scan of Eq. (63) is too degraded
#: to read the exponents off directly.
#:
#: The same substitution shows where every constant in the paper comes from,
#: and it is **one number carried to two significant figures**: the 0.67 of
#: Eq. (58)/(62). The printed 0.94 of Eq. (63) is just
#: :math:`0.67/0.71 = 0.9437` rounded. The footnote's :math:`\Pr^{-0.6}` is
#: the :math:`\Pr^{0.4}` of the Nusselt correlation (stated where Cohen &
#: Reshotko are corrected "by multiplying their result by
#: :math:`(0.71)^{0.4}`") against the :math:`\Pr^{-1}` of Eq. (45).
#:
#: Backing the Eq.-(63) constant out of the paper's own three statements
#: gives three different answers, spanning 1.1%:
#:
#: ===========================  ==========================  ========
#: statement                    back-substitution           constant
#: ===========================  ==========================  ========
#: Eq. (62), factor 0.67        :math:`0.67\,\Pr^{-0.4}`      0.7684
#: Eq. (63), factor 0.94        :math:`0.94\,\Pr^{0.6}`       0.7654
#: footnote to Eq. (63)         as printed                    0.76
#: ===========================  ==========================  ========
#:
#: The 0.763 in general circulation (and in Paper II, Eq. 4.1) appears
#: nowhere in the source, but it sits inside that band and is as defensible
#: as any of them. No reading supports a third significant figure. All four
#: are named so the choice is explicit rather than inherited.
FAY_RIDDELL_COEFFICIENT_SOURCE = 0.76
"""As printed in the footnote to Fay & Riddell Eq. (63)."""
FAY_RIDDELL_COEFFICIENT_EXACT_094 = 0.7654
"""Back-substituted from the printed 0.94 of Eq. (63) at Pr = 0.71."""
FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT = 0.7684
"""Back-substituted from the 0.67 of Eq. (58)/(62) through Eq. (45)."""
FAY_RIDDELL_COEFFICIENT_LITERATURE = 0.763
"""The widely quoted value; not in the source, but inside its own spread."""
#: Eq. (63) as printed, valid at Pr = 0.71 only.
FAY_RIDDELL_FACTOR_PR071 = 0.94
#: Eq. (58)/(62): the one fitted constant everything else descends from.
FAY_RIDDELL_NUSSELT_COEFFICIENT = 0.67
#: Sutton–Graves constant for Earth air, SI units (Paper II, §4.1 Remark).
SUTTON_GRAVES_EARTH = 1.7415e-4


def _positive(value: ArrayLike, name: str) -> _FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be finite and > 0")
    return arr


def newtonian_velocity_gradient(
    effective_radius: ArrayLike,
    stagnation_pressure: ArrayLike,
    freestream_pressure: ArrayLike,
    stagnation_density: ArrayLike,
) -> _FloatArray:
    """Stagnation velocity gradient :math:`(du_e/dx)_s` (Paper II, Eq. 4.2).

    .. math::

        \\left(\\frac{du_e}{dx}\\right)_s = \\frac{1}{R_{\\mathrm{eff}}}
        \\sqrt{\\frac{2(p_s - p_\\infty)}{\\rho_s}} .

    Units: 1/s. This is the ablation coupling point — recession updates
    :math:`R_{\\mathrm{eff}}` here within the same step.
    """
    r_eff = _positive(effective_radius, "effective_radius")
    p_s = _positive(stagnation_pressure, "stagnation_pressure")
    p_inf = np.asarray(freestream_pressure, dtype=np.float64)
    rho_s = _positive(stagnation_density, "stagnation_density")
    if np.any(p_inf < 0.0) or not np.all(np.isfinite(p_inf)):
        raise ValueError("freestream_pressure must be finite and >= 0")
    if np.any(p_s <= p_inf):
        raise ValueError(
            "stagnation pressure must exceed freestream pressure; the modified "
            "Newtonian estimate has no meaning otherwise"
        )
    return np.asarray(np.sqrt(2.0 * (p_s - p_inf) / rho_s) / r_eff)


def catalycity_bracket(
    dissociation_enthalpy: ArrayLike,
    total_enthalpy_edge: ArrayLike,
    catalycity: WallCatalycity = WallCatalycity.EQUILIBRIUM,
    lewis: float = 1.4,
) -> _FloatArray:
    """The chemical-energy bracket of Fay & Riddell, by wall condition.

    .. math::

        \\text{catalytic:}\\quad & 1 + (Le^{\\beta_{Le}} - 1)\\,h_D/h_{0e} \\\\
        \\text{noncatalytic:}\\quad & 1 - h_D/h_{0e}

    Separated from :func:`fay_riddell` because it is the whole of the
    catalycity effect: everything else in the correlation is identical
    across the three cases. Isolating it makes the factor-of-two Anderson
    reports directly inspectable rather than buried in a flux.

    Parameters
    ----------
    dissociation_enthalpy, total_enthalpy_edge:
        :math:`h_D` and :math:`h_{0e}` (J/kg). Their ratio is the fraction
        of the edge total enthalpy tied up as chemical energy in
        dissociated species.
    catalycity:
        Which correlation. See :class:`WallCatalycity`.
    lewis:
        :math:`Le \\approx 1.4` for air. Ignored for the noncatalytic case,
        where it does not appear.

    Returns
    -------
    numpy.ndarray
        The dimensionless bracket. Above 1 for a catalytic wall (Le > 1
        means atoms diffuse to the wall faster than heat conducts away
        from it, so recombination *adds* to the flux) and below 1 for a
        noncatalytic one.
    """
    h0e = _positive(total_enthalpy_edge, "total_enthalpy_edge")
    h_d = np.asarray(dissociation_enthalpy, dtype=np.float64)
    if np.any(h_d < 0.0) or np.any(h_d > h0e):
        raise ValueError("dissociation enthalpy must satisfy 0 <= h_D <= h_0e")
    if not (np.isfinite(lewis) and lewis > 0.0):
        raise ValueError(f"lewis must be finite and > 0, got {lewis}")

    fraction = h_d / h0e
    if catalycity is WallCatalycity.FROZEN_NONCATALYTIC:
        return np.asarray(1.0 - fraction)
    exponent = catalycity.lewis_exponent
    assert exponent is not None
    return np.asarray(1.0 + (lewis**exponent - 1.0) * fraction)


def fay_riddell(
    edge_density: ArrayLike,
    edge_viscosity: ArrayLike,
    wall_density: ArrayLike,
    wall_viscosity: ArrayLike,
    velocity_gradient: ArrayLike,
    total_enthalpy_edge: ArrayLike,
    wall_enthalpy: ArrayLike,
    dissociation_enthalpy: ArrayLike,
    prandtl: float = 0.71,
    lewis: float = 1.4,
    lewis_exponent: float | None = None,
    coefficient: float = FAY_RIDDELL_COEFFICIENT_LITERATURE,
    catalycity: WallCatalycity = WallCatalycity.EQUILIBRIUM,
) -> _FloatArray:
    """Fay–Riddell stagnation convective heat flux (Paper II, Eq. 4.1), W/m².

    .. math::

        \\dot q_{s} = 0.763\\,\\mathrm{Pr}^{-0.6}
        (\\rho_e \\mu_e)^{0.4} (\\rho_w \\mu_w)^{0.1}
        \\sqrt{(du_e/dx)_s}\\,(h_{0e} - h_w)
        \\Big[1 + \\big(Le^{\\beta_{Le}} - 1\\big) \\tfrac{h_D}{h_{0e}}\\Big].

    Parameters
    ----------
    edge_density, edge_viscosity:
        Boundary-layer edge :math:`\\rho_e` (kg/m³), :math:`\\mu_e`
        (Pa·s).
    wall_density, wall_viscosity:
        Gas properties at wall temperature.
    velocity_gradient:
        :math:`(du_e/dx)_s` (1/s), from
        :func:`newtonian_velocity_gradient`.
    total_enthalpy_edge, wall_enthalpy:
        :math:`h_{0e}`, :math:`h_w` (J/kg), with
        :math:`h_{0e} > h_w` (heating, not cooling).
    dissociation_enthalpy:
        Free-stream mixture dissociation enthalpy :math:`h_D` (J/kg) at
        edge conditions, :math:`0 \\le h_D \\le h_{0e}`.
    prandtl, lewis:
        :math:`\\Pr \\approx 0.71`, :math:`Le \\approx 1.4` for air.
    lewis_exponent:
        :math:`\\beta_{Le}`, overriding the value implied by
        ``catalycity``. Left as ``None`` normally — the exponent and the
        bracket form have to agree, and ``catalycity`` sets both. Supply it
        only to explore a value the three published correlations do not
        name, and not at all for a noncatalytic wall, where the Lewis
        number does not enter.
    coefficient:
        Leading constant, defaulting to the widely quoted 0.763. See
        :data:`FAY_RIDDELL_COEFFICIENT_SOURCE` for why the primary source
        supports either 0.76 or 0.7654 depending on which of its two
        mutually inconsistent statements is taken as authoritative, and
        why 0.763 is in neither.
    catalycity:
        Which of the three correlations applies — see
        :class:`WallCatalycity`. Defaults to the equilibrium boundary
        layer, which is what the framework assumed before the noncatalytic
        case was reachable at all.
    """
    if not (np.isfinite(prandtl) and prandtl > 0.0):
        raise ValueError(f"prandtl must be finite and > 0, got {prandtl}")
    if not (np.isfinite(lewis) and lewis > 0.0):
        raise ValueError(f"lewis must be finite and > 0, got {lewis}")
    if lewis_exponent is not None:
        if catalycity is WallCatalycity.FROZEN_NONCATALYTIC:
            raise ValueError(
                "a noncatalytic wall has no Lewis exponent: its bracket is "
                "1 - h_D/h_0e, with no Le dependence at all. Passing one "
                "would silently be ignored."
            )
        if not (np.isfinite(lewis_exponent) and 0.0 < lewis_exponent < 1.0):
            raise ValueError(f"lewis_exponent must be in (0, 1), got {lewis_exponent}")
    rho_mu_e = _positive(edge_density, "edge_density") * _positive(
        edge_viscosity, "edge_viscosity"
    )
    rho_mu_w = _positive(wall_density, "wall_density") * _positive(
        wall_viscosity, "wall_viscosity"
    )
    dudx = _positive(velocity_gradient, "velocity_gradient")
    h0e = _positive(total_enthalpy_edge, "total_enthalpy_edge")
    h_w = np.asarray(wall_enthalpy, dtype=np.float64)
    if np.any(h_w >= h0e):
        raise ValueError("wall enthalpy must be below edge total enthalpy (heating case)")

    if lewis_exponent is None:
        bracket = catalycity_bracket(dissociation_enthalpy, h0e, catalycity, lewis)
    else:
        h_d = np.asarray(dissociation_enthalpy, dtype=np.float64)
        if np.any(h_d < 0.0) or np.any(h_d > h0e):
            raise ValueError("dissociation enthalpy must satisfy 0 <= h_D <= h_0e")
        bracket = np.asarray(1.0 + (lewis**lewis_exponent - 1.0) * h_d / h0e)
    if not (np.isfinite(coefficient) and coefficient > 0.0):
        raise ValueError(f"coefficient must be finite and > 0, got {coefficient}")
    return np.asarray(
        coefficient
        * prandtl**-0.6
        * rho_mu_e**0.4
        * rho_mu_w**0.1
        * np.sqrt(dudx)
        * (h0e - h_w)
        * bracket
    )


def sutton_graves(
    freestream_density: ArrayLike,
    effective_radius: ArrayLike,
    freestream_velocity: ArrayLike,
    constant: float = SUTTON_GRAVES_EARTH,
) -> _FloatArray:
    """Sutton–Graves screening correlation
    :math:`\\dot q_s = k_{SG}\\sqrt{\\rho_\\infty/R_{\\mathrm{eff}}}\\,V_\\infty^3`
    (W/m²).

    This is the fallback of the Remark in Paper II §4.1 for use before a
    boundary-layer edge solution exists. It is a correlation, not a
    theory; results computed here must not be reported as Fay–Riddell
    results.
    """
    if not (np.isfinite(constant) and constant > 0.0):
        raise ValueError(f"constant must be finite and > 0, got {constant}")
    rho = _positive(freestream_density, "freestream_density")
    r_eff = _positive(effective_radius, "effective_radius")
    v = _positive(freestream_velocity, "freestream_velocity")
    return np.asarray(constant * np.sqrt(rho / r_eff) * v**3)
