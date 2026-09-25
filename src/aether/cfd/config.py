"""SU2 cases for the full flight envelope, written rather than edited.

A hand-maintained ``.cfg`` per flight condition does not survive an envelope
sweep. Mach, incidence, altitude and the moment reference all move together,
and the physics model has to move with them: a perfect gas is right at Mach 3
and wrong at Mach 20, a compressible solver is singular at Mach 0, and a
turbulence model that is a nuisance at Mach 20 is mandatory at Mach 0.6. This
module holds one description of a case and renders the config from it, so the
only thing that varies across a sweep is the flight condition.

Version, and why it is stated
-----------------------------

Written for **SU2 v8** and checked against v8.5.0 "Harrier". SU2 renamed a
large part of its configuration vocabulary at v7 and the old names are not
accepted as aliases — ``PHYSICAL_PROBLEM`` became ``SOLVER``, the
``LINEAR_SOLVER_*_FLOW`` family lost its suffix, ``REGIME_TYPE`` was absorbed
into the solver name, and ``GRID_MOVEMENT= PITCHING`` became
``GRID_MOVEMENT= RIGID_MOTION`` with the amplitude and frequency given as
three-vectors. A v6-era file does not degrade on v8; it aborts in the parser
before the mesh is read. :func:`~aether.cfd.config.SU2Case.render`
therefore emits v8 spelling only, and the test suite parses what it writes
with the installed binary rather than trusting this note.

The three regimes
-----------------

:class:`PerfectGas`
    Air as a calorically perfect gas with Sutherland viscosity. Correct while
    the flow is cold enough that vibrational modes stay unexcited and nothing
    dissociates, which on a re-entry body means roughly Mach 5 and below.
:class:`Nonequilibrium`
    SU2's NEMO path: five-species air, separate translational-rotational and
    vibrational-electronic temperatures, Wilke transport. What the stagnation
    region actually is above Mach 10, where a perfect gas overpredicts the
    shock-layer temperature by thousands of kelvin because it has nowhere to
    put the energy.
:class:`LowMach`
    A perfect gas with preconditioning, for the bottom of the envelope. A
    compressible solver's system is singular at zero Mach and stiff near it;
    preconditioning rescales the acoustic eigenvalues so the same code path
    reaches Mach 0.05 instead of stalling at 0.3.

The band between Mach 5 and 10 is genuinely ambiguous — real-gas effects are
present and not dominant — and :func:`regime_for` picks the perfect gas there
deliberately, because a NEMO run costs several times a perfect-gas one and the
error it removes is still small. :func:`regime_for` is a default, not a law;
pass a regime explicitly to overrule it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "FlowConditions",
    "LowMach",
    "Nonequilibrium",
    "Numerics",
    "PerfectGas",
    "PitchingMotion",
    "ReferenceFrame",
    "Regime",
    "SU2Case",
    "UnsteadyRun",
    "regime_for",
    "regime_named",
]

_GAMMA = 1.4
_GAS_CONSTANT = 287.058


@dataclass(frozen=True)
class FlowConditions:
    """Freestream state and attitude for one point of the envelope.

    Attributes
    ----------
    mach, alpha, beta:
        Mach number and incidence/sideslip (rad). Radians here and degrees in
        the rendered config, converted at the boundary — every angle elsewhere
        in this package is in radians, and a module that switches convention
        halfway is where sign errors live.
    temperature, pressure:
        Static freestream conditions (K, Pa). **Required.** They used to
        default to sea level, which is a defensible default for almost any
        other package and a trap for this one: a case asked for at Mach 20
        and nothing else silently ran at 101325 Pa and 288 K, which is
        internally consistent, converges, and answers a question nobody
        asked. The failure is invisible in the residual and shows up only as
        a wrong pressure scale in everything computed afterwards. Use
        :meth:`at_altitude` for the common case.
    wall_temperature:
        Isothermal wall temperature (K). A radiative-equilibrium wall would be
        more faithful and is what :mod:`aether.aerodynamics.friction` models,
        but SU2's isothermal condition is what the heat-flux output is defined
        against, so the wall is fixed here and the equilibrium temperature
        recovered afterwards if wanted.
    """

    mach: float
    temperature: float
    pressure: float
    alpha: float = 0.0
    beta: float = 0.0
    wall_temperature: float = 300.0

    def __post_init__(self) -> None:
        if not (np.isfinite(self.mach) and self.mach > 0.0):
            msg = f"Mach number must be finite and > 0, got {self.mach}"
            raise ValueError(msg)
        if not (np.isfinite(self.temperature) and self.temperature > 0.0):
            msg = f"freestream temperature must be finite and > 0, got {self.temperature}"
            raise ValueError(msg)
        if not (np.isfinite(self.pressure) and self.pressure > 0.0):
            msg = f"freestream pressure must be finite and > 0, got {self.pressure}"
            raise ValueError(msg)

    @property
    def speed(self) -> float:
        """Freestream speed (m/s)."""
        return float(self.mach * np.sqrt(_GAMMA * _GAS_CONSTANT * self.temperature))

    @property
    def density(self) -> float:
        """Freestream density (kg/m³)."""
        return float(self.pressure / (_GAS_CONSTANT * self.temperature))

    @property
    def dynamic_pressure(self) -> float:
        """:math:`q_\\infty = \\tfrac12 \\rho V^2` (Pa)."""
        return float(0.5 * self.density * self.speed**2)

    @property
    def viscosity(self) -> float:
        """Freestream dynamic viscosity from Sutherland's law (Pa·s)."""
        t = float(self.temperature)
        return float(1.716e-5 * (t / 273.15) ** 1.5 * (273.15 + 110.4) / (t + 110.4))

    @classmethod
    def at_altitude(
        cls,
        mach: float,
        altitude: float,
        alpha: float = 0.0,
        beta: float = 0.0,
        wall_temperature: float = 300.0,
        atmosphere: Any | None = None,
    ) -> FlowConditions:
        """Build the freestream from an altitude and the standard atmosphere.

        The constructor to reach for. Naming the altitude makes the flight
                condition explicit and derives the rest, which is what a trajectory
                produces and what a reader of the case can check.

                ``temperature`` and ``pressure`` used to default to sea level, and
                this function existed to talk callers out of relying on that. It did
                not work — a Mach 20 case that kept the defaults was not obviously
                wrong on inspection: it rendered, parsed, ran, and produced a Reynolds
                number four orders of magnitude too large and a boundary layer
                belonging to no part of any trajectory. The defaults are now gone and
                the freestream is required, so this is a convenience rather than a
                guardrail.

                ``atmosphere`` defaults to
                :func:`aether.atmosphere.earth_atmosphere` — the 1976 standard below
                86 km under NRLMSIS above it.
        """
        from aether.atmosphere import earth_atmosphere

        model = atmosphere if atmosphere is not None else earth_atmosphere()
        state = model.state(float(altitude))
        return cls(
            mach=float(mach),
            alpha=float(alpha),
            beta=float(beta),
            temperature=float(np.atleast_1d(state.temperature)[0]),
            pressure=float(np.atleast_1d(state.pressure)[0]),
            wall_temperature=float(wall_temperature),
        )

    def reynolds(self, length: float) -> float:
        """:math:`\\mathrm{Re} = \\rho V L / \\mu` on ``length``.

        Computed from the case's own freestream state rather than carried as a
        constant. The drafted template fixed ``REYNOLDS_NUMBER= 1.5E6`` and then
        swept Mach 0.05 to 27 past it, which does not merely lose accuracy: SU2
        non-dimensionalises the viscous terms with the Reynolds number it is
        given, so a fixed value silently changes the *altitude* of every point
        in the sweep, and the boundary layer solved at Mach 20 is the one that
        belongs to some other flight condition entirely."""
        return float(self.density * self.speed * float(length) / self.viscosity)


@dataclass(frozen=True)
class ReferenceFrame:
    """Non-dimensionalisation and the moment reference point.

    The reference area and length must be the **same values the panel tables
    use**, not each configuration's own, or coefficients from the two sources
    cannot be spliced. That is the argument made at length in
    :mod:`aether.aerodynamics.tables`, and it applies with more force here:
    a CFD point that silently used its own reference area looks like a
    disagreement with the panel method rather than like the unit error it is.
    """

    area: float
    length: float
    moment_reference: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.area) and self.area > 0.0):
            msg = f"reference area must be finite and > 0, got {self.area}"
            raise ValueError(msg)
        if not (np.isfinite(self.length) and self.length > 0.0):
            msg = f"reference length must be finite and > 0, got {self.length}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Regime:
    """Base for the gas and solver model. Subclasses render their own block."""

    name: str = "regime"

    def lines(self, flow: FlowConditions) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    @property
    def needs_reynolds(self) -> bool:
        """Whether SU2 demands ``REYNOLDS_NUMBER`` for this solver.

        It does for viscous perfect-gas runs, and refuses to start without one:
        *"Reynolds number required for NAVIER_STOKES and RANS"*. It does not for
        NEMO, which initialises from thermodynamic conditions and rejects the
        over-determination, nor for Euler, which has no viscous term to
        non-dimensionalise.
        """
        return False


@dataclass(frozen=True)
class PerfectGas(Regime):
    """Calorically perfect air, Sutherland viscosity, optional turbulence.

    ``turbulence`` selects the RANS model, or ``None`` for laminar
    Navier–Stokes. It is not a free choice of spelling: SU2 reads
    ``KIND_TURB_MODEL`` only when ``SOLVER= RANS``, and setting the model with
    ``SOLVER= NAVIER_STOKES`` — which is what the drafted config did — leaves
    the run laminar with no warning, because the key is simply never consulted.
    The two are therefore set together here and cannot disagree.
    """

    name: str = "perfect-gas"
    turbulence: str | None = "SST"
    inviscid: bool = False
    """Euler rather than Navier–Stokes: no boundary layer, and no wall heat flux."""

    hybrid_rans_les: str | None = None
    """Detached-eddy treatment, or ``None`` for pure RANS.

    ``SA_DDES`` is the working choice: delayed DES, which keeps RANS inside the
    attached boundary layer and switches to LES in the separated region, and
    the *delay* is what stops the switch happening prematurely inside a thick
    hypersonic boundary layer (modelled-stress depletion).

    A separated near wake is the case this exists for. RANS closures are
    calibrated on attached shear layers and are not reliable in a recirculating
    base region; DDES resolves the large-scale unsteadiness there instead of
    modelling it. It is only meaningful alongside
    :class:`UnsteadyRun` -- a detached-eddy model integrated to a steady state
    is a contradiction, and `SU2Case.__post_init__` refuses that pairing.

    SU2's hybrid options are all Spalart-Allmaras derived (``SA_DES``,
    ``SA_DDES``, ``SA_ZDES``, ``SA_EDDES``, verified against the 8.5 binary), so
    a k-omega model cannot carry one. Asking for SST with DDES is refused rather
    than quietly ignored, which is what SU2 itself would do with the key.
    """

    #: Hybrid RANS-LES models this SU2 build accepts, read out of the binary.
    HYBRID_MODELS: ClassVar[frozenset[str]] = frozenset(
        {"SA_DES", "SA_DDES", "SA_ZDES", "SA_EDDES"}
    )

    #: Turbulence models a hybrid treatment can be built on.
    SA_FAMILY: ClassVar[frozenset[str]] = frozenset(
        {"SA", "SA_NEG", "SA_COMP", "SA_E"}
    )

    def __post_init__(self) -> None:
        if self.hybrid_rans_les is None:
            return
        if self.hybrid_rans_les not in self.HYBRID_MODELS:
            raise ValueError(
                f"unknown hybrid RANS-LES model {self.hybrid_rans_les!r}; "
                f"this SU2 accepts {sorted(self.HYBRID_MODELS)}"
            )
        if self.inviscid:
            raise ValueError(
                "a detached-eddy model needs a viscous solver; inviscid=True "
                "solves Euler and has no turbulence model to detach from"
            )
        if self.turbulence not in self.SA_FAMILY:
            raise ValueError(
                f"{self.hybrid_rans_les} is Spalart-Allmaras derived and cannot "
                f"run on turbulence={self.turbulence!r}; choose one of "
                f"{sorted(self.SA_FAMILY)}"
            )

    @property
    def needs_reynolds(self) -> bool:
        return not self.inviscid

    def lines(self, flow: FlowConditions) -> list[str]:
        if self.inviscid:
            solver = ["SOLVER= EULER"]
        elif self.turbulence:
            solver = ["SOLVER= RANS", f"KIND_TURB_MODEL= {self.turbulence}"]
            if self.hybrid_rans_les:
                solver.append(f"HYBRID_RANSLES= {self.hybrid_rans_les}")
        else:
            solver = ["SOLVER= NAVIER_STOKES"]
        block = [*solver, "MATH_PROBLEM= DIRECT", "FLUID_MODEL= STANDARD_AIR"]
        if not self.inviscid:
            block += [
                "VISCOSITY_MODEL= SUTHERLAND",
                "MU_REF= 1.716E-5",
                "MU_T_REF= 273.15",
                "SUTHERLAND_CONSTANT= 110.4",
            ]
        return [*block, f"GAMMA_VALUE= {_GAMMA}", f"GAS_CONSTANT= {_GAS_CONSTANT}"]


@dataclass(frozen=True)
class LowMach(PerfectGas):
    """Perfect gas with low-Mach preconditioning, for the bottom of the sweep.

    The compressible equations are ill-conditioned as :math:`M \\to 0`: the
    acoustic eigenvalues run away from the convective one, the implicit system
    stiffens, and convergence stalls. Preconditioning rescales them back
    together. The drafted sweep reached for this below Mach 0.3, which is the
    right threshold; what it did not do is clamp the Mach number itself, and a
    request for exactly Mach 0 is singular no matter how well preconditioned —
    see :func:`regime_for`.
    """

    name: str = "low-mach"

    def lines(self, flow: FlowConditions) -> list[str]:
        return [*super().lines(flow), "LOW_MACH_PREC= YES"]


@dataclass(frozen=True)
class Nonequilibrium(Regime):
    """Five-species air in thermochemical nonequilibrium — SU2's NEMO solver.

    Four corrections to the drafted NEMO block, each of which stops the run
    rather than degrading it:

    ``CHEMISTRY_MODEL= PARK`` and ``TWO_TEMPERATURE_MODEL= YES``
        Neither is an SU2 v8 option. Park's rates are what ``GAS_MODEL= AIR-5``
        already carries, and the two-temperature model is not a switch — it is
        what the NEMO solver *is*, and it is engaged by giving the
        vibrational-electronic freestream temperature below.
    ``FREESTREAM_TEMPERATURE_VE``, ``INIT_OPTION``, ``FREESTREAM_OPTION``
        Required. Without them SU2 aborts inside
        ``CNEMOEulerSolver::SetNondimensionalization`` before iterating, because
        it has no second temperature to non-dimensionalise against.
    ``REYNOLDS_NUMBER``
        Must be absent. NEMO initialises from thermodynamic conditions
        (``TD_CONDITIONS``); a Reynolds number is over-determination and the
        combination is rejected.
    ``CONV_NUM_METHOD_FLOW= AUSMPLUSUP``
        Not implemented for NEMO — the run dies at "Invalid upwind scheme".
        NEMO accepts ``AUSM``, ``AUSMPLUSUP2``, ``MSW`` and ``ROE``;
        ``AUSMPLUSUP2`` is the default here for the same carbuncle-resistance
        argument the axisymmetric path makes for AUSM.

    ``vibrational_temperature`` defaults to the freestream translational
    temperature, which is the equilibrium assumption and the right one for
    undisturbed air ahead of the shock.
    """

    name: str = "nonequilibrium"
    gas_model: str = "AIR-5"
    composition: tuple[float, ...] = (0.767, 0.233, 0.0, 0.0, 0.0)
    transport: str = "WILKE"
    vibrational_temperature: float | None = None
    inviscid: bool = False

    def __post_init__(self) -> None:
        total = float(sum(self.composition))
        if not np.isclose(total, 1.0, atol=1e-6):
            msg = f"gas composition mass fractions must sum to 1, got {total:g}"
            raise ValueError(msg)

    #: Upwind schemes SU2's NEMO solver implements. Anything else dies at
    #: "Invalid upwind scheme" before the mesh is read -- including HLLC, which
    #: is the best choice on the perfect-gas path and therefore exactly what a
    #: caller who has read the flux matrix will reach for.
    #:
    #: Read from the SU2 8.5.0 source rather than inferred: the NEMO branch of
    #: ``CDriver::InstantiateNEMONumerics`` switches on ROE, AUSM, AUSMPLUSUP2,
    #: **AUSMPLUSM** and MSW. An earlier version of this set was missing
    #: AUSMPLUSM, which is the one scheme here designed for robustness at a
    #: strong shock.
    #:
    #: ``CUpwAUSMPWplus_NEMO`` is also implemented -- AUSMPW+ is the classic
    #: carbuncle cure in this family -- but it appears in no ``UPWIND`` enum
    #: entry and no driver case, so it cannot be reached from a config file.
    ACCEPTS: ClassVar[frozenset[str]] = frozenset(
        {"AUSM", "AUSMPLUSM", "AUSMPLUSUP2", "MSW", "ROE"}
    )

    #: What a scheme NEMO cannot run is replaced by. AUSM+M rather than
    #: AUSM+-up2: the latter is what the failing Mach 12 to 20 runs used, and it
    #: is the scheme that limit-cycles on the perfect-gas path. AUSM+M is newer
    #: (doi 10.1016/j.apm.2019.09.005) and NEMO-only, so it has no entry in the
    #: perfect-gas flux matrix and is **untested here** -- which is said out
    #: loud rather than hidden behind a default.
    SUBSTITUTE: ClassVar[str] = "AUSMPLUSM"

    def convective_for(self, requested: str) -> str:
        """The nearest scheme NEMO implements, and a note when it substitutes.

        Substituting silently would be the wrong trade -- a run that quietly
        used a different discretisation than its record claims is the failure
        this project has spent a day on. The caller is told; the alternative is
        refusing, and refusing an envelope sweep at its top end because one
        setting does not carry over is worse.
        """
        return requested if requested in self.ACCEPTS else self.SUBSTITUTE

    def lines(self, flow: FlowConditions) -> list[str]:
        t_ve = (
            flow.temperature
            if self.vibrational_temperature is None
            else float(self.vibrational_temperature)
        )
        solver = "NEMO_EULER" if self.inviscid else "NEMO_NAVIER_STOKES"
        composition = ", ".join(f"{x:.6g}" for x in self.composition)
        return [
            f"SOLVER= {solver}",
            "MATH_PROBLEM= DIRECT",
            "FLUID_MODEL= SU2_NONEQ",
            f"GAS_MODEL= {self.gas_model}",
            f"GAS_COMPOSITION= ({composition})",
            f"TRANSPORT_COEFF_MODEL= {self.transport}",
            f"FREESTREAM_TEMPERATURE_VE= {t_ve:.10g}",
            "INIT_OPTION= TD_CONDITIONS",
            "FREESTREAM_OPTION= TEMPERATURE_FS",
        ]


def regime_named(
    name: str,
    inviscid: bool = False,
    turbulence: str | None = "SST",
    hybrid_rans_les: str | None = None,
) -> Regime:
    """A regime by its :attr:`Regime.name`, for callers that carry a string.

    The queue stores its spec as JSON, so an override there is a name rather
    than a class. ``turbulence`` is dropped for nonequilibrium for the reason
    :func:`regime_for` gives: NEMO takes no RANS model.
    """
    key = str(name).strip().lower()
    if key in ("perfect-gas", "perfect", "perfect_gas"):
        return PerfectGas(
            inviscid=inviscid, turbulence=turbulence, hybrid_rans_les=hybrid_rans_les
        )
    if key in ("low-mach", "low_mach", "lowmach"):
        return LowMach(
            inviscid=inviscid, turbulence=turbulence, hybrid_rans_les=hybrid_rans_les
        )
    if key in ("nonequilibrium", "nemo", "noneq"):
        if hybrid_rans_les:
            msg = (
                f"{hybrid_rans_les} cannot run on the nonequilibrium solver: NEMO "
                "carries no turbulence model for a hybrid treatment to detach from"
            )
            raise ValueError(msg)
        return Nonequilibrium(inviscid=inviscid)
    msg = (
        f"unknown regime {name!r}; expected one of "
        "'perfect-gas', 'low-mach', 'nonequilibrium'"
    )
    raise ValueError(msg)


def regime_for(
    mach: float,
    inviscid: bool = False,
    turbulence: str | None = "SST",
    hybrid_rans_les: str | None = None,
) -> Regime:
    """The default gas model for a Mach number.

    The thresholds are the drafted sweep's, with the physics behind them
    stated: below **Mach 0.3** the compressible system needs preconditioning;
    above **Mach 10** the shock layer is hot enough that treating air as a
    perfect gas overpredicts its temperature badly, because a perfect gas has
    no vibrational modes to excite and no dissociation to absorb the enthalpy.

    Note what this function does *not* do. The drafted sweep clamped Mach to a
    floor of 0.05 inside the loop, silently substituting a different flight
    condition for the one asked for and recording the substitute in the table.
    A sweep that quietly moves its own grid points is worse than one that
    refuses them, so the clamp is not reproduced here — :class:`FlowConditions`
    rejects a non-positive Mach number outright, and a sweep that wants an
    incompressible anchor point should ask for Mach 0.05 and mean it.
    """
    mach = float(mach)
    if mach < 0.3:
        return LowMach(
            inviscid=inviscid, turbulence=turbulence, hybrid_rans_les=hybrid_rans_les
        )
    if mach < 10.0:
        return PerfectGas(
            inviscid=inviscid, turbulence=turbulence, hybrid_rans_les=hybrid_rans_les
        )
    if hybrid_rans_les:
        msg = (
            f"{hybrid_rans_les} cannot run above Mach 10, where regime_for "
            "selects NEMO: it carries no turbulence model to detach from. Pin "
            "regime='perfect-gas' if a detached-eddy run is what is wanted."
        )
        raise ValueError(msg)
    # ``turbulence`` is not carried into NEMO: its solver is NEMO_EULER or
    # NEMO_NAVIER_STOKES and takes no RANS model. Dropping it here is the
    # honest thing -- a caller asking for SST above Mach 10 is asking for
    # something that does not exist, and silently pretending otherwise is how
    # a setting stops meaning anything.
    return Nonequilibrium(inviscid=inviscid)


@dataclass(frozen=True)
class Numerics:
    """Discretisation, linear solver and convergence.

    The defaults follow the axisymmetric path's reasoning
    (:class:`aether.aerodynamics.cfd.su2.SU2Settings`) — robustness across a
    whole unattended sweep beats speed at any one point — with two hypersonic
    changes.

    ``convective``
        ``AUSMPLUSUP2`` rather than plain AUSM. Both avoid the carbuncle that
        makes Roe unusable at a strong bow shock; the ``+up2`` variant adds the
        low-Mach fix, which matters because the same scheme has to work at the
        subsonic end of the envelope.
    ``cfl`` and ``cfl_bounds``
        Start low. A hypersonic case initialised from freestream has a
        transient in which the bow shock forms and sweeps back across the
        mesh, and a CFL that is fine on the converged solution will produce a
        negative temperature during it. Adaptation raises it once the shock has
        settled.
    """

    convective: str = "AUSMPLUSUP2"
    limiter: str = "VENKATAKRISHNAN_WANG"
    limiter_coefficient: float = 0.05
    entropy_fix: float = 0.1
    muscl: bool = True
    first_order_iterations: int = 0
    """Iterations of a first-order prelude before the second-order run.

    Zero disables it. Non-zero is close to mandatory above about Mach 6, and
    the reason is the startup transient rather than the converged solution: a
    case initialised from uniform freestream has to form its bow shock and
    sweep it back across the mesh, and during that a MUSCL reconstruction
    across the forming discontinuity produces negative temperatures. On the
    waverider at Mach 8 the second-order run dies on a NaN at iteration 205 at
    CFL 0.5, and still dies at CFL 0.25, while first order survives — so
    lowering the CFL is not the fix, and the fix is not to run first order
    either, because first order smears the shock badly enough to cost several
    percent of drag.

    The prelude establishes the shock cheaply, writes a restart, and hands it
    to the second-order run as an initial condition it can survive.
    """
    cfl: float = 0.5
    cfl_adapt: bool = True
    cfl_bounds: tuple[float, float] = (0.1, 50.0)
    iterations: int = 5000
    linear_solver_iterations: int = 10
    force_tolerance: float = 5.0e-3
    """Drift of the force mean across ``force_window`` that counts as converged.

    Set relative to the *discretisation* error, not as small as possible. The
    grid study this was built for shows level-to-level differences of 5 to 14 %,
    so an iterative convergence of half a percent is an order of magnitude
    below the error that actually limits the answer, and tightening further
    buys nothing while costing iterations that could have gone into a finer
    mesh. A tolerance far below the discretisation error is not rigour — it is
    a run that never terminates and a table that never gets built.

    The corollary is that this number should move when the mesh does: a study
    resolved to 0.5 % wants a tolerance nearer 5e-4.
    """
    force_window: int = 2000
    """Iterations the convergence test looks back over.

    Long, because it has to average out the ripple rather than sample it: the
    drift a shorter window reports is dominated by where it happened to land
    in the oscillation. See :func:`~aether.cfd.solver._force_settled`.
    """
    minimum_iterations: int = 500
    extra: dict[str, str] = field(default_factory=dict)
    """Raw ``KEY= VALUE`` overrides, appended last so they win."""

    def lines(self) -> list[str]:
        low, high = self.cfl_bounds
        return [
            f"CONV_NUM_METHOD_FLOW= {self.convective}",
            f"MUSCL_FLOW= {'YES' if self.muscl else 'NO'}",
            f"SLOPE_LIMITER_FLOW= {self.limiter}",
            f"VENKAT_LIMITER_COEFF= {self.limiter_coefficient:.10g}",
            f"ENTROPY_FIX_COEFF= {self.entropy_fix:.10g}",
            "NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES",
            "TIME_DISCRE_FLOW= EULER_IMPLICIT",
            f"CFL_NUMBER= {self.cfl:.10g}",
            f"CFL_ADAPT= {'YES' if self.cfl_adapt else 'NO'}",
            f"CFL_ADAPT_PARAM= ( 0.1, 1.05, {low:.10g}, {high:.10g} )",
            "LINEAR_SOLVER= FGMRES",
            "LINEAR_SOLVER_PREC= ILU",
            "LINEAR_SOLVER_ERROR= 1E-6",
            f"LINEAR_SOLVER_ITER= {int(self.linear_solver_iterations)}",
        ]


@dataclass(frozen=True)
class UnsteadyRun:
    r"""Time-accurate integration of a **static** body.

    Why this exists separately from :class:`PitchingMotion`
    ------------------------------------------------------

    ``SU2Case.render`` decided time-accuracy with ``unsteady = self.motion is
    not None``. Dual-time stepping was therefore reachable only by attaching a
    forced oscillation, and the case where the *body* is still and the *flow*
    is unsteady -- a separated base, a shear layer, a shock-buffet -- could not
    be expressed at all. That is not an omission in the queue, it is an
    omission in the model: the physics of a recirculating near wake is
    unsteady whether or not anything is moving.

    A steady solver pointed at a separated base returns a number. It is the
    value of a limit cycle at whichever iteration the run stopped, it is not
    grid-convergent, and averaging the force history does not rescue it because
    the *spatial* field it came from never satisfied the steady equations. The
    base pressure has to be integrated in time and averaged over many shedding
    periods, which is what this class is for.

    Sizing it
    ---------

    Bluff-body shedding sits near :math:`St = f D / U \approx 0.2`. Resolving
    the cycle needs :math:`\Delta t \lesssim T/30`, and a converged *mean*
    needs tens of cycles, so

    .. math:: \Delta t \approx \frac{D}{6\,U}, \qquad
              T_{\text{total}} \approx 30\,\frac{D}{0.2\,U}

    :meth:`for_base_flow` does exactly that arithmetic from the base diameter
    and the freestream speed, rather than leaving a time step to be guessed.

    ``start_time`` marks where the starting transient ends and statistics may
    begin; it is carried for the reduction rather than given to SU2, which has
    no notion of it.
    """

    time_step: float
    """Physical time step (s)."""

    total_time: float
    """Physical time to integrate (s)."""

    inner_iterations: int = 20
    """Dual-time sub-iterations per physical step. Too few and the implicit
    solve never converges within the step, which shows up as a time history
    that depends on the step size."""

    start_time: float = 0.0
    """Physical time before which samples are discarded as starting transient."""

    @property
    def steps(self) -> int:
        return int(np.ceil(self.total_time / self.time_step))

    @property
    def sampled_steps(self) -> int:
        """Steps that survive the transient, i.e. what the mean is built from."""
        return max(0, self.steps - int(np.ceil(self.start_time / self.time_step)))

    def __post_init__(self) -> None:
        for name in ("time_step", "total_time"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if self.inner_iterations < 1:
            raise ValueError(
                f"inner_iterations must be >= 1, got {self.inner_iterations}"
            )
        if not (np.isfinite(self.start_time) and self.start_time >= 0.0):
            raise ValueError(f"start_time must be finite and >= 0, got {self.start_time}")
        if self.start_time >= self.total_time:
            raise ValueError(
                f"start_time {self.start_time:g} leaves no samples before "
                f"total_time {self.total_time:g}"
            )

    @classmethod
    def for_base_flow(
        cls,
        base_diameter: float,
        speed: float,
        strouhal: float = 0.2,
        steps_per_cycle: int = 30,
        cycles: float = 30.0,
        transient_cycles: float = 5.0,
        inner_iterations: int = 20,
    ) -> UnsteadyRun:
        """Size a base-flow run from the shedding scale rather than by guess.

        ``strouhal`` 0.2 is the bluff-body value; ``cycles`` 30 with the first
        five discarded is what it takes for the mean base pressure to stop
        moving. The defaults are a starting point to be justified per case, not
        a law -- quote the reduced statistics, not the settings.
        """
        for name, value in (
            ("base_diameter", base_diameter), ("speed", speed),
            ("strouhal", strouhal), ("cycles", cycles),
        ):
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        frequency = strouhal * float(speed) / float(base_diameter)
        period = 1.0 / frequency
        return cls(
            time_step=period / float(steps_per_cycle),
            total_time=cycles * period,
            inner_iterations=inner_iterations,
            start_time=transient_cycles * period,
        )

    def lines(self) -> list[str]:
        return [
            "TIME_DOMAIN= YES",
            "TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER",
            f"TIME_STEP= {self.time_step:.10g}",
            f"MAX_TIME= {self.total_time:.10g}",
            f"TIME_ITER= {self.steps}",
            f"INNER_ITER= {int(self.inner_iterations)}",
        ]


@dataclass(frozen=True)
class PitchingMotion:
    """Forced sinusoidal pitch oscillation, for the damping derivatives.

    Drives :math:`\\alpha(t) = \\alpha_0 + A\\sin(\\omega t)` by rotating the
    whole grid, which is what makes the run affordable: the mesh is rigid, so
    there is no deformation solve and no remeshing, and the boundary layer
    built in :func:`~aether.aerodynamics.cfd.meshing.viscous_domain` keeps its
    quality for the whole cycle.

    SU2 v8 spelling, none of which the drafted config used:

    ``GRID_MOVEMENT= RIGID_MOTION``
        ``PITCHING`` is not a valid value in v8 — it was the v6 name, and the
        parser rejects it while listing the four it accepts.
    ``PITCHING_AMPL`` / ``PITCHING_OMEGA`` / ``PITCHING_PHASE``
        Three-vectors, one component per body axis, not the scalars
        ``PITCH_AMPLITUDE`` / ``PITCH_OMEGA`` / ``PITCH_PHASE``. Pitch is
        rotation about **y**, so the amplitude and frequency go in the second
        slot; putting a scalar there would set the roll component.

    ``ITER`` must be absent from an unsteady case. Present alongside
    ``TIME_ITER`` it aborts the run in ``SetPostprocessing`` with no message,
    which is why :meth:`SU2Case.render` drops it whenever a motion is attached.

    Attributes
    ----------
    amplitude:
        Pitch amplitude :math:`A` (rad). Small enough to stay linear — the
        derivative being measured is defined on the linearisation — and large
        enough that the moment signal clears the convergence noise. Two degrees
        is the usual compromise.
    frequency:
        Angular frequency :math:`\\omega` (rad/s), **not** cycles per second.
        The drafted config wrote ``PITCH_OMEGA= 10.0`` next to a 10 Hz label;
        the two differ by :math:`2\\pi` and the reduced frequency with them.
    """

    amplitude: float = np.deg2rad(2.0)
    frequency: float = 10.0
    phase: float = 0.0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    time_step: float = 1.0e-4
    inner_iterations: int = 15
    cycles: float = 5.0
    """Oscillation cycles to run. The first is discarded as a starting transient."""

    @property
    def period(self) -> float:
        """One oscillation period (s)."""
        return float(2.0 * np.pi / self.frequency)

    def reduced_frequency(self, reference_length: float, speed: float) -> float:
        """:math:`k = \\omega L_{\\mathrm{ref}} / 2V` — the similarity parameter.

        The damping derivative is a function of :math:`k`, not of frequency,
        and quoting :math:`C_{m_q}` without it is quoting half a result. Values
        near 0.01–0.05 are the quasi-steady range a flight-dynamics model wants.
        """
        return float(self.frequency * float(reference_length) / (2.0 * float(speed)))

    def steps_per_cycle(self) -> int:
        return int(np.ceil(self.period / self.time_step))

    def lines(self) -> list[str]:
        x, y, z = self.origin
        return [
            "TIME_DOMAIN= YES",
            "TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER",
            f"TIME_STEP= {self.time_step:.10g}",
            f"MAX_TIME= {self.cycles * self.period:.10g}",
            f"TIME_ITER= {int(np.ceil(self.cycles * self.steps_per_cycle()))}",
            f"INNER_ITER= {int(self.inner_iterations)}",
            "GRID_MOVEMENT= RIGID_MOTION",
            f"MOTION_ORIGIN= {x:.10g} {y:.10g} {z:.10g}",
            f"PITCHING_OMEGA= 0.0 {self.frequency:.10g} 0.0",
            f"PITCHING_AMPL= 0.0 {np.rad2deg(self.amplitude):.10g} 0.0",
            f"PITCHING_PHASE= 0.0 {np.rad2deg(self.phase):.10g} 0.0",
        ]


@dataclass(frozen=True)
class SU2Case:
    """One complete SU2 case: flight condition, gas model, mesh and outputs."""

    flow: FlowConditions
    reference: ReferenceFrame
    mesh_filename: str
    regime: Regime | None = None
    """``None`` selects :func:`regime_for` on the case's own Mach number."""

    turbulence: str | None = "SST"
    """RANS model for the resolved regime, or ``None`` for laminar.

    Passed to :func:`regime_for` alongside :attr:`inviscid` rather than pinning
    a regime, so the **Mach-dependent gas model still applies**. Pinning
    ``PerfectGas`` to carry this setting silently disabled the switch to
    :class:`Nonequilibrium` above Mach 10, and Mach 12 to 20 runs went out as
    perfect gas at stagnation temperatures of 8000 to 21000 K.
    """

    inviscid: bool = False
    """Solve Euler rather than RANS, when the regime is resolved rather than pinned.

    :func:`regime_for` has always taken this and :meth:`resolved_regime` never
    passed it, so a case that did not pin a regime got a **viscous** one no
    matter what mesh it was about to be solved on. Every run through
    :mod:`~aether.cfd.worker` went out as ``SOLVER= RANS`` with
    ``KIND_TURB_MODEL= SST`` on a mesh built by ``inviscid_domain`` -- no prism
    layer, no wall clustering, nothing k-omega SST can compute a production
    term on. Its residuals are the ones that diverge.
    """
    numerics: Numerics = field(default_factory=Numerics)
    motion: PitchingMotion | None = None

    unsteady: UnsteadyRun | None = None
    """Time-accurate integration of a static body, when no motion is attached.

    Kept separate from :attr:`motion` because they answer different questions:
    a motion makes the *body* move to measure a derivative, this makes the
    *solver* time-accurate so an unsteady flow can be integrated at all. A
    separated near wake needs the second and not the first, and before this
    existed there was no way to ask for it -- ``render`` decided time-accuracy
    purely from ``motion is not None``.

    Setting both is refused: a forced oscillation already carries its own time
    step and duration, and two sources for them is how they come to disagree.
    """
    wall_marker: str = "vehicle"
    base_marker: str | None = None
    """Second wall marker for the base disc, monitored separately.

    Set it on any body with a blunt base. An Euler solution has no valid base
    pressure, and folding it into one force coefficient can dominate the
    answer — see :func:`~aether.aerodynamics.cfd.meshing.inviscid_domain`.
    With this set, SU2 reports per-marker coefficients and the forebody term
    is separable from the term that has to be replaced by a correlation.
    """
    farfield_marker: str = "farfield"
    imposed_inflow: bool = False
    """Impose freestream on :attr:`farfield_marker` instead of extrapolating it.

    ``MARKER_FAR`` is a **characteristic** condition: it decides how many
    variables to impose from the sign of the normal velocity. That is the right
    condition on a far field placed tens of body lengths out, where the boundary
    is broadside to the flow and the split is unambiguous.

    It is the wrong one on a **shock-fitted** outer boundary, which by
    construction follows the shock and therefore runs nearly *parallel* to the
    flow -- 13 degrees on the Mach 8 sphere-cone. The normal velocity there is a
    small difference of large numbers, the characteristic split is ill-posed,
    and the solution drifts. Measured on that case: **70 % of the 4 033
    boundary nodes above Mach 8.05**, a median of 8.53 against a freestream 8.0,
    and a worst node expanded to 0.00048 K and 0.00024 Pa -- vacuum.

    The wall was unaffected, which is why every force check passed. It is still
    a boundary condition producing states that do not exist.

    A shock-fitted domain is *defined* as the region in which the flow differs
    from freestream, so its outer boundary is a surface on which the flow **is**
    freestream. Imposing all of it there is not an approximation, it is the
    construction's own premise, and it is well posed whatever the angle.
    """
    outflow_markers: Sequence[str] = ()
    """Boundaries where the flow leaves supersonically, joined to ``MARKER_FAR``.

    A shock-fitted block is cut off at the base plane. The flow there is
    supersonic and outgoing, so every characteristic leaves the domain and the
    characteristic far-field condition extrapolates all of them -- which is what
    makes ``MARKER_FAR`` the right condition rather than a convenient one. Kept
    a separate field so the reasoning is visible at the call site.
    """
    symmetry_markers: Sequence[str] = ()
    restart: bool = False
    volume_every: int = 0
    """Write the volume solution every N iterations; 0 writes only at the end.

    Worth setting on a long run. A diverging case does not announce *where* it
    is going wrong -- SU2 reports "N points are not physical" and no location
    -- and on the Mach 20 biconic here the answer was a few millimetres of
    nose: four nodes at :math:`x \\approx 0.8` mm carrying 800 kPa against a
    pitot ceiling of 80 kPa, and 73,000 K against an equilibrium 6--10,000 K.
    A snapshot every fifty iterations shows that in the first minutes rather
    than at the end of six hours.

    The cost is small next to the benefit: the file is about 650 bytes a node,
    and writing it takes a second or so against tens of seconds an iteration.
    """

    write_volume: bool = True
    """Write the nodal volume solution, not only the restart.

    On by default, and the default matters. A restart carries conservative
    variables only, so pressure, temperature and Mach have to be
    reconstructed from it -- and reconstructing them needs a ratio of
    specific heats, which is exactly what stops existing above about Mach 10.
    The volume file carries those three as the solver computed them, under
    whichever gas model actually ran, so a nonequilibrium case can be read at
    all and a perfect-gas one need not be re-derived.

    Tecplot ASCII rather than ParaView: both carry the same variables, and
    this one is a variable list followed by a node-major block of numbers,
    which is a format a reader can be sure it has understood. It costs roughly
    400 bytes a node, so a 300k-node case adds about 120 MB -- the price of
    the run being visualisable at all, and cheap against the run itself.
    """

    def __post_init__(self) -> None:
        if self.motion is not None and self.unsteady is not None:
            msg = (
                "a case carries either a PitchingMotion or an UnsteadyRun, not "
                "both: the motion already fixes the time step and duration, and "
                "a second source for them is how the two come to disagree"
            )
            raise ValueError(msg)

    def resolved_regime(self) -> Regime:
        if self.regime is not None:
            return self.regime
        return regime_for(
            self.flow.mach, inviscid=self.inviscid, turbulence=self.turbulence
        )

    def at(self, mach: float | None = None, alpha: float | None = None) -> SU2Case:
        """The same case at another flight condition.

        The regime is re-resolved unless one was pinned explicitly, so moving a
        case from Mach 3 to Mach 20 moves it onto the NEMO solver rather than
        carrying a perfect gas into a dissociating shock layer.
        """
        flow = replace(
            self.flow,
            mach=self.flow.mach if mach is None else float(mach),
            alpha=self.flow.alpha if alpha is None else float(alpha),
        )
        return replace(self, flow=flow)

    def render(self) -> str:
        """The config file, in SU2 v8 spelling."""
        flow, reference = self.flow, self.reference
        regime = self.resolved_regime()
        unsteady = self.motion is not None or self.unsteady is not None
        x, y, z = reference.moment_reference

        lines: list[str] = [
            f"% {regime.name} case written by aether.cfd.config",
            f"% Mach {flow.mach:g}, alpha {np.rad2deg(flow.alpha):g} deg, "
            f"{flow.pressure:g} Pa, {flow.temperature:g} K",
            "",
            *regime.lines(flow),
            f"RESTART_SOL= {'YES' if self.restart else 'NO'}",
            "",
            f"MACH_NUMBER= {flow.mach:.10g}",
            f"AOA= {np.rad2deg(flow.alpha):.10g}",
            f"SIDESLIP_ANGLE= {np.rad2deg(flow.beta):.10g}",
            f"FREESTREAM_TEMPERATURE= {flow.temperature:.10g}",
            f"FREESTREAM_PRESSURE= {flow.pressure:.10g}",
            "",
            f"REF_AREA= {reference.area:.10g}",
            f"REF_LENGTH= {reference.length:.10g}",
            f"REF_ORIGIN_MOMENT_X= {x:.10g}",
            f"REF_ORIGIN_MOMENT_Y= {y:.10g}",
            f"REF_ORIGIN_MOMENT_Z= {z:.10g}",
            "REF_DIMENSIONALIZATION= DIMENSIONAL",
        ]
        if regime.needs_reynolds:
            lines += [
                f"REYNOLDS_NUMBER= {flow.reynolds(reference.length):.10g}",
                f"REYNOLDS_LENGTH= {reference.length:.10g}",
            ]
        lines.append("")

        # An inviscid case has no wall temperature to impose: MARKER_ISOTHERMAL
        # on an Euler wall is not a stricter boundary condition, it is an
        # invalid one.
        inviscid = bool(getattr(regime, "inviscid", False))
        walls = [self.wall_marker]
        if self.base_marker:
            walls.append(self.base_marker)
        joined = ", ".join(walls)
        if inviscid:
            lines.append(f"MARKER_EULER= ( {joined} )")
        else:
            lines.append(
                "MARKER_ISOTHERMAL= ( "
                + ", ".join(f"{name}, {flow.wall_temperature:.10g}" for name in walls)
                + " )"
            )
        if self.imposed_inflow:
            velocity = flow.speed * np.array(
                [
                    np.cos(flow.alpha) * np.cos(flow.beta),
                    -np.sin(flow.beta),
                    np.sin(flow.alpha) * np.cos(flow.beta),
                ]
            )
            lines += [
                f"MARKER_SUPERSONIC_INLET= ( {self.farfield_marker}, "
                f"{flow.temperature:.8g}, {flow.pressure:.8g}, "
                f"{velocity[0]:.8g}, {velocity[1]:.8g}, {velocity[2]:.8g} )",
            ]
            if self.outflow_markers:
                lines.append(
                    "MARKER_FAR= ( " + ", ".join(self.outflow_markers) + " )"
                )
        else:
            lines += [
                "MARKER_FAR= ( "
                + ", ".join([self.farfield_marker, *self.outflow_markers])
                + " )",
            ]
        lines += [
            f"MARKER_PLOTTING= ( {joined} )",
            # Order matters downstream: SU2 indexes per-marker coefficients
            # (`CD[0]`, `CD[1]`, ...) by position in this list, so the forebody
            # is always index 0 and the base index 1.
            f"MARKER_MONITORING= ( {joined} )",
        ]
        if self.symmetry_markers:
            lines.append(f"MARKER_SYM= ( {', '.join(self.symmetry_markers)} )")
        # NEMO implements a different set of upwind schemes, and asking it for
        # one it lacks kills the run before the mesh is read. Substitute, and
        # say so in the file -- a config that silently disagrees with the record
        # is the failure mode this pipeline has spent a day on.
        numerics = self.numerics
        allowed = getattr(regime, "convective_for", None)
        if allowed is not None:
            chosen = allowed(numerics.convective)
            if chosen != numerics.convective:
                lines.append(
                    f"% {numerics.convective} is not implemented for "
                    f"{regime.name}; using {chosen}"
                )
                numerics = replace(numerics, convective=chosen)
        lines += ["", *numerics.lines(), ""]

        if unsteady:
            # Deliberately no ITER in either branch: see PitchingMotion's note.
            block = self.motion if self.motion is not None else self.unsteady
            assert block is not None
            lines += [*block.lines(), ""]
        else:
            lines += [
                f"ITER= {int(self.numerics.iterations)}",
                # Convergence on the integrated force rather than the density
                # residual, for the reason given in aether's axisymmetric
                # driver: a limiter on an unstructured mesh stalls the residual
                # several decades above machine zero while the force is already
                # stable to five digits.
                "CONV_FIELD= DRAG",
                # **Four** orders below our own threshold, not two. SU2's Cauchy
                # criterion averages the *relative change per iteration*, which
                # goes small well before the mean has stopped moving -- and the
                # ratio between the two is not a constant, it depends on how
                # smoothly the case converges.
                #
                # Two orders was calibrated on the isotropic mesh, which
                # limit-cycles: large iterate-to-iterate differences keep the
                # Cauchy measure up and it never fires early. A wall-resolved
                # mesh converges *smoothly*, so its per-iteration change is tiny
                # while its mean is still walking. Measured 2026-09-11 at the
                # old margin: SU2 declared ``Cauchy[CD] 4.95e-05 < 5e-05`` at
                # iteration 1382 on a history whose two halves disagree by
                # **4.0 %**, and which our own drift test refuses at every window.
                #
                # The failure mode is backwards, which is why it went unnoticed:
                # it stops precisely the meshes that are behaving well. This is a
                # backstop now -- the criterion that means anything is
                # ``_force_settled``, applied by the worker to the whole history.
                f"CONV_CAUCHY_EPS= {1.0e-4 * self.numerics.force_tolerance:.10g}",
                # Clamped: SU2 stores the Cauchy series in a fixed buffer and
                # refuses more than 1000 elements, failing in the COutput
                # constructor before the mesh is read. Our own window may be
                # longer, because it is applied afterwards to the whole history
                # file rather than to a rolling buffer inside the solver.
                f"CONV_CAUCHY_ELEMS= {min(int(self.numerics.force_window), 1000)}",
                # Must be at least CONV_CAUCHY_ELEMS so the first evaluation
                # has a full window.  At STARTITER + 1 with a 1000-element
                # window, SU2 has 1 value and the criterion is trivially
                # satisfied — the M=25 second-order run exited at iteration
                # 3001 this way while the force was still moving.
                f"CONV_STARTITER= {max(int(self.numerics.minimum_iterations), min(int(self.numerics.force_window), 1000))}",
                "",
            ]

        screen = ["INNER_ITER", "RMS_DENSITY", "LIFT", "DRAG"]
        if unsteady:
            screen = ["TIME_ITER", *screen]
        lines += [
            f"MESH_FILENAME= {self.mesh_filename}",
            "MESH_FORMAT= SU2",
            "TABULAR_FORMAT= CSV",
            "CONV_FILENAME= history",
            "OUTPUT_FILES= ( RESTART, SURFACE_CSV, TECPLOT_ASCII )"
            if self.write_volume
            else "OUTPUT_FILES= ( RESTART, SURFACE_CSV )",
            "SURFACE_FILENAME= surface_flow",
            "VOLUME_FILENAME= volume_flow",
            "RESTART_FILENAME= restart_flow",
            # Named explicitly rather than left to SU2's default, which is
            # `solution.dat` and not the `solution_flow.dat` the symmetry with
            # RESTART_FILENAME suggests. A restart written to one name and
            # looked for under the other fails at the file open, after the
            # prelude has already been paid for.
            "SOLUTION_FILENAME= solution_flow.dat",
            f"SCREEN_OUTPUT= ( {', '.join(screen)} )",
            # AERO_COEFF_SURF is what writes the *per-marker* coefficients
            # (`CD[0]`, `CD[1]`, ...). Without it SU2 reports only the summed
            # force over every monitoring marker, which is precisely the number
            # that is not meaningful when one of those markers is an Euler base.
            "HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF, AERO_COEFF_SURF )"
            if self.base_marker
            else "HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )",
            # Y_PLUS so the mesh's wall spacing can be checked against what it
            # was sized for, rather than assumed to have worked.
            "VOLUME_OUTPUT= ( COORDINATES, SOLUTION, PRIMITIVE )"
            if inviscid
            else "VOLUME_OUTPUT= ( COORDINATES, SOLUTION, PRIMITIVE, Y_PLUS )",
            f"OUTPUT_WRT_FREQ= {self.volume_every if self.volume_every > 0 else 100000}",
            # Number the snapshots instead of overwriting one file. The reader
            # that renders the live frames cannot tell a finished 35 MB dump
            # from one being written, and every heuristic for it failed: on a
            # run writing a snapshot every four seconds, not one frame was
            # rendered for the whole run, silently, because a torn read looks
            # exactly like "nothing new yet". Numbered, a file with a
            # higher-numbered sibling is provably closed. The reader deletes
            # what it has passed, so the disk holds two.
            #
            # SU2 refuses this outright on a transient problem
            # (CConfig.cpp:4009, "incompatible with transient problems"), so a
            # case carrying a motion keeps the overwriting name and the reader
            # falls back to its quiescence heuristic there.
            "WRT_VOLUME_OVERWRITE= NO"
            if self.write_volume and not unsteady
            else "WRT_VOLUME_OVERWRITE= YES",
        ]
        lines += [f"{key}= {value}" for key, value in self.numerics.extra.items()]
        return "\n".join(lines) + "\n"
