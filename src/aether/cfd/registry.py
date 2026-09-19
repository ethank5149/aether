"""A run's identity, on disk, so no process has to be the one holding it.

:mod:`~aether.cfd.run` already detaches the solver and reads its
progress from ``history.csv``, which is what makes a run survive the death of
whatever started it. What it does not do is make that run *findable* again.
``SolveJob`` keeps the handle in a :class:`subprocess.Popen` and the start
instant in :func:`time.monotonic`, and neither of those crosses a process
boundary: a kernel that restarts leaves SU2 running on eight ranks with no way
to see it, stop it, or say how far along it is.

This module is the missing half. One JSON record per run, beside the results,
holding everything a *different* process needs to pick the run back up:

* the specification it was launched from, so the run can be described without
  the launching code still being alive;
* the case directory, so its history and log can be read;
* the pid **and the kernel's own start-time for that pid**, so liveness is a
  question with an answer rather than a guess;
* ``started_at`` as wall-clock rather than monotonic, so elapsed time and an
  ETA are recoverable by a reader that was not running when the job began.

Why ``pid_birth``
-----------------

A bare ``os.kill(pid, 0)`` is not a liveness check on a machine that has been
up for weeks. Pids are reused, and a record from a run three days ago can name
a pid that now belongs to somebody's editor -- which this would report as a
live solve, and which ``stop()`` would then kill. Linux publishes the process
start time in field 22 of ``/proc/<pid>/stat``, in clock ticks since boot, and
a (pid, start-time) pair is unique for the life of the boot. Recording it turns
"is this run alive" from a heuristic into a fact.

The writing discipline
----------------------

There are no locks, because there is no shared mutable state to lock. Exactly
one process writes any given record: the worker, once it has claimed the run.
A UI creates records only in the ``queued`` state, which no worker has touched
yet, and asks for a stop by *creating a separate file* rather than editing the
record -- an atomic create against an atomic replace, with no read-modify-write
in between for two writers to interleave.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import MISSING, asdict, dataclass, field, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, ClassVar, Literal

from aether.paths import cfd_results

__all__ = [
    "TERMINAL",
    "NewerSchema",
    "RunRecord",
    "RunSpec",
    "RunState",
    "archive_dir",
    "archived",
    "cancel_requested",
    "clear_cancel",
    "enqueue",
    "forget",
    "history_rows",
    "load",
    "load_all",
    "pid_birth",
    "pid_namespace",
    "process_alive",
    "queue_dir",
    "reconcile",
    "request_cancel",
    "restore",
    "save",
]

RunState = Literal["queued", "meshing", "solving", "done", "failed", "stopped"]

#: States from which a run does not move again. A reader that finds one of
#: these needs no liveness check and no reconciliation.
TERMINAL: frozenset[str] = frozenset({"done", "failed", "stopped"})


def queue_dir() -> Path:
    """Where run records live.

    Under the results share rather than beside the code, for the same reason
    the cases are: the records outlive any particular checkout, and a bind
    mount is what makes them visible to a worker in one container and a
    dashboard in another.
    """
    return cfd_results() / "queue"


class NewerSchema(ValueError):  # noqa: N818
    """A record written by a build that knows fields this one does not.

    Distinct from a malformed file, which :func:`load_all` is right to skip
    quietly -- a record caught mid-write must not take a dashboard down. This
    one must not be quiet: the run will never start, and an operator staring at
    a queue that is not draining deserves to be told why rather than left to
    infer it.
    """


@dataclass(frozen=True)
class RunSpec:
    """Everything needed to reproduce one run, and nothing about its progress.

    Deliberately flat and JSON-native. This is the notebook's ``_snapshot()``
    given a type and somewhere durable to live -- the same fields, so a queue
    built in the old notebook and one built in the dashboard describe the same
    job.
    """

    body: str
    mach: float
    alpha_deg: float = 0.0
    sideslip_deg: float = 0.0
    """Sideslip angle. Non-zero needs a body that is not axisymmetric to mean
    anything, and a mesh oriented against the flow rather than the axis --
    which :func:`~aether.cfd.geometries.build_domain` now does."""

    altitude_km: float = 45.0
    iterations: int = 6000
    geometry: dict[str, float] = field(default_factory=dict)
    wall_refinement: float = 0.05
    optimize: bool = True
    quality_threshold: float = 0.1
    ranks: int | None = None
    """MPI ranks; ``None`` sizes them from the mesh."""

    volume_every: int = 0
    """Iterations between volume snapshots; ``0`` writes only at the end.

    **Choose this in wall-clock seconds, not iterations.**
    :class:`~aether.cfd.frames.FrameWriter` renders a preview only once a
    snapshot has stopped changing --- quiescent for ``QUIESCENT`` *and* the same
    size across two consecutive polls, which the worker makes every three
    seconds. So a snapshot interval shorter than roughly five wall-clock seconds
    leaves no window to catch, and the preview stops updating while the solve
    runs normally.

    Measured on a 419k-cell sphere-cone at 0.28 s/iteration: ``25`` puts
    snapshots 7 s apart and **one frame in a thousand iterations** was caught,
    against 57 frames on an older, slower case where the same setting meant
    12 s apart. ``100`` restores a ~28 s interval and about ten frames per
    thousand iterations. **A faster solver needs a larger value, not a
    smaller one.**
    """

    nose_cell: float | None = None
    """Cell size for a refinement ball at the stagnation region, if wanted."""

    inviscid: bool = False
    """Solve Euler rather than RANS. **Deprecated: do not set this true.**

    Euler is the wrong simplification for this project. It produces no wall
    heat flux, and integrated heat load is a *carried state* that bounds the
    corridor and drives the ablation chain -- so an Euler solve cannot feed the
    thermal side at all. It also drops skin friction, which is a real part of
    the drag it would be used to measure.

    The field exists because the mismatch it was added to fix is real, and
    naming it is how the fix stays findable. **The mismatch is to be resolved on
    the mesh side, not here**: build the prism boundary layer that k-omega SST
    requires, which :func:`~aether.cfd.geometries.build_domain` now does
    under ``viscous=True``.

    Defaults to ``False`` so records written before this field existed still
    describe what they actually ran -- they ran RANS, and reloading one must
    keep saying so.

    :func:`~aether.cfd.geometries.build_domain` calls ``inviscid_domain``,
    whose own docstring says it is *"an isotropic tetrahedral domain with no
    prism layer, which is the right choice for an Euler run"*. Every run through
    this queue nonetheless went out as ``SOLVER= RANS`` with ``KIND_TURB_MODEL=
    SST``, because :class:`~aether.cfd.config.GasModel` defaults to
    ``inviscid=False`` and nothing here ever set it.

    k-omega SST needs wall-resolved prism layers at y+ of order one. On an
    isotropic tet mesh with no wall clustering its production terms are computed
    on cells orders of magnitude too coarse, and they blow up: across every
    diverged case measured on 2026-09-10, ``rms[k]`` and ``rms[w]`` are the
    fastest-growing residuals, frequently ahead of the momentum equations, while
    a converged case has every residual flat or falling.

    That mismatch accounts for the whole failure pattern -- incidence, Mach,
    altitude, geometric complexity, and the refined mesh being *worse* than the
    coarse one, which is otherwise inexplicable: across every diverged case
    measured on 2026-09-10, ``rms[k]`` and ``rms[w]`` are the fastest-growing
    residuals, frequently ahead of the momentum equations, while a converged
    case has every residual flat or falling.
    """

    viscous_mesh: bool = False
    """Extrude a prism boundary layer. **False here, and measured.**

    A prism stack needs room. At Mach 8 and 45 km the boundary layer is 0.109 m
    against a 0.06 m nose radius, so columns fold at the stagnation point and
    the extrusion is truncated exactly where the gradients are steepest --
    ``BoundaryLayerTruncated`` says so out loud. Every run on such a mesh
    diverged, **including alpha = 0, which converges on the isotropic one.**

    So the isotropic mesh is not a compromise here, it is the only one that
    builds cleanly on this geometry at this condition. Laminar Navier-Stokes
    tolerates it because there is no turbulence model whose source terms depend
    on wall resolution; the wall shear it returns is under-resolved, and that
    belongs in the error budget rather than in a crash.
    """

    turbulence: str | None = None
    """RANS model, or ``None`` for **laminar** Navier-Stokes. Laminar here.

    This defaulted to ``"SST"`` so that records written before the field existed
    would still describe what they actually ran. That is a correct requirement
    and it was solved in the wrong place: a dataclass default serves both
    *deserialising an old record* and *constructing a new run*, and those two
    want opposite answers. The legacy fill lives in :meth:`RunRecord.from_json`
    now, keyed on the field being **absent** from the payload, which is the
    thing that actually distinguishes an old record -- a new one always writes
    the key, including when it is null.

    Leaving it here cost a suite. The dashboard form does not expose the
    setting, so every run queued from it took the default, and on 2026-09-11 a
    whole baseline suite -- hemisphere, sphere-cone, biconic -- ran k-omega SST
    on a Mach 8 flow at Re = 1.4e4 and every one of them hit a NaN at around
    iteration 217 with a flat residual. Wrong physics that also does not run.

    ``Re_L`` is 6.3e5 at Mach 8 and 45 km on this body, well under the 1e6 where
    hypersonic boundary layers typically transition -- so k-omega SST was
    modelling turbulence that is not there. It was also being asked to do it on
    a mesh that cannot support it: the boundary layer is 0.109 m against a
    0.06 m nose radius, so a prism stack folds at the stagnation point, and at
    y+ = 30 -- the *minimum* for wall functions -- the first cell is already
    1.2x the whole boundary layer. There is no wall-resolved mesh and no
    wall-function mesh to build here.

    Laminar keeps what Euler throws away: skin friction, and wall heat flux for
    the ablation chain. → :attr:`inviscid`, which is deprecated for that reason.
    """

    variant: str = ""
    """Deliberate fork of an otherwise identical case, appended to the name.

    For side-by-side diagnostics that share every physical setting -- two
    prelude lengths, two iteration budgets -- where the point *is* to keep both
    on disk at once. Explicit, because a fork that happens by accident is how a
    mesh study overwrites its own baseline.
    """

    cfl_max: float = 50.0
    """Ceiling the CFL adapter may ramp to. The drafted default is 50."""

    limiter_coefficient: float = 0.05
    r"""Venkatakrishnan-Wang ``K``. **Smaller is more limiting**, hence more
    dissipative and more robust: the threshold is :math:`(K\,\Delta x)^3`, so a
    smaller ``K`` keeps the limiter active where a larger one lets the
    reconstruction run unlimited."""

    settle_tolerance: float = 0.0
    r"""Stop the solver when the force stops moving. ``0`` keeps the fixed count.

    A fixed iteration count is a **bound, not a criterion**, and it is wrong in
    both directions: the isotropic mesh settles into its limit cycle inside two
    hundred iterations and then burns eight hundred more, while a wall-resolved
    mesh was still falling one percent per hundred iterations at the cap
    → ``a-wall-resolved-mesh-does-not-settle-in-a-thousand-iterations``.

    The criterion is :func:`~aether.cfd.solver._force_settled`, which
    handles both regimes: a limit cycle is judged on the disagreement between
    half-window means once enough oscillation periods have been resolved, and a
    monotone convergence on the same disagreement once the ripple has fallen
    below the tolerance. Validated against three real histories -- it refuses a
    transient at 5.2 % drift and accepts a converged run at 1.7e-6.

    ``0.0`` by default because enabling it means **terminating a live MPI job**
    on a judgement about a number, and that should be asked for rather than
    inherited.
    """

    refinement: float = 1.0
    r"""Uniform grid-refinement level for the shock-fitted mesher.

    Scales the **surface** panel counts and the **wall-normal** layer count
    together, so a triplet has one refinement ratio :math:`r` and therefore one
    observed order to compute. Refining in a single direction does not.

    The wall cell is deliberately **not** scaled: it is set by the cell Reynolds
    number, which is physics rather than resolution, and shrinking it with the
    grid would confound two effects. The normal direction refines by lowering
    the growth ratio instead -- ``growth ** (1/refinement)`` -- which fits more
    cells into the same shock layer between the same wall cell and the same
    outer boundary.

    Only read when :attr:`shock_fitted`; ``wall_refinement`` sizes the isotropic
    path and has no effect on this one.
    """

    shock_fitted: bool = False
    """Build a wall-normal block marched to the shock, not an isotropic tet box.

    The isotropic mesher put **zero** cells inside the 9 mm shock standoff at
    1.36 million elements, and refining it 9.5-fold moved that count from 2 to
    0 because the cells went everywhere except where the physics is. A
    shock-fitted block puts 121 there for twice the cell count
    → :mod:`aether.cfd.extrude`.

    Defaults to ``False`` so archived records still describe the mesh they ran.
    """

    cell_reynolds: float = 1.0
    r"""Wall cell size, as :math:`\mathrm{Re}_{cell} = \rho a \Delta s/\mu`.

    LAURA's shipped default is 1, which is the criterion for **heat transfer**
    accuracy. Only read when :attr:`shock_fitted`; the isotropic path has no
    way to honour it. Relaxing it is linear in the wall cell, so 10 buys exactly
    one order of magnitude and no more.
    """

    convective: str = "AUSMPLUSUP2"
    """Upwind flux. The carbuncle is a property of the *scheme*, not only the mesh.

    Roe is the most carbuncle-prone at a strong bow shock; HLLE is immune and too
    dissipative to keep a boundary layer; the AUSM family sits between. SU2 v8
    accepts ``ROE``, ``AUSM``, ``AUSMPLUSUP2``, ``SLAU``, ``HLLC``, ``MSW`` and
    ``ROE_LOW_DISSIPATION`` on the perfect-gas path, and ``AUSM``,
    ``AUSMPLUSUP2``, ``MSW`` and ``ROE`` on NEMO.
    """

    limiter: str = "VENKATAKRISHNAN_WANG"
    """Slope limiter. See :attr:`limiter_iterations` before changing this."""

    limiter_iterations: int = 1000
    """Freeze the limiter after this many iterations; ``0`` never freezes it.

    Venkatakrishnan-family limiters are **not differentiable**, and the standard
    consequence is a limit cycle: the residual stalls one or two decades down and
    the forces oscillate forever instead of settling. That is this pipeline's
    exact signature --- measured 2026-09-11 across a 9.5x range in cell count, the
    density residual floors at -0.82, -0.98 and -1.29 while the drag coefficient
    ripples 2.3 to 3.2 percent of its mean with an erratic period. A steady flow
    refined nine times over does not do that.

    Freezing the limiter after the shock has formed is the textbook cure and is
    what ``LIMITER_ITER`` exists for. **1000 by default**, which is past the
    prelude and past the point the residual reaches its floor on every body
    measured; the legacy fill is in :meth:`RunRecord.from_json`. ``0`` was the
    historical behaviour and it was kept as the default for the sake of old
    records, which is the mistake :attr:`turbulence` documents
    → [[a-default-that-served-two-masters]].
    """

    entropy_fix: float = 0.1
    """Roe entropy-fix coefficient. Only read by the Roe-family schemes."""

    wake: bool = False
    """Resolve the base and near wake instead of correlating them.

    A shock-fitted block ends at the base plane, which is right for forebody
    aerodynamics -- outside the bow shock everything is freestream -- and cannot
    produce a base pressure. That has come from
    ``closure.base_axial_coefficient``, an empirical :math:`C_{p,b} = -1/M^2`,
    all along: **7 % of total drag at Mach 10 and 32 % at Mach 4** arriving from
    a one-line correlation rather than from the solver.

    With this the aft face is swept downstream eight base diameters, flaring
    along the shock envelope so the outer boundary stays outside it, and the
    base disc becomes wall. The two blocks are welded, not stacked: see
    ``extrude._join_wake``.

    A near wake is separated and unsteady, so a base pressure taken from a
    *steady* solve is the value of a limit cycle at whichever iteration it
    stopped. Pair this with :attr:`unsteady`, and with
    :attr:`hybrid_rans_les` if the separated region is to be resolved rather
    than modelled.
    """

    hybrid_rans_les: str | None = None
    """Detached-eddy model (``"SA_DDES"``), or ``None`` for pure RANS.

    Only meaningful with :attr:`unsteady`: a detached-eddy closure integrated to
    a steady state is a contradiction. See
    :attr:`aether.cfd.config.PerfectGas.hybrid_rans_les`.
    """

    unsteady: bool = False
    """Integrate in time on a static body, rather than marching to steady state.

    The third field in this class to expose a capability that was implemented
    and unreachable -- after :attr:`first_order_iterations` and :attr:`regime`.
    ``config.SU2Case`` has emitted dual-time stepping all along, but only when a
    :class:`~aether.cfd.config.PitchingMotion` was attached, so
    time-accuracy was reachable **only** by making the body oscillate. A static
    body in an unsteady flow -- a separated base, a shear layer, shock buffet --
    could not be expressed.

    That is not a convenience gap. A separated near wake is unsteady as a matter
    of physics, and a steady solver pointed at one returns the value of a limit
    cycle at whatever iteration it stopped: not grid-convergent, and not
    rescued by averaging the force history, because the field it came from never
    satisfied the steady equations. Base pressure has to be integrated in time.

    The step and duration are **derived**, not carried here: the worker sizes
    them from the base diameter and freestream speed through
    :meth:`~aether.cfd.config.UnsteadyRun.for_base_flow`, so the physics
    sets them rather than a number typed into a spec.
    """

    unsteady_cycles: float = 30.0
    """Shedding cycles to integrate. The mean base pressure is built from these
    less :attr:`unsteady_transient_cycles`."""

    unsteady_transient_cycles: float = 5.0
    """Cycles discarded before statistics begin."""

    unsteady_steps_per_cycle: int = 30
    """Physical steps per shedding cycle -- the time-step refinement knob, and
    the one a temporal convergence study varies."""

    regime: str | None = None
    """Gas model, overriding the Mach-number default: ``"perfect-gas"``,
    ``"nonequilibrium"``, ``"low-mach"``, or ``None`` for
    :func:`~aether.cfd.config.regime_for`.

    ``regime_for`` is documented as "a default, not a law; pass a regime
    explicitly to overrule it" -- and the queue had no way to. Every case at
    Mach 10 and above therefore went to NEMO, and **not one of them has ever
    produced a usable number**: Mach 10 and one Mach 20 case segfault before the
    first iteration, and Mach 12, 14, 16, 18 and 20 all diverge to residuals of
    +9 to +13 decades, returning drag coefficients of 50, 2642, 26, 228 and
    -4.4. The same one-implemented-two-unreachable shape as
    :attr:`first_order_iterations` before it was wired through.

    Forcing perfect gas above Mach 10 buys a number with a **stated** bias --- a
    perfect gas has no vibrational modes and no dissociation, so it overpredicts
    shock-layer temperature, and the drag it returns is an upper bound on the
    pressure contribution --- in place of no number at all. That is a
    substitution to make deliberately and record, not silently.
    """

    first_order_iterations: int = 500
    """First-order prelude length; ``0`` disables it.

    Non-zero is close to mandatory above about Mach 6. A case initialised from
    uniform freestream has to form its bow shock and sweep it back across the
    mesh, and a MUSCL reconstruction across the forming discontinuity produces
    negative temperatures --- so the second-order run dies on a NaN during the
    transient, at any CFL. See :attr:`aether.cfd.config.Numerics.first_order_iterations`,
    which documents the measurement this default was chosen against.

    **500 by default, and the legacy fill is in** :meth:`RunRecord.from_json`.
    It defaulted to 0 so that records written before the field existed would
    still describe what they ran -- the same one-default-two-callers mistake as
    :attr:`turbulence` → [[a-default-that-served-two-masters]], and it cost the
    same thing. This field's own docstring calls a prelude "close to mandatory
    above about Mach 6", and the note recording its addition ends with *"a
    default-valued field is silently disabled everywhere it is not explicitly
    passed"*. It was then left disabled for every run the queue ever made.
    """

    def label(self) -> str:
        """A short human name, the one the case directory is built from."""
        return f"{self.body} M{self.mach:g} a{self.alpha_deg:+g} h{self.altitude_km:g}"

    def case_name(self) -> str:
        """The directory name for this case.

        Matches the notebook's scheme, so an existing results tree is still
        addressed by the same names and a re-queued point lands on its own
        previous directory rather than a second one beside it.

        The geometry is emitted in **declared parameter order** -- the order
        ``ReferenceBody.parameters`` lists it and therefore the order the form
        builds the dict in -- and explicitly not sorted. Sorting looks tidier
        and is wrong: the notebook wrote ``nos0.06-len2-hal10`` for a
        sphere-cone, so an alphabetised ``hal10-len2-nos0.06`` names a
        directory that does not exist and silently starts the case again beside
        the one already solved.
        """
        tag = "-".join(f"{k[:3]}{v:g}" for k, v in self.geometry.items())
        stem = (
            f"m{self.mach:07.4f}-a{self.alpha_deg:+05.2f}-h{self.altitude_km:04.1f}"
        )
        name = f"{stem}-{tag}" if tag else stem
        return name + self._variant_tag()

    #: Fields that change *how* a run is computed or observed, not *what* it
    #: computes. Excluded from the case name so two runs differing only in rank
    #: count or snapshot cadence share a directory, which is correct.
    #:
    #: **Iteration budgets are in here deliberately.** A case run for longer is
    #: the same case, and :meth:`case_name`'s contract is that a re-queued point
    #: lands on its own previous directory. If two budgets disagree then one has
    #: not converged, and that is a question to answer from a single history
    #: rather than by forking a second directory. Use :attr:`variant` to fork on
    #: purpose.
    RUNTIME_ONLY: ClassVar[frozenset[str]] = frozenset(
        {"ranks", "volume_every", "iterations", "first_order_iterations", "variant"}
    )

    #: Fields already carried by the stem or the geometry tag.
    IDENTITY: ClassVar[frozenset[str]] = frozenset(
        {"body", "mach", "alpha_deg", "altitude_km", "geometry"}
    )

    #: Short names for everything else. **Every remaining field must appear
    #: here**, and :meth:`_variant_tag` raises if one does not -- which is the
    #: point. A field added without a tag would otherwise alias onto the run it
    #: was meant to be compared against, silently, which has already destroyed
    #: one converged baseline.
    ABBREVIATIONS: ClassVar[dict[str, str]] = {
        "sideslip_deg": "b",
        "iterations": "it",
        "wall_refinement": "wr",
        "optimize": "opt",
        "quality_threshold": "qt",
        "nose_cell": "nc",
        "viscous_mesh": "vmesh",
        "turbulence": "turb",
        "inviscid": "euler",
        "cfl_max": "cfl",
        "limiter_coefficient": "vk",
        "shock_fitted": "fit",
        "refinement": "res",
        "settle_tolerance": "settle",
        "cell_reynolds": "recell",
        "convective": "flux",
        "limiter": "lim",
        "limiter_iterations": "freeze",
        "entropy_fix": "efix",
        # The gas model is the single largest thing a spec can change, so it has
        # to reach the directory name: a perfect-gas Mach 10 and a NEMO Mach 10
        # are different answers to the same question.
        "regime": "gas",
        # Unsteadiness and the closure both change the answer, so both
        # reach the directory name: a DDES base-flow run and a steady
        # RANS run of the same condition are different results.
        "wake": "wake",
        "hybrid_rans_les": "des",
        "unsteady": "unst",
        "unsteady_cycles": "cyc",
        "unsteady_transient_cycles": "tran",
        "unsteady_steps_per_cycle": "spc",
    }

    def _variant_tag(self) -> str:
        """Suffix naming every field that **differs from its default**.

        Derived, not hand-written, and that is the whole point. The previous
        version tested each field against a default it restated inline, so when
        ``turbulence``'s default moved from ``None`` to ``"SST"`` its truthiness
        test silently began appending ``-turbSST`` to every case name and broke
        the documented scheme. Comparing against the dataclass's own default
        cannot drift that way.

        Two rules decide membership:

        * a field already in the stem or geometry tag is skipped;
        * a field that changes only *how* the answer is computed or observed --
          MPI ranks, snapshot cadence -- is skipped, so those do not fork a
          directory;

        and **anything else must have an abbreviation or this raises**, because
        a field that quietly fails to appear is how a mesh study overwrites the
        baseline it exists to be compared against.
        """
        defaults: dict[str, Any] = {}
        for f in dataclass_fields(self):
            if f.default is not MISSING:
                defaults[f.name] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                defaults[f.name] = f.default_factory()  # type: ignore[misc]

        parts: list[str] = []
        if self.variant:
            parts.append(str(self.variant))
        for f in dataclass_fields(self):
            name = f.name
            if name in self.IDENTITY or name in self.RUNTIME_ONLY:
                continue
            if name not in self.ABBREVIATIONS:
                msg = (
                    f"RunSpec.{name} has no entry in ABBREVIATIONS, so it cannot "
                    "appear in the case name and two runs differing only in it "
                    "would share a directory. Add an abbreviation, or list it in "
                    "RUNTIME_ONLY if it does not change the answer."
                )
                raise RuntimeError(msg)
            value = getattr(self, name)
            if name not in defaults or value == defaults[name]:
                continue
            short = self.ABBREVIATIONS[name]
            if name == "turbulence" and value is None:
                parts.append("lam")  # laminar reads better than "turbNone"
            elif isinstance(value, bool):
                parts.append(short)
            elif isinstance(value, float):
                signed = name == "sideslip_deg"
                parts.append(f"{short}{value:+g}" if signed else f"{short}{value:g}")
            else:
                parts.append(f"{short}{value}")
        return "-" + "-".join(parts) if parts else ""


@dataclass
class RunRecord:
    """One run's durable state. The worker owns it once it leaves ``queued``."""

    run_id: str
    spec: RunSpec
    state: RunState = "queued"
    case_dir: str | None = None
    queued_at: float = 0.0
    started_at: float | None = None
    """Wall-clock instant the *solver* started, not the mesh.

    Wall-clock on purpose. :func:`time.monotonic` is only comparable within one
    process, so a monotonic start instant is exactly the field that a reader
    which was not running at launch cannot use -- which is the whole failure
    this module exists to fix.
    """

    finished_at: float | None = None
    pid: int | None = None
    pid_birth: int | None = None
    pid_ns: str | None = None
    """The PID namespace :attr:`pid` was issued in.

    Recorded because the panel and the worker are usually different containers.
    A pid from another namespace cannot be checked and cannot be signalled, so
    a reader that finds one must say so rather than guess -- and must certainly
    not signal the same number in its own namespace.
    """

    ranks: int | None = None
    elements: int | None = None
    total: int | None = None
    returncode: int | None = None
    message: str = ""
    """Why it ended, when that is not obvious from the state alone."""

    @property
    def path(self) -> Path:
        return queue_dir() / f"{self.run_id}.json"

    @property
    def directory(self) -> Path | None:
        return None if self.case_dir is None else Path(self.case_dir)

    @property
    def history(self) -> Path | None:
        directory = self.directory
        return None if directory is None else directory / "history.csv"

    @property
    def log(self) -> Path | None:
        directory = self.directory
        return None if directory is None else directory / "su2.log"

    @property
    def active(self) -> bool:
        return self.state not in TERMINAL

    @property
    def elapsed(self) -> float | None:
        """Seconds the solver has been running, or ran for.

        Recoverable from any process, which is the point of storing a
        wall-clock start.
        """
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def alive(self) -> bool:
        """Whether the solver this record names is still running.

        Answerable only from the namespace that issued the pid. From anywhere
        else this is ``False`` -- not because the solver is known to be dead,
        but because this process cannot know, and ``reconcile`` must never
        settle a run on the strength of a check it was not entitled to make.
        Use :attr:`checkable` to tell the two apart.
        """
        return self.checkable and process_alive(self.pid, self.pid_birth)

    @property
    def checkable(self) -> bool:
        """Whether this process is in a position to judge :attr:`pid` at all."""
        if self.pid is None:
            return False
        return self.pid_ns is None or self.pid_ns == pid_namespace()

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = asdict(self.spec)
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RunRecord:
        data = dict(payload)
        spec = data.pop("spec", {})
        # A record written by an *older* version is fine: missing fields take
        # their defaults, so a queue is not invalidated by a schema that grew.
        #
        # A record written by a *newer* version is not fine, and dropping its
        # unknown fields silently is how this went wrong on 2026-09-11. The
        # long-lived worker process was holding a RunSpec from before
        # ``convective`` existed. It read eight queued records, dropped that
        # field on the way in, and then ``save`` wrote each record back
        # **without it** -- so the flux scheme was erased from disk, every case
        # name lost its ``-fluxHLLC`` suffix, all eight collided with the
        # baseline directory, and the first of them overwrote a converged
        # result while running a scheme nobody asked for.
        #
        # Refusing is strictly better. A worker that stops is a worker you
        # restart; a worker that silently rewrites your queue to match its own
        # idea of the schema destroys the thing it was asked to protect.
        known = set(cls.__dataclass_fields__)
        spec_known = set(RunSpec.__dataclass_fields__)
        # A record written before ``turbulence`` existed ran k-omega SST, and
        # reloading one must keep saying so. Keyed on absence, not on the value:
        # every record written since carries the key, so a null there is a
        # deliberate laminar run and must not be overwritten.
        # Fields whose default was changed after the fact, and what a record
        # written before each existed actually ran. Keyed on absence, because
        # every record written since carries the key -- so an explicit value,
        # including a null or a zero, is a deliberate choice and is kept.
        legacy: dict[str, Any] = {
            "turbulence": "SST",
            "first_order_iterations": 0,
            "limiter_iterations": 0,
        }
        if isinstance(spec, dict):
            missing = {k: v for k, v in legacy.items() if k not in spec}
            if missing:
                spec = {**spec, **missing}
        unknown = sorted(set(spec) - spec_known)
        if unknown:
            msg = (
                f"run record {data.get('run_id', '?')} carries RunSpec fields this "
                f"build does not know: {', '.join(unknown)}. It was written by a "
                "newer version. Loading it would drop those fields and saving it "
                "would erase them from disk, which silently merges distinct runs "
                "into one case directory. Restart this process on the current code."
            )
            raise NewerSchema(msg)
        return cls(
            spec=RunSpec(**{k: v for k, v in spec.items() if k in spec_known}),
            **{k: v for k, v in data.items() if k in known and k != "spec"},
        )


def pid_namespace() -> str | None:
    """This process's PID namespace, as the kernel's inode string.

    A pid is only meaningful inside the namespace that issued it, and this
    subsystem is routinely spread across several: the worker in one container,
    the panel in another, both on ``--net=host`` but each with its own pid
    space. Without this, a reader checks ``os.kill(pid, 0)`` against *its own*
    namespace and gets an answer about an unrelated process -- or, far more
    often, no process at all, and reports a perfectly healthy worker as absent.

    ``None`` where ``/proc`` does not answer, in which case callers fall back
    to the checks that do not need it.
    """
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def _stat_fields(pid: int) -> list[str] | None:
    """``/proc/<pid>/stat`` from the state character on, or ``None``.

    The comm field is cut at the *last* closing parenthesis rather than by
    splitting on whitespace: a process is free to call itself ``S U 2 (b)``,
    and every parser that splits the line naively gets that wrong.

    What comes back is field 3 onward, so index 0 is the state character and
    index 19 is ``starttime`` -- field 22 counting from one, less the two that
    were just cut away.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (OSError, ValueError):
        return None
    try:
        return stat[stat.rindex(")") + 1 :].split()
    except ValueError:
        return None


def pid_birth(pid: int) -> int | None:
    """The kernel's start time for ``pid``, in clock ticks since boot.

    ``None`` when the process is gone, or on a platform without ``/proc``
    (where the pair degrades to a bare pid check, which is what the rest of
    this module already tolerates).
    """
    fields = _stat_fields(pid)
    if fields is None or len(fields) < 20:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def process_alive(pid: int | None, birth: int | None = None) -> bool:
    """Whether ``pid`` names a live process, and the *same* one as before.

    Two things this refuses to call alive, because reporting a finished run as
    running is the more dangerous of the two errors -- a stop request built on
    it signals whatever holds the pid now:

    **A recycled pid.** When ``birth`` is given it must match. Pids wrap, and a
    record from three days ago can name one that belongs to somebody's editor
    today.

    **A zombie.** ``os.kill(pid, 0)`` succeeds against a process that has
    exited but not yet been reaped, so a bare signal probe reports a finished
    solver as still solving for as long as its parent takes to wait on it --
    and if the parent has itself died, until init gets round to it.
    """
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, owned by somebody else. Believe the existence, not the access.
        pass
    except OSError:
        return False

    fields = _stat_fields(pid)
    if fields is None:
        # No /proc, or it vanished between the signal and the read. The bare
        # signal probe is all there is.
        return True
    if fields and fields[0] == "Z":
        return False
    if birth is None:
        return True
    current = pid_birth(pid)
    return current is None or current == birth


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace ``path`` in one step, so no reader ever sees half a record.

    ``os.replace`` is atomic within a filesystem, and the temporary is made in
    the same directory to keep it there -- a temporary in ``/tmp`` would make
    this a copy across devices and lose the guarantee.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # **Not** ``sort_keys``. It reads as tidiness and it is data loss: the keys
    # of ``spec.geometry`` are in the order the body declares its parameters,
    # and :meth:`RunSpec.case_name` builds the case directory name from that
    # order. Sorting them here re-sorted them behind that method's back, so a
    # run queued as `nos0.06-len2-hal10` was solved into `hal10-len2-nos0.06`
    # and a second directory appeared beside the one already on disk. The
    # dataclass field order gives stable output without touching the contents.
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def save(record: RunRecord) -> RunRecord:
    """Persist a record. Only its owner should call this."""
    _write_atomic(record.path, record.to_json())
    return record


def enqueue(spec: RunSpec, *, run_id: str | None = None) -> RunRecord:
    """Add a run to the queue and return its record.

    The id carries the queue instant in front of a random suffix: readable at a
    glance, and unique even for two runs queued inside the same second. Order
    comes from ``queued_at`` rather than from the id, because that second of
    resolution is exactly where a lexical sort stops being submission order.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    record = RunRecord(
        run_id=run_id or f"{stamp}-{uuid.uuid4().hex[:6]}",
        spec=spec,
        state="queued",
        queued_at=time.time(),
        total=spec.iterations,
    )
    return save(record)


def load(run_id: str) -> RunRecord | None:
    """One record by id, or ``None`` if it is not there.

    :class:`NewerSchema` **propagates**. It is the one parse failure that is
    not an absent record: the file is there, it is well formed, and this build
    is too old to read it. Returning ``None`` for that is indistinguishable
    from "no such run", which is precisely the quiet answer the guard exists to
    prevent -- and :func:`load_all` already reports it loudly, so a point
    lookup that swallowed it would make the two disagree about the same file.
    """
    path = queue_dir() / f"{run_id}.json"
    try:
        return RunRecord.from_json(json.loads(path.read_text()))
    except NewerSchema:
        raise
    except (OSError, ValueError, TypeError):
        return None


def load_all(*, active_only: bool = False) -> list[RunRecord]:
    """Every record, in submission order.

    Ordered by ``queued_at``, which is a float and therefore actually
    distinguishes two runs queued in the same second; the id is only a
    tie-break.

    A record that fails to parse is skipped rather than raised on. This is
    read by a dashboard on a timer, and one malformed file -- a record caught
    mid-write by a filesystem that does not honour the replace, say -- must not
    take the whole view down.
    """
    directory = queue_dir()
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(RunRecord.from_json(json.loads(path.read_text())))
        except NewerSchema as newer:
            # Loud, once per record per scan. A silently skipped run looks
            # exactly like a queue that has stopped for no reason.
            print(f"registry: skipping {path.name} -- {newer}", file=sys.stderr)
        except (OSError, ValueError, TypeError):
            continue
    if active_only:
        records = [r for r in records if r.active]
    records.sort(key=lambda r: (r.queued_at, r.run_id))
    return records


def _cancel_path(run_id: str) -> Path:
    return queue_dir() / f"{run_id}.cancel"


def request_cancel(run_id: str) -> None:
    """Ask the worker to stop a run.

    A separate file rather than a field on the record, because the requester is
    not the record's owner. Editing the record from a second process means a
    read-modify-write racing the worker's own writes, and the loser of that
    race is a silently dropped stop.
    """
    path = _cancel_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def cancel_requested(run_id: str) -> bool:
    return _cancel_path(run_id).exists()


def clear_cancel(run_id: str) -> None:
    _cancel_path(run_id).unlink(missing_ok=True)


def archive_dir() -> Path:
    """Where forgotten records go. Outside the queue, so scans do not see them."""
    return queue_dir() / "archive"


def forget(run_id: str) -> None:
    """Retire a record. The case directory and the record are both kept.

    Refuses to forget a live run: the record is the only handle on that
    process, and deleting it is how a solve becomes unkillable.

    The record is **moved to** :func:`archive_dir`, not deleted. Clearing
    finished runs is a tidying action and reads like one, but the record is the
    only index into a case directory -- the results survive a delete and become
    invisible to the report and the dashboard alike, which is a worse outcome
    than the button suggests. Archiving costs a few kilobytes and makes the
    action reversible with :func:`restore`.
    """
    try:
        record = load(run_id)
    except NewerSchema:
        # A record this build cannot read is a record this build did not
        # start, so it is not running and archiving it is safe. Refusing here
        # would leave it wedged in the queue with no way to clear it, which
        # turns a loud guard into a trap.
        record = None
    if record is not None and record.active and record.alive():
        raise RuntimeError(f"{run_id} is still running -- cancel it first")
    source = queue_dir() / f"{run_id}.json"
    if source.exists():
        target = archive_dir()
        target.mkdir(parents=True, exist_ok=True)
        source.replace(target / source.name)
    clear_cancel(run_id)


def archived() -> list[str]:
    """Run ids that have been retired, newest first."""
    directory = archive_dir()
    if not directory.is_dir():
        return []
    return sorted((p.stem for p in directory.glob("*.json")), reverse=True)


def restore(run_id: str) -> bool:
    """Put an archived record back in the queue. ``False`` if there is none."""
    source = archive_dir() / f"{run_id}.json"
    if not source.exists():
        return False
    source.replace(queue_dir() / source.name)
    return True


def reconcile(record: RunRecord) -> RunRecord:
    """Settle a record whose owner died without finishing it.

    A worker killed mid-solve leaves a record saying ``solving`` forever. On
    the next look the pid is checked: still alive means the solver outlived its
    supervisor and the record is honest, so it is left alone and can be picked
    back up. Dead means the run ended unobserved, and it is settled from what
    it wrote -- reaching the requested iteration count is the only evidence of
    success available after the fact, since the exit status died with the
    parent that could have read it.

    A run still in ``meshing`` with a dead worker is a different case and is
    marked failed: meshing happens *in* the worker, so there is no detached
    process to have survived it. Only a process that is not the one meshing
    ever sees that state, since the build is synchronous inside a worker pass.

    A ``queued`` run is left alone, and the distinction matters: it has no pid
    for the same reason a dead run has none -- it has not started yet -- so
    treating "no live process" as evidence of death would settle every waiting
    run as a failure the instant a worker looked at the queue.

    A run whose pid belongs to *another* PID namespace is settled rather than
    left, and that is right rather than merely convenient: the only way a
    record carries a foreign namespace is that the container which launched the
    solver is gone, and a container taking its pid namespace with it takes
    every process in it too. Call this only from the worker; a panel that
    reconciled would be settling runs it is not entitled to judge.
    """
    if record.state == "queued" or not record.active or record.alive():
        return record

    if record.state == "meshing":
        return save(
            replace(
                record,
                state="failed",
                finished_at=time.time(),
                message="the worker died while meshing",
            )
        )

    reached = history_rows(record.history)
    complete = record.total is not None and reached >= record.total
    return save(
        replace(
            record,
            state="done" if complete else "failed",
            finished_at=time.time(),
            message=(
                ""
                if complete
                else f"the solver exited unobserved at iteration {reached}"
            ),
        )
    )


def history_rows(history: Path | None) -> int:
    """Iterations written, counted without parsing the numbers.

    A long run's history reaches tens of megabytes and this is called on a
    timer, so it counts newlines over a buffer instead of going through
    :mod:`csv` -- the value wanted is a row count, not the rows.
    """
    if history is None or not history.exists():
        return 0
    try:
        with history.open("rb") as handle:
            lines = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 20), b""))
    except OSError:
        return 0
    return max(0, lines - 1)
