r"""Refuse a run that cannot work, before it costs an hour.

Nearly every failure on this pipeline has been found the same way: queue a run,
wait ten minutes, inspect the wreckage. That is a terrible loop, and most of
what it caught did not need a solver to catch.

Measured against one day's failures, these are the ones that were findable in
seconds and were not:

============================================  ==========================
failure                                       what would have caught it
============================================  ==========================
prelude aborts on a marker the mesh lacks     markers in config vs mesh
forces read as NaN                            columns the config will emit
two specs sharing a case directory            case name against the queue
``resolution`` wired to the wrong call site   cell count against the level
a mesh with inverted cells                    quality of the built mesh
a body no shock provider covers               the provider's own refusal
============================================  ==========================

So this builds the mesh, renders the config, and checks them **against each
other** -- which is where the failures live, because each half is correct alone.
It costs seconds and it runs before anything is queued.

What it deliberately does **not** do is predict whether the solve converges. A
scheme that limit-cycles and a shock layer that clips are properties of the
flow, not of the inputs, and claiming to foresee them is how a green light stops
meaning anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Check", "Preflight", "preflight"]


@dataclass(frozen=True)
class Check:
    """One thing that was looked at, and what was found."""

    name: str
    passed: bool
    detail: str
    advisory: bool = False
    """True for something worth seeing that is not a reason to refuse.

    Snapshots cost a second each and a run that skips them is a legitimate
    choice -- but it is a choice whose consequence is invisible until you sit
    watching an empty panel for an hour, so it is said out loud.
    """

    def __str__(self) -> str:
        mark = "ok" if self.passed else ("warn" if self.advisory else "FAIL")
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class Preflight:
    """Everything checked about one spec."""

    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str, advisory: bool = False) -> None:
        self.checks.append(Check(name, passed, detail, advisory))

    @property
    def ok(self) -> bool:
        """Advisories do not refuse a run; they are reported alongside it."""
        return all(check.passed or check.advisory for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and not c.advisory]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.advisory]

    def report(self) -> str:
        return "\n".join(str(check) for check in self.checks)


def preflight(
    spec: Any, *, root: Path | None = None, exclude: str | None = None, deep: bool = True
) -> Preflight:
    """Build what a run would build and check it agrees with itself.

    ``exclude`` is the run id being checked, when it is already in the queue.
    Without it the name check finds the record itself and calls it a collision,
    which refuses every run -- caught by the worker's own tests before it
    reached anything real.

    ``deep`` builds the mesh. The queue form wants that, because it is deciding
    whether to commit an hour. The worker does not: it is about to build the
    same mesh anyway, and the extruder already refuses an inverted one.
    """
    import numpy as np

    result = Preflight()

    # ---- the surface
    try:
        from aether.cfd.geometries import build_surface

        surface = build_surface(spec.body, resolution=spec.refinement, **spec.geometry)
        faces = len(np.asarray(surface.faces))
        result.add("surface", bool(surface.is_closed), f"{faces} faces, closed={surface.is_closed}")
    except Exception as error:
        result.add("surface", False, f"{type(error).__name__}: {error}")
        return result

    # ---- the case name, against every live record
    try:
        from aether.cfd.registry import load_all

        name = spec.case_name()
        clash = [
            r.run_id[-6:]
            for r in load_all()
            if r.spec.case_name() == name
            and r.state in {"queued", "meshing", "solving"}
            and r.run_id != exclude
        ]
        result.add(
            "case name",
            not clash,
            name if not clash else f"{name} -- already claimed by {', '.join(clash)}",
        )
    except Exception as error:
        result.add("case name", False, f"{type(error).__name__}: {error}")

    # ---- will there be anything to watch
    result.add(
        "previews",
        spec.volume_every > 0,
        f"a field snapshot every {spec.volume_every} iterations"
        if spec.volume_every > 0
        else "volume_every is 0, so this run writes its field only at the end "
        "and the flow view stays empty for its whole length",
        advisory=True,
    )

    if not spec.shock_fitted:
        result.add(
            "mesher",
            True,
            "isotropic path -- mesh checks skipped, they belong to the shock-fitted one",
        )
        return result

    # ---- the shock envelope and its own validity
    try:
        from aether.cfd.shock import envelope_for

        envelope = envelope_for(spec.body, spec.geometry, spec.mach)
        envelope.validity.check(spec.mach)
        trust = "prediction" if envelope.validity.trustworthy else "envelope only"
        result.add(
            "shock provider",
            True,
            f"{envelope.validity.provider} ({trust}) -- {envelope.validity.why}",
        )
    except Exception as error:
        result.add("shock provider", False, f"{type(error).__name__}: {error}")
        return result

    # ---- the flow condition, needed by both of the checks below
    from aether.cfd.config import FlowConditions

    flow = FlowConditions.at_altitude(
        mach=spec.mach, altitude=1.0e3 * spec.altitude_km
    )

    # ---- the mesh, and where its cells went
    #
    # Built by the same function the worker builds it with. It used to be built
    # by a second implementation here, which is how this check came to report
    # 8 160 poor cells for a mesh the worker produces with 102: the nose cap was
    # applied behind a ``geometry["nose_radius"]`` lookup in both, and only one
    # of them was fixed. A gate that models what it is gating drifts from it.
    if not deep:
        result.add("mesh", True, "not built here -- the worker builds it next")
    try:
        if deep:
            from aether.cfd.shockfit import fit_shock_layer

            fitted = fit_shock_layer(spec, flow, strict=False)
            quality = fitted.quality
            result.add(
                "mesh built",
                fitted.layer.crossings == 0 and quality.inverted == 0,
                f"{quality.elements} prisms, {fitted.layer.crossings} crossings, "
                f"{quality.inverted} inverted, {quality.poor} below {quality.threshold:g}",
            )
            # A column that never crossed the envelope was truncated, and over
            # it the outer boundary is an offset surface carrying an imposed
            # freestream rather than a shock. `march_thickness` has counted
            # these since it was written and nothing had ever read the count:
            # the hemisphere's is 42.6 %, and its drag drifted 30 % upward over
            # 800 iterations without ever settling.
            share = fitted.unenclosed_fraction
            result.add(
                "domain encloses the shock",
                share <= 0.005,
                f"{share:.2%} of columns end behind the shock"
                + (
                    f" ({fitted.capped_fraction:.1%} capped, at the free edge "
                    "where the normal tilts aft -- which is the cap working)"
                    if share <= 0.005
                    else " -- freestream is being imposed inside the shock layer "
                    "over that share of the wall, which is what made the "
                    "hemisphere's drag drift 30 % without settling"
                ),
            )
            tightest = fitted.cells_in_shock_layer()
            result.add(
                "shock layer resolved",
                tightest >= 20,
                f"{tightest} cells between body and shock on the tightest column "
                f"(wall cell {fitted.requirement.wall_cell * 1e6:.3g} um)",
            )
    except Exception as error:
        result.add("mesh built", False, f"{type(error).__name__}: {error}")
        return result

    # ---- the turbulence model, against the Reynolds number it would model
    #
    # This is the check that was missing on 2026-09-11, when a whole baseline
    # suite ran k-omega SST on a Mach 8 flow at Re = 1.4e4 -- wrong physics that
    # also NaN'd at around iteration 217 on every body. Nothing refused it: the
    # dashboard form does not expose the setting, so all three runs took the
    # RunSpec default, and the only place the choice was written down was a
    # docstring saying laminar is what to set.
    if spec.turbulence:
        vertices = np.asarray(surface.vertices, dtype=np.float64)
        length = float(np.ptp(vertices[:, 0])) or 1.0
        reynolds = flow.reynolds(length)
        # 1e6 is where hypersonic boundary layers typically begin to transition;
        # below it a RANS model is generating eddy viscosity for turbulence that
        # is not there, and it has to do it in a shock layer, which is where it
        # goes unstable.
        result.add(
            "turbulence model",
            reynolds >= 1.0e6,
            f"{spec.turbulence} at Re_L = {reynolds:.3g} on {length:.3g} m"
            + (
                ""
                if reynolds >= 1.0e6
                else " -- below the ~1e6 where hypersonic boundary layers "
                "transition, so this models turbulence that is not there. "
                "Set turbulence=None for laminar Navier-Stokes."
            ),
        )

    # ---- the startup transient
    #
    # A case initialised from uniform freestream has to form its bow shock and
    # sweep it back across the mesh, and a MUSCL reconstruction across the
    # forming discontinuity produces negative temperatures. Measured on the
    # waverider at Mach 8: the second-order run dies at iteration 205 at CFL
    # 0.5 and still dies at CFL 0.25, so lowering the CFL is not the fix.
    #
    # Advisory rather than fatal: a prelude that fails is not fatal either --
    # the second-order run simply starts from freestream as it always did -- so
    # refusing here would be stronger than the consequence warrants.
    if spec.mach > 6.0 and spec.first_order_iterations <= 0:
        result.add(
            "first-order prelude",
            False,
            f"none, at Mach {spec.mach:g}. Above about Mach 6 the second-order "
            "run has been measured dying on a NaN during the startup transient "
            "at every CFL tried. Set first_order_iterations.",
            advisory=True,
        )

    # ---- the config, against the mesh it will be solved on
    try:
        from aether.cfd.config import Numerics, SU2Case
        from aether.cfd.geometries import reference_frame
        from aether.cfd.worker import case_markers

        rendered = SU2Case(
            flow=flow,
            inviscid=spec.inviscid,
            turbulence=spec.turbulence,
            reference=reference_frame(spec.body, **spec.geometry),
            mesh_filename=f"{spec.body}.su2",
            numerics=Numerics(convective=spec.convective),
            **case_markers(spec),
        ).render()

        wanted = set()
        for line in rendered.splitlines():
            if line.startswith("MARKER_") and "=" in line:
                body = line.split("=", 1)[1].strip().strip("()")
                for token in body.split(","):
                    token = token.strip()
                    if token and not token.replace(".", "").replace("-", "").isdigit():
                        wanted.add(token)
        # What the shock-fitted writer emits, which is not a fixed set: with a
        # resolved wake the base disc goes out under its own tag, because a base
        # pressure folded into the vehicle wall cannot be separated from the
        # forebody force and is then exactly the correlation it replaced.
        present = {"vehicle", "farfield", "outflow"}
        if spec.wake:
            present.add("vehicle_base")
        missing = sorted(wanted - present)
        result.add(
            "markers",
            not missing,
            f"config names {sorted(wanted)}; mesh provides {sorted(present)}"
            + (f" -- MISSING {missing}" if missing else ""),
        )

        # And the force reader has to be able to read what SU2 will write. A
        # shock-fitted block monitors one marker -- *unless* a wake was built,
        # and then the base is a second one. Asserting the first unconditionally
        # made a resolved base look like a configuration error.
        single = "vehicle_base" not in wanted
        expected_single = bool(spec.shock_fitted) and not spec.wake
        result.add(
            "forces readable",
            single == expected_single,
            "single monitored marker, so resolved_forces needs forebody_only"
            if single
            else "per-marker breakdown available, so the base is separable",
        )
    except Exception as error:
        result.add("markers", False, f"{type(error).__name__}: {error}")

    return result
