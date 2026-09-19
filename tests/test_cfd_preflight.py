"""The gate that refuses a run which cannot work.

Every check here corresponds to a failure that actually happened on
2026-09-11 and cost a ten-minute run to discover.
"""

from __future__ import annotations

import pytest

from aether.cfd.preflight import preflight
from aether.cfd.registry import RunSpec

CONE = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}


def _spec(**kw) -> RunSpec:
    base = dict(  # noqa: C408
        body="sphere-cone", mach=8.0, altitude_km=45.0, geometry=CONE,
        turbulence=None, shock_fitted=True, cell_reynolds=100.0, convective="HLLC",
    )
    return RunSpec(**{**base, **kw})


def test_a_sound_spec_passes_every_check() -> None:
    report = preflight(_spec())
    assert report.ok, report.report()
    names = {check.name for check in report.checks}
    assert {"surface", "shock provider", "mesh built", "shock layer resolved", "markers"} <= names


def test_the_config_and_the_mesh_must_agree_on_markers() -> None:
    """The prelude died on `vehicle_base`, which a shock-fitted mesh lacks.

    Both halves were correct alone: a valid config, a valid mesh. The gate
    exists because the fault was that they disagreed.
    """
    report = preflight(_spec())
    markers = next(c for c in report.checks if c.name == "markers")
    assert markers.passed
    assert "vehicle_base" not in markers.detail


def test_a_body_with_no_shock_provider_is_refused_before_meshing() -> None:
    """``lifting-body`` used to be the example here and now has a provider, so
    the refusal is exercised on a name that is genuinely uncovered. Refusing
    before meshing is what stops a domain being built around the wrong place."""
    report = preflight(_spec(body="blended-wing-body", geometry={"length": 2.9}))
    assert not report.ok
    assert any(
        "no shock provider" in c.detail or "blended-wing-body" in c.detail
        for c in report.failures
    )


def test_the_shock_layer_must_actually_be_resolved() -> None:
    """The isotropic mesher put zero cells there at 1.36 million elements."""
    report = preflight(_spec())
    check = next(c for c in report.checks if c.name == "shock layer resolved")
    assert check.passed
    assert int(check.detail.split()[0]) >= 20


def test_the_isotropic_path_says_which_checks_do_not_apply() -> None:
    """Silence about a check that was skipped reads as a check that passed."""
    report = preflight(_spec(shock_fitted=False))
    skipped = next(c for c in report.checks if c.name == "mesher")
    assert "skipped" in skipped.detail


def test_a_failure_is_reported_rather_than_raised() -> None:
    """The gate is called from a queue form; it must not throw at a user."""
    report = preflight(_spec(body="not-a-body"))
    assert not report.ok
    assert report.failures
    assert isinstance(report.report(), str)


@pytest.fixture
def queue_in_tmp(tmp_path, monkeypatch):
    """Point the registry at a scratch results tree, not the real share.

    This test enqueues, and it used to enqueue into whatever queue the process
    was pointed at -- which for a bare ``pytest`` run is ``results/cfd/queue``
    inside the repo. It cleans up in a ``finally``, so it is tidy right up until
    the run is interrupted, and then it leaves a record behind that makes every
    later preflight fail its case-name check with a collision against a run that
    does not exist. That cost a diagnosis on 2026-09-11.
    """
    monkeypatch.setenv("AETHER_RESULTS", str(tmp_path))
    return tmp_path / "cfd" / "queue"


def test_a_run_already_in_the_queue_does_not_collide_with_itself(queue_in_tmp) -> None:
    """The worker checks a record that is already queued, so it finds itself.

    Without excluding it the name check calls that a collision and refuses
    every run -- which is what it did, and the worker's own tests caught it
    before it reached anything real.
    """
    import dataclasses

    from aether.cfd.registry import enqueue, forget

    spec = _spec()
    record = enqueue(dataclasses.replace(spec))
    try:
        blind = preflight(spec)
        assert not blind.ok
        assert any(c.name == "case name" for c in blind.failures)

        aware = preflight(spec, exclude=record.run_id)
        assert aware.ok, aware.report()
    finally:
        forget(record.run_id)


def test_the_shallow_gate_skips_the_mesh_it_is_about_to_build() -> None:
    """The worker is seconds from building the same mesh; the form is not."""
    shallow = preflight(_spec(), deep=False)
    assert shallow.ok
    assert any("not built here" in c.detail for c in shallow.checks)
    assert not any(c.name == "shock layer resolved" for c in shallow.checks)

    deep = preflight(_spec())
    assert any(c.name == "shock layer resolved" for c in deep.checks)


def test_the_gate_measures_the_mesh_the_worker_will_actually_build() -> None:
    """Otherwise the gate is a model of the run, and models drift.

    It did. The nose cap was applied behind a ``geometry["nose_radius"]`` lookup
    in two places; the fix to measure the radius instead landed in one of them,
    and the preflight went on reporting 8 160 poor cells for a mesh the worker
    builds with 102. Both call `shockfit.fit_shock_layer` now, and this asserts
    it by building through each entry point and comparing.
    """
    from aether.cfd.config import FlowConditions
    from aether.cfd.preflight import preflight
    from aether.cfd.registry import RunSpec
    from aether.cfd.shockfit import fit_shock_layer

    spec = RunSpec(
        body="hemisphere",
        mach=8.0,
        altitude_km=45.0,
        geometry={"radius": 0.045},
        shock_fitted=True,
        cell_reynolds=100.0,
    )
    flow = FlowConditions.at_altitude(mach=spec.mach, altitude=1.0e3 * spec.altitude_km)
    fitted = fit_shock_layer(spec, flow)

    checks = {check.name: check.detail for check in preflight(spec, deep=True).checks}
    assert f"{fitted.quality.elements} prisms" in checks["mesh built"]
    assert f"{fitted.quality.poor} below" in checks["mesh built"]
    assert f"{fitted.cells_in_shock_layer()} cells between" in checks["shock layer resolved"]


def test_the_cap_reaches_a_body_that_does_not_spell_it_nose_radius() -> None:
    """The hemisphere's parameter is ``radius``, and it kept its pole for it."""
    from aether.cfd.config import FlowConditions
    from aether.cfd.registry import RunSpec
    from aether.cfd.shockfit import fit_shock_layer

    spec = RunSpec(
        body="hemisphere",
        mach=8.0,
        altitude_km=45.0,
        geometry={"radius": 0.045},
        shock_fitted=True,
        cell_reynolds=100.0,
    )
    fitted = fit_shock_layer(
        spec, FlowConditions.at_altitude(mach=8.0, altitude=45.0e3)
    )
    assert fitted.capped is not None
    assert fitted.capped.peak_valence_after <= 8
    # Without the cap this is 9 792.
    assert fitted.quality.poor < 500


def test_a_turbulence_model_below_transition_is_refused() -> None:
    """The check that was missing when a whole baseline suite burned.

    k-omega SST at Re_L = 1.4e4 is generating eddy viscosity for turbulence that
    is not there, and it has to do it inside a shock layer, which is where it
    goes unstable. Nothing refused it: the dashboard form did not expose the
    setting, so hemisphere, sphere-cone and biconic all took the default and all
    three NaN'd at around iteration 217 with a flat residual.
    """
    from aether.cfd.preflight import preflight
    from aether.cfd.registry import RunSpec

    common = {
        "body": "hemisphere",
        "mach": 8.0,
        "altitude_km": 45.0,
        "geometry": {"radius": 0.045},
        "shock_fitted": True,
        "cell_reynolds": 100.0,
    }
    turbulent = preflight(RunSpec(**common, turbulence="SST"), deep=False)
    assert not turbulent.ok
    assert any(
        check.name == "turbulence model" and "1e6" in check.detail
        for check in turbulent.failures
    )

    laminar = preflight(RunSpec(**common, turbulence=None), deep=False)
    assert laminar.ok
    # Nothing to say when there is no model to say it about.
    assert not any(check.name == "turbulence model" for check in laminar.checks)


def test_every_campaign_body_is_meshed_to_its_shock_not_to_a_cap() -> None:
    """A count computed correctly since it was written, and never read.

    ``march_thickness`` truncates a column whose normal does not reach the shock
    envelope, and its docstring says a large count "means the envelope does not
    bound this body and the mesh should be questioned rather than used". Nothing
    questioned it. The hemisphere's was **42.6 %**, and its drag drifted 30 %
    upward across 800 iterations without ever settling.

    The envelope was not the fault: Billig's hyperbola encloses that body
    everywhere -- at the equator *radius* the shock stands at x = 11.9 mm
    against a body station of 45 mm, and it does not reach the equator plane
    until r = 75 mm. The cap was. It defaulted to a quarter of the body's axial
    **length**, which is a meaningless scale for a blunt body: 11.25 mm on a
    45 mm hemisphere, against a shoulder shock layer genuinely 30 mm thick. The
    same rule gives 515 mm on a 2 m sphere-cone and never binds, which is how it
    survived.
    """
    from aether.cfd.geometries import REFERENCE_BODIES
    from aether.cfd.preflight import preflight
    from aether.cfd.registry import RunSpec

    for body in ("hemisphere", "sphere-cone", "biconic"):
        spec = RunSpec(
            body=body, mach=8.0, altitude_km=45.0,
            geometry=REFERENCE_BODIES[body].defaults(),
            shock_fitted=True, cell_reynolds=100.0,
        )
        report = preflight(spec, deep=True)
        check = next(c for c in report.checks if c.name == "domain encloses the shock")
        assert check.passed, f"{body}: {check.detail}"
        assert report.ok, report.report()


def test_a_capped_column_is_not_the_same_as_an_unenclosed_shock() -> None:
    """The gate has to ask the geometry, not the bookkeeping.

    A column caps when its normal does not reach the envelope within the allowed
    length. Near the free edge at the base that happens for a reason that says
    nothing about the envelope -- the smoothed normal tilts aft and the column
    runs nearly parallel to the shock. On the sphere-cone every one of the 240
    capped columns is in the last 6 % of the body, where the shock is 60 mm off
    a wall whose column was cut at 512: enclosed nine times over.

    So capping is reported and enclosure is gated. The hemisphere's original
    configuration -- a cap at a quarter of a 45 mm body against a 30 mm shoulder
    shock layer -- reads 32.77 % unenclosed and is refused; the repaired one and
    both slender bodies read 0.00 %.
    """
    from aether.cfd.config import FlowConditions
    from aether.cfd.geometries import REFERENCE_BODIES
    from aether.cfd.registry import RunSpec
    from aether.cfd.shockfit import fit_shock_layer

    flow = FlowConditions.at_altitude(mach=8.0, altitude=45.0e3)
    for body in ("sphere-cone", "biconic"):
        spec = RunSpec(
            body=body, mach=8.0, altitude_km=45.0,
            geometry=REFERENCE_BODIES[body].defaults(),
            shock_fitted=True, cell_reynolds=100.0,
        )
        fitted = fit_shock_layer(spec, flow)
        # Capping at the rim is allowed; leaving the shock outside is not.
        assert fitted.unenclosed_fraction == 0.0, body


def test_the_column_cap_is_scaled_to_the_shock_not_to_the_body_length() -> None:
    """The hemisphere needs 4.4 standoffs at the shoulder; a quarter of its
    45 mm span is 1.7. The cap has to come from the envelope's own length
    scale or it binds on exactly the bodies the envelope describes best."""
    import numpy as np

    from aether.cfd.config import FlowConditions
    from aether.cfd.registry import RunSpec
    from aether.cfd.shockfit import CAP_STANDOFFS, fit_shock_layer

    spec = RunSpec(
        body="hemisphere", mach=8.0, altitude_km=45.0, geometry={"radius": 0.045},
        shock_fitted=True, cell_reynolds=100.0,
    )
    fitted = fit_shock_layer(spec, FlowConditions.at_altitude(mach=8.0, altitude=45.0e3))
    assert fitted.layer.capped == 0

    nodes = fitted.layer.nodes
    thickness = np.linalg.norm(nodes[:, -1, :] - nodes[:, 0, :], axis=1)
    # Thickest where the wall normal is perpendicular to the axis, thinnest at
    # the stagnation point, and the cap is never the thing that decides it.
    assert thickness.max() / fitted.standoff > 4.0
    assert thickness.max() < CAP_STANDOFFS * fitted.standoff
