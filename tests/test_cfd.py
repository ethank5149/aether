"""The CFD case writer, the sweep glue, and the damping reduction.

The config tests are written against the **installed SU2 binary** wherever one
is present, not against a list of keywords copied from documentation. That is
the only check that means anything here: SU2 renamed most of its configuration
vocabulary at v7 and rejects the old spelling in the parser, so a config that
"looks right" and a config that runs are different claims, and only the second
one is worth asserting.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aether.aerodynamics.cfd.meshing import (
    ViscousSizing,
    boundary_layer_thickness,
    wall_spacing_for_y_plus,
)
from aether.aerodynamics.panels import (
    blunted_multiconic,
    smooth_bent_biconic,
    spatular_wedge,
)
from aether.cfd import (
    FlowConditions,
    LowMach,
    Nonequilibrium,
    PerfectGas,
    PitchingMotion,
    ReferenceFrame,
    SU2Case,
    cfd_grid,
    pitch_damping_from_history,
    regime_for,
)
from aether.geometry import VehicleMesh

GENERATORS = (spatular_wedge, blunted_multiconic, smooth_bent_biconic)


def _su2() -> str | None:
    found = shutil.which("SU2_CFD")
    if found:
        return found
    candidate = Path("/config/miniconda3/envs/su2/bin/SU2_CFD")
    return str(candidate) if candidate.exists() else None


@pytest.fixture(scope="module")
def reference() -> ReferenceFrame:
    return ReferenceFrame(area=0.44, length=0.75, moment_reference=(1.2, 0.0, 0.0))


# -- geometry ---------------------------------------------------------------


@pytest.mark.parametrize("generator", GENERATORS, ids=lambda f: f.__name__)
def test_surface_grid_closes_into_a_watertight_outward_body(generator) -> None:
    """Every parametric generator must produce a body a mesher can use.

    Two properties, and the second is the one that fails silently: the surface
    has to be closed, and its normals have to point *out*. An inward-wound body
    is closed, has the right area, integrates to the right panel loads, and
    makes the boundary-layer extruder grow prisms into the vehicle.
    """
    mesh = VehicleMesh.from_surface_grid(generator().surface, name=generator.__name__)
    assert mesh.is_closed, f"open at stations {mesh.boundary_stations()}"

    triangles = mesh.triangles
    signed_volume = float(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        ).sum()
        / 6.0
    )
    assert signed_volume > 0.0, "normals wound inward"


def test_open_surface_grid_is_refused() -> None:
    grid = spatular_wedge().surface
    from aether.aerodynamics.panels import SurfaceGrid

    slit = SurfaceGrid(vertices=grid.vertices[:, :-1, :].copy())
    assert not slit.seam_closed
    with pytest.raises(ValueError, match="does not repeat"):
        VehicleMesh.from_surface_grid(slit)


# -- flight conditions ------------------------------------------------------


def test_reynolds_number_is_derived_not_assumed() -> None:
    """The freestream, not a constant, sets the Reynolds number.

    The drafted template pinned ``REYNOLDS_NUMBER= 1.5E6`` and swept the whole
    envelope past it. Across the altitude band a re-entry body actually flies,
    the true value moves by more than three orders of magnitude.
    """
    values = [
        FlowConditions.at_altitude(mach=20.0, altitude=h).reynolds(1.0)
        for h in (0.0, 20.0e3, 40.0e3, 60.0e3)
    ]
    assert all(np.diff(values) < 0.0), "Reynolds number must fall with altitude"
    assert values[0] / values[-1] > 1.0e3


def test_zero_and_negative_mach_are_refused_not_clamped() -> None:
    """A sweep must not silently substitute a different flight condition."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="Mach number"):
            FlowConditions(mach=bad, temperature=226.5, pressure=1197.0)


def test_regime_selection_covers_the_envelope() -> None:
    assert isinstance(regime_for(0.05), LowMach)
    assert isinstance(regime_for(3.0), PerfectGas)
    assert not isinstance(regime_for(3.0), LowMach)
    assert isinstance(regime_for(20.0), Nonequilibrium)


# -- rendered configuration -------------------------------------------------


def test_rendered_config_uses_no_su2_v6_keywords(reference) -> None:
    """None of the drafts' v6 spellings may survive into a rendered case."""
    retired = (
        "PHYSICAL_PROBLEM",
        "REGIME_TYPE",
        "CHEMISTRY_MODEL",
        "TWO_TEMPERATURE_MODEL",
        "LINEAR_SOLVER_FLOW",
        "LINEAR_SOLVER_PREC_FLOW",
        "LINEAR_SOLVER_ERROR_FLOW",
        "LINEAR_SOLVER_ITER_FLOW",
        "PITCH_AMPLITUDE",
        "PITCH_OMEGA",
        "PITCH_PHASE",
    )
    cases = [
        SU2Case(FlowConditions.at_altitude(m, 30.0e3), reference, "m.su2")
        for m in (0.05, 3.0, 20.0)
    ]
    cases.append(
        SU2Case(
            FlowConditions.at_altitude(10.0, 30.0e3),
            reference,
            "m.su2",
            motion=PitchingMotion(origin=(1.2, 0.0, 0.0)),
        )
    )
    for case in cases:
        text = case.render()
        for keyword in retired:
            assert f"\n{keyword}=" not in f"\n{text}", f"{keyword} in {case.resolved_regime().name}"


def test_turbulence_model_implies_the_rans_solver(reference) -> None:
    """KIND_TURB_MODEL is ignored unless SOLVER is RANS — they must agree."""
    text = SU2Case(
        FlowConditions.at_altitude(3.0, 30.0e3),
        reference,
        "m.su2",
        regime=PerfectGas(turbulence="SST"),
    ).render()
    assert "SOLVER= RANS" in text
    assert "KIND_TURB_MODEL= SST" in text

    laminar = SU2Case(
        FlowConditions.at_altitude(3.0, 30.0e3),
        reference,
        "m.su2",
        regime=PerfectGas(turbulence=None),
    ).render()
    assert "SOLVER= NAVIER_STOKES" in laminar
    assert "KIND_TURB_MODEL" not in laminar


def test_reynolds_emitted_only_where_su2_wants_it(reference) -> None:
    flow = FlowConditions.at_altitude(10.0, 30.0e3)
    viscous = SU2Case(flow, reference, "m.su2", regime=PerfectGas()).render()
    euler = SU2Case(flow, reference, "m.su2", regime=PerfectGas(inviscid=True)).render()
    nemo = SU2Case(flow, reference, "m.su2", regime=Nonequilibrium()).render()
    assert "REYNOLDS_NUMBER=" in viscous and "REYNOLDS_LENGTH=" in viscous
    assert "REYNOLDS_NUMBER=" not in euler
    assert "REYNOLDS_NUMBER=" not in nemo
    assert "INIT_OPTION= TD_CONDITIONS" in nemo
    assert "FREESTREAM_TEMPERATURE_VE=" in nemo


def test_unsteady_case_omits_iter(reference) -> None:
    """ITER alongside TIME_ITER aborts SU2 in SetPostprocessing with no message."""
    text = SU2Case(
        FlowConditions.at_altitude(10.0, 30.0e3),
        reference,
        "m.su2",
        motion=PitchingMotion(origin=(1.2, 0.0, 0.0)),
    ).render()
    assert "\nITER=" not in f"\n{text}"
    assert "TIME_ITER=" in text
    assert "GRID_MOVEMENT= RIGID_MOTION" in text
    assert "PITCHING_AMPL= 0.0 2" in text


def test_inviscid_wall_is_euler_not_isothermal(reference) -> None:
    text = SU2Case(
        FlowConditions.at_altitude(6.0, 30.0e3),
        reference,
        "m.su2",
        regime=PerfectGas(inviscid=True),
    ).render()
    assert "MARKER_EULER=" in text
    assert "MARKER_ISOTHERMAL=" not in text


def test_reference_quantities_are_the_vehicles_not_unity(reference) -> None:
    text = SU2Case(FlowConditions.at_altitude(5.0, 30.0e3), reference, "m.su2").render()
    assert "REF_AREA= 0.44" in text
    assert "REF_LENGTH= 0.75" in text
    assert "REF_ORIGIN_MOMENT_X= 1.2" in text


def test_nonequilibrium_composition_must_be_normalised() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        Nonequilibrium(composition=(0.5, 0.2, 0.0, 0.0, 0.0))


@pytest.mark.skipif(_su2() is None, reason="SU2_CFD not installed")
def test_every_regime_is_accepted_by_the_installed_su2(reference, tmp_path) -> None:
    """The check that actually matters: SU2 parses and preprocesses each case."""
    mesh = tmp_path / "m.su2"
    _write_trivial_mesh(mesh)
    cases = {
        "low-mach": SU2Case(FlowConditions.at_altitude(0.05, 0.0), reference, "m.su2"),
        "perfect-gas": SU2Case(
            FlowConditions.at_altitude(3.0, 30.0e3, alpha=np.deg2rad(10.0)),
            reference,
            "m.su2",
        ),
        "laminar": SU2Case(
            FlowConditions.at_altitude(3.0, 30.0e3),
            reference,
            "m.su2",
            regime=PerfectGas(turbulence=None),
        ),
        "euler": SU2Case(
            FlowConditions.at_altitude(6.0, 30.0e3),
            reference,
            "m.su2",
            regime=PerfectGas(inviscid=True),
        ),
        "nemo": SU2Case(FlowConditions.at_altitude(20.0, 40.0e3), reference, "m.su2"),
        "nemo-euler": SU2Case(
            FlowConditions.at_altitude(25.0, 50.0e3),
            reference,
            "m.su2",
            regime=Nonequilibrium(inviscid=True),
        ),
        "urans": SU2Case(
            FlowConditions.at_altitude(10.0, 30.0e3),
            reference,
            "m.su2",
            motion=PitchingMotion(origin=(1.2, 0.0, 0.0), frequency=62.8),
        ),
    }
    binary = _su2()
    assert binary is not None
    for name, case in cases.items():
        (tmp_path / "case.cfg").write_text(case.render())
        finished = subprocess.run(
            [binary, "-d", "case.cfg"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert finished.returncode == 0, f"{name} rejected by SU2:\n" + "\n".join(
            line
            for line in (finished.stdout + finished.stderr).splitlines()
            if "Error" in line or "invalid" in line.lower() or "!!" in line
        )


def _write_trivial_mesh(path: Path) -> None:
    """A sphere in a box: the smallest 3-D mesh with the right marker names."""
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("t")
        box = gmsh.model.occ.addBox(-3, -3, -3, 6, 6, 6)
        ball = gmsh.model.occ.addSphere(0, 0, 0, 0.5)
        cut, _ = gmsh.model.occ.cut([(3, box)], [(3, ball)])
        gmsh.model.occ.synchronize()
        wall, farfield = [], []
        for dim, tag in gmsh.model.getEntities(2):
            target = wall if gmsh.model.occ.getMass(dim, tag) < 10.0 else farfield
            target.append(tag)
        gmsh.model.addPhysicalGroup(2, wall, name="vehicle")
        gmsh.model.addPhysicalGroup(2, farfield, name="farfield")
        gmsh.model.addPhysicalGroup(3, [tag for _, tag in cut], name="fluid")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.5)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


# -- wall spacing and the boundary layer ------------------------------------


def test_wall_spacing_rises_with_mach_through_the_reference_temperature() -> None:
    """The Eckert correction is the whole point of the estimate.

    Without it the spacing would fall monotonically with Mach, because the
    incompressible Reynolds number rises. With it the reference temperature
    rises faster, the reference density falls, and the required first cell
    grows — by more than a factor of ten between Mach 3 and Mach 20.
    """
    spacings = [
        wall_spacing_for_y_plus(m, 3.0, temperature=220.0, pressure=1100.0)
        for m in (3.0, 10.0, 20.0)
    ]
    assert spacings[0] < spacings[1] < spacings[2]
    assert spacings[2] / spacings[0] > 10.0


def test_layer_count_spans_the_estimated_boundary_layer() -> None:
    sizing = ViscousSizing()
    first = wall_spacing_for_y_plus(10.0, 3.0, temperature=220.0, pressure=1100.0)
    delta = boundary_layer_thickness(10.0, 3.0, temperature=220.0, pressure=1100.0)
    layers = sizing.layers_to_span(first, delta)
    assert sizing.with_layers(layers).total_thickness(first) >= delta
    assert sizing.with_layers(layers - 1).total_thickness(first) < delta


@pytest.mark.slow
@pytest.mark.filterwarnings(
    # Truncation is documented behaviour, not a failure: the layer is reported
    # short rather than silently accepted. What this test asserts is that a
    # genuinely three-dimensional anisotropic layer was built at all.
    "ignore::aether.aerodynamics.cfd.meshing.BoundaryLayerTruncated"
)
def test_viscous_domain_builds_a_prism_boundary_layer(tmp_path) -> None:
    """A genuinely three-dimensional, anisotropic boundary layer.

    The assertion is on the *prism* count, not merely on the mesh existing.
    gmsh's ``BoundaryLayer`` field is two-dimensional only — it takes a
    ``CurvesList`` and has no ``SurfacesList`` — so a 3-D layer has to come
    from ``extrudeBoundaryLayer``, and a mesh built without it would be all
    tetrahedra and would still look like a valid answer.
    """
    pytest.importorskip("gmsh")
    from aether.aerodynamics.cfd.meshing import viscous_domain

    body = VehicleMesh.from_surface_grid(blunted_multiconic().surface, name="body")
    result = viscous_domain(
        body,
        tmp_path / "body.su2",
        mach=10.0,
        temperature=220.0,
        pressure=1100.0,
        sizing=ViscousSizing(n_layers=10),
    )
    assert result.dimension == 3
    assert result.n_prisms > 0
    # At least a few full layers over the whole wall. Not the requested count:
    # the extruder terminates locally where a column would degenerate, and on
    # a body with a shoulder fillet that legitimately costs layers.
    assert result.n_prisms >= 4 * body.n_faces
    assert result.first_cell_height < 1.0e-2
    assert result.path.exists()


# -- damping ----------------------------------------------------------------


def test_damping_recovers_a_known_derivative() -> None:
    """Reduce a synthetic history whose answer is known exactly.

    The stiffness term is deliberately included and is three orders of
    magnitude larger than the damping term it sits on top of. A reduction that
    fails to close the hysteresis loop exactly does not lose a little accuracy
    here; it returns the wrong number.
    """
    chord, speed = 0.75, 3000.0
    truth, stiffness, offset = -0.85, -0.42, 0.03
    motion = PitchingMotion(amplitude=np.deg2rad(2.0), frequency=60.0, time_step=2.0e-5, cycles=6.0)
    time = np.arange(0.0, 6.0 * motion.period, motion.time_step)
    alpha = motion.amplitude * np.sin(motion.frequency * time)
    rate = motion.amplitude * motion.frequency * np.cos(motion.frequency * time)
    moment = offset + stiffness * alpha + (chord / (2 * speed)) * truth * rate

    result = pitch_damping_from_history(time, moment, motion, chord, speed)
    assert result.damping == pytest.approx(truth, rel=1.0e-4)
    assert result.stable
    assert result.cycles_used == 4  # arange stops short of the 6th period
    assert result.cycle_scatter < 1.0e-5
    assert result.reduced_frequency == pytest.approx(motion.frequency * chord / (2 * speed))


def test_damping_refuses_a_partial_cycle() -> None:
    motion = PitchingMotion(frequency=60.0, time_step=1.0e-4)
    time = np.arange(0.0, 1.4 * motion.period, motion.time_step)
    with pytest.raises(ValueError, match="whole cycles"):
        pitch_damping_from_history(time, np.zeros_like(time), motion, 0.75, 3000.0)


def test_dynamic_instability_is_reported_as_such() -> None:
    chord, speed = 0.75, 3000.0
    motion = PitchingMotion(amplitude=np.deg2rad(2.0), frequency=60.0, time_step=2.0e-5, cycles=5.0)
    time = np.arange(0.0, 5.0 * motion.period, motion.time_step)
    rate = motion.amplitude * motion.frequency * np.cos(motion.frequency * time)
    moment = (chord / (2 * speed)) * 0.6 * rate
    result = pitch_damping_from_history(time, moment, motion, chord, speed)
    assert not result.stable
    assert "DYNAMICALLY UNSTABLE" in result.summary()


# -- sweep ------------------------------------------------------------------


def test_cfd_grid_is_coarser_than_the_panel_default() -> None:
    from aether.aerodynamics.tables import SweepGrid

    panel = SweepGrid(mach=SweepGrid.default_mach(), alpha=SweepGrid.default_alpha())
    assert cfd_grid().size < panel.size / 5


def test_sweep_records_altitude_in_metadata(tmp_path, reference) -> None:
    """A CFD table is only meaningful at the Reynolds number it was built at."""
    from aether.aerodynamics.tables import Coefficients, SweepGrid, SweepRun

    class _Stub:
        name = "stub"

        def solve(self, mach: float, alpha: float) -> Coefficients:
            return Coefficients(0.1, 0.2 * alpha, -0.01 * alpha)

    grid = SweepGrid(mach=np.array([2.0, 5.0]), alpha=np.array([0.0, 0.1]))
    run = SweepRun(
        name="t",
        grid=grid,
        solver=_Stub(),
        store=tmp_path / "t.jsonl",
        reference_area=reference.area,
        reference_length=reference.length,
        metadata={"altitude_m": 30.0e3},
    )
    table = run.run()
    assert table.complete
    assert table.metadata["altitude_m"] == 30.0e3


def test_hdf5_export_carries_its_own_reference_quantities(tmp_path, reference) -> None:
    """A table that has been moved away from its checkpoint must still be readable.

    The drafted export wrote a bare frame of Mach/AoA/CL/CD, which cannot be
    used without knowing what it was non-dimensionalised on — and that
    information lived only in the config file of whichever run produced it.
    """
    h5py = pytest.importorskip("h5py")
    from aether.aerodynamics.tables import AeroTable
    from aether.cfd import write_hdf5

    table = AeroTable(
        name="body",
        mach=np.array([2.0, 5.0]),
        alpha=np.array([0.0, 0.1]),
        axial=np.array([[0.1, 0.11], [0.12, 0.13]]),
        normal=np.array([[0.0, 0.2], [0.0, 0.21]]),
        pitching_moment=np.array([[0.0, -0.01], [0.0, -0.011]]),
        reference_area=reference.area,
        reference_length=reference.length,
        solver="su2-3d",
        metadata={"altitude_m": 30.0e3},
    )
    path = write_hdf5(table, tmp_path / "t.h5")
    with h5py.File(path, "r") as handle:
        group = handle["aero_table"]
        assert group.attrs["reference_area_m2"] == pytest.approx(reference.area)
        assert group.attrs["reference_length_m"] == pytest.approx(reference.length)
        assert group.attrs["altitude_m"] == pytest.approx(30.0e3)
        assert group.attrs["solver"] == "su2-3d"
        assert np.allclose(group["axial"][()], table.axial)
        assert np.allclose(group["alpha_rad"][()], table.alpha)


def test_mpirun_is_resolved_beside_the_su2_binary(tmp_path) -> None:
    """SU2 lives in its own conda environment, and so does its MPI.

    Resolving ``mpirun`` from PATH finds either nothing — the usual case, since
    that environment is not activated — or a foreign MPI, under which every
    rank gets a communicator of size one and the domain is solved N times over
    while appearing to work.
    """
    from aether.cfd.solver import find_mpirun

    beside = tmp_path / "mpirun"
    beside.write_text("#!/bin/sh\n")
    assert find_mpirun(tmp_path / "SU2_CFD") == str(beside)

    with pytest.raises(FileNotFoundError, match="processes > 1"):
        import shutil as _shutil

        real = _shutil.which
        try:
            _shutil.which = lambda _name: None  # type: ignore[assignment]
            find_mpirun(tmp_path / "empty" / "SU2_CFD")
        finally:
            _shutil.which = real


def test_convergence_is_not_fooled_by_a_ripple_straddling_the_window() -> None:
    """A drifting solution must fail the test at every window length.

    The half-mean construction passes this history whenever the window happens
    to cover a whole number of ripple periods, which is what makes it unsafe:
    the verdict depends on where the window landed rather than on the physics.
    """
    from aether.cfd.solver import _force_settled

    iteration = np.arange(20000, dtype=float)
    ripple = 0.15 * 0.005 * np.sin(iteration * 2 * np.pi / 197.0)
    drifting = 0.15 + ripple + 3.0e-6 * iteration
    for window in (2000, 4000, 8000):
        settled, drift, _ = _force_settled(drifting, window, 1.0e-3)
        assert not settled, f"declared converged at window {window} (drift {drift:.2e})"

    settled_flat, drift_flat, _ = _force_settled(0.15 + ripple, 4000, 1.0e-3)
    assert settled_flat, f"a rippling but trend-free history must pass ({drift_flat:.2e})"

    # Too noisy to assess is refused, not passed.
    loud = 0.15 + 30.0 * ripple
    settled_loud, _, ripple_loud = _force_settled(loud, 4000, 1.0e-3)
    assert not settled_loud, f"ripple {ripple_loud:.2e} should have failed the guard"


def test_divergence_is_detected_from_the_log_not_the_exit_status(tmp_path) -> None:
    """This Open MPI build returns zero after MPI_ABORT.

    So a run that died on a NaN at iteration 205 comes back looking like a
    clean exit, and its last finite iterate goes into the table. On the
    waverider that iterate was a negative lift coefficient at +6 degrees of
    incidence — a number no downstream consumer would question the sign of.
    """
    from aether.cfd.solver import SU2Solver

    (tmp_path / "su2.log").write_text(
        "|  205| 2.24| 0.34| 0.08|\n\n"
        'Error in "void CSolver::SetResidual_RMS(...)": \n'
        "SU2 has diverged (NaN detected).\n"
    )
    assert SU2Solver._detect_divergence(tmp_path, 0) is True
    assert SU2Solver._detect_divergence(tmp_path, 1) is True

    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "su2.log").write_text("|  8000| -6.10| 0.34| 0.08|\nExit Success\n")
    assert SU2Solver._detect_divergence(clean, 0) is False
    # A missing mesh or a bad marker is a failure, but not this one: reporting
    # it as divergence would send a caller to lower the CFL over a typo.
    assert SU2Solver._detect_divergence(clean, 1) is False


def test_a_first_order_prelude_renders_two_distinct_cases(reference) -> None:
    """The prelude must actually be first order, and the main run must restart."""
    from aether.cfd import FlowConditions, Numerics, PerfectGas, SU2Case

    flow = FlowConditions.at_altitude(8.0, 30.0e3)
    numerics = Numerics(first_order_iterations=1500, iterations=20000)
    prelude = SU2Case(
        flow,
        reference,
        "m.su2",
        regime=PerfectGas(inviscid=True),
        numerics=replace(numerics, muscl=False, iterations=1500),
    ).render()
    main = SU2Case(
        flow, reference, "m.su2", regime=PerfectGas(inviscid=True), numerics=numerics, restart=True
    ).render()
    assert "MUSCL_FLOW= NO" in prelude
    assert "RESTART_SOL= NO" in prelude
    assert "MUSCL_FLOW= YES" in main
    assert "RESTART_SOL= YES" in main
    # SU2's default SOLUTION_FILENAME is `solution.dat`, not the
    # `solution_flow.dat` that symmetry with RESTART_FILENAME suggests, so the
    # name has to be stated or the restart fails at the file open.
    assert "SOLUTION_FILENAME= solution_flow.dat" in main


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::aether.aerodynamics.cfd.meshing.BoundaryLayerTruncated")
def test_the_inviscid_domain_carries_no_prisms(tmp_path) -> None:
    """An Euler run has no viscous gradient, so a prism layer is only harm.

    On this package's sphere-cone the viscous mesher was asked for a 0.199 m
    boundary layer around a 0.05 m nose radius: the extrusion self-intersects,
    gets truncated, and leaves skewed cells exactly at the stagnation region.
    Every Mach 8 case run against it diverged.
    """
    pytest.importorskip("gmsh")
    from aether.aerodynamics.cfd.meshing import ViscousSizing, inviscid_domain
    from aether.geometry import bodies, brep

    surface = brep.surface_mesh(bodies.sphere_cone(), size_max=0.12, curvature_nodes=12)
    result = inviscid_domain(
        surface,
        tmp_path / "euler.su2",
        mach=8.0,
        sizing=ViscousSizing(upstream=4.0, downstream=6.0, transverse=5.0, farfield_size=5.0),
        wall_refinement=0.05,
        wall_band=1.0,
    )
    assert result.n_prisms == 0
    assert result.dimension == 3
    assert result.path.exists()
    # The refinement field has to actually bite. A Distance field on a discrete
    # surface returns nothing usable and the mesher silently produces a uniform
    # farfield-sized mesh, which converges to a drag coefficient of the wrong
    # sign — so the cell count is the check that the sizing was applied.
    low, high = surface.bounds
    span = float(high[0] - low[0])
    domain = (span * 11.0) * (float(high[1] - low[1]) + 10.0 * span) ** 2
    implied_edge = (domain / result.n_elements) ** (1.0 / 3.0)
    assert implied_edge < 0.5 * span, (
        f"mean cell edge {implied_edge:.3f} m over a {span:.3f} m body — the "
        f"background field was ignored"
    )


def test_base_and_forebody_are_reported_separately(reference) -> None:
    """A blunt-based body can have a base force larger than its forebody force.

    On a Mach 8 sphere-cone the Euler base came back at ``CA = -0.28`` against a
    forebody of ``+0.05``: folded together the reported drag is *negative*, which
    is not a small error but an impossible one. The base term is not physical —
    real base pressure is set by a separated viscous shear layer the Euler
    equations do not contain — so it has to be separable, not merely small.
    """
    from aether.aerodynamics.cfd.su2 import SurfaceForces

    forces = SurfaceForces(
        forebody_axial=0.0527,
        base_axial=-0.2813,
        normal=-0.30,
        forebody_normal=0.041,
        base_normal=-0.341,
        pitching_moment=0.0,
        forebody_moment=0.0,
        forebody_area=2.72,
        base_area=0.51,
    )
    assert forces.axial < 0.0, "the total is the number that misleads"
    assert forces.forebody_axial > 0.0, "the forebody is the number that means something"
    assert abs(forces.base_axial) > abs(forces.forebody_axial)

    # The same trap at incidence: an Euler base carries a normal force too, and
    # folding it in gave a negative lift coefficient at positive alpha.
    lift_all, _ = forces.wind_axes(np.deg2rad(6.0), forebody_only=False)
    lift_fore, drag_fore = forces.wind_axes(np.deg2rad(6.0), forebody_only=True)
    assert lift_all < 0.0, "including the base inverts the lift"
    assert lift_fore > 0.0, "the forebody lifts, as it must at positive incidence"
    assert drag_fore > 0.0


def test_the_convergence_test_reads_the_forebody_not_the_total() -> None:
    """The base never settles, so judging the total never reports convergence.

    Synthesised from the real failure: a forebody that has settled to a steady
    value, plus a base oscillating with no trend. Judged on the sum, the case
    looks unconverged forever and no iteration count helps, because the drift
    lives entirely in a quantity nobody should be reading.
    """
    from aether.cfd.solver import _force_settled

    step = np.arange(6000, dtype=float)
    forebody = 0.079 + 0.0002 * np.exp(-step / 800.0)
    base = -0.28 + 0.05 * np.sin(step / 90.0) + 0.02 * np.sin(step / 37.0)

    settled_fore, drift_fore, _ = _force_settled(forebody, 3000, 1.0e-3)
    settled_total, drift_total, _ = _force_settled(forebody + base, 3000, 1.0e-3)

    assert settled_fore, f"the forebody has settled (drift {drift_fore:.2e})"
    assert not settled_total, "the total must not pass; its drift is all base"
    assert drift_total > 10.0 * drift_fore


@pytest.mark.slow
def test_the_base_split_survives_the_volume_mesher(tmp_path) -> None:
    """Splitting the wall must happen before meshing, and stay conformal.

    Re-tagging faces after the volume is meshed leaves surface elements with no
    adjacent volume element, and SU2 refuses the mesh outright with exactly that
    message. Both patches therefore share one node set.
    """
    pytest.importorskip("gmsh")
    from aether.aerodynamics.cfd.meshing import ViscousSizing, inviscid_domain
    from aether.geometry import bodies, brep

    body = bodies.sphere_cone(half_angle=np.deg2rad(10.0), nose_radius=0.05, length=2.0)
    surface = brep.surface_mesh(body, size_max=0.18, size_min=0.02, curvature_nodes=10)
    result = inviscid_domain(
        surface,
        tmp_path / "split.su2",
        mach=8.0,
        sizing=ViscousSizing(upstream=3.0, downstream=5.0, transverse=4.0, farfield_size=6.0),
        wall_refinement=0.22,
        wall_band=0.5,
        base_station=1.985,
    )
    text = result.path.read_text()
    assert "MARKER_TAG= vehicle\n" in text
    assert "MARKER_TAG= vehicle_base\n" in text
    # The base disc is a small fraction of the wetted area; if the station were
    # wrong it would take most of the body with it.
    forebody_elems = int(text.split("MARKER_TAG= vehicle\nMARKER_ELEMS= ")[1].split("\n")[0])
    base_elems = int(text.split("MARKER_TAG= vehicle_base\nMARKER_ELEMS= ")[1].split("\n")[0])
    assert 0 < base_elems < forebody_elems / 5


def test_a_base_station_outside_the_body_is_refused(tmp_path) -> None:
    pytest.importorskip("gmsh")
    from aether.aerodynamics.cfd.meshing import inviscid_domain
    from aether.geometry import bodies, brep

    surface = brep.surface_mesh(bodies.sphere_cone(), size_max=0.25, curvature_nodes=6)
    with pytest.raises(ValueError, match="base_station"):
        inviscid_domain(surface, tmp_path / "x.su2", mach=8.0, base_station=99.0)


def test_a_rippling_but_settled_solution_is_recognised() -> None:
    """The failure that made every hypersonic case look unconverged.

    A shock-dominated Euler solution oscillates by several percent forever; its
    *mean* is what converges and what gets reported. The real sphere-cone
    history settles to a forebody mean stable to 0.2 % while rippling 5.8 %
    peak-to-peak, and the previous criterion rejected it on the ripple alone —
    so no iteration count could ever have passed.
    """
    from aether.cfd.solver import _force_settled

    step = np.arange(24000, dtype=float)
    ripple = 0.058 * 0.0717 * np.sin(step / 31.0) + 0.01 * 0.0717 * np.sin(step / 7.3)
    settled_signal = 0.0717 + ripple

    ok, drift, spread = _force_settled(settled_signal, 12000, 5.0e-3)
    assert ok, f"a trend-free 5.8 % ripple must pass (drift {drift:.2e})"
    assert spread > 0.05, "the test signal should actually be rippling"
    assert drift < 1.0e-3


def test_a_window_too_short_to_resolve_the_ripple_is_refused() -> None:
    """Not 'unconverged' — *unmeasurable*, which is a different verdict.

    Over one or two oscillations every statistic reads whatever the window
    happened to catch. The guard counts periods rather than thresholding the
    amplitude, so a short window is rejected as a measurement while the same
    signal passes over a long one.
    """
    from aether.cfd.solver import _force_settled

    step = np.arange(24000, dtype=float)
    # Period 2*pi*100 ~ 628 iterations: 800 spans barely one oscillation,
    # 16000 spans twenty-five.
    signal = 0.0717 + 0.058 * 0.0717 * np.sin(step / 100.0)

    short, _, _ = _force_settled(signal, 800, 5.0e-3)
    long, _, _ = _force_settled(signal, 16000, 5.0e-3)
    assert not short, "one period cannot resolve a mean"
    assert long, "twenty-five periods can"


def test_a_genuinely_drifting_signal_still_fails() -> None:
    """The guard must not turn into a way of passing everything."""
    from aether.cfd.solver import _force_settled

    step = np.arange(24000, dtype=float)
    drifting = 0.0717 + 0.058 * 0.0717 * np.sin(step / 31.0) + 4.0e-7 * step
    ok, drift, _ = _force_settled(drifting, 12000, 5.0e-3)
    assert not ok, f"a 6 % climb across the window must fail (drift {drift:.2e})"


def test_the_base_is_split_by_normal_not_by_station() -> None:
    """A station cut sweeps forebody into the base once a shoulder is blunted.

    On a 10 mm-shouldered sphere-cone, cutting at 0.985 of the length made the
    base patch 1.14 times the area of the disc — a band of cone counted as
    base, inflating a correlated base drag by the same fraction and removing
    that area's contribution from the forebody. Splitting on the outward normal
    needs no magic number and falls where the flow separates: the widest point,
    where the fillet's normal turns aft.
    """
    from aether.geometry import bodies, brep

    body = bodies.sphere_cone(
        half_angle=np.deg2rad(10.0), nose_radius=0.05, length=2.0, shoulder_radius=0.010
    )
    surface = brep.surface_mesh(body, size_max=0.15, size_min=0.007, curvature_nodes=24)
    properties = brep.solid_properties(body)
    disc = np.pi * (properties.diameter / 2.0) ** 2

    normals = np.asarray(surface.normals)
    areas = np.asarray(surface.areas)
    centroids = np.asarray(surface.centroids)
    low, high = surface.bounds

    by_normal = float(areas[normals[:, 0] > 0.0].sum())
    station = float(low[0] + 0.985 * (high[0] - low[0]))
    by_station = float(areas[centroids[:, 0] >= station].sum())

    assert by_normal / disc < 1.05, f"normal split gives {by_normal / disc:.2f}x the disc"
    assert by_station / disc > 1.10, "the station cut should over-collect, as it did"
    assert by_normal < by_station


def test_nemo_substitutes_a_flux_it_implements_and_says_so() -> None:
    """HLLC is the best perfect-gas choice and NEMO does not implement it.

    So a caller who has read the flux matrix reaches for exactly the setting
    that kills a Mach 16 run at "Invalid upwind scheme", before the mesh is
    read. Substituting is right; substituting *silently* is not -- a config
    that disagrees with its own record is the failure this pipeline spent a day
    on, so the file says what it did.
    """
    from aether.cfd.config import FlowConditions, Numerics, SU2Case
    from aether.cfd.geometries import reference_frame

    reference = reference_frame(
        "sphere-cone", nose_radius=0.06, length=2.0, half_angle=10.0
    )

    def render(mach: float) -> str:
        return SU2Case(
            flow=FlowConditions.at_altitude(mach=mach, altitude=35_000.0),
            reference=reference,
            mesh_filename="m.su2",
            turbulence=None,
            numerics=Numerics(convective="HLLC"),
        ).render()

    perfect = render(8.0)
    assert "CONV_NUM_METHOD_FLOW= HLLC" in perfect
    assert "not implemented" not in perfect

    nonequilibrium = render(16.0)
    # AUSM+M, not AUSM+-up2: the latter is what every failing Mach 12-to-20 run
    # used, and it is the scheme that limit-cycles on the perfect-gas path.
    assert "CONV_NUM_METHOD_FLOW= AUSMPLUSM" in nonequilibrium
    assert "HLLC is not implemented for nonequilibrium" in nonequilibrium


def test_a_scheme_nemo_does_implement_is_left_alone() -> None:
    from aether.cfd.config import Nonequilibrium

    # Read from SU2 8.5.0's CDriver NEMO branch, not inferred. An earlier set
    # was missing AUSMPLUSM -- the one scheme here built for robustness at a
    # strong shock.
    regime = Nonequilibrium()
    for scheme in ("AUSM", "AUSMPLUSM", "AUSMPLUSUP2", "MSW", "ROE"):
        assert regime.convective_for(scheme) == scheme
    for scheme in ("HLLC", "SLAU", "LAX-FRIEDRICH"):
        assert regime.convective_for(scheme) == "AUSMPLUSM"
