"""Plasma/RF blackout and ablation: the two signature capabilities.

Both are checked against things that are known independently of the code —
textbook critical densities, an analytic energy balance — rather than against
their own previous output, because both were transcribed from drafts that
looked right and were not.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from aether.cfd import (
    AblationSolver,
    BlackoutWindow,
    TPSMaterial,
    UnderResolvedGrid,
    attenuation_db,
    collision_frequency,
    critical_density,
    plasma_frequency,
    saha_electron_density,
)


class TestPlasma:
    @pytest.mark.parametrize(
        ("frequency", "expected"),
        [(2.2e9, 6.0e16), (8.4e9, 8.75e17), (32.0e9, 1.27e19)],
    )
    def test_critical_density_matches_the_textbook_values(self, frequency, expected):
        """S-, X- and Ka-band cut-off densities are standard numbers."""
        assert critical_density(frequency) == pytest.approx(expected, rel=0.02)

    def test_plasma_frequency_inverts_critical_density(self):
        """The two are the same relation read in opposite directions."""
        for f in (1.0e9, 1.0e10, 1.0e11):
            n = critical_density(f)
            assert float(plasma_frequency(n)[0]) == pytest.approx(2.0 * np.pi * f, rel=1e-9)

    def test_saha_is_written_in_one_unit_system(self):
        """The drafted version mixed eV and SI and produced plausible nonsense.

        Two independent anchors: ionisation is negligible at 1500 K, and by
        6000 K equilibrium air is well past S-band cut-off — which is what
        makes re-entry blackout a phenomenon at all.
        """
        cold = float(saha_electron_density(1500.0, 5000.0)[0])
        hot = float(saha_electron_density(6000.0, 5000.0)[0])
        assert cold < 1.0e12
        assert hot > critical_density(2.2e9)
        assert hot / max(cold, 1.0) > 1.0e6

    def test_electron_density_rises_with_temperature(self):
        t = np.array([2000.0, 3000.0, 4000.0, 5000.0, 6000.0])
        n = saha_electron_density(t, np.full_like(t, 5000.0))
        assert np.all(np.diff(n) > 0.0)

    def test_the_mole_fraction_is_not_optional(self):
        """n_e goes as sqrt(n_species); taking the whole mixture is 10x out."""
        trace = float(saha_electron_density(5000.0, 5000.0, mole_fraction=0.01)[0])
        whole = float(saha_electron_density(5000.0, 5000.0, mole_fraction=1.0)[0])
        assert whole / trace == pytest.approx(10.0, rel=0.01)

    def test_cut_off_is_infinite_loss_not_a_large_number(self):
        """A reflected link is a different situation from an attenuated one."""
        dense = np.array([1.0e21])
        nu = collision_frequency(np.array([6000.0]), np.array([5000.0]))
        loss = attenuation_db(dense, nu, 2.2e9, np.array([0.01]))
        assert np.isinf(loss[0])

        window = BlackoutWindow(2.2e9, float(loss[0]), 1.0e21, critical_density(2.2e9))
        assert window.cut_off
        assert np.isinf(window.margin_db)
        assert "CUT OFF" in window.summary()

    def test_attenuation_falls_with_frequency(self):
        """The whole argument for a Ka-band telemetry link."""
        n = np.full(20, 1.0e17)
        nu = collision_frequency(np.full(20, 4000.0), np.full(20, 5000.0))
        ds = np.full(20, 0.002)
        losses = [float(np.sum(attenuation_db(n, nu, f, ds)))
                  for f in (8.4e9, 20.0e9, 32.0e9)]
        assert all(np.isfinite(losses))
        assert losses[0] > losses[1] > losses[2]


@pytest.fixture(scope="module")
def material() -> TPSMaterial:
    """Emissivity zero so the analytic balance is exact.

    With re-radiation the surface sheds several MW/m^2 and there is no closed
    form left to check the solver against.
    """
    return TPSMaterial(density=1800.0, specific_heat=1200.0, conductivity=8.0,
                       heat_of_ablation=25.0e6, ablation_temperature=3300.0,
                       emissivity=0.0)


@pytest.fixture(scope="module")
def columns():
    from aether.geometry import bodies, brep, wall_columns

    surface = brep.surface_mesh(
        bodies.caret_waverider(design_mach=8.0, wedge_angle=np.deg2rad(6.0), length=4.0),
        size_max=0.15, curvature_nodes=10,
    )
    return wall_columns(surface, n_cells=32, wall_cell=1.0e-4, max_depth=0.06)


class TestAblation:
    """Through-thickness recession on wall-normal graded columns."""

    @staticmethod
    def _steady_rate(material: TPSMaterial, flux: float, initial: float = 300.0) -> float:
        """Closed-form steady recession rate (m/s).

        Latent heat *plus* the sensible heat to bring virgin material up to the
        ablation temperature. Omitting the second term over-predicts by 14 % at
        these properties.
        """
        rise = material.ablation_temperature - initial
        return flux / (
            material.density * (material.heat_of_ablation + material.specific_heat * rise)
        )

    def test_recession_matches_the_steady_energy_balance(self, material, columns):
        flux = 6.0e6
        solver = AblationSolver(material, columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnderResolvedGrid)
            first = solver.advance(np.full(columns.n_columns, flux), 120.0)
            second = solver.advance(np.full(columns.n_columns, flux), 120.0)

        measured = (second.peak_recession - first.peak_recession) / 120.0
        expected = self._steady_rate(material, flux)
        assert measured == pytest.approx(expected, rel=0.05), (
            f"{1e3 * measured:.4f} mm/s against {1e3 * expected:.4f} mm/s"
        )

    def test_the_wall_settles_at_the_ablation_temperature(self, material, columns):
        """A receding surface sits just above it; far above means dead cells heating."""
        solver = AblationSolver(material, columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnderResolvedGrid)
            state = solver.advance(np.full(columns.n_columns, 6.0e6), 60.0)
        assert material.ablation_temperature <= state.peak_wall_temperature
        assert state.peak_wall_temperature < material.ablation_temperature + 100.0

    def test_consumed_cells_stop_absorbing(self, material, columns):
        """Frozen out of the linear system, or their temperature runs away.

        Leaving them in gave 221,000 K on this vehicle, and the energy they
        swallowed went missing from the recession it should have driven.
        """
        solver = AblationSolver(material, columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnderResolvedGrid)
            solver.advance(np.full(columns.n_columns, 6.0e6), 90.0)
        assert np.any(solver.phi <= 0.0), "nothing ablated, so the test proves nothing"
        assert np.all(solver.temperature[solver.phi <= 0.0] < 1.0e4)

    def test_grid_convergence_of_the_recession(self, material):
        """Refining the wall cell must not move the answer much."""
        from aether.geometry import bodies, brep, wall_columns

        surface = brep.surface_mesh(
            bodies.caret_waverider(design_mach=8.0, wedge_angle=np.deg2rad(6.0), length=4.0),
            size_max=0.15, curvature_nodes=10,
        )
        answers = []
        for wall_cell, n_cells in ((2.0e-4, 24), (5.0e-5, 40)):
            grid = wall_columns(surface, n_cells=n_cells, wall_cell=wall_cell,
                                max_depth=0.06)
            solver = AblationSolver(material, grid)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UnderResolvedGrid)
                answers.append(
                    solver.advance(np.full(grid.n_columns, 6.0e6), 90.0).peak_recession
                )
        assert answers[0] == pytest.approx(answers[1], rel=0.02)

    def test_ablation_never_creates_material(self, material, columns):
        """The bug the smoke test caught: a vehicle that gained mass."""
        solver = AblationSolver(material, columns)
        start = solver.mass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnderResolvedGrid)
            state = solver.advance(np.full(columns.n_columns, 3.0e6), 60.0)
        assert state.mass <= start, f"gained mass: {start:.2f} -> {state.mass:.2f} kg"
        assert 0.0 <= state.mass_lost_fraction <= 1.0
        assert np.all(solver.phi >= 0.0) and np.all(solver.phi <= 1.0)

    def test_the_centre_of_gravity_moves_away_from_the_heating(self, material, columns):
        """Heat only the forward third; the shield CG must move aft."""
        solver = AblationSolver(material, columns)
        x = columns.origin[:, 0]
        forward = x < x.min() + 0.33 * (x.max() - x.min())
        flux = np.where(forward, 6.0e6, 0.0)
        before = solver.centroid()[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnderResolvedGrid)
            state = solver.advance(flux, 90.0)
        assert state.centroid[0] > before
        assert state.mass_lost_fraction > 0.0
        assert state.peak_recession > 0.0

    def test_vehicle_centroid_dilutes_the_shield_shift(self, material, columns):
        """The number a trajectory needs; the shield's own shift overstates it."""
        solver = AblationSolver(material, columns)
        x = columns.origin[:, 0]
        flux = np.where(x < x.min() + 0.33 * (x.max() - x.min()), 6.0e6, 0.0)
        inert_cg = np.array([float(np.mean(x)), 0.0, 0.0])
        before = solver.vehicle_centroid(4000.0, inert_cg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnderResolvedGrid)
            solver.advance(flux, 90.0)
        after = solver.vehicle_centroid(4000.0, inert_cg)
        shield_shift = abs(solver.centroid()[0] - inert_cg[0])
        assert abs(after[0] - before[0]) < shield_shift

    def test_a_coarse_wall_cell_says_so(self, material):
        from aether.geometry import bodies, brep, wall_columns

        surface = brep.surface_mesh(bodies.sphere_cone(), size_max=0.2,
                                    curvature_nodes=6)
        grid = wall_columns(surface, n_cells=6, growth=1.0, max_depth=0.05)
        solver = AblationSolver(material, grid)
        with pytest.warns(UnderResolvedGrid, match="thermal penetration"):
            solver.advance(np.full(grid.n_columns, 1.0e6), 1.0)

    def test_a_step_beyond_the_stable_limit_is_refused(self, material, columns):
        solver = AblationSolver(material, columns)
        flux = np.full(columns.n_columns, 6.0e6)
        with pytest.raises(ValueError, match="stable step"):
            solver.step(flux, 100.0 * solver.stable_step(flux))

