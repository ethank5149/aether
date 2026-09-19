"""Blended pressure closure and panel loads (Paper II, §3.3)."""

import numpy as np
import pytest

from aether.aerodynamics import (
    PanelModel,
    blended_pressure_coefficient,
    curved_lifting_body,
    newtonian_pressure_coefficient,
    prandtl_meyer_angle,
    prandtl_meyer_pressure_coefficient,
    rayleigh_pitot_cp_max,
    smoothstep,
    vacuum_pressure_coefficient,
)

MACH = 8.0


class TestRayleighPitot:
    def test_hypersonic_limit(self):
        """Cp_max -> 1.8394 for gamma = 1.4 as M -> infinity."""
        assert rayleigh_pitot_cp_max(1000.0) == pytest.approx(1.8394, abs=1e-4)

    def test_monotone_in_mach(self):
        vals = [rayleigh_pitot_cp_max(m) for m in (2, 3, 5, 10, 20, 50)]
        assert np.all(np.diff(vals) > 0)
        assert all(1.0 < v < 1.84 for v in vals)

    def test_vacuum_limit_formula(self):
        assert vacuum_pressure_coefficient(MACH) == pytest.approx(
            -2.0 / (1.4 * MACH**2)
        )

    def test_validation(self):
        with pytest.raises(ValueError, match="supersonic"):
            rayleigh_pitot_cp_max(0.5)
        with pytest.raises(ValueError, match="gamma"):
            rayleigh_pitot_cp_max(5.0, gamma=0.9)


class TestPrandtlMeyer:
    def test_zero_at_sonic(self):
        assert float(prandtl_meyer_angle(1.0)) == pytest.approx(0.0, abs=1e-15)

    def test_textbook_value_at_mach_two(self):
        """nu(2) = 26.38 deg for gamma = 1.4."""
        assert np.rad2deg(float(prandtl_meyer_angle(2.0))) == pytest.approx(
            26.3798, abs=1e-3
        )

    def test_monotone(self):
        m = np.linspace(1.0, 10.0, 50)
        assert np.all(np.diff(prandtl_meyer_angle(m)) > 0)

    def test_expansion_lowers_pressure(self):
        cp = prandtl_meyer_pressure_coefficient(np.deg2rad([-1.0, -5.0, -15.0]), MACH)
        assert np.all(np.diff(cp) < 0), "stronger expansion must lower Cp"
        assert np.all(cp < 0.0)

    def test_zero_turn_gives_freestream(self):
        assert float(prandtl_meyer_pressure_coefficient(0.0, MACH)) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_floored_at_vacuum(self):
        cp = prandtl_meyer_pressure_coefficient(np.deg2rad(-80.0), MACH)
        assert float(cp) == pytest.approx(vacuum_pressure_coefficient(MACH), rel=1e-12)

    def test_compression_branch_is_smooth_extension(self):
        """Positive incidence continues the isentropic relation into
        compression — the smooth extension the blend needs."""
        cp = prandtl_meyer_pressure_coefficient(np.deg2rad([1.0, 3.0]), MACH)
        assert np.all(cp > 0.0)
        assert cp[1] > cp[0]

    def test_validation(self):
        with pytest.raises(ValueError, match="M >= 1"):
            prandtl_meyer_angle(0.5)


class TestSmoothstep:
    def test_endpoints_and_midpoint(self):
        assert float(smoothstep(0.0)) == 0.0
        assert float(smoothstep(1.0)) == 1.0
        assert float(smoothstep(0.5)) == pytest.approx(0.5)

    def test_first_and_second_derivatives_vanish_at_ends(self):
        """C2 continuity at the band edges needs S' = S'' = 0 there;
        a cubic smoothstep would only give C1."""
        h = 1e-4
        for t0 in (0.0, 1.0):
            t = np.array([t0 - 2 * h, t0 - h, t0, t0 + h, t0 + 2 * h])
            s = smoothstep(np.clip(t, 0.0, 1.0))
            inner = np.clip(np.array([t0 - h, t0, t0 + h]), 0.0, 1.0)
            si = smoothstep(inner)
            d1 = (si[2] - si[0]) / (2 * h)
            d2 = (s[3] - 2 * s[2] + s[1]) / h**2
            assert abs(d1) < 1e-6
            assert abs(d2) < 1e-2

    def test_clamps_outside_unit_interval(self):
        assert float(smoothstep(-3.0)) == 0.0
        assert float(smoothstep(4.0)) == 1.0


class TestBlendedClosure:
    def test_unblended_branches_match_their_formulas(self):
        cp_max = rayleigh_pitot_cp_max(MACH)
        d = np.deg2rad(np.array([-10.0, 10.0]))
        cp = blended_pressure_coefficient(d, MACH, blend_width=0.0)
        assert cp[1] == pytest.approx(
            float(newtonian_pressure_coefficient(d[1], cp_max))
        )
        assert cp[0] == pytest.approx(
            float(prandtl_meyer_pressure_coefficient(d[0], MACH))
        )

    def test_unblended_closure_is_c0_but_not_c1(self):
        """The premise of the whole blending exercise (Paper II, Remark 2)."""
        h = 1e-7
        cp_m = float(blended_pressure_coefficient(-h, MACH, blend_width=0.0))
        cp_p = float(blended_pressure_coefficient(+h, MACH, blend_width=0.0))
        assert abs(cp_m - cp_p) < 1e-5, "must be continuous"
        slope_lee = (float(blended_pressure_coefficient(-h, MACH, blend_width=0.0))
                     - float(blended_pressure_coefficient(-3 * h, MACH, blend_width=0.0))) / (2 * h)
        slope_wind = (float(blended_pressure_coefficient(3 * h, MACH, blend_width=0.0))
                      - float(blended_pressure_coefficient(h, MACH, blend_width=0.0))) / (2 * h)
        assert abs(slope_wind) < 1e-3, "Newtonian branch has zero slope at the seam"
        assert abs(slope_lee) > 0.1, "expansion branch does not"
        assert abs(slope_lee) > 100 * abs(slope_wind), "the slopes must not match"

    def test_blending_is_c1_across_the_seam(self):
        """With a blend the first derivative must be continuous through
        delta = 0, which the unblended closure is not."""
        width = 0.05
        h = width / 200.0

        def slope(d0):
            return (
                float(blended_pressure_coefficient(d0 + h, MACH, blend_width=width))
                - float(blended_pressure_coefficient(d0 - h, MACH, blend_width=width))
            ) / (2 * h)

        assert slope(-h) == pytest.approx(slope(h), rel=0.05)

    def test_blend_recovers_pure_branches_outside_the_band(self):
        width = 0.03
        for d in (-0.2, -0.05, 0.05, 0.2):
            assert float(blended_pressure_coefficient(d, MACH, blend_width=width)) == (
                pytest.approx(
                    float(blended_pressure_coefficient(d, MACH, blend_width=0.0)),
                    rel=1e-12,
                )
            )

    def test_blend_endpoints_match_at_band_edges(self):
        width = 0.04
        cp_max = rayleigh_pitot_cp_max(MACH)
        assert float(blended_pressure_coefficient(width, MACH, blend_width=width)) == (
            pytest.approx(float(newtonian_pressure_coefficient(width, cp_max)))
        )
        assert float(blended_pressure_coefficient(-width, MACH, blend_width=width)) == (
            pytest.approx(float(prandtl_meyer_pressure_coefficient(-width, MACH)))
        )

    def test_never_below_vacuum(self):
        d = np.deg2rad(np.linspace(-89.0, 89.0, 200))
        cp = blended_pressure_coefficient(d, MACH, blend_width=0.02)
        assert np.all(cp >= vacuum_pressure_coefficient(MACH) - 1e-12)

    def test_validation(self):
        with pytest.raises(ValueError, match="blend_width"):
            blended_pressure_coefficient(0.0, MACH, blend_width=-0.1)
        with pytest.raises(ValueError, match="blend_width"):
            blended_pressure_coefficient(0.0, MACH, blend_width=2.0)


class TestPanelModel:
    # A class-scoped fixture defined as an *instance* method is deprecated:
    # the fixture runs once per class while each test gets a fresh instance,
    # so anything it set on `self` would be invisible. With
    # `filterwarnings = ["error"]` that deprecation is an error, which is
    # why these seven tests had been erroring at setup.
    @pytest.fixture(scope="class")
    @classmethod
    def body(cls):
        return curved_lifting_body()

    def test_geometry_sane(self, body):
        assert body.n_panels > 100
        assert body.total_area > 0.0
        assert np.allclose(np.linalg.norm(body.normals, axis=1), 1.0)

    def test_closed_body_normals_sum_to_zero(self, body):
        """A closed surface has sum(A_i n_i) = 0; the demonstration body
        is closed up to its open trailing edge, so the residual must be
        small relative to the total area."""
        residual = np.linalg.norm(np.sum(body.areas[:, None] * body.normals, axis=0))
        assert residual / body.total_area < 0.05

    def test_incidence_increases_with_alpha_on_lower_surface(self, body):
        lower = body.centroids[:, 2] < 0.0
        d0 = np.mean(body.incidences(0.0)[lower])
        d10 = np.mean(body.incidences(np.deg2rad(10.0))[lower])
        assert d10 > d0

    def test_normal_force_grows_with_incidence(self, body):
        q = 0.5 * 0.02 * (MACH * 300.0) ** 2
        n0 = body.loads(np.deg2rad(2.0), MACH, q)[0][2]
        n10 = body.loads(np.deg2rad(12.0), MACH, q)[0][2]
        assert n10 > n0

    def test_trim_solution_zeroes_the_moment(self, body):
        q = 0.5 * 0.02 * (MACH * 300.0) ** 2
        trim = body.trim(MACH, q, blend_width=0.02)
        assert trim.converged
        scale = q * body.total_area * 6.0
        assert abs(trim.pitching_moment) < 1e-8 * scale
        assert abs(body.pitching_moment(trim.incidence, MACH, q, blend_width=0.02)) < (
            1e-8 * scale
        )

    def test_trim_insensitive_to_blend_width(self, body):
        """The II-V4 criterion: trim incidence must not shift by more
        than 0.1 deg across a blend-width sweep."""
        q = 0.5 * 0.02 * (MACH * 300.0) ** 2
        alphas = [
            body.trim(MACH, q, blend_width=bw).incidence
            for bw in (0.0, 0.005, 0.01, 0.02, 0.04)
        ]
        shift = np.rad2deg(max(alphas) - min(alphas))
        assert shift < 0.1, f"trim shifted {shift:.4f} deg across the blend sweep"

    def test_untrimmable_bracket_raises(self, body):
        q = 0.5 * 0.02 * (MACH * 300.0) ** 2
        with pytest.raises(ValueError, match="no trim point"):
            body.trim(MACH, q, bracket=(np.deg2rad(20.0), np.deg2rad(30.0)))

    def test_validation(self):
        with pytest.raises(ValueError, match="unit vectors"):
            PanelModel(
                centroids=np.zeros((2, 3)),
                normals=np.full((2, 3), 0.5),
                areas=np.ones(2),
            )
        with pytest.raises(ValueError, match="areas"):
            PanelModel(
                centroids=np.zeros((2, 3)),
                normals=np.tile([0.0, 0.0, 1.0], (2, 1)),
                areas=np.array([1.0, -1.0]),
            )
        with pytest.raises(ValueError, match="n_chord"):
            curved_lifting_body(n_chord=2)
