"""6-DOF state: attitude kinematics and deformed-surface incidence."""

import numpy as np
import pytest
import scipy.integrate

from aether.dynamics import (
    dcm_from_quaternion,
    deformed_normal,
    local_incidence,
    quaternion_derivative,
    quaternion_norm_error,
)


class TestQuaternionKinematics:
    def test_norm_is_invariant_of_the_exact_flow(self):
        """Without stabilization ||q|| is conserved by the exact flow, so
        d/dt(||q||^2) must vanish identically."""
        rng = np.random.default_rng(0)
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        w = rng.standard_normal(3)
        dq = quaternion_derivative(q, w, baumgarte_gain=0.0)
        assert abs(2.0 * q @ dq) < 1e-15

    def test_constant_rate_gives_exact_rotation(self):
        """Rotating at a constant body rate about z for time t must equal
        the analytic half-angle quaternion."""
        omega = np.array([0.0, 0.0, 1.7])
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        t_end = 1.1
        sol = scipy.integrate.solve_ivp(
            lambda _t, q: quaternion_derivative(q, omega, 0.0),
            (0.0, t_end),
            q0,
            rtol=1e-12,
            atol=1e-14,
        )
        angle = 1.7 * t_end
        expected = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        assert np.allclose(sol.y[:, -1], expected, atol=1e-10)

    def test_baumgarte_pulls_norm_back_to_unity(self):
        """The unit sphere must be attracting, not merely invariant."""
        q0 = 1.10 * np.array([1.0, 0.0, 0.0, 0.0])
        k_q = 5.0
        sol = scipy.integrate.solve_ivp(
            lambda _t, q: quaternion_derivative(q, np.array([0.3, -0.2, 0.5]), k_q),
            (0.0, 6.0 / k_q),
            q0,
            rtol=1e-11,
            atol=1e-13,
        )
        assert abs(quaternion_norm_error(q0)) > 0.09
        assert abs(float(quaternion_norm_error(sol.y[:, -1]))) < 1e-4

    def test_baumgarte_leaves_unit_quaternion_untouched(self):
        q = np.array([0.6, 0.8, 0.0, 0.0])
        w = np.array([0.1, 0.2, 0.3])
        assert np.allclose(
            quaternion_derivative(q, w, 0.0), quaternion_derivative(q, w, 10.0)
        )

    def test_batched(self):
        rng = np.random.default_rng(3)
        q = rng.standard_normal((5, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w = rng.standard_normal((5, 3))
        batch = quaternion_derivative(q, w, 2.0)
        for i in range(5):
            assert np.allclose(batch[i], quaternion_derivative(q[i], w[i], 2.0))

    def test_validation(self):
        with pytest.raises(ValueError, match="trailing dimension 4"):
            quaternion_derivative(np.zeros(3), np.zeros(3))
        with pytest.raises(ValueError, match="trailing dimension 3"):
            quaternion_derivative(np.array([1.0, 0, 0, 0]), np.zeros(4))
        with pytest.raises(ValueError, match="baumgarte_gain"):
            quaternion_derivative(np.array([1.0, 0, 0, 0]), np.zeros(3), -1.0)


class TestDirectionCosineMatrix:
    def test_identity_quaternion(self):
        assert np.allclose(dcm_from_quaternion([1.0, 0, 0, 0]), np.eye(3))

    def test_orthonormal_and_proper(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            q = rng.standard_normal(4)
            c = dcm_from_quaternion(q)
            assert np.allclose(c @ c.T, np.eye(3), atol=1e-13)
            assert np.linalg.det(c) == pytest.approx(1.0, abs=1e-13)

    def test_hamilton_convention_not_jpl(self):
        """A +90 deg rotation about z maps the ECI x-axis to the body
        -y axis under the Hamilton ECI->body convention. The JPL
        convention gives the transpose, which is also a valid rotation —
        which is exactly why this is asserted rather than assumed."""
        q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        c = dcm_from_quaternion(q)
        assert np.allclose(c @ np.array([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-14)
        assert np.allclose(c @ np.array([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-14)

    def test_composition_matches_quaternion_product(self):
        """C(q) for a 2*theta rotation equals C(q_theta) applied twice."""
        theta = 0.37
        q1 = np.array([np.cos(theta / 2), 0, np.sin(theta / 2), 0])
        q2 = np.array([np.cos(theta), 0, np.sin(theta), 0])
        c1 = dcm_from_quaternion(q1)
        assert np.allclose(c1 @ c1, dcm_from_quaternion(q2), atol=1e-13)

    def test_normalizes_input(self):
        q = np.array([2.0, 0.0, 0.0, 0.0])
        assert np.allclose(dcm_from_quaternion(q), np.eye(3))
        with pytest.raises(ValueError, match="non-zero"):
            dcm_from_quaternion(np.zeros(4))


class TestDeformedNormal:
    def test_flat_undeformed_surface(self):
        n = deformed_normal(0.0, 0.0)
        assert np.allclose(n, [0.0, 0.0, 1.0])

    def test_midsurface_normal_formula(self):
        """offset = 0 must give n ∝ (-w_x, -w_y, 1)."""
        w_x, w_y = 0.3, -0.2
        n = deformed_normal(w_x, w_y)
        expected = np.array([-w_x, -w_y, 1.0])
        expected /= np.linalg.norm(expected)
        assert np.allclose(n, expected)

    def test_offset_uses_rotation_fields(self):
        """A non-zero half-thickness makes the normal depend on the
        rotations, not just the midsurface slope."""
        common = {"w_x": 0.1, "w_y": 0.05}
        n_mid = deformed_normal(**common)
        n_off = deformed_normal(
            **common,
            phi_x=0.2, phi_y=-0.1,
            phi_x_x=0.4, phi_x_y=0.1, phi_y_x=-0.2, phi_y_y=0.3,
            offset=0.05,
        )
        assert not np.allclose(n_mid, n_off)
        assert np.linalg.norm(n_off) == pytest.approx(1.0)

    def test_offset_requires_derivatives(self):
        with pytest.raises(ValueError, match="rotation derivatives"):
            deformed_normal(0.1, 0.1, offset=0.05)

    def test_batched_shape(self):
        w_x = np.linspace(-0.3, 0.3, 7)
        w_y = np.zeros(7)
        n = deformed_normal(w_x, w_y)
        assert n.shape == (7, 3)
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


class TestLocalIncidence:
    def test_sign_convention_windward_positive(self):
        """Paper II, Remark 1: with n the OUTWARD normal, a panel facing
        into the flow must give delta_c > 0. Dropping the negation
        inverts the entire pressure and heating distribution."""
        normal = np.array([0.0, 0.0, 1.0])  # upward-facing panel
        v_from_above = np.array([0.0, 0.0, -1.0])  # flow arriving downward
        assert float(local_incidence(normal, v_from_above)) == pytest.approx(np.pi / 2)
        v_from_below = np.array([0.0, 0.0, 1.0])
        assert float(local_incidence(normal, v_from_below)) == pytest.approx(-np.pi / 2)

    def test_edge_on_flow_is_zero(self):
        assert float(
            local_incidence([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
        ) == pytest.approx(0.0, abs=1e-15)

    def test_known_angle(self):
        # a panel whose outward normal is tilted so that sin(delta) = sin(theta)
        theta = np.deg2rad(30.0)
        normal = np.array([0.0, -np.cos(theta), np.sin(theta)])
        v = np.array([0.0, 0.0, -1.0])
        assert float(local_incidence(normal, v)) == pytest.approx(theta, abs=1e-12)

    def test_velocity_need_not_be_normalized(self):
        n = np.array([0.0, 0.0, 1.0])
        assert float(local_incidence(n, [0.0, 0.0, -7.3])) == pytest.approx(np.pi / 2)

    def test_validation(self):
        with pytest.raises(ValueError, match="non-zero"):
            local_incidence([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="trailing dimension 3"):
            local_incidence([0.0, 1.0], [0.0, 0.0, 1.0])



class TestTheAttitudeConventionIsPinned:
    """Which quaternion and rotation convention this kernel uses, stated as a test.

    :func:`dcm_from_quaternion` is correct, and that is exactly why this exists.
    The convention it implements -- scalar-first ``(w, x, y, z)`` with the
    *passive* sense, so the matrix maps inertial components to body components --
    is a choice, and it is currently recorded nowhere an importer would see. The
    alternatives are equally standard: SciPy takes scalar-last and returns the
    *active* rotation, which is this matrix transposed.

    A mismatch between the two is silent. Both are orthonormal with unit
    determinant, both round-trip, and a trajectory flown with the transpose
    simply goes somewhere else. Nothing about the arithmetic complains. The risk
    is not in this function but at its boundaries: the moment any external
    library -- SciPy, Orekit, Basilisk -- is brought alongside it, one side must
    be transposed, and this test is what says which.
    """

    @staticmethod
    def _scipy_rotation(quaternion):
        from scipy.spatial.transform import Rotation

        w, x, y, z = quaternion
        return Rotation.from_quat([x, y, z, w])

    def test_it_is_scalar_first_and_passive(self):
        """The convention itself, against an independent implementation.

        Agreement to machine precision over random attitudes, and specifically
        with SciPy's *transpose* -- the assertion that would fail first if the
        convention were ever changed.
        """
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(500):
            quaternion = rng.normal(size=4)
            quaternion /= np.linalg.norm(quaternion)
            ours = np.asarray(dcm_from_quaternion(quaternion))
            theirs = self._scipy_rotation(quaternion).as_matrix()
            worst = max(worst, float(np.max(np.abs(ours - theirs.T))))
        assert worst < 1e-14

    def test_the_active_reading_is_not_what_this_returns(self):
        """Stated positively so the failure mode is documented, not merely absent.

        If a future change made this return SciPy's matrix rather than its
        transpose, the test above would fail and this one would too -- and the
        pair together say *which* direction the convention moved.
        """
        rng = np.random.default_rng(1)
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        ours = np.asarray(dcm_from_quaternion(quaternion))
        theirs = self._scipy_rotation(quaternion).as_matrix()
        assert np.max(np.abs(ours - theirs)) > 1e-3

    def test_it_maps_inertial_components_into_the_body_frame(self):
        """What 'passive' means operationally, on a case with an obvious answer.

        A 90 degree rotation about z: a vector along inertial x has body
        components along -y under the passive reading. Getting +y instead would
        mean the active convention.
        """
        half = np.sqrt(0.5)
        quaternion = np.array([half, 0.0, 0.0, half])  # w, x, y, z
        matrix = np.asarray(dcm_from_quaternion(quaternion))
        body = matrix @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(body, [0.0, -1.0, 0.0], atol=1e-12)

    def test_it_is_a_proper_rotation(self):
        """Orthonormal with unit determinant -- true of both conventions.

        Included to make the point that these checks cannot distinguish them,
        which is why the convention needs pinning separately.
        """
        rng = np.random.default_rng(2)
        for _ in range(100):
            quaternion = rng.normal(size=4)
            quaternion /= np.linalg.norm(quaternion)
            matrix = np.asarray(dcm_from_quaternion(quaternion))
            assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-14)
            assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-14)


class TestTheFallbackMatchesTheLibrary:
    """The closed form is retained only for arithmetics SciPy cannot take.

    It must therefore agree with SciPy exactly on the floats where both work,
    or the certified layer and the simulator would be using different rotations
    and nothing would say so.
    """

    def test_it_agrees_with_the_scipy_path_on_floats(self):
        from aether.dynamics.attitude import _dcm_expression

        rng = np.random.default_rng(3)
        worst = 0.0
        for _ in range(500):
            quaternion = rng.normal(size=4)
            quaternion /= np.linalg.norm(quaternion)
            worst = max(
                worst,
                float(
                    np.max(
                        np.abs(
                            _dcm_expression(quaternion)
                            - np.asarray(dcm_from_quaternion(quaternion))
                        )
                    )
                ),
            )
        assert worst < 1e-14

    def test_the_batched_shape_survives_the_swap(self):
        """SciPy is inherently batched; the reshape around it must preserve shape."""
        rng = np.random.default_rng(4)
        quaternions = rng.normal(size=(7, 4))
        quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
        assert np.asarray(dcm_from_quaternion(quaternions)).shape == (7, 3, 3)

    def test_a_zero_quaternion_is_still_refused(self):
        with pytest.raises(ValueError, match="non-zero"):
            dcm_from_quaternion([0.0, 0.0, 0.0, 0.0])
