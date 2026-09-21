"""The occupation-measure certificate for the downrange bound.

The moment relaxation is tested first on the 1D system — constant density,
no altitude dynamics — where the answer is known in closed form. That test
validates the SDP plumbing, after which the full polynomial entry system
shows the improvement over the box bound.
"""

from __future__ import annotations

import numpy as np
import pytest

cvxpy = pytest.importorskip("cvxpy")


class TestMomentRelaxation1D:
    r"""The range bound on a 1D system: dV/dt = -D₀V², R = ∫V dt.

    The closed-form answer is (1/D₀)ln(V₀/V_f), which is the limit the
    speed-banded box bound converges to. The moment relaxation must converge
    to it from above, and it must do so faster than the box banding does —
    otherwise there is no point.
    """

    @staticmethod
    def _closed_form(V0: float, Vf: float, D0: float) -> float:
        return np.log(V0 / Vf) / D0

    @staticmethod
    def _moment_bound(V0: float, Vf: float, D0: float, order: int) -> float:
        r"""Moment relaxation at the given order, in normalised coordinates.

        Normalise v = (V - V_f)/(V_0 - V_f) \in [0,1].  Real-time dynamics
        dV/dt = -D_0 V^2 become dv/dt = -\alpha (V_f + v \Delta V)^2 with
        \alpha = D_0 / \Delta V.  The range R = \int V\,dt becomes the
        objective \max V_f y_0 + \Delta V\, y_1.

        Liouville for \psi = v^n (n \ge 1):
          n \alpha [a_0 y_{n-1} + a_1 y_n + a_2 y_{n+1}] = 1
        where a_0 = V_f^2, a_1 = 2 V_f \Delta V, a_2 = \Delta V^2.
        """
        import cvxpy as cp

        dV = V0 - Vf
        a0 = Vf ** 2
        a1 = 2.0 * Vf * dV
        a2 = dV ** 2
        alpha = D0 / dV

        n_moments = 2 * order + 1
        y = cp.Variable(n_moments, name="y")
        constraints = []

        max_test = min(2 * order - 1, n_moments - 2)
        for n in range(1, max_test + 1):
            terms = 0.0
            for k, ak in enumerate([a0, a1, a2]):
                idx = n - 1 + k
                if idx < n_moments:
                    terms = terms + ak * y[idx]
            constraints.append(n * alpha * terms == 1.0)

        size = order + 1
        M = cp.Variable((size, size), symmetric=True, name="M")
        for i in range(size):
            for j in range(i, size):
                k = i + j
                if k < n_moments:
                    constraints.append(M[i, j] == y[k])
                    if i != j:
                        constraints.append(M[j, i] == y[k])
        constraints.append(M >> 0)

        loc_size = order
        if loc_size >= 1:
            L1 = cp.Variable((loc_size, loc_size), symmetric=True, name="L1")
            for i in range(loc_size):
                for j in range(i, loc_size):
                    k = i + j + 1
                    if k < n_moments:
                        constraints.append(L1[i, j] == y[k])
                        if i != j:
                            constraints.append(L1[j, i] == y[k])
            constraints.append(L1 >> 0)

        if loc_size >= 1:
            L2 = cp.Variable((loc_size, loc_size), symmetric=True, name="L2")
            for i in range(loc_size):
                for j in range(i, loc_size):
                    k = i + j
                    if k + 1 < n_moments:
                        constraints.append(L2[i, j] == y[k] - y[k + 1])
                        if i != j:
                            constraints.append(L2[j, i] == y[k] - y[k + 1])
            constraints.append(L2 >> 0)

        objective = cp.Maximize(Vf * y[0] + dV * y[1])
        problem = cp.Problem(objective, constraints)
        problem.solve(solver="CLARABEL")
        if problem.status not in ("optimal", "optimal_inaccurate"):
            return float("inf")
        return float(problem.value)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_it_converges_from_above_to_the_closed_form(self):
        """The relaxation hierarchy must be monotonically decreasing
        and must approach the closed form. It may not dip below it."""
        V0, Vf = 7387.0, 304.8
        rho = 1e-4
        beta = 281.0
        D0 = rho / (2.0 * beta)
        truth = self._closed_form(V0, Vf, D0)

        bounds = [self._moment_bound(V0, Vf, D0, order=d) for d in (2, 3, 4, 5)]

        for d, b in zip((2, 3, 4, 5), bounds):
            assert b >= truth * 0.999, f"degree {d} dipped below: {b:.1f} < {truth:.1f}"

        assert bounds == sorted(bounds, reverse=True), f"not monotone: {bounds}"

        assert bounds[-1] < truth * 1.10, (
            f"degree 5 still {bounds[-1] / truth:.2f}× the closed form"
        )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_the_box_bound_is_weaker_at_the_same_subdivision_count(self):
        """The moment relaxation at degree d uses O(d) moments — the same
        information as O(d) speed bands — but gets a tighter answer because
        it exploits the positive-semidefinite structure."""
        V0, Vf = 7387.0, 304.8
        rho = 1e-4
        beta = 281.0
        D0 = rho / (2.0 * beta)

        moment_d3 = self._moment_bound(V0, Vf, D0, order=3)

        edges = np.linspace(V0, Vf, 4)
        box_3 = sum(
            (fast**2 - slow**2) / (2.0 * D0 * slow**2)
            for fast, slow in zip(edges[:-1], edges[1:])
        )

        assert moment_d3 < box_3, (
            f"moment {moment_d3:.0f} should beat box {box_3:.0f} at matched complexity"
        )


class TestMomentBound:
    """The full 5-variable primal moment relaxation."""

    @pytest.fixture
    def corridor(self):
        from aether.certification.occupation import CorridorParams

        return CorridorParams(
            speed_high_ms=7387.0,
            speed_low_ms=304.8,
            altitude_ceiling_m=50e3,
            altitude_floor_m=20e3,
            ballistic_coefficient=281.0,
            lift_to_drag=0.30,
        )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_the_sdp_converges_at_order_three(self, corridor):
        """The primal moment relaxation gives a finite bound at d=3."""
        from aether.certification.occupation import corridor_moment_bound

        result = corridor_moment_bound(corridor, order=3)
        assert result.sdp_status in ("optimal", "optimal_inaccurate"), (
            f"SDP did not converge: {result.sdp_status}"
        )
        assert 0 < result.max_downrange_m < float("inf"), (
            f"bound not finite positive: {result.max_downrange_m}"
        )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_order_four_tightens_below_the_constant_density_bound(self, corridor):
        """At d=4 the dynamics-aware bound beats the 1D constant-density bound."""
        from aether.certification.occupation import corridor_moment_bound

        result = corridor_moment_bound(corridor, order=4)
        assert result.sdp_status in ("optimal", "optimal_inaccurate"), (
            f"SDP did not converge: {result.sdp_status}"
        )

        rho_min = 1.225 * np.exp(-corridor.altitude_ceiling_m / 8500.0)
        D0_min = rho_min / (2.0 * corridor.ballistic_coefficient)
        R_1d = np.log(corridor.speed_high_ms / corridor.speed_low_ms) / D0_min

        assert result.max_downrange_m < R_1d, (
            f"d=4 bound {result.max_downrange_m/1e3:.0f} km should be below "
            f"1D constant-density {R_1d/1e3:.0f} km"
        )


class TestOccupationBound:
    """Smoke test: the full 5-variable SOS certificate SDP converges."""

    @pytest.fixture
    def corridor(self):
        from aether.certification.occupation import CorridorParams

        return CorridorParams(
            speed_high_ms=7387.0,
            speed_low_ms=304.8,
            altitude_ceiling_m=50e3,
            altitude_floor_m=20e3,
            ballistic_coefficient=281.0,
            lift_to_drag=0.30,
        )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_the_sdp_converges_at_degree_two(self, corridor):
        """The SOS certificate SDP solves at the lowest useful order."""
        from aether.certification.occupation import corridor_certificate

        result = corridor_certificate(corridor, degree=2)
        assert result.sdp_status in ("optimal", "optimal_inaccurate"), (
            f"SDP did not converge: {result.sdp_status}"
        )
        assert result.max_downrange_m < float("inf")
