"""Tests for rigorous enclosures and verified reachable tubes.

The property that matters is **soundness**, not accuracy: an enclosure may be as
wide as it likes and still be useful, but it may never exclude a value the true
field attains. A too-wide enclosure costs subdivision effort; a too-narrow one
costs the proof, silently, in the one direction nothing checks.

Every field here has a closed-form solution, so "did it work" has an exact
answer rather than a plausible-looking curve. Nothing in this package knows about
any application, and these tests are written to keep it that way.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.certification import (
    NUMPY_OPS,
    enclose_field,
    interval,
    jacobian_enclosure,
    mean_value_step,
    reachable_step,
    reachable_tube,
    rough_enclosure,
    tube_widths,
)


def _harmonic(*frequencies: float):
    r"""Independent oscillators: :math:`\ddot y = -\omega^2 y`.

    Solution :math:`y(t) = y_0\cos\omega t + (v_0/\omega)\sin\omega t`, so where
    a certified tube must lie is known in advance.
    """

    def field(state, _control, _ops):
        rates = []
        for k, omega in enumerate(frequencies):
            rates.extend([state[2 * k + 1], -(omega**2) * state[2 * k]])
        return rates

    return field


def _damped(rate: float):
    r""":math:`\dot y = -k y`, solution :math:`y_0 e^{-kt}`."""

    def field(state, _control, _ops):
        return [-rate * state[0]]

    return field


def _driven():
    """A field that uses the control, so control quantification is exercised."""

    def field(state, control, ops):
        return [control * ops.cos(state[0])]

    return field


class TestInterval:
    def test_it_covers_the_requested_endpoints(self):
        ball = interval(-2.0, 3.0)
        assert float(ball.lower()) <= -2.0
        assert float(ball.upper()) >= 3.0

    def test_a_point_interval_is_the_point(self):
        ball = interval(1.5, 1.5)
        assert float(ball.lower()) == pytest.approx(1.5)
        assert float(ball.upper()) == pytest.approx(1.5)

    def test_a_reversed_interval_is_refused(self):
        with pytest.raises(ValueError, match="reversed"):
            interval(5.0, 1.0)

    def test_a_non_finite_interval_is_refused(self):
        with pytest.raises(ValueError, match="finite"):
            interval(0.0, np.inf)


class TestEncloseField:
    def test_it_contains_every_sampled_evaluation(self):
        field = _harmonic(1.0, 2.0)
        box = [(0.5, 1.5), (-0.5, 0.5), (0.5, 1.5), (-0.5, 0.5)]
        enclosure = enclose_field(field, box, (0.0, 0.0))
        rng = np.random.default_rng(0)
        for _ in range(2000):
            state = [rng.uniform(lo, hi) for lo, hi in box]
            for value, (lower, upper) in zip(
                field(state, 0.0, NUMPY_OPS), enclosure, strict=True
            ):
                assert lower <= value <= upper

    def test_it_narrows_as_the_box_narrows(self):
        """Enclosures must converge on the point value, or subdivision buys nothing."""
        field = _harmonic(1.0)
        widths = []
        for half in (0.4, 0.1, 0.025):
            enclosure = enclose_field(
                field, [(1.0 - half, 1.0 + half), (-half, half)], (0.0, 0.0)
            )
            widths.append(max(hi - lo for lo, hi in enclosure))
        assert widths == sorted(widths, reverse=True)
        assert widths[-1] < widths[0] / 10.0

    def test_the_control_is_quantified_over(self):
        """An interval control must widen the enclosure, or a tube means nothing."""
        field = _driven()
        narrow = enclose_field(field, [(0.0, 0.1)], (1.0, 1.0))
        wide = enclose_field(field, [(0.0, 0.1)], (-1.0, 1.0))
        assert (wide[0][1] - wide[0][0]) > (narrow[0][1] - narrow[0][0])


class TestReachableTube:
    def test_it_contains_the_analytic_solution(self):
        field = _harmonic(1.0, 2.0, 3.0)
        box = [(1.0, 1.0), (0.0, 0.0)] * 3
        dt, steps = 0.02, 25
        tube = reachable_tube(field, box, (0.0, 0.0), dt, steps)
        for k in range(steps):
            t = (k + 1) * dt
            exact = []
            for omega in (1.0, 2.0, 3.0):
                exact.extend([np.cos(omega * t), -omega * np.sin(omega * t)])
            for value, (lower, upper) in zip(exact, tube[k], strict=True):
                assert lower <= value <= upper

    def test_it_contains_an_exponential_decay(self):
        field = _damped(0.5)
        dt, steps = 0.05, 20
        tube = reachable_tube(field, [(1.0, 1.0)], (0.0, 0.0), dt, steps)
        for k in range(steps):
            lower, upper = tube[k][0]
            assert lower <= np.exp(-0.5 * (k + 1) * dt) <= upper

    def test_boxes_only_grow(self):
        """Axis-aligned boxes cannot contract under this scheme."""
        widths = tube_widths(
            reachable_tube(_harmonic(1.0, 2.0), [(0.9, 1.1), (-0.1, 0.1)] * 2,
                           (0.0, 0.0), 0.05, 10)
        )
        for component in range(widths.shape[1]):
            assert np.all(np.diff(widths[:, component]) >= -1e-12)

    def test_a_shorter_step_gives_a_tighter_tube_at_the_same_horizon(self):
        field = _harmonic(1.0)
        box = [(0.9, 1.1), (-0.1, 0.1)]
        coarse = tube_widths(reachable_tube(field, box, (0.0, 0.0), 0.1, 10))
        fine = tube_widths(reachable_tube(field, box, (0.0, 0.0), 0.05, 20))
        assert fine[-1].max() < coarse[-1].max()


class TestRoughEnclosure:
    def test_it_satisfies_the_picard_containment_it_claims(self):
        r"""White-box: :math:`X + [0,h] f(Y) \subseteq Y` is the whole proof.

        Checked directly rather than trusted, because everything downstream
        inherits it -- if this containment does not hold, the tube is bounding
        something not yet known to exist.
        """
        field = _harmonic(1.0, 2.0)
        box = [(0.9, 1.1), (-0.1, 0.1), (0.9, 1.1), (-0.1, 0.1)]
        dt = 0.05
        enclosure = rough_enclosure(field, box, (0.0, 0.0), dt)
        rates = enclose_field(field, enclosure, (0.0, 0.0))
        for (lo_y, hi_y), (lo_x, hi_x), (lo_f, hi_f) in zip(
            enclosure, box, rates, strict=True
        ):
            assert lo_y <= lo_x + min(0.0, dt * lo_f)
            assert hi_x + max(0.0, dt * hi_f) <= hi_y

    def test_it_refuses_a_step_it_cannot_verify(self):
        """Refusal is the correct outcome, not a failure to work around."""
        with pytest.raises(ValueError, match="no rough enclosure verified"):
            rough_enclosure(_harmonic(1.0), [(0.9, 1.1), (-0.1, 0.1)], (0.0, 0.0), 500.0)

    def test_a_non_positive_step_is_refused(self):
        with pytest.raises(ValueError, match="step must be positive"):
            rough_enclosure(_harmonic(1.0), [(0.9, 1.1), (-0.1, 0.1)], (0.0, 0.0), 0.0)


class TestJacobian:
    def test_the_jacobian_of_a_linear_field_is_its_matrix(self):
        """Constant and exactly known, so the enclosure should be a point."""
        enclosure = jacobian_enclosure(
            _harmonic(1.0, 2.0, 3.0), [(0.0, 1.0)] * 6, (0.0, 0.0)
        )
        expected = np.zeros((6, 6))
        for block, omega in enumerate((1.0, 2.0, 3.0)):
            expected[2 * block, 2 * block + 1] = 1.0
            expected[2 * block + 1, 2 * block] = -(omega**2)
        for i in range(6):
            for j in range(6):
                lower, upper = enclosure[i][j]
                assert lower <= expected[i, j] <= upper
                assert upper - lower < 1e-9

    def test_it_sizes_itself_to_the_state(self):
        """No hardcoded dimension: the package serves whatever field it is given."""
        for n, field in ((2, _harmonic(1.0)), (4, _harmonic(1.0, 2.0)), (1, _damped(0.5))):
            enclosure = jacobian_enclosure(field, [(0.0, 1.0)] * n, (0.0, 0.0))
            assert len(enclosure) == n
            assert all(len(row) == n for row in enclosure)

    def test_an_empty_box_is_refused(self):
        with pytest.raises(ValueError, match="no components"):
            jacobian_enclosure(_damped(0.5), [], (0.0, 0.0))


class TestMeanValueStep:
    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_it_contains_the_analytic_solution(self, dt):
        r"""Soundness at several step sizes.

        The Taylor remainder scales as :math:`h^2` while the linear part scales
        as :math:`h`, so a missing remainder shows up first at the largest step
        -- which is why this is parameterised rather than checked once.
        """
        field = _harmonic(1.0, 2.0)
        box = [(1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
        result = mean_value_step(field, box, (0.0, 0.0), dt)
        exact = [np.cos(dt), -np.sin(dt), np.cos(2 * dt), -2 * np.sin(2 * dt)]
        for value, (lower, upper) in zip(exact, result, strict=True):
            assert lower <= value <= upper

    def test_it_is_tighter_than_the_naive_step(self):
        """Keeping the correlation must pay for the Jacobian."""
        field = _harmonic(1.0, 2.0)
        box = [(0.9, 1.1), (-0.1, 0.1), (0.9, 1.1), (-0.1, 0.1)]
        dt = 0.05
        naive = reachable_step(field, box, (0.0, 0.0), dt)
        refined = mean_value_step(field, box, (0.0, 0.0), dt)
        assert sum(hi - lo for lo, hi in refined) < sum(hi - lo for lo, hi in naive)

    def test_it_still_grows(self):
        """Honest about what it does not fix.

        The mean-value form slows the wrapping; it does not remove it. Boxes are
        still axis-aligned and a rotating set is still re-covered each step.
        """
        field = _harmonic(1.0, 2.0)
        box = [(0.9, 1.1), (-0.1, 0.1), (0.9, 1.1), (-0.1, 0.1)]
        first = mean_value_step(field, box, (0.0, 0.0), 0.05)
        second = mean_value_step(field, first, (0.0, 0.0), 0.05)
        assert sum(hi - lo for lo, hi in second) > sum(hi - lo for lo, hi in first)
