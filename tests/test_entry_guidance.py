"""Entry guidance: the generic law, with its tuning stated by each test."""

import numpy as np
import pytest
import scipy.optimize

from aether.guidance.entry import _MU as MU
from aether.guidance.entry import _R_EARTH as R_EARTH
from aether.guidance.entry import (
    DragTracker,
    EntryBody,
    EntryState,
    atmospheric_density,
    bank_reversal_needed,
    crossrange_deadband,
    crossrange_deadband_from_range,
    entry_dynamics,
    entry_ocp_dynamics,
    equilibrium_glide_drag,
    equilibrium_glide_profile,
    range_to_go,
    simulate_entry,
)

#: The speed schedule the deadband tests are written against (m/s).
SPEEDS = {"speed_high": 7000.0, "speed_low": 1000.0}

#: Test tuning. None of it is a default of the law; these are the values the
#: expectations below were measured with.
TRACKER = DragTracker(
    gain_proportional=0.02, gain_derivative=0.5, gain_integral=0.004, reference_cosine=0.5
)


def _fly(body, initial, reference, **overrides):
    kwargs = {
        "tracker": TRACKER,
        "terminal_speed": 1000.0,
        "deadband_high": np.deg2rad(12.0),
        "deadband_low": np.deg2rad(2.0),
        "deadband_speeds": (SPEEDS["speed_high"], SPEEDS["speed_low"]),
    }
    kwargs.update(overrides)
    return simulate_entry(body, initial, reference, **kwargs)


class TestEntryGuidance:
    """Entry guidance: drag tracking and bank-angle modulation."""

    def _body(self):
        return EntryBody(ballistic_coefficient=200.0, lift_to_drag=2.0)

    def _entry(self):
        return EntryState(
            radius=R_EARTH + 80e3,
            longitude=0.0,
            latitude=0.0,
            speed=7000.0,
            flight_path_angle=np.deg2rad(-1.0),
            heading=np.deg2rad(90.0),
        )

    def test_range_integral_is_exact_for_constant_drag(self):
        """The range-energy relation dR/de = -1/D is exact, so at constant
        drag the integral must reproduce the closed form to quadrature
        precision. This pins the quantity the whole architecture rests on:
        commanding a drag profile is commanding a range."""
        entry = self._entry()
        final = 0.5 * 1000.0**2 - MU / (R_EARTH + 30e3)
        for drag in (5.0, 20.0, 80.0):
            predicted = range_to_go(lambda _e, d=drag: d, entry.specific_energy, final)
            assert predicted == pytest.approx((entry.specific_energy - final) / drag, rel=1e-12)

    def test_range_falls_as_commanded_drag_rises(self):
        """Higher drag sheds the same energy over less ground. This is the
        lever the tracker actually pulls."""
        entry = self._entry()
        final = 0.5 * 1000.0**2 - MU / (R_EARTH + 30e3)
        ranges = [
            range_to_go(lambda _e, d=drag: d, entry.specific_energy, final)
            for drag in (5.0, 20.0, 80.0)
        ]
        assert ranges[0] > ranges[1] > ranges[2]

    def test_range_rejects_a_profile_that_would_diverge(self):
        entry = self._entry()
        final = 0.5 * 1000.0**2 - MU / (R_EARTH + 30e3)
        with pytest.raises(ValueError, match="strictly positive"):
            range_to_go(lambda _e: 0.0, entry.specific_energy, final)
        with pytest.raises(ValueError, match="energy must exceed"):
            range_to_go(lambda _e: 20.0, final, entry.specific_energy)

    def test_deadband_is_wide_when_fast_and_tight_when_slow(self):
        """Scheduling direction is the whole point: a constant deadband
        either reverses constantly at entry or stops correcting at
        handover."""
        high, low = np.deg2rad(12.0), np.deg2rad(2.0)
        assert crossrange_deadband(7000.0, high, low, **SPEEDS) == pytest.approx(high)
        assert crossrange_deadband(1000.0, high, low, **SPEEDS) == pytest.approx(low)
        assert low < crossrange_deadband(4000.0, high, low, **SPEEDS) < high
        # Clamped outside the scheduled range rather than extrapolated.
        assert crossrange_deadband(9000.0, high, low, **SPEEDS) == pytest.approx(high)
        assert crossrange_deadband(300.0, high, low, **SPEEDS) == pytest.approx(low)

    def test_deadband_refuses_an_inverted_schedule(self):
        with pytest.raises(ValueError, match="0 < low < high"):
            crossrange_deadband(5000.0, np.deg2rad(2.0), np.deg2rad(12.0), **SPEEDS)

    def test_reversal_triggers_only_outside_the_deadband(self):
        high, low = np.deg2rad(12.0), np.deg2rad(2.0)
        assert not bank_reversal_needed(np.deg2rad(5.0), 7000.0, high, low, **SPEEDS)
        assert bank_reversal_needed(np.deg2rad(5.0), 1000.0, high, low, **SPEEDS)
        assert bank_reversal_needed(np.deg2rad(20.0), 7000.0, high, low, **SPEEDS)

    def test_tracker_reduces_vertical_lift_when_flying_too_much_drag(self):
        """The sign that reads backwards until you notice why. Above the
        reference drag the body is too deep, so the command must raise
        it — which means *more* vertical lift, hence a smaller bank."""
        tracker = DragTracker(
            gain_proportional=0.02, gain_derivative=0.0, gain_integral=0.0, reference_cosine=0.5
        )
        on_profile = tracker.command(20.0, 20.0)
        too_much_drag = tracker.command(30.0, 20.0)
        too_little_drag = tracker.command(10.0, 20.0)
        assert too_much_drag < on_profile < too_little_drag

    def test_tracker_saturates_on_bank_angle_not_on_cosine(self):
        """A saturated command must still be a physically meaningful
        attitude. Clipping the cosine instead would let the loop ask for a
        bank with no vertical lift at all."""
        tracker = DragTracker(
            gain_proportional=10.0, gain_derivative=0.0, gain_integral=0.0, reference_cosine=0.5
        )
        limit = np.deg2rad(60.0)
        # Far below the reference drag: bank all the way over to the limit,
        # spilling vertical lift so the body descends into denser air.
        assert tracker.command(0.0, 100.0, max_bank=limit) == pytest.approx(limit)
        # Far above it: wings level, all lift vertical, climb out. The
        # command saturates at zero bank rather than at a clipped cosine.
        assert tracker.command(1000.0, 0.0, max_bank=limit) == pytest.approx(0.0)
        for drag in (0.0, 5.0, 50.0, 1000.0):
            assert 0.0 <= tracker.command(drag, 20.0, max_bank=limit) <= limit

    def _flyable(self, body, bank_deg=45.0):
        """An equilibrium-glide reference the body can actually hold."""
        return equilibrium_glide_profile(body, R_EARTH + 50e3, np.deg2rad(bank_deg))

    def test_equilibrium_drag_is_small_at_entry_and_grows_as_speed_falls(self):
        """The constraint that decides which reference profiles exist at
        all. Near orbital speed centrifugal relief cancels most of the
        weight, leaving little lift to spare for drag; by handover the full
        weight must be carried. A constant-drag reference ignores this and
        is simply unflyable over the first half of the glide."""
        body = self._body()
        radius = R_EARTH + 50e3
        fast = equilibrium_glide_drag(body, 7000.0, radius, np.deg2rad(60.0))
        slow = equilibrium_glide_drag(body, 1000.0, radius, np.deg2rad(60.0))
        assert fast == pytest.approx(2.0, abs=0.3)
        assert slow == pytest.approx(9.5, abs=0.5)
        assert slow > 4.0 * fast
        with pytest.raises(ValueError, match="no usable vertical lift"):
            equilibrium_glide_drag(body, 5000.0, radius, np.pi / 2.0)

    def test_bank_reversals_bound_the_crossrange(self):
        """The result the lateral logic exists to produce. Holding a single
        bank sign accumulates over a thousand kilometres of crossrange;
        reversing on the scheduled deadband cuts it by more than an order
        of magnitude.

        The target sits at the range the longitudinal profile actually
        delivers. That is load-bearing, not tidiness: the lateral law
        steers on bearing, so a profile that overflies inverts the bearing
        part-way through and the deadband degenerates."""
        body, entry = self._body(), self._entry()
        reference = self._flyable(body)
        drifting = _fly(body, entry, reference, target=None)
        matched = (float(drifting.downrange / R_EARTH), 0.0)
        corrected = _fly(body, entry, reference, target=matched)
        assert drifting.reversals == 0
        assert abs(drifting.crossrange) > 1.0e6
        assert corrected.reversals > 0
        assert abs(corrected.crossrange) < 0.1 * abs(drifting.crossrange)

    def test_terminal_range_ends_the_glide_relative_to_the_target(self):
        """The structural fix. A speed gate ends the flight wherever the
        body happens to have got to, because it is defined on the
        body's own state; handing over at a fixed range-to-go makes the
        arrival point the handover point by construction."""
        body, entry = self._body(), self._entry()
        reference = self._flyable(body)
        drifting = _fly(body, entry, reference, target=None)
        arc = float(drifting.downrange / R_EARTH)
        handed = _fly(
            body,
            entry,
            reference,
            target=(arc, 0.0),
            range_gain=20.0,
            terminal_range=150e3,
        )
        remaining = R_EARTH * np.arccos(
            np.clip(
                np.cos(handed.states[2, -1]) * np.cos(handed.states[1, -1] - arc),
                -1.0,
                1.0,
            )
        )
        assert remaining <= 160e3
        # And it stops earlier than the speed gate would.
        assert handed.terminal_speed > 1000.0

    def test_an_unreachable_terminal_range_falls_back_to_the_speed_gate(self):
        """A profile that under-ranges announces itself this way rather
        than flying forever."""
        body, entry = self._body(), self._entry()
        reference = self._flyable(body)
        drifting = _fly(body, entry, reference, target=None)
        arc = float(drifting.downrange / R_EARTH)
        handed = _fly(
            body,
            entry,
            reference,
            target=(arc, 0.0),
            range_gain=20.0,
            terminal_range=1e3,
        )
        assert handed.terminal_speed == pytest.approx(1000.0, abs=25.0)

    def test_terminal_range_requires_a_target_and_a_positive_value(self):
        body, entry = self._body(), self._entry()
        reference = self._flyable(body)
        with pytest.raises(ValueError, match="terminal range needs a target"):
            _fly(body, entry, reference, terminal_range=100e3)
        with pytest.raises(ValueError, match="terminal_range must be finite"):
            _fly(body, entry, reference, target=(1.0, 0.0), terminal_range=-1.0)

    def test_a_range_scheduled_deadband_bounds_the_miss_not_the_angle(self):
        """A deadband on heading error is the wrong invariant once the
        glide hands over on range: the speed schedule spans 7000 to 1000
        m/s, the flight ends near 1700, and the deadband is still 3.2
        degrees when it does. What matters is the lateral distance that
        error becomes, and that shrinks with range."""
        far = crossrange_deadband_from_range(600e3, 2e3, np.deg2rad(12.0), np.deg2rad(0.1))
        near = crossrange_deadband_from_range(50e3, 2e3, np.deg2rad(12.0), np.deg2rad(0.1))
        # Tight far out and permissive close in -- the opposite of the
        # speed schedule, and correct: a given heading error becomes a
        # larger miss the further away it is held.
        assert far < near
        # Very close in it saturates wide, keeping the reversal count down.
        assert crossrange_deadband_from_range(
            1e3, 2e3, np.deg2rad(12.0), np.deg2rad(0.1)
        ) == pytest.approx(np.deg2rad(12.0))
        # The bound is the miss it permits, not the angle.
        assert 50e3 * np.tan(near) == pytest.approx(2e3, rel=1e-9)
        assert 600e3 * np.tan(far) == pytest.approx(2e3, rel=1e-9)

    def test_range_scheduled_deadband_validates_its_inputs(self):
        with pytest.raises(ValueError, match="range_to_go"):
            crossrange_deadband_from_range(0.0, 2e3, 0.2, 0.01)
        with pytest.raises(ValueError, match="allowed_crossrange"):
            crossrange_deadband_from_range(50e3, 0.0, 0.2, 0.01)
        with pytest.raises(ValueError, match="0 < low < high"):
            crossrange_deadband_from_range(50e3, 2e3, 0.01, 0.2)

    def test_range_scheduling_reduces_terminal_crossrange(self):
        """The measured effect, on a single perturbed case rather than the
        full Monte Carlo so the suite stays fast."""
        body, entry = self._body(), self._entry()
        reference = self._flyable(body)
        drifting = _fly(body, entry, reference, target=None)
        arc = float(drifting.downrange / R_EARTH)
        perturbed = EntryState(
            radius=entry.radius + 2000.0,
            longitude=entry.longitude,
            latitude=entry.latitude + 2000.0 / R_EARTH,
            speed=entry.speed,
            flight_path_angle=entry.flight_path_angle,
            heading=entry.heading,
        )
        common = {
            "target": (arc, 0.0),
            "range_gain": 20.0,
            "terminal_range": 150e3,
        }
        speed_scheduled = _fly(body, perturbed, reference, **common)
        range_scheduled = _fly(
            body,
            perturbed,
            reference,
            crossrange_tolerance=2e3,
            deadband_low=np.deg2rad(0.75),
            **common,
        )
        assert abs(range_scheduled.crossrange) < abs(speed_scheduled.crossrange)
        assert range_scheduled.reversals > speed_scheduled.reversals

    def test_roll_rate_limit_prices_the_bank_reversals(self):
        """Reversals are free only with an unlimited roll rate. A reversal
        sweeps the bank through zero, where vertical lift is maximum, so
        the body lofts each time; a slower roll holds that excursion
        longer."""
        body, entry = self._body(), self._entry()
        reference = self._flyable(body)
        drifting = _fly(body, entry, reference, target=None)
        arc = float(drifting.downrange / R_EARTH)
        common = {
            "target": (arc, 0.0),
            "range_gain": 20.0,
            "terminal_range": 150e3,
            "crossrange_tolerance": 2e3,
            "deadband_low": np.deg2rad(0.25),
        }
        fast = _fly(body, entry, reference, roll_rate_limit=np.deg2rad(30.0), **common)
        slow = _fly(body, entry, reference, roll_rate_limit=np.deg2rad(3.0), **common)
        # A rate too slow to complete a reversal before the next is demanded
        # leaves the body permanently slewing, and the reversal count
        # *falls* rather than rises — which is the signature of the failure.
        assert slow.reversals < fast.reversals
        assert np.max(np.abs(np.diff(slow.bank))) <= np.deg2rad(3.0) + 1e-9

    def test_roll_rate_limit_is_validated(self):
        body, entry = self._body(), self._entry()
        reference = self._flyable(body)
        with pytest.raises(ValueError, match="roll_rate_limit"):
            _fly(body, entry, reference, roll_rate_limit=0.0)

    def test_overflying_the_target_degrades_the_lateral_channel(self):
        """The coupling between the two channels, stated as a measurement
        rather than as a caveat."""
        body, entry = self._body(), self._entry()
        reference = self._flyable(body)
        drifting = _fly(body, entry, reference, target=None)
        arc = float(drifting.downrange / R_EARTH)
        matched = _fly(body, entry, reference, target=(arc, 0.0))
        short = _fly(body, entry, reference, target=(0.95 * arc, 0.0))
        assert abs(short.crossrange) > 3.0 * abs(matched.crossrange)

    def test_entry_terminates_on_the_speed_gate_and_stays_airborne(self):
        body, entry = self._body(), self._entry()
        result = _fly(body, entry, lambda _e: 20.0, target=(np.deg2rad(60.0), 0.0))
        assert result.terminal_speed == pytest.approx(1000.0, abs=25.0)
        assert np.all(result.altitudes > 0.0)
        assert result.times[-1] < 3000.0
        assert result.downrange > 1.0e6

    def test_higher_commanded_drag_shortens_the_flown_range(self):
        """The prediction of `range_to_go`, checked against a flown
        trajectory rather than against itself."""
        body, entry = self._body(), self._entry()
        shallow = _fly(body, entry, self._flyable(body, 30.0), target=None)
        steep = _fly(body, entry, self._flyable(body, 70.0), target=None)
        assert shallow.downrange > steep.downrange

    def test_lift_to_drag_sets_crossrange_capability(self):
        """Crossrange is what lift buys, so a higher-L/D body flown with
        the bank sign held must drift further off its initial great
        circle."""
        entry = self._entry()
        reference = lambda _e: 20.0  # noqa: E731
        low = _fly(
            EntryBody(ballistic_coefficient=200.0, lift_to_drag=1.0),
            entry,
            reference,
            target=None,
        )
        high = _fly(
            EntryBody(ballistic_coefficient=200.0, lift_to_drag=2.5),
            entry,
            reference,
            target=None,
        )
        assert abs(high.crossrange) > abs(low.crossrange)

    def test_density_is_clamped_at_the_surface(self):
        """Below the surface the model has no meaning, and extrapolating it
        keeps a diverging trajectory numerically alive long past the point
        where it stopped describing anything."""
        assert float(atmospheric_density(0.0)) == pytest.approx(1.225)
        assert float(atmospheric_density(-5000.0)) == pytest.approx(1.225)
        assert float(atmospheric_density(8500.0)) == pytest.approx(1.225 / np.e)

    def test_body_rejects_a_bank_limit_with_no_vertical_authority(self):
        with pytest.raises(ValueError, match="max_bank"):
            EntryBody(ballistic_coefficient=200.0, lift_to_drag=2.0, max_bank=np.pi / 2.0)
        with pytest.raises(ValueError, match="lift_to_drag"):
            EntryBody(ballistic_coefficient=200.0, lift_to_drag=0.0)
        with pytest.raises(ValueError, match="ballistic_coefficient"):
            EntryBody(ballistic_coefficient=-1.0, lift_to_drag=2.0)

    def test_refuses_an_entry_that_is_already_over(self):
        body = self._body()
        slow = EntryState(
            radius=R_EARTH + 80e3,
            longitude=0.0,
            latitude=0.0,
            speed=800.0,
            flight_path_angle=0.0,
            heading=0.0,
        )
        with pytest.raises(ValueError, match="nothing to fly"):
            _fly(body, slow, self._flyable(body))


class TestEntryDynamics:
    """The six-state entry EOM, extracted from seven duplicated copies.

    These are invariants the equations must satisfy, not restatements of the
    expressions -- a test written by copying the implementation would have
    passed against every one of the seven copies including any that had drifted.
    """

    @staticmethod
    def _body():
        return EntryBody(ballistic_coefficient=4000.0, lift_to_drag=2.5)

    @staticmethod
    def _state():
        return np.array([
            R_EARTH + 45_000.0,     # radius
            np.deg2rad(12.0),       # longitude
            np.deg2rad(31.0),       # latitude
            5200.0,                 # speed
            np.deg2rad(-1.4),       # flight path angle
            np.deg2rad(70.0),       # heading
        ])

    def test_the_kinematic_block_has_the_speed_as_its_magnitude(self):
        """The first three rates are the velocity in spherical components.

        Their magnitude must come back as V exactly. This catches a dropped
        cos(gamma), a missing cos(lat) in the longitude rate, or a swapped
        sin/cos between the latitude and longitude channels -- none of which
        changes the shape of a trajectory enough to be obvious in a plot.
        """
        state = self._state()
        rates = entry_dynamics(state, 0.4, self._body())
        radius, _lon, lat, speed = state[0], state[1], state[2], state[3]
        ground = np.hypot(radius * np.cos(lat) * rates[1], radius * rates[2])
        assert np.hypot(rates[0], ground) == pytest.approx(speed, rel=1e-12)

    @pytest.mark.parametrize("bank", [0.0, 0.5, -0.9, 1.3])
    def test_lift_does_no_work_whatever_the_bank(self, bank):
        r"""Specific energy decays at exactly :math:`-VD`, independent of bank.

        :math:`\dot e = V\dot V + \mu\dot r/r^2 = V(-D - g\sin\gamma) + gV\sin\gamma
        = -VD`. Lift is perpendicular to the velocity, so banking redistributes
        it without changing the energy budget. A sign error in the gravity term
        of either the speed or the radius rate breaks this and nothing else
        obviously.
        """
        state = self._state()
        body = self._body()
        rates = entry_dynamics(state, bank, body)
        radius, speed = state[0], state[3]
        energy_rate = speed * rates[3] + MU * rates[0] / radius**2
        drag = body.drag_acceleration(radius - R_EARTH, speed)
        assert energy_rate == pytest.approx(-speed * drag, rel=1e-10)

    def test_it_reproduces_the_equilibrium_glide_condition(self):
        r"""At the equilibrium drag, a level glide has :math:`\dot\gamma = 0`.

        Cross-checks the EOM against :func:`equilibrium_glide_drag`, which was
        derived separately from the vertical force balance. Setting
        :math:`\gamma = 0` and putting the body at the altitude where its
        drag equals the equilibrium value must leave the flight path angle
        stationary; if the two disagree, one of them is wrong.
        """
        body = self._body()
        speed, bank = 5200.0, np.deg2rad(35.0)

        def pitch_rate(altitude):
            state = np.array([R_EARTH + altitude, 0.0, 0.0, speed, 0.0, 0.0])
            return entry_dynamics(state, bank, body, body_radius=R_EARTH)[4]

        # The altitude the EOM says this body trims at: too low and the
        # excess lift pitches it up, too high and it falls out of the glide.
        altitude = scipy.optimize.brentq(pitch_rate, 20_000.0, 90_000.0)
        radius = R_EARTH + altitude
        assert body.drag_acceleration(altitude, speed) == pytest.approx(
            equilibrium_glide_drag(body, speed, radius, bank), rel=1e-9
        )

    def test_bank_turns_the_heading_and_its_sign_decides_which_way(self):
        state = self._state()
        body = self._body()
        right = entry_dynamics(state, 0.6, body)
        left = entry_dynamics(state, -0.6, body)
        # The lift-driven part of the heading rate flips; the Coriolis-free
        # curvature term (which does not depend on bank) does not.
        assert right[5] > left[5]
        assert right[4] == pytest.approx(left[4], rel=1e-12)

    def test_zero_bank_puts_all_the_lift_into_the_vertical(self):
        r"""At :math:`\sigma = 0` the heading rate keeps only its curvature term."""
        state = self._state()
        body = self._body()
        rates = entry_dynamics(state, 0.0, body)
        radius, lat, speed, gamma, heading = (
            state[0], state[2], state[3], state[4], state[5]
        )
        curvature = (
            speed * np.cos(gamma) * np.sin(heading) * np.tan(lat) / radius
        )
        assert rates[5] == pytest.approx(curvature, rel=1e-12)

    def test_a_scalar_and_a_one_element_control_agree(self):
        """Both call styles exist: solve_ivp passes a float, an OCP an array."""
        state, body = self._state(), self._body()
        assert np.allclose(
            entry_dynamics(state, 0.42, body),
            entry_dynamics(state, np.array([0.42]), body),
            rtol=0.0, atol=0.0,
        )

    def test_the_ocp_wrapper_is_the_same_function(self):
        state, body = self._state(), self._body()
        dynamics = entry_ocp_dynamics(body)
        assert np.allclose(
            dynamics(state, np.array([0.3])),
            entry_dynamics(state, 0.3, body),
            rtol=0.0, atol=0.0,
        )

    def test_a_higher_lift_to_drag_pulls_up_harder(self):
        state = self._state()
        blunt = entry_dynamics(state, 0.0, EntryBody(4000.0, 0.5))
        sharp = entry_dynamics(state, 0.0, EntryBody(4000.0, 3.0))
        assert sharp[4] > blunt[4]
        # Same ballistic coefficient, so the drag deceleration is unchanged.
        assert sharp[3] == pytest.approx(blunt[3], rel=1e-12)
