"""Skip entry guidance: the predictor, the corrector, and a closed-loop lunar return."""

from __future__ import annotations

import numpy as np
import pytest

from aether.atmosphere.model import earth_atmosphere, tabulate
from aether.guidance.entry import _R_EARTH, EntryBody, EntryState
from aether.guidance.skip import (
    EARTH_ROTATION_RATE,
    SKIP_OUT_ERROR,
    SkipGuidance,
    SkipGuidanceConfig,
    SkipPhase,
    fly_skip_entry,
    predict_constant_bank,
)

FT = 0.3048
NMI = 1852.0

_ATMOSPHERE = tabulate(earth_atmosphere())


def _density(altitude: float) -> float:
    return float(_ATMOSPHERE.state(altitude).density)


#: A lunar-return capsule: L/D 0.30, beta 280 kg/m², flown through a
#: standard atmosphere and allowed to bank lift-down.
CAPSULE = EntryBody(280.0, 0.30, max_bank=np.deg2rad(165.0), density=_density)

#: Artemis I's entry interface (AAS 24-174, Table 1) and the splashdown point
#: (Table 4), 2 nmi from the target the guidance flew to.
INTERFACE = EntryState(
    radius=_R_EARTH + 400000.0 * FT,
    longitude=np.deg2rad(-120.08071),
    latitude=np.deg2rad(-25.82847),
    speed=36062.6568 * FT,
    flight_path_angle=np.deg2rad(-5.66367),
    heading=np.deg2rad(4.65389),
)
TARGET = (np.deg2rad(-118.10181), np.deg2rad(27.34852))

CONFIG = SkipGuidanceConfig(
    initial_bank=np.deg2rad(15.0),
    initial_roll_exit_altitude_rate=-700.0 * FT,
    terminal_speed=1000.0 * FT,
    range_tolerance=25.0 * NMI,
    max_iterations=5,
    estimator_drag_threshold=1.6 * FT,
    corridor_constant=5.0e-5,
    corridor_quadratic=1.2e-10,
    roll_rate=np.deg2rad(15.0),
    final_phase_drag=6.0 * FT,
    final_range_tolerance=1.0 * NMI,
)


def _predict(u: float, state: EntryState = INTERFACE, **kw):
    return predict_constant_bank(
        state.as_array(), 0.0, TARGET, CAPSULE, u, terminal_speed=1000.0 * FT, **kw
    )


class TestPredictor:
    def test_full_lift_up_from_a_lunar_return_skips_out(self):
        """Near escape speed, holding all the lift up climbs out for good. That
        is not a range to compare, and must not be reported as one."""
        assert _predict(1.0).range_error == SKIP_OUT_ERROR

    def test_range_rises_with_vertical_lift(self):
        """The monotone ordering the corrector brackets on."""
        errors = [_predict(u).range_error for u in (-0.9, -0.6, -0.3, 0.0)]
        assert errors == sorted(errors)

    def test_the_reserve_segment_changes_the_prediction(self):
        """Two-segment prediction: the lift flown after the second entry is part
        of the plan, and more of it flies further."""
        base = {"final_phase_drag": 6.0 * FT}
        low = _predict(-0.1, final_vertical_lift_fraction=0.2, **base)
        high = _predict(-0.1, final_vertical_lift_fraction=0.8, **base)
        assert high.range_error > low.range_error

    def test_the_target_is_aimed_at_where_it_will_be(self):
        """A later epoch rotates the site further east; for a body flying
        north, that changes the distance still to go."""
        now = predict_constant_bank(
            INTERFACE.as_array(), 0.0, TARGET, CAPSULE, -0.3, terminal_speed=1000.0 * FT
        )
        later = predict_constant_bank(
            INTERFACE.as_array(), 3600.0, TARGET, CAPSULE, -0.3, terminal_speed=1000.0 * FT
        )
        assert now.range_error != later.range_error
        assert pytest.approx(7.292115e-5, rel=1e-6) == EARTH_ROTATION_RATE


class TestCorrector:
    def test_an_unreachable_overshoot_saturates_lift_down(self):
        """If even the bank limit overshoots, command the bank limit."""
        near = (INTERFACE.longitude + np.deg2rad(5.0), INTERFACE.latitude)
        guidance = SkipGuidance(CAPSULE, near, CONFIG)
        guidance.phase = SkipPhase.PREDICTOR_CORRECTOR
        guidance._correct(INTERFACE.as_array(), 0.0)
        assert guidance.vertical_lift_fraction == pytest.approx(np.cos(CAPSULE.max_bank))


@pytest.fixture(scope="module")
def flight():
    truth = EntryBody(
        280.0, 0.30 * 0.95, max_bank=np.deg2rad(165.0), density=lambda h: 0.9 * _density(h)
    )
    guidance = SkipGuidance(CAPSULE, TARGET, CONFIG)
    return fly_skip_entry(truth, INTERFACE, guidance), guidance


class TestArtemisI:
    """The Artemis I case, flown through an atmosphere 10 % thinner and with an
    L/D 5 % lower than the guidance's model -- the differences the flight's
    own estimators reported (AAS 24-174)."""

    def test_lands_inside_the_orion_requirement(self, flight):
        """Orion is required to land within 5.4 nmi of its target."""
        result, _ = flight
        assert result.miss_distance < 5.4 * NMI

    def test_flies_a_skip(self, flight):
        """Out of the atmosphere and back: Artemis I peaked at 287.4 kft."""
        result, _ = flight
        assert 250e3 * FT < result.skip_apogee_altitude < 320e3 * FT
        names = [phase for _, phase in result.phases]
        assert SkipPhase.FINAL in names

    def test_phase_timing_tracks_the_flight(self, flight):
        """Final began 551 s after entry interface and Terminal 882 s on Artemis I."""
        result, _ = flight
        at = {phase: t for t, phase in result.phases}
        assert at[SkipPhase.FINAL] == pytest.approx(551.0, abs=60.0)
        assert at[SkipPhase.TERMINAL] == pytest.approx(882.0, abs=60.0)

    def test_the_estimators_find_the_atmosphere_and_the_lift(self, flight):
        _, guidance = flight
        assert guidance.density_factor == pytest.approx(0.90, abs=0.02)
        assert guidance.lift_to_drag_estimate == pytest.approx(0.285, abs=0.01)

    def test_reversals_bunch_as_the_corridor_closes(self, flight):
        """Artemis I reversed six times (AAS 24-174, Table 3), most of them in
        the last 150 s as the V² corridor narrows.

        The count is deliberately not pinned here. It is sensitive: this
        capsule's ballistic coefficient is 0.4 % from the example's and reverses
        a seventh time, which is the non-determinism the lateral-logic
        literature describes (NTRS 20160001182). The example test pins the
        count for the flight's own parameters.
        """
        result, _ = flight
        assert 5 <= result.reversals <= 8
        times = result.times[: len(result.bank)]
        flips = times[np.nonzero(np.diff(np.sign(result.bank)))[0] + 1]
        assert sum(t > result.times[-1] - 150.0 for t in flips) >= 3

    def test_a_commanded_reversal_can_be_superseded_mid_roll(self, flight):
        """The body is still rolling when the next command arrives, so the
        guidance asks for more reversals than the bank actually completes."""
        result, _ = flight
        assert result.commanded_reversals >= result.reversals

    def test_the_load_stays_humane(self, flight):
        result, _ = flight
        assert result.peak_deceleration_g < 6.0
