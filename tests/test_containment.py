"""Containment radii of a bivariate normal, against closed forms and Siouris."""

import numpy as np
import pytest

from aether.batch.containment import (
    SIOURIS_TABLE_5_2,
    SIOURIS_TABLE_5_2_PROBABILITIES,
    SIOURIS_TABLE_5_2_RATIOS,
    containment_probability,
    containment_radius,
    containment_ratio,
)

#: Circular-case closed forms.
R50_OVER_SIGMA = float(np.sqrt(2.0 * np.log(2.0)))
R95_OVER_SIGMA = float(np.sqrt(-2.0 * np.log(0.05)))


class TestContainment:
    """The exact elliptical integral, and why the usual conversion factor is wrong."""

    def test_circular_case_is_the_rayleigh_closed_form(self):
        assert containment_radius(0.5, 100.0) == pytest.approx(100.0 * R50_OVER_SIGMA, rel=1e-12)
        assert containment_radius(0.95, 100.0, 100.0) == pytest.approx(
            100.0 * R95_OVER_SIGMA, rel=1e-9
        )

    def test_containment_probability_is_consistent_with_its_own_radius(self):
        """Round-trip: the radius containing p must contain p."""
        for major, minor in ((100.0, 100.0), (300.0, 100.0), (500.0, 50.0)):
            for p in (0.3, 0.5, 0.95, 0.99):
                r = containment_radius(p, major, minor)
                assert containment_probability(r, major, minor) == pytest.approx(p, abs=1e-8)
    def test_the_ratio_rises_as_the_ellipse_elongates(self):
        """The measured behaviour, and the reason a computed ratio beats an
        assumed one. Scaling a 50% radius by the circular 2.079 *under-states* the
        95% radius for any real dispersion."""
        ratios = [containment_ratio(100.0, 100.0 * f) for f in (1.0, 0.5, 0.3, 0.1, 0.02)]
        assert ratios == sorted(ratios)
        assert ratios[0] == pytest.approx(2.0789, abs=1e-3)
        # The one-dimensional limit is the normal-distribution equivalent,
        # 1.96 / 0.6745.
        assert ratios[-1] == pytest.approx(1.959964 / 0.674490, rel=2e-3)
    def test_a_degenerate_axis_reduces_to_the_normal_distribution(self):
        """With one sigma zero the disc becomes an interval, and the
        containment radius must be the ordinary normal quantile."""
        assert containment_radius(0.95, 100.0, 0.0) == pytest.approx(195.9964, rel=1e-4)
        assert containment_radius(0.5, 100.0, 0.0) == pytest.approx(67.449, rel=1e-4)

    def test_inputs_are_validated(self):
        with pytest.raises(ValueError, match="at least one sigma"):
            containment_radius(0.5, 0.0, 0.0)
        with pytest.raises(ValueError, match="must be finite"):
            containment_radius(0.5, -1.0, 100.0)
        with pytest.raises(ValueError, match="probability must lie"):
            containment_radius(1.0, 100.0)


class TestAgainstSiourisTable52:
    """The containment integral against 126 published numbers.

    Before this, `containment_radius` was checked only against its own
    circular closed form and its own consistency with
    `containment_probability`. Siouris Table 5.2 spans the entire domain —
    21 aspect ratios from degenerate to circular, six probability levels —
    and is an entirely independent computation.
    """

    def test_reproduces_every_non_degenerate_entry(self):
        """All 120 two-dimensional entries, to one unit in the table's last
        printed place.

        The bound is 1e-4 rather than the 5e-5 of ideal rounding because
        three of these 120 entries are one unit high in the source — see
        `test_the_four_entries_the_source_rounds_up`. Every other entry is
        inside 5e-5, i.e. agreement to every digit published.
        """
        worst = 0.0
        for i, ratio in enumerate(SIOURIS_TABLE_5_2_RATIOS):
            if ratio == 0.0:
                continue  # degenerate row, handled separately
            for j, probability in enumerate(SIOURIS_TABLE_5_2_PROBABILITIES):
                ours = containment_radius(probability, 1.0, ratio)
                worst = max(worst, abs(ours - SIOURIS_TABLE_5_2[i][j]))
        assert worst < 1.0e-4, f"worst deviation from Table 5.2 is {worst:.2e}"

    def test_the_four_entries_the_source_rounds_up(self):
        """Four of the 126 entries sit just over half a unit in the last
        place from the exact value, and in every case the exact value
        rounds to a *different* last digit — so the discrepancy is the
        table's rounding, not ours.

        Pinned rather than absorbed into a loose tolerance: if our integral
        ever drifted, these four would move and the rest would not, which
        distinguishes a real regression from this known artefact.
        """
        from scipy.stats import norm

        known = {
            (0.00, 0.75): 1.1504,
            (0.05, 0.50): 0.6764,
            (0.75, 0.90): 1.9034,
            (1.00, 0.95): 2.4478,
        }
        for (ratio, probability), published in known.items():
            if ratio == 0.0:
                exact = float(norm.ppf(0.5 + 0.5 * probability))
            else:
                exact = containment_radius(probability, 1.0, ratio)
            deviation = published - exact
            assert 5.0e-5 < deviation < 6.0e-5
            # The tell: the exact value rounds to a different last digit.
            assert round(exact, 4) != published

        # And they are the *only* four. Everything else is inside ideal
        # rounding, so this is not a systematic bias in our integral.
        outliers = 0
        for i, ratio in enumerate(SIOURIS_TABLE_5_2_RATIOS):
            for j, probability in enumerate(SIOURIS_TABLE_5_2_PROBABILITIES):
                if ratio == 0.0:
                    exact = float(norm.ppf(0.5 + 0.5 * probability))
                else:
                    exact = containment_radius(probability, 1.0, ratio)
                if abs(SIOURIS_TABLE_5_2[i][j] - exact) > 5.0e-5:
                    outliers += 1
        assert outliers == 4

    def test_degenerate_row_is_the_one_dimensional_normal(self):
        """The sigma_S = 0 row is not a two-dimensional result at all: with
        no spread on one axis the miss distance is a folded normal on the
        other, so the entries must be normal quantiles. That the table and
        the closed form agree here is what makes the rest of it credible."""
        from scipy.stats import norm

        for j, probability in enumerate(SIOURIS_TABLE_5_2_PROBABILITIES):
            quantile = float(norm.ppf(0.5 + 0.5 * probability))
            assert quantile == pytest.approx(SIOURIS_TABLE_5_2[0][j], abs=1e-4)
        # The two entries everyone recognises.
        assert SIOURIS_TABLE_5_2[0][1] == pytest.approx(0.6745, abs=5e-5)
        assert SIOURIS_TABLE_5_2[0][4] == pytest.approx(1.9600, abs=5e-5)

    def test_circular_column_is_the_rayleigh_closed_form(self):
        """The other endpoint. The published 1.1774 and 2.4478 must be
        sqrt(2 ln 2) and sqrt(-2 ln 0.05) exactly."""
        circular = SIOURIS_TABLE_5_2[-1]
        assert circular[1] == pytest.approx(R50_OVER_SIGMA, abs=5e-5)
        # 2.4478 is one of the four entries the source rounds up.
        assert circular[4] == pytest.approx(R95_OVER_SIGMA, abs=1e-4)
        for j, probability in enumerate(SIOURIS_TABLE_5_2_PROBABILITIES):
            closed_form = float(np.sqrt(-2.0 * np.log(1.0 - probability)))
            assert closed_form == pytest.approx(circular[j], abs=1e-4)

    def test_table_is_monotone_in_both_directions(self):
        """Guards the transcription against a transposed or shuffled row.
        K must rise with probability across a row, and rise with aspect
        ratio down a column — a rounder distribution needs a larger radius
        to contain the same fraction."""
        for row in SIOURIS_TABLE_5_2:
            assert list(row) == sorted(row)
        for j in range(len(SIOURIS_TABLE_5_2_PROBABILITIES)):
            column = [row[j] for row in SIOURIS_TABLE_5_2]
            assert column == sorted(column)
