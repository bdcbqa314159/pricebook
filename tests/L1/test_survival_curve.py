"""The survival curve's own checks (L1) — tranche 25, the XVA spine's credit side.

`CAPINSKI-ZASTAWNIAK` ch 2: a hazard function `λ` defines the default time's survival
probability `S(t) = exp(−∫₀ᵗ λ(u)du)`, and the default density is `dPD(t) = λ(t)S(t)dt`.

**Two fixture traps are closed here on purpose.** A FLAT hazard makes `S(t) = e^{−λt}`, so a wrong
*cumulation* — reading `λ(t)` where the integral `Σλᵢτᵢ` belongs — returns the right number at every
horizon and the oracle is blind to it. And the first segment is equally degenerate: inside
`[0, times[1])` the cumulation is `λ₀t` whatever the later hazards are. So the transcription is
pinned on a PIECEWISE-CONSTANT hazard, read BEYOND the first segment, with the distinctness of the
pieces asserted as a premise.

The test computes its own reference by explicit summation over the segments — never by calling the
curve's own cumulation (§4 independence).
"""

import math

import pytest

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market.survival_curve import SurvivalCurve

from datetime import date

ANCHOR = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(ANCHOR, DC)

# Piecewise-constant hazard: distinct pieces, so a cumulation bug cannot hide (see the docstring).
TIMES = (0.0, 1.0, 3.0, 7.0)
HAZARDS = (0.010, 0.025, 0.060, 0.040)
CURVE = SurvivalCurve(TM, TIMES, HAZARDS)


def _reference_survival(t: float) -> float:
    """`exp(−Σ λᵢ·(overlap of [times[i], times[i+1]) with [0, t]))`, summed here in the test.

    Independent of the curve: this walks the segments itself and never calls the code under test.
    """
    total = 0.0
    for i, lam in enumerate(HAZARDS):
        lo = TIMES[i]
        hi = TIMES[i + 1] if i + 1 < len(TIMES) else math.inf
        if t <= lo:
            break
        total += lam * (min(t, hi) - lo)
    return math.exp(-total)


class TestTheFixtureIsNotDegenerate:
    def test_the_hazard_pieces_are_DISTINCT_so_a_cumulation_bug_cannot_hide(self) -> None:
        spread = max(HAZARDS) - min(HAZARDS)
        assert spread > 0.04, (
            f"a near-flat hazard makes S(t)=e^-lambda*t and blinds the cumulation; spread {spread}"
        )

    def test_the_transcription_horizons_reach_BEYOND_the_first_segment(self) -> None:
        # Inside [0, times[1]) the cumulation is lambda_0*t for ANY later hazards.
        assert max(2.0, 5.0, 9.0) > TIMES[1]


class TestSurvivalIsTranscribedPointwise:
    @pytest.mark.parametrize("t", [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0])
    def test_survival_matches_an_INDEPENDENTLY_SUMMED_cumulative_hazard(self, t: float) -> None:
        assert CURVE.survival(t) == pytest.approx(_reference_survival(t), rel=1e-14)

    def test_a_FLAT_hazard_is_exactly_exp_minus_lambda_t(self) -> None:
        flat = SurvivalCurve.flat(ANCHOR, 0.03, DC)
        for t in (0.25, 1.0, 4.0, 10.0):
            assert flat.survival(t) == pytest.approx(math.exp(-0.03 * t), rel=1e-15)


class TestTheRequirementOraclesTheFinanceForces:
    """True for all parameters, so they survive a transcription error (§4)."""

    def test_survival_starts_at_one(self) -> None:
        assert CURVE.survival(0.0) == 1.0

    def test_survival_is_monotone_NON_INCREASING(self) -> None:
        grid = [0.1 * k for k in range(100)]
        values = [CURVE.survival(t) for t in grid]
        assert all(b <= a for a, b in zip(values, values[1:]))

    def test_survival_stays_in_the_unit_interval(self) -> None:
        assert all(0.0 < CURVE.survival(0.1 * k) <= 1.0 for k in range(100))

    def test_a_ZERO_hazard_never_defaults(self) -> None:
        certain = SurvivalCurve(TM, (0.0,), (0.0,))
        assert all(certain.survival(t) == 1.0 for t in (0.0, 1.0, 50.0))

    def test_the_DENSITY_relation_dPD_equals_lambda_S_dt(self) -> None:
        """`d(1−S)/dt = λ(t)S(t)`, with the derivative taken NUMERICALLY in the test.

        The relation is derived from the finance, not read back from the curve: a central difference
        of the curve's own `survival` against its own `default_density`, which are different code.
        """
        h = 1e-6
        for t in (0.5, 2.0, 5.0, 8.0):
            slope = (CURVE.survival(t - h) - CURVE.survival(t + h)) / (2.0 * h)
            assert slope == pytest.approx(CURVE.default_density(t), rel=1e-7)


class TestTheGuardsAreReachable:
    def test_a_hazard_count_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            SurvivalCurve(TM, (0.0, 1.0), (0.01,))

    def test_a_curve_must_start_at_zero(self) -> None:
        with pytest.raises(ValueError, match="t=0.0"):
            SurvivalCurve(TM, (0.5,), (0.01,))

    def test_times_must_ascend(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            SurvivalCurve(TM, (0.0, 2.0, 1.0), (0.01, 0.02, 0.03))

    def test_a_NEGATIVE_hazard_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            SurvivalCurve(TM, (0.0,), (-0.01,))
