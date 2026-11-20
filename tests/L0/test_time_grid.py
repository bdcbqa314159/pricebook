"""`checked_ascending_times` (L0) — tranche 65, re-audit #7's F-7.2.

One piece of knowledge, one home. Re-audit #7 found this guard written **byte-identically in four
modules across three layers** — `market/curve.py` (T1), `market/survival_curve.py` (T38),
`risk/credit_valuation_adjustment.py` (T25), `risk/mc_exposure.py` (T26) — each with the identical
condition and the identical message. §6's rule-of-two had fired four times over on a shared *shape*,
and six prior duplication passes did not name it, because an identical two-line block in four files
reads as idiom rather than as duplication.

A **guard**, not a `TimeGrid` value object: the T62 precedent. What the four share is one piece of
knowledge, not a field group — none of them is near the §7 field limit, so no bundle is *compelled*,
and a value object would have changed `self.times` to `self.grid.times` at every site for nothing.

The extraction is pure: op order is preserved (the same `any`/`zip` over the same tuple, just one
call frame further out), so the proof is **bit-identity in-process** — 22 curve and survival-curve
values byte-for-byte unchanged across all four rewired consumers. That measurement is *not*
committed: §9 forbids promoting an in-process bit pattern to a cross-platform claim (T49), so what
lands is this file plus the four consumers' own oracles staying green at their own tolerances.

Provenance: tests only — see `src/pricebook/foundation/time_grid.py`.
"""

import pytest

from pricebook.foundation.time_grid import checked_ascending_times


class TestTheRefusals:
    def test_a_REPEATED_time_is_refused(self) -> None:
        """Strict, not non-decreasing, and the strictness is the content: two pillars at one instant
        make interpolation ambiguous and a pairwise difference zero — both silent failures, which is
        why this is a construction-time refusal rather than a runtime check."""
        with pytest.raises(ValueError, match="strictly ascending"):
            checked_ascending_times((0.0, 1.0, 1.0, 2.0))

    def test_a_DECREASING_time_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly ascending"):
            checked_ascending_times((0.0, 2.0, 1.0))

    def test_the_refusal_is_detected_at_ANY_position(self) -> None:
        """A guard that only looked at the first pair, or only at the last, would pass one of these.
        The premise behind the `zip` pairwise form, asserted rather than read off the code."""
        for bad in ((1.0, 0.0, 2.0, 3.0), (0.0, 1.0, 3.0, 2.0), (0.0, 2.0, 1.0, 3.0)):
            with pytest.raises(ValueError, match="strictly ascending"):
                checked_ascending_times(bad)

    def test_the_MESSAGE_is_ours(self) -> None:
        """The T44b reachability shape. Four other modules raise their own ascending refusals with
        their own wording, so a test that only caught *some* `ValueError` could not tell this guard
        from theirs — the message is what discriminates."""
        with pytest.raises(ValueError) as caught:
            checked_ascending_times((1.0, 1.0))
        assert str(caught.value) == "times must be strictly ascending"


class TestTheAcceptances:
    """The non-vacuity half: a guard that refused everything would pass every test above."""

    def test_a_STRICTLY_ASCENDING_grid_is_accepted(self) -> None:
        assert checked_ascending_times((0.0, 0.25, 1.0, 2.5, 30.0)) is None

    def test_a_SINGLE_point_is_accepted(self) -> None:
        """One pillar has no pair to compare, so `zip` is empty and the grid is trivially ascending
        — which is what a flat one-pillar curve needs."""
        assert checked_ascending_times((0.0,)) is None

    def test_an_EMPTY_grid_is_accepted_here_and_refused_by_its_OWNERS(self) -> None:
        """A deliberate boundary. Emptiness is not an *ordering* property, and the four consumers do
        not agree on it — a curve needs at least one pillar, an exposure profile needs its times to
        match its values. So each keeps its own emptiness rule and this guard checks ordering only,
        which is the §6 line: share the knowledge that is genuinely shared, not the knowledge that
        differs per consumer."""
        assert checked_ascending_times(()) is None

    def test_NEGATIVE_times_are_accepted_here_too(self) -> None:
        """Same reasoning: a time grid's *sign* is its owner's business (a curve's first pillar must
        be `t = 0`, which `curve.py` still asserts for itself). This guard is about order alone."""
        assert checked_ascending_times((-2.0, -1.0, 0.0, 1.0)) is None


class TestTheFourConsumersSTILLRefuse:
    """F-7.2's actual content: the extraction must not have weakened any of the four call sites.
    Stated as a test rather than trusted to the refactor, because the whole finding was that four
    identical guards went unnoticed — and an extraction that silently dropped one would go unnoticed
    the same way."""

    def test_the_DISCOUNT_CURVE_still_refuses_a_non_ascending_grid(self) -> None:
        from datetime import date

        from pricebook.foundation.day_count import DayCountConvention
        from pricebook.foundation.time_measure import TimeMeasure
        from pricebook.market.curve import DiscountCurve

        measure = TimeMeasure(date(2026, 1, 15), DayCountConvention.ACT_365_FIXED)
        with pytest.raises(ValueError, match="strictly ascending"):
            DiscountCurve(measure, (0.0, 2.0, 1.0), (0.02, 0.03, 0.04))

    def test_the_SURVIVAL_CURVE_still_refuses_a_non_ascending_grid(self) -> None:
        from datetime import date

        from pricebook.foundation.day_count import DayCountConvention
        from pricebook.foundation.time_measure import TimeMeasure
        from pricebook.market.survival_curve import SurvivalCurve

        measure = TimeMeasure(date(2026, 1, 15), DayCountConvention.ACT_365_FIXED)
        with pytest.raises(ValueError, match="strictly ascending"):
            SurvivalCurve(measure, (0.0, 5.0, 1.0), (0.0, 0.01, 0.02))

    def test_the_EXPOSURE_PROFILE_still_refuses_a_non_ascending_grid(self) -> None:
        from pricebook.risk.credit_valuation_adjustment import ExposureProfile

        with pytest.raises(ValueError, match="strictly ascending"):
            ExposureProfile(times=(0.5, 2.0, 1.0), epe=(1.0, 2.0, 3.0))

    def test_the_SIMULATED_EXPOSURE_still_refuses_a_non_ascending_grid(self) -> None:
        from pricebook.risk.mc_exposure import SimulatedExposure

        with pytest.raises(ValueError, match="strictly ascending"):
            SimulatedExposure(times=(0.5, 2.0, 1.0), values=((1.0, 2.0, 3.0),))
