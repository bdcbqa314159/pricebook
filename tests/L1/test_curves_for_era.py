"""`CurveSet.curves_for_era` (L1) — re-audit #6, F-6.6.

The era selection used to be written out twice: in the swap pricer (L4, tranche 55) and in the
collateralised capstone (L5, tranche 58). A shared **shape**, not a shared word, so §6's
rule-of-two fired and the lookup moved here — where `projections` lives, and below both consumers.

**Why it matters beyond tidiness, which is the whole of the ruling:** the arc's single-curve design
invariant — that single-curve and multi-curve are *co-equal selectable modes*, chosen by the shape
of the market rather than by a flag — **is realised through exactly this lookup**. Two copies meant
the invariant was re-derived per consumer and the swap pricer's "the one place the mode is
selected" docstring was simply false. One home makes the claim true and gives the invariant
somewhere to be guarded.

The refactor is pure: dict membership and a branch over curve handles, **no floating-point work
reordered**, so the purity proof is bit-identity in-process (§4) — 18 swap PVs across both eras and
6 capstone exposure grids with their CVAs, byte-for-byte unchanged. That measurement is *not*
committed: §9's "transcendental results differ 1–2 ULP across runtimes" forbids promoting an
in-process bit pattern to a cross-platform assertion (T49), so what lands is the existing oracle
set at its own tolerances, plus the two branches asserted directly below.

Provenance: tests only — see `src/pricebook/market/curve_set.py`.
"""

from datetime import date

import pytest

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.money import Currency
from pricebook.foundation.tenor import Tenor
from pricebook.market.curve import DiscountCurve
from pricebook.market.curve_set import CurveKey, CurveRole, CurveSet, ProjectionKey

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
EUR, USD = Currency("EUR"), Currency("USD")
SIX_M, THREE_M = Tenor(6), Tenor(3)

OIS = DiscountCurve.flat(VAL, 0.020, DC)
PROJ = DiscountCurve.flat(VAL, 0.026, DC)


def _single() -> CurveSet:
    return CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): OIS})


def _multi() -> CurveSet:
    return CurveSet(
        {CurveKey(CurveRole.DISCOUNT, EUR): OIS},
        projections={ProjectionKey(EUR, SIX_M): PROJ},
    )


class TestTheTwoBranchesAreCoEqual:
    """Both halves of the invariant — *agree* where they must and *differ* where they must — at the
    one place the choice is now made. A sweep that only ever saw one branch would be the T33b
    input-population degeneracy: uniformly well-formed market data, so the other branch never runs.
    """

    def test_a_snapshot_WITHOUT_projections_is_the_PRE_CRISIS_era(self) -> None:
        """Both legs read the discount curve, and they are the **same object** — not merely equal.
        Identity is the strong form: it is what makes `_basis_factor`'s `projection is discount`
        shortcut structurally sound rather than a numerical coincidence."""
        projection, discount = _single().curves_for_era(EUR, SIX_M)
        assert projection is discount is OIS

    def test_a_snapshot_WITH_a_projection_is_the_POST_CRISIS_era(self) -> None:
        projection, discount = _multi().curves_for_era(EUR, SIX_M)
        assert projection is PROJ
        assert discount is OIS

    def test_the_two_eras_actually_DIFFER_so_neither_test_is_vacuous(self) -> None:
        """The premise §4 demands, asserted rather than assumed: if the fixture's OIS and
        projection curves happened to coincide, both tests above would pass against a single-curve
        implementation. Measured as a real basis."""
        single_p, _ = _single().curves_for_era(EUR, SIX_M)
        multi_p, multi_d = _multi().curves_for_era(EUR, SIX_M)
        assert multi_p is not multi_d
        assert single_p.df_at(5.0) != pytest.approx(multi_p.df_at(5.0), rel=1e-6)
        assert abs(multi_p.df_at(5.0) / multi_d.df_at(5.0) - 1.0) > 0.01


class TestTheSelectionIsKeyedOnBOTHDimensions:
    """A `ProjectionKey` is `(currency, tenor)`, so the era is per-currency **and** per-tenor. A
    lookup keyed on only one would silently hand a 6M projection curve to a 3M trade."""

    def test_a_DIFFERENT_TENOR_falls_back_to_the_pre_crisis_branch(self) -> None:
        projection, discount = _multi().curves_for_era(EUR, THREE_M)
        assert projection is discount is OIS

    def test_a_DIFFERENT_CURRENCY_raises_rather_than_defaulting(self) -> None:
        """No discount curve at all is not an era choice — it is a missing input, and the accessor
        refuses it. Falling back to a flat curve would price the trade at its undiscounted
        notional, a wrong number wearing the right shape."""
        with pytest.raises(KeyError, match="no discount curve"):
            _multi().curves_for_era(USD, SIX_M)


class TestItIsTheSameLookupBothConsumersNowShare:
    def test_the_swap_pricers_helper_DELEGATES_rather_than_re_deriving(self) -> None:
        """F-6.6's actual content. `_curves` must return exactly what `curves_for_era` returns —
        same objects — or the two have drifted apart again and the "one place" claim is false a
        second time."""
        from pricebook.engine.interest_rate_swap import _curves
        from pricebook.market.snapshot import MarketSnapshot

        for curves in (_single(), _multi()):
            market = MarketSnapshot(valuation_date=VAL, curves=curves)
            assert _curves(market, EUR, SIX_M) == curves.curves_for_era(EUR, SIX_M)

    def test_the_capstone_no_longer_carries_its_own_copy(self) -> None:
        """The duplication guard, stated structurally rather than by reading the file once: the
        capstone must not re-implement the membership test. A string check is a weak instrument,
        but the failure it guards is precisely textual — a second copy reappearing — and the T30
        lesson is to count the copies of a guard, not the values."""
        from pathlib import Path

        import pricebook.risk.collateralised_capstone as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "in market.curves.projections" not in source, (
            "the capstone re-derives the era lookup again; it belongs in CurveSet.curves_for_era"
        )
        assert "curves_for_era" in source
