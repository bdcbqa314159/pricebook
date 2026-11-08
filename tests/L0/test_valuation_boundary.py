"""The historical-flow boundary as ONE atom (L0) — tranche 26, closing C10's trigger.

Invariant 6: *a flow on or before `valuation_date` is historical* — excluded from the mark, PV 0,
never discounted with a non-positive `t`. The comparison is therefore **strict** (`pay >
valuation_date` is live) and the boundary case is a payment landing exactly today.

**Why this atom exists, and why only now.** Until tranche 25 the comparison was inlined in five
engine modules, so register row C10 could only ever be `per-module`: a divergence between them was
*expressible*, and tranche 24 measured that it mattered — flipping `>` to `>=` was a **real 41%
mispricing** in `capfloor` (5,929 -> 8,364) while being a proved equivalent mutant in two others.
One atom makes the divergence inexpressible, which is what upgrades C10 to `guarded - structurally`.

It is extracted now because tranche 26's exposure grid is a real consumer (§6: not speculatively).
"""

from datetime import date, timedelta

from pricebook.foundation.time_measure import is_historical

TODAY = date(2026, 11, 8)


class TestTheBoundaryIsStrict:
    def test_a_flow_landing_EXACTLY_today_is_historical(self) -> None:
        """The case tranche 24 found mispriced by 41% when the comparison slipped."""
        assert is_historical(TODAY, TODAY) is True

    def test_a_flow_TOMORROW_is_live(self) -> None:
        assert is_historical(TODAY + timedelta(days=1), TODAY) is False

    def test_a_flow_YESTERDAY_is_historical(self) -> None:
        assert is_historical(TODAY - timedelta(days=1), TODAY) is True


class TestTheAtomIsTotalAndPure:
    def test_every_date_is_either_historical_or_live_and_never_both(self) -> None:
        for offset in range(-400, 401, 37):
            pay = TODAY + timedelta(days=offset)
            historical = is_historical(pay, TODAY)
            assert historical == (pay <= TODAY)
            assert historical != (pay > TODAY)

    def test_the_predicate_returns_a_BOOL_not_a_truthy_value(self) -> None:
        # A filter spelled `if not is_historical(...)` must not depend on truthiness of a date.
        assert isinstance(is_historical(TODAY, TODAY), bool)
