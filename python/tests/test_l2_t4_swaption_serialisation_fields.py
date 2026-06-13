"""Regression for L2 Wave-2 audit — `Swaption` serialisation dropped
``convention``, ``stub``, ``eom`` from the round-trip.

Pre-fix the `_serialisable` field list was:

    ["expiry", "swap_end", "strike", "swaption_type", "notional",
     "fixed_frequency", "float_frequency",
     "fixed_day_count", "float_day_count"]

Missing: `calendar`, `convention`, `stub`, `eom`.  All four are
constructor arguments that affect the underlying swap's schedule
generation.  A user serializing a Swaption with non-default
`convention=PRECEDING` (or non-default `stub` / `eom`), then
deserializing it, got a Swaption that PRICED DIFFERENTLY than the
original.

Post-fix `convention`, `stub`, `eom` are part of the field list.
`calendar` remains excluded because Calendar instances hold runtime
holiday data that isn't currently part of the serialisable type system
— the caller re-attaches a calendar via `from_convention` on load.
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.core.calendar import BusinessDayConvention
from pricebook.core.schedule import Frequency, StubType
from pricebook.core.serialisable import from_dict
from pricebook.options.swaption import Swaption, SwaptionType


def _make_swaption(convention=BusinessDayConvention.MODIFIED_FOLLOWING,
                    stub=StubType.SHORT_FRONT, eom=True):
    return Swaption(
        expiry=date(2026, 1, 15),
        swap_end=date(2031, 1, 15),
        strike=0.04,
        swaption_type=SwaptionType.PAYER,
        notional=1_000_000.0,
        convention=convention,
        stub=stub,
        eom=eom,
    )


class TestRoundTripPreservesConvention:
    def test_modified_following(self):
        s = _make_swaption(convention=BusinessDayConvention.MODIFIED_FOLLOWING)
        s2 = from_dict(s.to_dict())
        assert s2.convention == BusinessDayConvention.MODIFIED_FOLLOWING

    def test_preceding(self):
        s = _make_swaption(convention=BusinessDayConvention.PRECEDING)
        s2 = from_dict(s.to_dict())
        assert s2.convention == BusinessDayConvention.PRECEDING

    def test_following(self):
        s = _make_swaption(convention=BusinessDayConvention.FOLLOWING)
        s2 = from_dict(s.to_dict())
        assert s2.convention == BusinessDayConvention.FOLLOWING


class TestRoundTripPreservesStub:
    @pytest.mark.parametrize("stub", [
        StubType.SHORT_FRONT,
        StubType.LONG_FRONT,
        StubType.SHORT_BACK,
        StubType.LONG_BACK,
    ])
    def test_stub_preserved(self, stub):
        s = _make_swaption(stub=stub)
        s2 = from_dict(s.to_dict())
        assert s2.stub == stub


class TestRoundTripPreservesEOM:
    def test_eom_true(self):
        s = _make_swaption(eom=True)
        s2 = from_dict(s.to_dict())
        assert s2.eom is True

    def test_eom_false(self):
        s = _make_swaption(eom=False)
        s2 = from_dict(s.to_dict())
        assert s2.eom is False
