"""Regression for L2 Wave-2 audit — `InterestRateSwap` serialisation
dropped ``convention``, ``stub``, ``eom`` (same gap as Swaption v0.976).

Pre-fix the `_serialisable` field list was:

    ["start", "end", "fixed_rate", "direction", "notional",
     "fixed_frequency", "float_frequency",
     "fixed_day_count", "float_day_count", "spread"]

Missing: `calendar`, `convention`, `stub`, `eom`.  All four affect the
leg schedules.  A `to_dict → from_dict` on a non-default IRS produced
a swap that priced differently.

Post-fix `convention`, `stub`, `eom` are part of the field list.
`calendar` remains excluded (runtime-only holiday data).
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.core.calendar import BusinessDayConvention
from pricebook.core.schedule import StubType
from pricebook.core.serialisable import from_dict
from pricebook.fixed_income.swap import InterestRateSwap, SwapDirection


def _make_irs(convention=BusinessDayConvention.MODIFIED_FOLLOWING,
              stub=StubType.SHORT_FRONT, eom=True):
    return InterestRateSwap(
        start=date(2024, 1, 1),
        end=date(2029, 1, 1),
        fixed_rate=0.04,
        direction=SwapDirection.PAYER,
        notional=1_000_000.0,
        convention=convention,
        stub=stub,
        eom=eom,
    )


class TestRoundTripPreservesConvention:
    @pytest.mark.parametrize("conv", [
        BusinessDayConvention.MODIFIED_FOLLOWING,
        BusinessDayConvention.PRECEDING,
        BusinessDayConvention.FOLLOWING,
    ])
    def test_convention_preserved(self, conv):
        s = _make_irs(convention=conv)
        s2 = from_dict(s.to_dict())
        assert s2.convention == conv


class TestRoundTripPreservesStub:
    @pytest.mark.parametrize("stub", [
        StubType.SHORT_FRONT,
        StubType.LONG_FRONT,
        StubType.SHORT_BACK,
        StubType.LONG_BACK,
    ])
    def test_stub_preserved(self, stub):
        s = _make_irs(stub=stub)
        s2 = from_dict(s.to_dict())
        assert s2.stub == stub


class TestRoundTripPreservesEOM:
    def test_eom_true(self):
        s = _make_irs(eom=True)
        s2 = from_dict(s.to_dict())
        assert s2.eom is True

    def test_eom_false(self):
        s = _make_irs(eom=False)
        s2 = from_dict(s.to_dict())
        assert s2.eom is False
