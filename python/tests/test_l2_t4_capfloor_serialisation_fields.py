"""Regression for L2 Wave-2 audit — `CapFloor` serialisation dropped
``convention`` (same pattern as v0.976 Swaption / v0.977 IRS / v0.978 CDS).
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.core.calendar import BusinessDayConvention
from pricebook.core.serialisable import from_dict
from pricebook.options.capfloor import CapFloor


class TestRoundTripPreservesConvention:
    @pytest.mark.parametrize("conv", [
        BusinessDayConvention.MODIFIED_FOLLOWING,
        BusinessDayConvention.PRECEDING,
        BusinessDayConvention.FOLLOWING,
    ])
    def test_convention_preserved(self, conv):
        c = CapFloor(start=date(2024, 1, 1), end=date(2029, 1, 1),
                     strike=0.04, notional=1_000_000.0,
                     convention=conv)
        c2 = from_dict(c.to_dict())
        assert c2.convention == conv
