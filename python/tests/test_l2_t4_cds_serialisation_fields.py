"""Regression for L2 Wave-2 audit — `CDS` serialisation dropped
``convention`` (same shape as Swaption v0.976 / IRS v0.977).

Pre-fix the `_serialisable` field list missed `convention`, which
controls the business-day rolling rule applied to coupon dates.  A
CDS with non-default ``convention=PRECEDING`` round-tripped to one
with the default MODIFIED_FOLLOWING, changing the payment schedule
and therefore the price.

Post-fix `convention` is part of the field list.  `calendar` remains
excluded (runtime-only holiday data).
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.core.calendar import BusinessDayConvention
from pricebook.core.serialisable import from_dict
from pricebook.credit.cds import CDS


def _make_cds(convention=BusinessDayConvention.MODIFIED_FOLLOWING):
    return CDS(
        start=date(2024, 1, 1),
        end=date(2029, 1, 1),
        spread=0.02,
        notional=1_000_000.0,
        convention=convention,
    )


class TestRoundTripPreservesConvention:
    @pytest.mark.parametrize("conv", [
        BusinessDayConvention.MODIFIED_FOLLOWING,
        BusinessDayConvention.PRECEDING,
        BusinessDayConvention.FOLLOWING,
    ])
    def test_convention_preserved(self, conv):
        c = _make_cds(convention=conv)
        c2 = from_dict(c.to_dict())
        assert c2.convention == conv

    def test_default_round_trip_still_works(self):
        """Default-convention CDS still serialises and deserialises."""
        c = CDS(
            start=date(2024, 1, 1), end=date(2029, 1, 1),
            spread=0.02, notional=1_000_000.0,
        )
        c2 = from_dict(c.to_dict())
        assert c2.convention == BusinessDayConvention.MODIFIED_FOLLOWING
