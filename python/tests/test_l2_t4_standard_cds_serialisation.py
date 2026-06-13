"""Regression for L2 Wave-2 audit — `StandardCDS` had a hand-written
``to_dict`` / ``from_dict`` that wasn't updated when `convention` was
added to the parent ``CDS._SERIAL_FIELDS`` (v0.978).

Pre-fix:
- The hand-written `to_dict` omitted `convention` from the params dict.
- The hand-written `from_dict` did not pass `convention=` to the
  constructor.

Result: a StandardCDS round-trip silently lost any non-default
convention.  The introspection sweep in v0.981 flagged this: the
declared `_SERIAL_FIELDS` (inherited from CDS) listed `convention`,
but the actual hand-written serialisation did not emit it.

Post-fix both methods handle `convention` (with a default for
backwards-compat with pre-fix dicts that don't carry the field).
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.core.calendar import BusinessDayConvention
from pricebook.core.serialisable import from_dict
from pricebook.credit.cds import StandardCDS


def _make(conv=BusinessDayConvention.MODIFIED_FOLLOWING):
    return StandardCDS(
        start=date(2024, 1, 1), end=date(2029, 1, 1),
        spread=0.02, grade="IG", notional=1_000_000.0,
        convention=conv,
    )


class TestStandardCDSRoundTrip:
    @pytest.mark.parametrize("conv", [
        BusinessDayConvention.MODIFIED_FOLLOWING,
        BusinessDayConvention.PRECEDING,
        BusinessDayConvention.FOLLOWING,
    ])
    def test_convention_preserved(self, conv):
        c = _make(conv)
        c2 = from_dict(c.to_dict())
        assert c2.convention == conv

    def test_legacy_dict_without_convention_defaults(self):
        """A dict missing the new `convention` field (legacy pre-v0.981)
        should still load successfully, defaulting to MODIFIED_FOLLOWING."""
        c = _make(BusinessDayConvention.PRECEDING)
        d = c.to_dict()
        # Simulate a legacy dict by stripping `convention`.
        del d["params"]["convention"]
        c2 = from_dict(d)
        assert c2.convention == BusinessDayConvention.MODIFIED_FOLLOWING

    def test_other_fields_still_round_trip(self):
        c = _make(BusinessDayConvention.PRECEDING)
        c2 = from_dict(c.to_dict())
        assert c2.spread == c.spread
        assert c2.grade == c.grade
        assert c2.notional == c.notional
        assert c2.standard_coupon == c.standard_coupon
