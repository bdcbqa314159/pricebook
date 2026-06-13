"""Regression for L2 Wave-2 audit — `VanillaCLN` was registered as
``_serialisable`` but its constructor did NOT store `frequency` as a
class attribute.  ``to_dict()`` raised ``AttributeError``, making the
class effectively unserialisable despite the declaration.

Pre-fix:

    def __init__(self, ..., frequency=Frequency.SEMI_ANNUAL, ...):
        self.start = start
        self.end = end
        # ... no `self.frequency = frequency` ...
        self.schedule = generate_schedule(start, end, frequency)

    _serialisable("vanilla_cln", [..., "frequency", ...])(VanillaCLN)

Calling ``vcln.to_dict()`` raised:
    AttributeError: 'VanillaCLN' object has no attribute 'frequency'

Post-fix: `self.frequency = frequency` is set in the constructor.
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.core.schedule import Frequency
from pricebook.core.serialisable import from_dict
from pricebook.credit.cds_index import VanillaCLN


class TestVanillaCLNSerialisation:
    def test_to_dict_does_not_raise(self):
        cln = VanillaCLN(
            start=date(2024, 1, 1), end=date(2029, 1, 1),
            coupon_rate=0.05, frequency=Frequency.QUARTERLY,
        )
        # Pre-fix this raised AttributeError.
        d = cln.to_dict()
        assert d is not None

    @pytest.mark.parametrize("freq", [
        Frequency.MONTHLY,
        Frequency.QUARTERLY,
        Frequency.SEMI_ANNUAL,
        Frequency.ANNUAL,
    ])
    def test_frequency_round_trips(self, freq):
        cln = VanillaCLN(
            start=date(2024, 1, 1), end=date(2029, 1, 1),
            coupon_rate=0.05, frequency=freq,
        )
        cln2 = from_dict(cln.to_dict())
        assert cln2.frequency == freq
        assert cln2.start == cln.start
        assert cln2.end == cln.end
        assert cln2.coupon_rate == cln.coupon_rate
