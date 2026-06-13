"""Regression for L2 Wave-2 audit — `AmortisingBond._serialisable` field
list was completely STALE.

Pre-fix declaration:

    _serialisable("amortising_bond", ['face_value', 'coupon_rate',
                                       'n_periods', 'frequency'])(AmortisingBond)

But the actual dataclass fields were `notional, coupon_rate,
maturity_years, n_payments, amortisation_type`.  None of the four
declared field names existed on the class.  The serialisable framework
emitted ``UserWarning`` on import for each mismatch.  A ``to_dict()``
call would either drop everything or raise ``AttributeError`` depending
on how the framework handled missing attributes; either way the round-
trip was broken.

Post-fix the field list matches the dataclass.  The ``from_convention``
factory was also broken (it tried to pass `frequency=` to a constructor
that doesn't accept it) and has been updated.
"""

from __future__ import annotations

import pytest

from pricebook.core.serialisable import from_dict
from pricebook.fixed_income.amortising_bond import AmortisingBond


class TestAmortisingBondSerialisation:
    def test_to_dict_runs(self):
        b = AmortisingBond(
            notional=1_000_000.0, coupon_rate=0.05,
            maturity_years=30.0, n_payments=360,
            amortisation_type="mortgage",
        )
        d = b.to_dict()
        assert d is not None

    def test_round_trip_preserves_all_fields(self):
        b = AmortisingBond(
            notional=500_000.0, coupon_rate=0.04,
            maturity_years=15.0, n_payments=180,
            amortisation_type="linear",
        )
        b2 = from_dict(b.to_dict())
        assert b2.notional == 500_000.0
        assert b2.coupon_rate == 0.04
        assert b2.maturity_years == 15.0
        assert b2.n_payments == 180
        assert b2.amortisation_type == "linear"

    @pytest.mark.parametrize("amort_type", ["mortgage", "linear"])
    def test_amortisation_type_round_trips(self, amort_type):
        b = AmortisingBond(
            notional=100_000.0, coupon_rate=0.03,
            maturity_years=10.0, n_payments=120,
            amortisation_type=amort_type,
        )
        b2 = from_dict(b.to_dict())
        assert b2.amortisation_type == amort_type
