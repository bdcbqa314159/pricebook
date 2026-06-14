"""Regression for L2 T4 audit of `options.barrier_option`:

Pre-fix two silent-no-op API params that produced wrong PV:

1. **`rebate` ignored**: constructor accepts and round-trips ``rebate``,
   but neither the PDE (``fd_barrier_knockout/knockin``) nor the MC
   path honoured it.  Any ``rebate != 0`` was silently dropped, giving
   a PV indistinguishable from the no-rebate case.

2. **`pv_ctx` hardcoded `spot=100.0`**: when the pricing engine calls
   ``instrument.pv_ctx(ctx)``, the prior implementation silently used
   ``spot=100.0`` regardless of the actual underlying — because
   ``PricingContext`` has no ``equity_spots`` field.  Any equity at
   a different spot got the PV of a 100-spot underlying.

Fix: raise ``NotImplementedError`` in both cases.  Better a loud
failure than a silently-wrong price.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.pricing_context import PricingContext
from pricebook.options.barrier_option import BarrierOption


REF = date(2026, 4, 28)
MAT = REF + timedelta(days=365)


def _curve():
    return DiscountCurve.flat(REF, 0.03)


class TestRebateRaises:
    def test_nonzero_rebate_raises_pde(self):
        """Pre-fix: rebate=5.0 priced same as rebate=0.0.
        Post-fix: raises NotImplementedError."""
        opt = BarrierOption(strike=100, barrier=120, barrier_type="up_out",
                            maturity=MAT, rebate=5.0)
        with pytest.raises(NotImplementedError, match="rebate"):
            opt.price(spot=100, curve=_curve(), vol=0.20, method="pde")

    def test_nonzero_rebate_raises_mc(self):
        opt = BarrierOption(strike=100, barrier=120, barrier_type="up_in",
                            maturity=MAT, rebate=3.0)
        with pytest.raises(NotImplementedError, match="rebate"):
            opt.price(spot=100, curve=_curve(), vol=0.20, method="mc")

    def test_zero_rebate_still_prices(self):
        """Default rebate=0 must continue to price normally."""
        opt = BarrierOption(strike=100, barrier=120, barrier_type="up_out",
                            maturity=MAT)
        result = opt.price(spot=100, curve=_curve(), vol=0.20, method="pde")
        assert result.price > 0


class TestPvCtxRaises:
    def test_pv_ctx_raises_not_implemented(self):
        """Pre-fix: pv_ctx silently used spot=100.
        Post-fix: raises NotImplementedError with explanation."""
        opt = BarrierOption(strike=100, barrier=120, barrier_type="up_out",
                            maturity=MAT)
        ctx = PricingContext(valuation_date=REF, discount_curve=_curve())
        with pytest.raises(NotImplementedError, match="equity spot"):
            opt.pv_ctx(ctx)
