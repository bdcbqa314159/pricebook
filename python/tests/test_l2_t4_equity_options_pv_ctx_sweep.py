"""Regression sweep for L2 T4 audit of equity-option ``pv_ctx`` impls.

Five equity options silently hardcoded ``spot=100.0`` (or returned 0)
in ``pv_ctx``, because ``PricingContext`` carries no ``equity_spots``
field.  Any engine consumer got the PV of a 100-spot underlying for
every trade — same recurring "silent no-op API param" bug pattern
already fixed in ``BarrierOption.pv_ctx`` (v1.042 slice).

Fix: raise ``NotImplementedError`` in each.  Loud failure over
silently-wrong price until ``PricingContext.equity_spots`` is added.

Coverage: ``american_option``, ``autocallable``, ``asian_option``,
``cliquet``, ``basket_option``.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.pricing_context import PricingContext
from pricebook.models.black76 import OptionType


REF = date(2026, 4, 28)
MAT = REF + timedelta(days=365)


@pytest.fixture
def ctx():
    curve = DiscountCurve.flat(REF, 0.03)
    return PricingContext(valuation_date=REF, discount_curve=curve)


def test_american_option_pv_ctx_raises(ctx):
    from pricebook.options.american_option import AmericanOption
    opt = AmericanOption(strike=100, maturity=MAT, option_type=OptionType.PUT)
    with pytest.raises(NotImplementedError, match="equity spot"):
        opt.pv_ctx(ctx)


def test_autocallable_pv_ctx_raises(ctx):
    from pricebook.options.autocallable import Autocallable
    obs = [REF + timedelta(days=90 * k) for k in range(1, 5)]
    opt = Autocallable(observation_dates=obs)
    with pytest.raises(NotImplementedError, match="equity spot"):
        opt.pv_ctx(ctx)


def test_asian_option_pv_ctx_raises(ctx):
    from pricebook.options.asian_option import AsianOption, AsianSchedule
    schedule = AsianSchedule.monthly(REF, MAT)
    opt = AsianOption(schedule=schedule, strike=100)
    with pytest.raises(NotImplementedError, match="equity spot"):
        opt.pv_ctx(ctx)


def test_cliquet_pv_ctx_raises(ctx):
    from pricebook.options.cliquet import Cliquet
    reset_dates = [REF + timedelta(days=90 * k) for k in range(1, 5)]
    opt = Cliquet(reset_dates=reset_dates)
    with pytest.raises(NotImplementedError, match="equity spot"):
        opt.pv_ctx(ctx)


def test_basket_option_pv_ctx_raises(ctx):
    from pricebook.options.basket_option import BasketOption
    opt = BasketOption(
        strike=100, maturity=MAT,
        weights=[0.5, 0.5], payoff_type="basket",
        option_type=OptionType.CALL,
    )
    with pytest.raises(NotImplementedError, match="per-asset equity spots"):
        opt.pv_ctx(ctx)
