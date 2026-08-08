"""L3 oracle — the dependency-ordered dual-curve bootstrap.

`calibrate` builds a curve SET in dependency order: the ESTR discount curve (self-
discounting) first, then the EURIBOR_3M projection curve discounted on it. Every
calibrating swap — OIS and EURIBOR — reprices to its quoted rate through the same
building blocks the engine composes (§3d). The two curves are genuinely distinct.
"""

from datetime import date

from pricebook_ng.calibration.calibrate import (
    CalibrationSpec,
    CurveBuild,
    ParSwapQuote,
    calibrate,
    par_swap,
)
from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    Tenor,
    TenorUnit,
    get_rate_index,
)
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01

VAL = date(2026, 1, 15)
ESTR = get_rate_index("ESTR")
EURIBOR_3M = get_rate_index("EURIBOR_3M")
DC = DayCountConvention.ACT_365_FIXED

OIS = (
    ParSwapQuote(Tenor(1, TenorUnit.YEAR), 0.030),
    ParSwapQuote(Tenor(2, TenorUnit.YEAR), 0.032),
    ParSwapQuote(Tenor(3, TenorUnit.YEAR), 0.034),
    ParSwapQuote(Tenor(5, TenorUnit.YEAR), 0.036),
)
EURIBOR = (  # a few bp above OIS ⇒ a genuinely distinct projection curve
    ParSwapQuote(Tenor(1, TenorUnit.YEAR), 0.0312),
    ParSwapQuote(Tenor(2, TenorUnit.YEAR), 0.0332),
    ParSwapQuote(Tenor(3, TenorUnit.YEAR), 0.0352),
    ParSwapQuote(Tenor(5, TenorUnit.YEAR), 0.0372),
)
DISCOUNT = CurveBuild(index=ESTR, frequency=Frequency.ANNUAL, day_count=DC, quotes=OIS)
PROJECTION = CurveBuild(index=EURIBOR_3M, frequency=Frequency.ANNUAL, day_count=DC, quotes=EURIBOR)
SPEC = CalibrationSpec.single_currency(valuation_date=VAL, currency=Currency.EUR, discount=DISCOUNT, projection=PROJECTION)


def _par(swap, discount, projection):
    return float_leg_pv(swap.float_leg.schedule, swap.float_leg.day_count, discount, projection) / rpv01(
        swap.fixed_leg.schedule, swap.fixed_leg.day_count, discount
    )


def test_both_curves_reprice_their_calibrating_swaps_to_par() -> None:
    model, result = calibrate(SPEC)
    assert result.converged
    assert len(result.residuals) == len(OIS) + len(EURIBOR)
    assert max(abs(r) for r in result.residuals) < 1e-10
    curves = model.market.curves
    discount = curves.discount(Currency.EUR)
    for q in OIS:
        swap = par_swap(SPEC, DISCOUNT, q.tenor, q.rate)
        assert abs(_par(swap, discount, curves.projection(ESTR)) - q.rate) < 1e-10
    for q in EURIBOR:
        swap = par_swap(SPEC, PROJECTION, q.tenor, q.rate)
        assert abs(_par(swap, discount, curves.projection(EURIBOR_3M)) - q.rate) < 1e-10


def test_projection_curve_is_distinct_from_discount() -> None:
    curves = calibrate(SPEC)[0].market.curves
    five_y = VAL + Tenor(5, TenorUnit.YEAR)
    assert abs(curves.discount(Currency.EUR).df(five_y) - curves.projection(EURIBOR_3M).df(five_y)) > 1e-4


def test_ois_projection_is_the_discount_curve_itself() -> None:
    # degenerate config: an OIS index projects off its own (discount) curve — same object
    curves = calibrate(SPEC)[0].market.curves
    assert curves.projection(ESTR) is curves.discount(Currency.EUR)
