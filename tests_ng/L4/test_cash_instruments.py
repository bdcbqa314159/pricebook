"""L4 oracle — cash instruments (deposits + FRAs + IMM futures) on both curves.

The discount (ESTR) curve is built from OIS deposits + OIS swaps; the projection (EURIBOR_3M)
curve from FRAs (front anchored by a spot FRA) + IMM futures + par swaps. Every calibrating
instrument must reprice to par THROUGH THE L4 ENGINE (registry-dispatched). The futures oracle
checks the forward APPROXIMATION is applied (curve reproduces 1 − price at the IMM segment),
never that a market price is right (doc 18 §2; convexity re-open trigger recorded in code).
"""

from datetime import date

from pricebook_ng.calibration.calibrate import (
    CalibrationSpec,
    CurveBuild,
    calibrate,
    product,
)
from pricebook_ng.engine import price
from pricebook_ng.foundation import (
    Accrual,
    Currency,
    DayCountConvention,
    Frequency,
    PricingResult,
    Tenor,
    TenorUnit,
    get_rate_index,
    next_imm,
)
from pricebook_ng.market.building_blocks import forward
from pricebook_ng.market.quotes import DepositQuote, FRAQuote, FutureQuote, ParSwapQuote

VAL = date(2026, 1, 15)
ESTR = get_rate_index("ESTR")
EURIBOR_3M = get_rate_index("EURIBOR_3M")
DC = DayCountConvention.ACT_365_FIXED
Y, M = TenorUnit.YEAR, TenorUnit.MONTH
IMM1 = next_imm(VAL + Tenor(1, Y))
IMM2 = next_imm(IMM1 + Tenor(1, M))
FUT = ((IMM1, 0.9660), (IMM2, 0.9655))

DISCOUNT = CurveBuild(
    index=ESTR,
    frequency=Frequency.ANNUAL,
    day_count=DC,
    quotes=(
        DepositQuote(Tenor(3, M), 0.0290),
        DepositQuote(Tenor(6, M), 0.0295),
        ParSwapQuote(Tenor(1, Y), 0.0300),
        ParSwapQuote(Tenor(2, Y), 0.0320),
        ParSwapQuote(Tenor(3, Y), 0.0340),
        ParSwapQuote(Tenor(5, Y), 0.0360),
    ),
)
PROJECTION = CurveBuild(
    index=EURIBOR_3M,
    frequency=Frequency.ANNUAL,
    day_count=DC,
    quotes=(
        FRAQuote(None, Tenor(3, M), 0.0312),
        FRAQuote(Tenor(3, M), Tenor(6, M), 0.0316),
        FRAQuote(Tenor(6, M), Tenor(9, M), 0.0320),
        FRAQuote(Tenor(9, M), Tenor(12, M), 0.0324),
        FutureQuote(IMM1, 0.9660),
        FutureQuote(IMM2, 0.9655),
        ParSwapQuote(Tenor(2, Y), 0.0332),
        ParSwapQuote(Tenor(3, Y), 0.0352),
        ParSwapQuote(Tenor(5, Y), 0.0372),
    ),
)
SPEC = CalibrationSpec.single_currency(valuation_date=VAL, currency=Currency.EUR, discount=DISCOUNT, projection=PROJECTION)


def test_every_cash_instrument_reprices_to_zero_through_the_engine() -> None:
    model, result = calibrate(SPEC)
    assert result.converged
    for build in (DISCOUNT, PROJECTION):
        for quote in build.quotes:
            result_pv = price(product(SPEC, build, quote), model)
            assert isinstance(result_pv, PricingResult), f"{quote!r} -> {result_pv!r}"
            assert abs(result_pv.pv.amount) < 1e-9, f"{quote!r} PV={result_pv.pv.amount}"


def test_future_reproduces_one_minus_price_at_the_imm_segment() -> None:
    # doc 18 §2: the oracle tests the forward approximation is APPLIED, not that a price is right.
    model, _ = calibrate(SPEC)
    projection = model.market.curves.projection(EURIBOR_3M)
    for imm, px in FUT:
        accrual = Accrual(imm, imm + EURIBOR_3M.id.tenor, DC)
        assert abs(forward(projection, accrual) - (1.0 - px)) < 1e-10
