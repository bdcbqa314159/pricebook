"""L6 audit Batch A — #2: a multi-currency book returns a value, not a TypeError (invariant 4).

Ports repro_M: a Book of one EUR swap + one USD swap, both individually priceable, used to raise
`TypeError: cannot mix EUR and USD` out of `mark_book`. The shell now returns a per-currency
`Mapping[Currency, Money]` (single-currency = the degenerate 1-entry map).
"""

from collections.abc import Mapping
from datetime import date

from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    RollRule,
    ScheduleTerms,
    Tenor,
    TenorUnit,
    TimeMeasure,
    build_schedule,
    get_rate_index,
)
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap
from pricebook_ng.shell import Book, Trade, mark_book

VAL = date(2026, 1, 15)
EURIBOR = get_rate_index("EURIBOR_3M")
SOFR = get_rate_index("SOFR")
EUR, USD = EURIBOR.id.currency, SOFR.id.currency
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
TM = TimeMeasure(VAL, DC)
FAR = VAL + Tenor(6, Y)
_SCHED = build_schedule(VAL, VAL + Tenor(5, Y), ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None)))
EUR_SWAP = VanillaSwap(1.0, EUR, FixedLeg(_SCHED, DC, 0.02), FloatLeg(_SCHED, DC, EURIBOR))
USD_SWAP = VanillaSwap(1.0, USD, FixedLeg(_SCHED, DC, 0.02), FloatLeg(_SCHED, DC, SOFR))


def _model() -> DiscountingModel:
    def flat(r: float) -> DiscountCurve:
        return DiscountCurve.flat(TM, r, until=FAR)
    curves = CurveSet({
        CurveKey(CurveRole.DISCOUNT, EUR): flat(0.03), CurveKey(CurveRole.PROJECTION, EURIBOR): flat(0.035),
        CurveKey(CurveRole.DISCOUNT, USD): flat(0.04), CurveKey(CurveRole.PROJECTION, SOFR): flat(0.045),
    })
    return DiscountingModel(MarketSnapshot(VAL, curves))


def test_mixed_currency_book_returns_a_two_entry_map() -> None:  # #2, repro_M
    book = Book((Trade((EUR_SWAP,), VAL), Trade((USD_SWAP,), VAL)))
    result = mark_book(book, _model())  # used to raise TypeError
    assert isinstance(result, Mapping)
    assert set(result.keys()) == {EUR, USD}
    assert result[EUR].currency == EUR and result[USD].currency == USD


def test_single_currency_book_is_a_one_entry_map() -> None:
    book = Book((Trade((EUR_SWAP,), VAL),))
    result = mark_book(book, _model())
    assert isinstance(result, Mapping)
    assert list(result.keys()) == [EUR]
