"""Property-based serialisation oracle (CP-3 ruling §4.2).

Strengthens the per-type round-trip checks from "round-trips *this* example" to
"round-trips *any* generated instance". The contract every serialisable type owes:

    T.from_dict(x.to_dict()) == x        for all valid x

Dict round-trip preserves float objects exactly (no text encoding here), so this is
exact equality, not a tolerance check. Covers the whole CP-3 encoder stack —
`Money`/`Accrual`/`Cashflow` (shared) and the five retired-against products.
"""

from datetime import date, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from pricebook_ng.foundation.cashflow import Accrual, Cashflow
from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention
from pricebook_ng.products.deposit import Deposit, deposit
from pricebook_ng.products.fixed_cashflow import FixedCashflow
from pricebook_ng.products.fra import ForwardRateAgreement
from pricebook_ng.products.ois import OvernightIndexSwap, overnight_index_swap
from pricebook_ng.products.swap import SwapTerms

# ── building-block strategies ──────────────────────────────────────────────────
currencies = st.sampled_from(list(Currency))
day_counts = st.sampled_from(list(DayCountConvention))
# builders call year_fraction at construction; ACT_ACT_ICMA needs coupon anchors and
# BUS_252 needs a calendar, so a plain builder can't use them — exclude for those types
# (types that only *store* the day-count keep the full set).
_DC = DayCountConvention
builder_day_counts = st.sampled_from(
    [d for d in _DC if d not in (_DC.ACT_ACT_ICMA, _DC.BUS_252)]
)
frequencies = st.sampled_from(list(Frequency))
# bounded + finite so instrument builders (amount·(1+rate·τ)) stay finite and exact
amounts = st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False)
rates = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False)
money = st.builds(Money, amount=amounts, currency=currencies)


@st.composite
def periods(draw):
    """A (start, end) pair with start < end."""
    start = draw(st.dates(min_value=date(2000, 1, 1), max_value=date(2060, 1, 1)))
    end = start + timedelta(days=draw(st.integers(min_value=1, max_value=20_000)))
    return start, end


@st.composite
def accruals(draw):
    start, end = draw(periods())
    return Accrual(start, end, draw(day_counts))


@st.composite
def cashflows(draw):
    return Cashflow(
        draw(st.dates(min_value=date(2000, 1, 1), max_value=date(2080, 1, 1))),
        draw(money),
        draw(st.one_of(st.none(), accruals())),
    )


@st.composite
def deposits(draw):
    start, maturity = draw(periods())
    return deposit(draw(money), draw(rates), start, maturity, draw(builder_day_counts))


@st.composite
def fras(draw):
    return ForwardRateAgreement(draw(money), draw(rates), draw(accruals()), draw(st.booleans()))


@st.composite
def swap_terms(draw):
    leg = ScheduleTerms(draw(frequencies), draw(builder_day_counts))
    return SwapTerms(leg, ScheduleTerms(draw(frequencies), draw(builder_day_counts)), draw(st.booleans()))


@st.composite
def ois(draw):
    start, maturity = draw(periods())
    return overnight_index_swap(draw(money), draw(rates), start, maturity, draw(swap_terms()))


# ── the property: every type round-trips exactly ──────────────────────────────
_CASES = [
    (NumericalConfig, st.builds(
        NumericalConfig,
        fd_bump=st.floats(min_value=1e-12, max_value=1.0, allow_nan=False, allow_infinity=False),
        mc_paths=st.integers(min_value=1, max_value=10_000_000),
        mc_seed=st.integers(min_value=0, max_value=2**31),
    )),
    (FixedCashflow, st.builds(FixedCashflow, cashflow=cashflows())),
    (Deposit, deposits()),
    (ForwardRateAgreement, fras()),
    (OvernightIndexSwap, ois()),
]


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_every_serialisable_type_round_trips(data):
    for cls, strat in _CASES:
        x = data.draw(strat, label=cls.__name__)
        assert cls.from_dict(x.to_dict()) == x, cls.__name__


@settings(max_examples=200, deadline=None)
@given(cf=cashflows())
def test_cashflow_round_trips(cf):
    # the shared atom directly (its optional Accrual both present and absent)
    assert Cashflow.from_dict(cf.to_dict()) == cf
