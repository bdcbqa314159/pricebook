"""Tranche 2 — the zero-coupon inflation swap, model-free. BERHAN-2012 §1.2.2.

Equation numbers are the thesis's: inflation leg payout `N[I(TM)/I0 − 1]` (1.2.6), fixed leg
`N[(1+b)^TM − 1]` (1.2.7), leg value `N[I(t)/I0·Pr(t,TM) − Pn(t,TM)]` (1.2.13) which at `t=0`
with `I(0)=I0` is `N[Pr−Pn]` (1.2.14), and the break-even relation `Pr = Pn(1+b)^TM` (1.2.16).

**No number here is taken from a figure.** The thesis's empirical results are plots, so the oracles
are its closed forms, an exact reduction, and one pointwise identity — all computed in the test
from quantities the test itself chose.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.foundation import (
    Currency,
    DayCountConvention,
    InflationIndex,
    Money,
    PricingFailure,
    PricingResult,
    Term,
)
from pricebook.market import (
    CurveKey,
    CurveRole,
    CurveSet,
    DiscountCurve,
    MarketSnapshot,
    real_curve_from_breakeven,
)
from pricebook.models import DiscountingModel
from pricebook.products import ZeroCouponInflationSwap

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
HICP = InflationIndex("HICPxt")
MATURITY = VAL + timedelta(days=365 * 5)
TERM = Term(VAL, MATURITY, DC)
N, I0 = 1_000_000.0, 100.0

NOMINAL = CurveKey(CurveRole.DISCOUNT, EUR)
REAL = CurveKey(CurveRole.REAL, EUR)


def _market(nominal: float, real: float, index_level: float = I0) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet(
            {
                NOMINAL: DiscountCurve.flat(VAL, nominal, DC),
                REAL: DiscountCurve.flat(VAL, real, DC),
            }
        ),
        index_fixings={HICP: index_level},
    )


def _swap(fixed_rate: float = 0.02, term: Term = TERM) -> ZeroCouponInflationSwap:
    return ZeroCouponInflationSwap(HICP, I0, fixed_rate, term, Money(N, EUR))


def _pv(swap: ZeroCouponInflationSwap, market: MarketSnapshot) -> float:
    out = price(swap, DiscountingModel(market))
    assert isinstance(out, PricingResult), out
    return out.pv.amount


# ── oracle 1: the leg's closed form, 1.2.14, off curves the TEST specifies ──────────────────
def test_the_inflation_leg_at_inception_is_N_times_Pr_minus_Pn() -> None:
    """`ZCIIS(0,TM) = N[Pr(0,TM) − Pn(0,TM)]` (1.2.14). `Pr` and `Pn` are flat curves the test
    chose, so both sides are arithmetic here — nothing is read back from the engine."""
    from pricebook.engine.inflation import inflation_leg_value

    for rn, rr in ((0.03, 0.01), (0.05, 0.05), (0.02, 0.04)):
        market = _market(rn, rr)
        t = TERM.years()
        pn, pr = math.exp(-rn * t), math.exp(-rr * t)
        got = inflation_leg_value(_swap(), market)
        assert got == pytest.approx(N * (pr - pn), abs=1e-9, rel=1e-12), (rn, rr)


# ── oracle 2: the EXACT reduction — 1.2.13 collapses to 1.2.14 at I(t)=I0 ───────────────────
def test_the_seasoned_formula_REDUCES_to_the_inception_one_when_the_index_is_at_its_base() -> None:
    """1.2.13 carries `I(t)/I0`; at `I(t)=I0` that factor is 1 and the formula *is* 1.2.14. One
    formula, not two — so if the index term were applied as a difference, or to the wrong leg,
    the two would diverge. This is the reduction shape the extraction doc says to reach for."""
    from pricebook.engine.inflation import inflation_leg_value

    rn, rr = 0.03, 0.01
    t = TERM.years()
    at_base = inflation_leg_value(_swap(), _market(rn, rr, index_level=I0))
    assert at_base == pytest.approx(N * (math.exp(-rr * t) - math.exp(-rn * t)), abs=1e-9)


def test_the_index_enters_as_a_RATIO_so_the_real_leg_scales_with_it() -> None:
    """`I(t)/I0` — double the index and the real-bond term doubles while the `−Pn` does not. An
    index entering as a difference would move the leg by a constant instead."""
    from pricebook.engine.inflation import inflation_leg_value

    rn, rr = 0.03, 0.01
    t = TERM.years()
    pn, pr = math.exp(-rn * t), math.exp(-rr * t)
    doubled = inflation_leg_value(_swap(), _market(rn, rr, index_level=2.0 * I0))
    assert doubled == pytest.approx(N * (2.0 * pr - pn), abs=1e-9)


# ── oracle 3a (MUST-FIX 1): the CONSTRUCTOR, pointwise — does not telescope ─────────────────
def test_the_real_curve_constructor_satisfies_1_2_16_POINTWISE() -> None:
    """`Pr(0,T) = Pn(0,T)·(1+b)^T` at every horizon, to 1e-12, with no swap involved.

    **Why this exists separately from the par oracle below.** A par reprice *telescopes*: build
    `Pr` by 1.2.16 and price a swap struck at `b`, and the PV is zero algebraically as long as
    construction and pricing share their conventions — *even if both are wrong the same way*. So
    the par test guards the shared-`Pr` identity and cannot see a wrong compounding or day count.
    This one tests the new structure directly, which is the structural-oracle rule applied to the
    one structure this tranche builds."""
    nominal = DiscountCurve.flat(VAL, 0.03, DC)
    breakevens = {1.0: 0.018, 2.5: 0.021, 5.0: 0.0235, 10.0: 0.025}
    real = real_curve_from_breakeven(nominal, breakevens)
    for t, b in breakevens.items():
        when = VAL + timedelta(days=round(t * 365))
        expected = nominal.df(when) * (1.0 + b) ** when_t(when)
        assert real.df(when) == pytest.approx(expected, abs=1e-12, rel=1e-12), t


def when_t(d: date) -> float:
    """The test's own year fraction, stated here rather than asked of the library."""
    return (d - VAL).days / 365.0


# ── oracle 3b: the par identity — the shared-`Pr` guard (telescopes, and that is understood) ─
def test_a_swap_struck_AT_the_break_even_reprices_to_ZERO() -> None:
    """1.2.16's defining property. It telescopes (see the constructor oracle above), so its job is
    narrower than it looks: it proves the *construction* of `Pr` and the *pricing* with `Pr` are
    one `Pr` and not two that could drift — the `rpv01` lesson, guarded before divergence."""
    nominal = DiscountCurve.flat(VAL, 0.03, DC)
    for b in (0.005, 0.02, 0.035):
        real = real_curve_from_breakeven(nominal, {TERM.years(): b})
        market = MarketSnapshot(
            valuation_date=VAL,
            curves=CurveSet({NOMINAL: nominal, REAL: real}),
            index_fixings={HICP: I0},
        )
        assert _pv(_swap(fixed_rate=b), market) == pytest.approx(0.0, abs=1e-6), b


def test_the_net_swap_is_the_inflation_leg_MINUS_the_fixed_leg() -> None:
    """The decomposition, asserted so the net PV cannot be a formula of its own."""
    from pricebook.engine.inflation import inflation_leg_value

    rn, rr, b = 0.03, 0.01, 0.02
    market = _market(rn, rr)
    t = TERM.years()
    fixed = N * ((1.0 + b) ** t - 1.0) * math.exp(-rn * t)
    assert _pv(_swap(b), market) == pytest.approx(inflation_leg_value(_swap(b), market) - fixed, abs=1e-9)


# ── oracle 5: a degenerate case with a known answer ─────────────────────────────────────────
def test_zero_inflation_everywhere_is_a_zero_PV() -> None:
    """`Pr = Pn`, `b = 0`, `I(t) = I0`: both legs pay nothing and the PV is exactly zero,
    independent of the level of rates."""
    for r in (0.0, 0.03, 0.09):
        assert _pv(_swap(fixed_rate=0.0), _market(r, r)) == pytest.approx(0.0, abs=1e-9)


# ── oracle 6: sign ──────────────────────────────────────────────────────────────────────────
def test_the_inflation_RECEIVER_gains_when_realised_inflation_beats_the_strike() -> None:
    """Higher real-bond price (lower real rate) means more inflation, so the receiver gains; and a
    higher strike costs them. The sign error no magnitude check catches."""
    base = _pv(_swap(0.02), _market(0.03, 0.01))
    more_inflation = _pv(_swap(0.02), _market(0.03, 0.005))  # lower real rate ⇒ higher break-even
    assert more_inflation > base
    assert _pv(_swap(0.03), _market(0.03, 0.01)) < base  # a higher strike costs the receiver


# ── oracle 7a (MUST-FIX 2): matured is HISTORICAL — PV 0, not a failure ─────────────────────
def test_a_MATURED_swap_is_PV_ZERO_not_a_failure() -> None:
    """Invariant 6, and tranche 1's precedent: a flow on or before the valuation date is the
    shell's business, excluded from the mark. `PricingFailure` is for inputs that *cannot* be
    priced — conflating the two would make a settled trade look broken."""
    for maturity in (VAL, VAL - timedelta(days=1), VAL - timedelta(days=400)):
        swap = _swap(term=Term(maturity - timedelta(days=365 * 5), maturity, DC))
        out = price(swap, DiscountingModel(_market(0.03, 0.01)))
        assert isinstance(out, PricingResult), out
        assert out.pv.amount == pytest.approx(0.0, abs=1e-12)


# ── oracle 7b: genuinely unpriceable inputs ARE failures ────────────────────────────────────
def test_a_missing_REAL_curve_is_a_failure() -> None:
    market = MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({NOMINAL: DiscountCurve.flat(VAL, 0.03, DC)}),
        index_fixings={HICP: I0},
    )
    assert isinstance(price(_swap(), DiscountingModel(market)), PricingFailure)


def test_a_missing_INDEX_FIXING_is_a_failure_not_an_assumed_base() -> None:
    """Defaulting `I(t)` to `I0` would silently price every seasoned trade as if at inception."""
    market = MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet(
            {NOMINAL: DiscountCurve.flat(VAL, 0.03, DC), REAL: DiscountCurve.flat(VAL, 0.01, DC)}
        ),
        index_fixings={},
    )
    out = price(_swap(), DiscountingModel(market))
    assert isinstance(out, PricingFailure)
    assert "fixing" in out.reason.lower()


# ── oracle 8: the representation ────────────────────────────────────────────────────────────
def test_the_swap_round_trips_and_refuses_an_unknown_schema_version() -> None:
    swap = _swap()
    assert ZeroCouponInflationSwap.from_dict(swap.to_dict()) == swap
    bad = swap.to_dict()
    bad["schema_version"] = 99
    with pytest.raises(ValueError, match="reads up to"):
        ZeroCouponInflationSwap.from_dict(bad)
