"""The forward CDS and the knockout default swaption (L4) — `OKANE-2008` ch 9.2/9.3. RED, T40.

**The correctness anchor is a cross-product identity, not a formula restatement.** A forward CDS is
long the spot CDS to `T` and short the spot CDS to `tF` at the same contractual spread, so

    V_forward(S₀) = price(CDS to T, S₀) − price(CDS to tF, S₀)

with **different code on each side**: the left is `(S(0,tF,T) − S₀)·RPV01(0,tF,T)` out of
`credit_legs`, the right is two tranche-38 engine calls through `price()`. That is what pins the
whole composition — the forward spread, the forward annuity, and the sign — rather than any single
number.

**Two things the spec offered as oracles are near-vacuous, and are labelled as such here.**

* Eq 9.2's `[S(t,T)·RPV01(t,T) − S(t,tF)·RPV01(t,tF)] / [RPV01(t,T) − RPV01(t,tF)]` and the 9.2.1
  derivation's `[Prot(t,T) − Prot(t,tF)] / [...]` are **the same expression**: the par spread *is*
  `Prot/RPV01`, so substituting gives one form from the other identically (measured: relative gap
  exactly 0.0). Asserting both is an algebraic tautology, not transcription independence — it is
  recorded as an identity and the independence comes from the cross-product above.
* "At `S₀ = S(0,tF,T)` the forward CDS is worth zero" **telescopes** — the forward spread is
  *defined* as the ratio that zeroes it. It is kept as a guard, and the non-telescoping version is
  the same statement asserted through the **engine difference**, where nothing was constructed to
  cancel.

The genuinely independent checks are therefore: the cross-product, the forward annuity against an
independent summation over the forward periods, and put-call parity tying the swaption back to the
forward CDS.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.foundation import CouponSchedule, Currency, Money, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.credit_legs import CdsLegs, forward_annuity, forward_spread, risky_pv01
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.models.credit import CreditModel
from pricebook.models.credit_swaption import CreditSwaptionModel
from pricebook.products.credit_default_swap import CreditDefaultSwap, ProtectionDirection
from pricebook.products.credit_default_swaption import CreditDefaultSwaption, SwaptionDirection
from pricebook.products.forward_cds import ForwardCDS

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
EUR = Currency("EUR")
DISCOUNT = CurveKey(CurveRole.DISCOUNT, EUR)
RECOVERY = 0.40
NOTIONAL = Money(10_000_000.0, EUR)
VOL = 0.45

#: NON-FLAT, read well beyond its first segment.
CURVE = DiscountCurve(TM, (0.0, 1.0, 2.0, 3.0, 5.0), (0.020, 0.026, 0.030, 0.033, 0.036))
#: TERM-STRUCTURED, and rising — a flat hazard makes the forward spread equal the spot one, which
#: is exactly where the forward-annuity *difference* stops carrying information (§4, the
#: reduction-anchor blind spot ratified at tranche 39).
HAZARD = SurvivalCurve(TM, (0.0, 1.0, 3.0), (0.010, 0.025, 0.045))

#: Semi-annual, so accruals are ~0.5 and unequal — the τ=1 history baked in.
def _dates(start_years: float, end_years: float) -> tuple[date, ...]:
    count = round((end_years - start_years) * 2)
    return tuple(VAL + timedelta(days=round(182.5 * (start_years * 2 + i)))
                 for i in range(count + 1))


EXPIRY_YEARS, MATURITY_YEARS = 2.0, 5.0
FULL_DATES = _dates(0.0, MATURITY_YEARS)
EXPIRY = FULL_DATES[round(EXPIRY_YEARS * 2)]


def _market() -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({DISCOUNT: CURVE}, survivals={EUR: HAZARD}))


def _credit() -> CreditModel:
    return CreditModel(_market(), RECOVERY)


def _swaption_model(volatility: float = VOL) -> CreditSwaptionModel:
    return CreditSwaptionModel(_credit(), volatility)


def _spot_cds(end_years: float, spread: float) -> CreditDefaultSwap:
    return CreditDefaultSwap(
        CouponSchedule(_dates(0.0, end_years), DC), spread, NOTIONAL,
        ProtectionDirection.BUYER)


def _forward_cds(spread: float, expiry: date = EXPIRY) -> ForwardCDS:
    return ForwardCDS(
        CouponSchedule(FULL_DATES, DC), expiry, spread, NOTIONAL, ProtectionDirection.BUYER)


def _swaption(strike: float, direction: SwaptionDirection) -> CreditDefaultSwaption:
    return CreditDefaultSwaption(
        CouponSchedule(FULL_DATES, DC), EXPIRY, strike, NOTIONAL, direction)


def _legs(end_years: float) -> CdsLegs:
    """The spot legs to one maturity, built here from the shared L1 functions."""
    from pricebook.market.credit_legs import protection_leg

    return CdsLegs(
        annuity=risky_pv01(CouponSchedule(_dates(0.0, end_years), DC).periods(), CURVE, HAZARD),
        protection=protection_leg(CURVE, HAZARD, RECOVERY, end_years, 2000))


def _par_forward() -> tuple[float, float]:
    """`(forward spread, forward annuity)` at the fixture's expiry and maturity."""
    near, far = _legs(EXPIRY_YEARS), _legs(MATURITY_YEARS)
    return forward_spread(near, far), forward_annuity(near, far)


class TestTheFixtureIsNotDegenerate:
    def test_the_hazard_is_TERM_STRUCTURED_so_the_forward_differs_from_both_spots(self) -> None:
        near, far = _legs(EXPIRY_YEARS), _legs(MATURITY_YEARS)
        spot_near, spot_far = near.protection / near.annuity, far.protection / far.annuity
        fwd, _ = _par_forward()
        assert fwd / spot_far - 1.0 > 0.2, (fwd, spot_far)
        assert fwd / spot_near - 1.0 > 0.8, (fwd, spot_near)

    def test_the_forward_annuity_is_a_REAL_subtraction(self) -> None:
        """Both terms material, so a dropped term cannot hide."""
        near, far = _legs(EXPIRY_YEARS), _legs(MATURITY_YEARS)
        assert near.annuity / far.annuity > 0.3, (near.annuity, far.annuity)
        assert forward_annuity(near, far) > 1.0

    def test_the_accruals_are_NON_UNIT_and_UNEQUAL(self) -> None:
        accruals = [a for _, _, a in CouponSchedule(FULL_DATES, DC).periods()]
        assert all(abs(a - 1.0) > 0.4 for a in accruals)
        assert len(set(round(a, 12) for a in accruals)) > 1

    def test_the_strike_is_NOT_at_the_money_so_parity_is_not_zero_equals_zero(self) -> None:
        fwd, _ = _par_forward()
        assert abs(STRIKE / fwd - 1.0) > 0.1, (STRIKE, fwd)

    def test_the_discount_curve_is_NON_FLAT(self) -> None:
        assert len(set(CURVE.zero_rates)) > 1


STRIKE = 0.0180


class TestTheForwardCdsAgainstTheSpotEngine:
    """The anchor: a cross-product identity with different code on each side."""

    @pytest.mark.parametrize("spread", [0.0150, 0.0229148072, 0.0300])
    def test_the_forward_cds_IS_the_difference_of_two_spot_cds(self, spread: float) -> None:
        model = _credit()
        far = price(_spot_cds(MATURITY_YEARS, spread), model)
        near = price(_spot_cds(EXPIRY_YEARS, spread), model)
        assert isinstance(far, PricingResult) and isinstance(near, PricingResult)
        want = far.pv.amount - near.pv.amount
        got = price(_forward_cds(spread), model)
        assert isinstance(got, PricingResult)
        assert got.pv.amount == pytest.approx(want, rel=1e-12, abs=1e-06), (got.pv.amount, want)

    def test_the_PAR_forward_spread_makes_the_forward_cds_worth_zero_through_the_ENGINE(
        self) -> None:
        """The non-telescoping form of the par-forward identity.

        Asserting `V = 0` against the formula telescopes — the forward spread is defined as the
        ratio that zeroes it. Asserting it against the **engine difference**, where nothing was
        constructed to cancel, does not."""
        fwd, _ = _par_forward()
        model = _credit()
        far = price(_spot_cds(MATURITY_YEARS, fwd), model)
        near = price(_spot_cds(EXPIRY_YEARS, fwd), model)
        assert abs(far.pv.amount - near.pv.amount) < 1e-03

    def test_the_SELLER_is_the_exact_negative_of_the_BUYER(self) -> None:
        buyer = price(_forward_cds(STRIKE), _credit())
        seller = price(
            ForwardCDS(CouponSchedule(FULL_DATES, DC), EXPIRY, STRIKE, NOTIONAL,
                       ProtectionDirection.SELLER), _credit())
        assert seller.pv.amount == pytest.approx(-buyer.pv.amount, rel=1e-14)


class TestTheForwardSpreadReductions:
    def test_a_FORWARD_START_at_the_first_schedule_date_reduces_to_the_SPOT_par_spread(
        self) -> None:
        """`tF → t` is the exact reduction (the near annuity vanishes), and the reference is
        tranche 38's `par_spread` — a cross-product, different code."""
        from pricebook.engine.credit_default_swap import par_spread

        empty = CdsLegs(annuity=0.0, protection=0.0)
        got = forward_spread(empty, _legs(MATURITY_YEARS))
        want = par_spread(_spot_cds(MATURITY_YEARS, STRIKE), _credit())
        assert got == pytest.approx(want, rel=1e-12), (got, want)

    def test_a_FLAT_hazard_makes_the_forward_spread_EQUAL_the_spot_one(self) -> None:
        """O'Kane 9.2.1, exact — and the degenerate case the companion above exists for: a flat
        hazard is precisely where the forward-annuity difference stops discriminating."""
        from pricebook.market.credit_legs import protection_leg

        flat = SurvivalCurve.flat(VAL, 0.03, DC)

        def legs(end_years: float) -> CdsLegs:
            return CdsLegs(
                annuity=risky_pv01(
                    CouponSchedule(_dates(0.0, end_years), DC).periods(), CURVE, flat),
                protection=protection_leg(CURVE, flat, RECOVERY, end_years, 2000))

        near, far = legs(EXPIRY_YEARS), legs(MATURITY_YEARS)
        spot = far.protection / far.annuity
        assert forward_spread(near, far) == pytest.approx(spot, rel=2e-04)

    def test_the_forward_ANNUITY_matches_an_INDEPENDENT_sum_over_the_forward_periods(self) -> None:
        """Genuinely independent: a difference of two sums versus one sum over the tail periods."""
        near, far = _legs(EXPIRY_YEARS), _legs(MATURITY_YEARS)
        direct = risky_pv01(
            CouponSchedule(_dates(EXPIRY_YEARS, MATURITY_YEARS), DC).periods(), CURVE, HAZARD)
        assert forward_annuity(near, far) == pytest.approx(direct, rel=1e-14)

    def test_eq_9_2_and_the_9_2_1_DERIVATION_are_the_same_expression(self) -> None:
        """An identity, NOT transcription independence — recorded so nobody mistakes it for one.

        `S(t,M) = Prot(t,M)/RPV01(t,M)` by definition, so substituting into 9.2 gives 9.2.1
        identically. The relative gap is exactly zero, which is the tell: two independent
        transcriptions of different formulas would differ in the last bits."""
        near, far = _legs(EXPIRY_YEARS), _legs(MATURITY_YEARS)
        spot_near, spot_far = near.protection / near.annuity, far.protection / far.annuity
        form_9_2 = ((spot_far * far.annuity - spot_near * near.annuity)
                    / (far.annuity - near.annuity))
        form_9_2_1 = (far.protection - near.protection) / (far.annuity - near.annuity)
        assert form_9_2 == form_9_2_1


class TestTheDefaultSwaption:
    def test_PUT_CALL_PARITY_ties_the_swaption_back_to_the_FORWARD_CDS(self) -> None:
        """The headline requirement oracle: it survives any slip in Black's `d±`, holds for every
        `σ` and `K`, and is a cross-product tie with different code each side."""
        payer = price(_swaption(STRIKE, SwaptionDirection.PAYER), _swaption_model())
        receiver = price(_swaption(STRIKE, SwaptionDirection.RECEIVER), _swaption_model())
        forward = price(_forward_cds(STRIKE), _credit())
        assert (payer.pv.amount - receiver.pv.amount) == pytest.approx(
            forward.pv.amount, rel=1e-12)

    @pytest.mark.parametrize("volatility", [0.15, 0.45, 0.95])
    def test_parity_holds_for_EVERY_vol(self, volatility: float) -> None:
        payer = price(_swaption(STRIKE, SwaptionDirection.PAYER), _swaption_model(volatility))
        receiver = price(
            _swaption(STRIKE, SwaptionDirection.RECEIVER), _swaption_model(volatility))
        forward = price(_forward_cds(STRIKE), _credit())
        assert (payer.pv.amount - receiver.pv.amount) == pytest.approx(
            forward.pv.amount, rel=1e-12)

    def test_ZERO_vol_gives_the_INTRINSIC_exactly(self) -> None:
        fwd, annuity = _par_forward()
        intrinsic = max(fwd - STRIKE, 0.0) * annuity * NOTIONAL.amount
        got = price(_swaption(STRIKE, SwaptionDirection.PAYER), _swaption_model(0.0))
        assert got.pv.amount == pytest.approx(intrinsic, rel=1e-12)

    def test_THE_COMPANION_a_positive_vol_adds_MATERIAL_time_value(self) -> None:
        """A reduction a no-op satisfies is vacuous (§4), so the non-degenerate case is asserted."""
        fwd, annuity = _par_forward()
        intrinsic = max(fwd - STRIKE, 0.0) * annuity * NOTIONAL.amount
        got = price(_swaption(STRIKE, SwaptionDirection.PAYER), _swaption_model())
        assert got.pv.amount / intrinsic - 1.0 > 0.05, (got.pv.amount, intrinsic)

    def test_the_value_is_MONOTONE_increasing_in_the_vol(self) -> None:
        values = [price(_swaption(STRIKE, SwaptionDirection.PAYER),
                        _swaption_model(v)).pv.amount for v in (0.0, 0.2, 0.5, 0.9)]
        assert values == sorted(values) and len(set(values)) == len(values), values

    def test_the_BLACK_COMPOSITION_matches_an_independent_transcription_of_9_3(self) -> None:
        """Black is reused and verified since tranche 5; the COMPOSITION is new, so it is
        transcribed here: the forward spread in, `σ²tE` as total variance, `× RPV01(0,tE,T)` out.

        The reference computes Black's bracket from `norm_cdf` directly rather than calling
        `black_price`, so a wrong total variance or a missing annuity scaling cannot cancel."""
        from pricebook.foundation.distributions import norm_cdf

        fwd, annuity = _par_forward()
        expiry_years = (EXPIRY - VAL).days / 365.0
        total_variance = VOL * VOL * expiry_years
        v = math.sqrt(total_variance)
        d1 = (math.log(fwd / STRIKE) + 0.5 * total_variance) / v
        d2 = d1 - v
        want = (fwd * norm_cdf(d1) - STRIKE * norm_cdf(d2)) * annuity * NOTIONAL.amount
        got = price(_swaption(STRIKE, SwaptionDirection.PAYER), _swaption_model())
        assert got.pv.amount == pytest.approx(want, rel=1e-12), (got.pv.amount, want)

    def test_both_swaption_values_are_NON_NEGATIVE(self) -> None:
        for direction in SwaptionDirection:
            got = price(_swaption(STRIKE, direction), _swaption_model())
            assert got.pv.amount >= 0.0, (direction, got.pv.amount)


class TestTheGuardsAreReachable:
    def test_a_FORWARD_START_that_is_not_a_schedule_date_is_refused(self) -> None:
        with pytest.raises(ValueError, match="schedule date"):
            ForwardCDS(CouponSchedule(FULL_DATES, DC), VAL + timedelta(days=7), STRIKE,
                       NOTIONAL, ProtectionDirection.BUYER)

    def test_a_FORWARD_START_at_or_after_maturity_is_refused(self) -> None:
        with pytest.raises(ValueError, match="before the maturity"):
            ForwardCDS(CouponSchedule(FULL_DATES, DC), FULL_DATES[-1], STRIKE, NOTIONAL,
                       ProtectionDirection.BUYER)

    def test_an_EXPIRY_that_is_not_a_schedule_date_is_refused(self) -> None:
        with pytest.raises(ValueError, match="schedule date"):
            CreditDefaultSwaption(CouponSchedule(FULL_DATES, DC), VAL + timedelta(days=7),
                                  STRIKE, NOTIONAL, SwaptionDirection.PAYER)

    def test_a_NEGATIVE_vol_is_refused(self) -> None:
        with pytest.raises(ValueError, match="volatility"):
            CreditSwaptionModel(_credit(), -0.1)

    def test_a_NON_POSITIVE_STRIKE_is_refused(self) -> None:
        """Black takes `log(F/K)`, so a zero or negative strike is not a wide answer but an
        undefined one — refused in the product's vocabulary rather than raised by `math.log`."""
        with pytest.raises(ValueError, match="strike"):
            CreditDefaultSwaption(CouponSchedule(FULL_DATES, DC), EXPIRY, 0.0, NOTIONAL,
                                  SwaptionDirection.PAYER)

    def test_a_NON_POSITIVE_forward_annuity_is_refused(self) -> None:
        """`S(t,tF,T)` divides by it, so a degenerate forward window must fail, not return inf."""
        same = _legs(MATURITY_YEARS)
        with pytest.raises(ValueError, match="forward annuity"):
            forward_spread(same, same)
