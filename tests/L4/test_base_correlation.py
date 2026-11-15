"""Base correlation (L4) — `OKANE-2008` ch 20.2-20.3. Pre-registered RED, tranche 44b.

The market quotes index tranches by **base correlation**: the single `ρ_B(K)` that, applied to the
**equity `[0,K]`** tranche, reprices the quote. The curve is bootstrapped shortest-strike-first, and
a `[K₁,K₂]` tranche is the **difference of two base-correlation equity tranches**:

    EL([K₁,K₂]) = EL_{ρ_B(K₂)}([0,K₂]) − EL_{ρ_B(K₁)}([0,K₁])

**The no-arbitrage monotonicity is a THEOREM here, not a test, and saying so is the point.**
`EL([K₁,K₂]) > 0 ⟺ Q < 1 and decreasing ⟺ protection > 0 ⟺ par > 0` — exactly. So *every positive
quote* bootstraps to a monotone curve by construction: measured down to a **1bp** mezzanine quote
and still monotone. "Refuse a non-monotone positive quote set" would ask the suite to reject
something that cannot exist — an oracle that cannot fail, on the spec's own headline requirement.
It is asserted below as the theorem, over extreme quotes, with the implication chain written out, so
the suite documents *why* it holds rather than appearing to check it.

**The non-vacuous guard is the other side of that iff: a quote with no base correlation at all.**
Given `ρ_B(3%) = 0.25`, the `[3%,7%]` par spread reachable over `ρ ∈ (0.02, 0.98)` is
`[−729, +1467] bp`; a 1600bp quote has **no root**. That is refused in the bootstrap's own domain
vocabulary before the root-finder sees it — `bracketed_root`'s contract says a non-straddling bracket
is the caller's bug, and the caller is expected to have refused an out-of-range target first.

**Interpolation is deliberately absent.** Every tranche priced here is bounded by *quoted* strikes,
so an unquoted strike is refused rather than interpolated: picking a convention nothing consumes
would be speculative (§6), and interpolating silently would be a plausible wrong number (§3.4).
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine.base_correlation import (
    BaseCorrelationCurve,
    HomogeneousPool,
    base_tranche_expected_loss,
    base_tranche_par_spread,
    bootstrap_base_correlation,
)
from pricebook.foundation import CouponSchedule, Currency, Money
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.survival_curve import ReferenceEntity, SurvivalCurve
from pricebook.models.lhp import LHPModel
from pricebook.models.tranche_loss import tranche_expected_loss
from pricebook.products.synthetic_cdo_tranche import TrancheBounds, TrancheQuote

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
EUR = Currency("EUR")
DISCOUNT_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
CURVE = DiscountCurve(TM, (0.0, 1.0, 2.0, 3.0, 5.0), (0.020, 0.026, 0.030, 0.033, 0.036))
INDEX = ReferenceEntity("CDX-IG")
POOL = SurvivalCurve(TM, (0.0, 1.0, 3.0), (0.009, 0.016, 0.028))
RECOVERY = 0.40
NOTIONAL = Money(10_000_000.0, EUR)
HORIZON = 5.0
#: Semi-annual AND **straddling the valuation date** — §4's tenth degenerate shape. The first
#: version of this fixture began *at* `VAL`, so no period was historical and the invariant-6 filter
#: in `base_tranche_par_spread` had nothing to filter: dropping the filter survived the whole file.
#: I had told Cowork this fixture straddled; it did not, and only re-deriving the claim found it.
DATES = tuple(VAL + timedelta(days=round(182.5 * i)) for i in range(-2, 11))
SCHEDULE = CouponSchedule(DATES, DC)

#: The capital structure the market quotes, and a deliberately **NON-FLAT** base-correlation curve
#: behind it — a flat one makes every strike's `ρ_B` equal and hides a per-strike bug (the T8
#: flat-curve lesson).
STRIKES = (0.03, 0.07, 0.15)
TRUE_CORRELATIONS = (0.25, 0.45, 0.62)


def _market() -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({DISCOUNT_KEY: CURVE}, survivals={INDEX: POOL}))


def _pool() -> HomogeneousPool:
    """Which pool, on what curves — bundled because §7's limit compels it (the bootstrap would
    otherwise take six arguments) and because the four always travel together."""
    return HomogeneousPool(_market(), INDEX, RECOVERY, EUR)


def _known_curve() -> BaseCorrelationCurve:
    """The curve the synthetic quotes are generated from — the round trip's independent truth."""
    return BaseCorrelationCurve(_pool(), STRIKES, TRUE_CORRELATIONS)


def _synthetic_quotes() -> tuple[TrancheQuote, ...]:
    """Par spreads off the known curve, so the round trip has something independent to recover."""
    curve = _known_curve()
    quotes = []
    lower = 0.0
    for strike in STRIKES:
        bounds = TrancheBounds(lower, strike)
        quotes.append(TrancheQuote(bounds, base_tranche_par_spread(curve, bounds, SCHEDULE)))
        lower = strike
    return tuple(quotes)


class TestTheFixtureIsNotDegenerate:
    def test_there_are_at_least_TWO_quoted_strikes(self) -> None:
        """One strike is the reduction, not a curve — the bootstrap must be read beyond its first."""
        assert len(STRIKES) >= 2

    def test_the_base_correlation_curve_is_NON_FLAT_and_RISING(self) -> None:
        assert len(set(TRUE_CORRELATIONS)) == len(TRUE_CORRELATIONS)
        assert list(TRUE_CORRELATIONS) == sorted(TRUE_CORRELATIONS)

    def test_the_schedule_STRADDLES_the_valuation_date(self) -> None:
        """§4's tenth shape, asserted as the premise rather than assumed.

        Without a historical period the invariant-6 filter in `base_tranche_par_spread` has nothing
        to filter, and dropping it survives every other oracle in this file — measured. This is the
        assertion whose absence made the first version of the fixture degenerate."""
        historical = [pay for _, pay, _ in SCHEDULE.periods() if pay <= VAL]
        assert len(historical) >= 2, historical
        live = [pay for _, pay, _ in SCHEDULE.periods() if pay > VAL]
        assert len(live) >= 2, "and there must still be a live leg to price"

    def test_the_SETTLED_premiums_are_EXCLUDED_from_the_annuity(self) -> None:
        """Invariant 6 on this module's leg, which the straddling fixture makes reachable."""
        from pricebook.engine.base_correlation import _BaseTrancheSurvival
        from pricebook.market.credit_legs import risky_pv01

        curve = _known_curve()
        bounds = TrancheBounds(0.03, 0.07)
        survival = _BaseTrancheSurvival(curve, bounds)
        live = [p for p in SCHEDULE.periods() if p[1] > VAL]
        whole = risky_pv01(SCHEDULE.periods(), CURVE, survival)
        want = risky_pv01(live, CURVE, survival)
        assert whole / want - 1.0 > 0.2, "the premise: the settled periods are material"
        # the par spread must be built on the live annuity, so a wider-than-par quote results
        from pricebook.engine.base_correlation import base_tranche_par_spread

        on_live = base_tranche_par_spread(curve, bounds, SCHEDULE)
        assert on_live == pytest.approx(
            base_tranche_par_spread(curve, bounds, CouponSchedule(
                tuple(d for d in DATES if d >= VAL), DC)), rel=1e-09)

    def test_the_quotes_are_a_REALISTIC_SKEW(self) -> None:
        """Equity far wider than mezzanine far wider than senior — measured 4254 / 661 / 131 bp."""
        spreads = [quote.spread for quote in _synthetic_quotes()]
        # measured 4254 / 661 / 131 bp — ratios of 6.4x and 5.1x, which is the shape of a real
        # index skew and the reason a flat correlation curve cannot reproduce one
        assert spreads[0] > 5.0 * spreads[1] > 0.0, spreads
        assert spreads[1] > 4.0 * spreads[2] > 0.0, spreads


class TestTheBootstrapRoundTrip:
    """The correctness anchor: a calibration must invert (the T8/T39 round-trip)."""

    def test_the_KNOWN_curve_is_RECOVERED_from_its_own_quotes(self) -> None:
        """Non-telescoping: an independently constructed curve in, the curve out. Measured at
        6.7e-16 — the bootstrap is an exact inverse, not an approximation."""
        bootstrapped = bootstrap_base_correlation(_synthetic_quotes(), _pool(), SCHEDULE)
        assert bootstrapped.strikes == STRIKES
        for strike, got, want in zip(STRIKES, bootstrapped.correlations, TRUE_CORRELATIONS):
            assert got == pytest.approx(want, rel=1e-12), (strike, got, want)

    def test_the_bootstrapped_curve_REPRICES_the_quotes(self) -> None:
        """The other direction of the same inverse, and the one a desk would check."""
        quotes = _synthetic_quotes()
        curve = bootstrap_base_correlation(quotes, _pool(), SCHEDULE)
        for quote in quotes:
            assert base_tranche_par_spread(curve, quote.bounds, SCHEDULE) == pytest.approx(
                quote.spread, rel=1e-10)

    def test_a_SINGLE_quoted_strike_reduces_to_the_LHP(self) -> None:
        """One `[0,K]` quote at correlation `ρ` bootstraps to exactly `ρ` — the bootstrap of a
        single strike is the LHP inverse, so this is a cross-tranche reduction onto tranche 43."""
        single = TrancheBounds(0.0, 0.03)
        model = LHPModel(_market(), INDEX, 0.31, RECOVERY)
        quote = TrancheQuote(
            single,
            base_tranche_par_spread(
                BaseCorrelationCurve(_pool(), (0.03,), (0.31,)), single, SCHEDULE))
        curve = bootstrap_base_correlation((quote,), _pool(), SCHEDULE)
        assert curve.correlations[0] == pytest.approx(0.31, rel=1e-12)
        # and the equity tranche's expected loss is tranche 43's, on the same model
        assert base_tranche_expected_loss(curve, 0.0, 0.03, HORIZON) == pytest.approx(
            tranche_expected_loss(model, 0.0, 0.03, HORIZON), rel=1e-11)


class TestTheMonotonicityIsATheoremNotATest:
    """§4's state-it-plainly discipline, applied to a theorem.

    The spec asked for a refusal of a non-monotone positive quote set. There is no such thing: the
    implication chain below is exact, so the suite would be rejecting something that cannot exist —
    an oracle that cannot fail. What it does instead is **assert the theorem** and say why.
    """

    def test_the_IMPLICATION_CHAIN_that_makes_monotonicity_automatic(self) -> None:
        """`EL([K₁,K₂]) > 0 ⟺ Q < 1 and decreasing ⟺ protection > 0 ⟺ par > 0`.

        Asserted link by link on a real quote set, so the reasoning is in the suite rather than in
        a comment: this is *why* a positive quote cannot produce an arbitrageable curve."""
        quotes = _synthetic_quotes()
        curve = bootstrap_base_correlation(quotes, _pool(), SCHEDULE)
        for quote in quotes:
            attachment, detachment = quote.bounds.attachment, quote.bounds.detachment
            expected_loss = base_tranche_expected_loss(
                curve, attachment, detachment, HORIZON)
            assert quote.spread > 0.0
            assert expected_loss > 0.0, (attachment, detachment, expected_loss)
            outstanding = 1.0 - expected_loss / quote.bounds.width
            assert outstanding < 1.0

    @pytest.mark.parametrize("mezzanine_bp", [1.0, 50.0, 661.0, 1400.0])
    def test_EVERY_positive_quote_bootstraps_MONOTONICALLY(self, mezzanine_bp: float) -> None:
        """The theorem, over extreme quotes — a 1bp mezzanine is as monotone as a 1400bp one.

        This is what base correlation buys and compound correlation cannot promise: there is always
        a consistent curve for a positive quote set."""
        quotes = _synthetic_quotes()
        stressed = (quotes[0], TrancheQuote(quotes[1].bounds, mezzanine_bp / 1e4), quotes[2])
        curve = bootstrap_base_correlation(stressed[:2], _pool(), SCHEDULE)
        equity_loss = base_tranche_expected_loss(curve, 0.0, 0.03, HORIZON)
        mezzanine_loss = base_tranche_expected_loss(curve, 0.03, 0.07, HORIZON)
        assert mezzanine_loss > 0.0, mezzanine_loss
        # the base losses are increasing in the strike, which is the monotonicity itself
        assert (equity_loss + mezzanine_loss) > equity_loss > 0.0


class TestAQuoteWithNoBaseCorrelationIsREFUSED:
    """The non-vacuous guard — the other side of the iff."""

    def test_an_UNREACHABLE_quote_is_refused_in_DOMAIN_vocabulary(self) -> None:
        """Measured: given `ρ_B(3%) = 0.25`, the `[3%,7%]` spread reachable over `ρ ∈ (0.02,0.98)`
        is `[−729, +1467] bp`. A 1600bp quote has no base correlation at all, and the bootstrap
        must say *that* — not surface `bracketed_root`'s "the bracket does not straddle", which is
        the root-finder reporting a caller's bug (its own docstring says so)."""
        quotes = _synthetic_quotes()
        unreachable = (quotes[0], TrancheQuote(quotes[1].bounds, 0.16))
        with pytest.raises(ValueError, match="no base correlation"):
            bootstrap_base_correlation(unreachable, _pool(), SCHEDULE)

    def test_the_refusal_REPORTS_THE_RANGE_which_only_the_bootstrap_knows(self) -> None:
        """A second killer for the reachability guard, and the reason it exists at all.

        Dropping the guard still raises — `bracketed_root` refuses a non-straddling bracket — so a
        test that only checked "it raises" would pass. What distinguishes our refusal is that it is
        **actionable**: it names the admissible correlation range and the spread interval the
        tranche actually spans, which the root-finder cannot know and which tells a desk whether to
        re-quote or to look at the strike below. The root-finder's own message describes a caller's
        bug in the caller's absence."""
        quotes = _synthetic_quotes()
        unreachable = (quotes[0], TrancheQuote(quotes[1].bounds, 0.16))
        with pytest.raises(ValueError) as caught:
            bootstrap_base_correlation(unreachable, _pool(), SCHEDULE)
        message = str(caught.value)
        assert "no base correlation" in message
        assert "3.00%" in message and "7.00%" in message, "the tranche is named"
        assert "spans" in message, "and the reachable interval is reported"
        assert "bracket" not in message, "the root-finder's vocabulary must not leak"

    def test_THE_PREMISE_the_same_strike_accepts_a_REACHABLE_quote(self) -> None:
        """So the refusal is about the quote and not the fixture's shape."""
        quotes = _synthetic_quotes()
        reachable = (quotes[0], TrancheQuote(quotes[1].bounds, 0.1400))
        curve = bootstrap_base_correlation(reachable, _pool(), SCHEDULE)
        assert 0.0 < curve.correlations[1] < 1.0

    def test_a_NON_POSITIVE_quote_is_refused(self) -> None:
        """A zero or negative spread implies a tranche that loses nothing, or gains — which is the
        configuration where monotonicity *would* fail, so it is refused at the door."""
        quotes = _synthetic_quotes()
        with pytest.raises(ValueError, match="positive"):
            bootstrap_base_correlation(
                (quotes[0], TrancheQuote(quotes[1].bounds, 0.0)), _pool(), SCHEDULE)


class TestTheGuardsAreReachable:
    def test_an_UNQUOTED_strike_is_REFUSED_not_interpolated(self) -> None:
        """Interpolation is a modelling choice the quotes do not pin and nothing in scope needs,
        so an unquoted strike fails rather than silently inventing a correlation (§3.4, §6)."""
        curve = _known_curve()
        with pytest.raises(KeyError, match="not a quoted strike"):
            curve.correlation_at(0.05)

    def test_a_QUOTED_strike_resolves(self) -> None:
        curve = _known_curve()
        for strike, correlation in zip(STRIKES, TRUE_CORRELATIONS):
            assert curve.correlation_at(strike) == correlation

    def test_quotes_OUT_OF_STRIKE_ORDER_cannot_be_EXPRESSED(self) -> None:
        """There is no order check, because it would be unreachable — and that is worth asserting.

        A first draft of the bootstrap had one. Contiguity forces `attachment ==
        previous_detachment` and `TrancheBounds` forces `detachment > attachment`, so the
        detachments increase *already*: `detachment <= strikes[-1]` cannot hold. Two guards
        implying a third makes the third dead code (the T15 rule), so it was deleted and this test
        pins the implication instead — an out-of-order quote set is refused, just not by the guard
        a reader might expect."""
        quotes = _synthetic_quotes()
        with pytest.raises(ValueError, match="equity"):
            bootstrap_base_correlation((quotes[1], quotes[0]), _pool(), SCHEDULE)
        # and swapping the two later strikes breaks contiguity, which is the guard that carries it
        with pytest.raises(ValueError, match="contiguous"):
            bootstrap_base_correlation((quotes[0], quotes[2], quotes[1]), _pool(), SCHEDULE)

    def test_a_GAPPED_capital_structure_is_refused(self) -> None:
        """Each quote must attach where the previous detached, or the bootstrap has no base
        correlation to subtract and would silently treat a gap as absorbed loss."""
        with pytest.raises(ValueError, match="attach"):
            bootstrap_base_correlation(
                (TrancheQuote(TrancheBounds(0.0, 0.03), 0.42),
                 TrancheQuote(TrancheBounds(0.05, 0.07), 0.06)), _pool(), SCHEDULE)

    def test_an_EMPTY_quote_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            bootstrap_base_correlation((), _pool(), SCHEDULE)

    def test_the_FIRST_quote_must_be_an_EQUITY_tranche(self) -> None:
        """The bootstrap starts from `[0,K]`: there is nothing below it to subtract."""
        with pytest.raises(ValueError, match="equity"):
            bootstrap_base_correlation(
                (TrancheQuote(TrancheBounds(0.03, 0.07), 0.06),), _pool(), SCHEDULE)
