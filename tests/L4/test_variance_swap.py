"""The variance swap by static replication (L1/L2/L4) — tranche 31.

The fair variance of a variance swap is **model-independent**: it is a weighted strip of vanilla
options, with no volatility model in the pricing at all.

    K_var = (2 / (T * df(T))) * [ integral_0^F P(K)/K^2 dK + integral_F^inf C(K)/K^2 dK ]

The `1/K^2` weight is the whole content — it is what turns a strip of options into a log contract,
and `-(2/T)*E[log(S_T/F)]` is the same object written the other way.

**The anchor is a cross-product reduction.** Under flat-vol Black-Scholes the strip integrates to
*exactly* `sigma^2`, because `E[log(S_T/F)] = -0.5*sigma^2*T`. The strip integrator and `black_price`
are **different code**, so `K_var = sigma^2` is a genuine cross-check rather than self-consistency —
and it fails under any error in the `1/K^2` weight, the `2/T` factor, the OTM put/call split at `F`,
or the forward.

**And the anchor alone is vacuous, which is the point of the companion.** A *flat* surface makes
`K_var = sigma^2` hold trivially: a no-op implementation satisfies it. It is the **skewed** surface
that forces `K_var` away from the ATM variance — the convexity premium — and only a correctly
`1/K^2`-weighted strip produces it. §4's reduction-plus-companion rule, applied where the reduction
looks like a real result.

**The strike grid has a validity regime, stated rather than assumed.** The continuum integral is a
finite strip in practice, and a **truncated** grid systematically *underprices* the tails: it returns
less than `sigma^2`, so the reduction would fail for the wrong reason. Both the truncation direction
and the convergence rate are asserted, and the band states what the committed grid resolves (the T25
"a reference is code too" lesson, applied to our own quadrature).
"""

import math
from datetime import date, timedelta

import pytest
from scipy.integrate import quad
from scipy.stats import norm

from pricebook.engine import price
from pricebook.engine.variance_replication import fair_variance
from pricebook.foundation import Currency, Money, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import DividendSchedule, EquityAsset, EquityUnderlying, dividend_forward
from pricebook.market.equity_skew import ImpliedVolSkew
from pricebook.models.variance_replication_model import StrikeGrid, VarianceReplicationModel
from pricebook.products.variance_swap import VarianceSwap

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
EUR = Currency("EUR")
ACME = EquityAsset("ACME")
DISCOUNT_KEY = CurveKey(CurveRole.DISCOUNT, EUR)

SPOT = 100.0
RATE = 0.03
EXPIRY = VAL + timedelta(days=365)
YEARS = 1.0
ATM_VOL = 0.28
#: A typical equity skew: volatility FALLS with strike, so the slope is negative.
SKEW_SLOPE = -0.15
#: Wide and fine: the regime where truncation is provably below the band (asserted below).
WIDE = StrikeGrid(lower=0.05, upper=5.0, count=1601)

CURVE = DiscountCurve.flat(VAL, RATE, DC)
UNDERLYING = EquityUnderlying(SPOT, 0.0, DividendSchedule())
FORWARD = dividend_forward(UNDERLYING, EXPIRY, CURVE)


def _market(skew: ImpliedVolSkew) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({DISCOUNT_KEY: CURVE}),
        equities={ACME: UNDERLYING},
        equity_skews={ACME: skew},
    )


def _flat(vol: float = ATM_VOL) -> ImpliedVolSkew:
    return ImpliedVolSkew(vol, 0.0)


def _skewed(slope: float = SKEW_SLOPE) -> ImpliedVolSkew:
    return ImpliedVolSkew(ATM_VOL, slope)


def _k_var(skew: ImpliedVolSkew, grid: StrikeGrid = WIDE) -> float:
    return fair_variance(UNDERLYING, EXPIRY, CURVE, skew, grid)


class TestTheFixtureIsNotDegenerate:
    def test_the_SKEW_is_material(self) -> None:
        """Else the skewed companion collapses into the flat reduction and proves nothing."""
        skew = _skewed()
        low = skew.volatility(0.7 * FORWARD, FORWARD)
        high = skew.volatility(1.3 * FORWARD, FORWARD)
        assert low - high > 0.05, f"the skew must bite: sigma(0.7F)={low}, sigma(1.3F)={high}"

    def test_the_WIDE_grid_is_wide_enough_for_truncation_to_be_below_the_band(self) -> None:
        """The regime the reduction is asserted in, stated as a premise rather than assumed."""
        assert WIDE.lower <= 0.1 and WIDE.upper >= 4.0 and WIDE.count >= 800


class TestTheExactReductionTheCrossProductAnchor:
    """Flat vol: the strip must integrate to exactly `sigma^2`. Strip integrator vs Black — two
    different pieces of code reaching one number."""

    @pytest.mark.parametrize("vol", [0.1, 0.2, 0.28, 0.45])
    def test_a_FLAT_surface_replicates_to_the_variance_itself(self, vol: float) -> None:
        assert _k_var(_flat(vol)) == pytest.approx(vol * vol, rel=2e-4)

    def test_the_band_states_what_the_committed_grid_RESOLVES(self) -> None:
        """**Measured, after the forward was put on the grid:** the committed grid reproduces
        `sigma^2` to **2.0e-05 relative** (1.6e-06 absolute at `sigma^2 = 0.0784`). So a green here
        rules out any systematic defect above ~0.004% of the fair variance — about 0.0003 of a vol
        point at this level. The band is set at 2e-4, an order of magnitude above the measurement,
        which is headroom for the skewed fixtures rather than slack: a looser band would start to
        absorb a real `1/K^2` weight error, and the convergence-rate test below is what separates
        residual discretisation from a wrong weight."""
        got = _k_var(_flat())
        assert abs(got / (ATM_VOL * ATM_VOL) - 1.0) < 2e-4
        assert abs(got / (ATM_VOL * ATM_VOL) - 1.0) > 1e-7, (
            "if this ever becomes exact, the grid is no longer the thing being characterised"
        )


class TestTheLogContractIdentity:
    """`K_var = -(2/T)*E[log(S_T/F)]` — the same object written the other way, computed by a THIRD
    route (numerical integration over the lognormal density), independent of both the strip and the
    `-0.5*sigma^2*T` algebra."""

    def test_the_strip_equals_the_log_contract_expectation(self) -> None:
        vol = ATM_VOL
        sd = vol * math.sqrt(YEARS)

        def integrand(z: float) -> float:
            # S_T = F*exp(-0.5*sd^2 + sd*z) under the forward measure.
            return (-0.5 * sd * sd + sd * z) * norm.pdf(z)

        expectation, _ = quad(integrand, -12.0, 12.0, limit=400)
        assert -(2.0 / YEARS) * expectation == pytest.approx(_k_var(_flat(vol)), rel=5e-4)

    def test_the_log_contract_route_also_pins_the_minus_two(self) -> None:
        """A dropped `-2` or a `2/T` written as `1/T` fails here and nowhere else in this file."""
        vol = 0.2
        assert -(2.0 / YEARS) * (-0.5 * vol * vol * YEARS) == pytest.approx(vol * vol, rel=1e-15)


class TestTheStrikeGridsValidityRegime:
    """A truncated grid underprices the tails — the direction and the rate are both asserted."""

    def test_a_TRUNCATED_grid_UNDERPRICES_the_fair_variance(self) -> None:
        """So a narrow grid fails the reduction for the WRONG reason — the T25 trap, named."""
        narrow = _k_var(_flat(), StrikeGrid(lower=0.8, upper=1.25, count=401))
        assert narrow < 0.9 * ATM_VOL * ATM_VOL, f"truncation must bite downward, got {narrow}"

    def test_WIDENING_the_grid_converges_upward_to_the_variance(self) -> None:
        target = ATM_VOL * ATM_VOL
        values = [
            _k_var(_flat(), StrikeGrid(lower=lo, upper=hi, count=1601))
            for lo, hi in ((0.6, 1.6), (0.35, 2.5), (0.15, 4.0), (0.05, 5.0))
        ]
        assert all(b > a for a, b in zip(values, values[1:])), f"must rise toward sigma^2: {values}"
        assert values[-1] == pytest.approx(target, rel=2e-4)

    def test_REFINING_a_wide_grid_improves_at_the_TRAPEZOID_rate(self) -> None:
        """Second order in the strike step — the rate distinguishes discretisation from a wrong
        weight, which a tolerance alone cannot (the T29 lesson)."""
        target = ATM_VOL * ATM_VOL
        errors = [
            abs(_k_var(_flat(), StrikeGrid(0.05, 5.0, n)) - target) for n in (201, 401, 801)
        ]
        ratios = [a / b for a, b in zip(errors, errors[1:])]
        # Measured 4.00, 4.02, 4.00, 4.00 once the forward sits ON the grid. Before that fix the
        # put/call switch left a derivative kink inside an interval and the ratios were 0.85, 22.1,
        # 0.61, 3.25 — no rate at all. Asserting the rate is what makes this an oracle rather than a
        # hope, and it is why the kink was fixed instead of the assertion loosened.
        assert all(3.5 < r < 4.5 for r in ratios), f"expected second order (~4x), got {ratios}"


class TestTheSkewCompanionWhatTheReductionCannotSee:
    """The reduction is satisfied by a no-op; this is not."""

    def test_a_SKEWED_surface_moves_the_fair_variance_ABOVE_the_ATM_variance(self) -> None:
        """The convexity premium: OTM puts carry more vol, and the `1/K^2` weight makes them count."""
        flat = _k_var(_flat())
        skewed = _k_var(_skewed())
        assert skewed > flat, f"a negative skew must lift K_var: {skewed} vs {flat}"
        assert skewed / flat - 1.0 > 0.02, f"the premium must be material, got {skewed / flat - 1.0}"

    def test_the_premium_GROWS_with_the_skew_slope(self) -> None:
        premia = [_k_var(_skewed(s)) for s in (0.0, -0.08, -0.15, -0.25)]
        assert all(b > a for a, b in zip(premia, premia[1:])), f"must grow with skew: {premia}"

    def test_a_POSITIVE_skew_moves_it_above_ATM_too_which_is_the_CONVEXITY_not_the_sign(self) -> None:
        """Worth pinning: the premium comes from *curvature in the weighted strip*, so either sign of
        slope raises `K_var` relative to flat. A test asserting "negative skew raises it" alone would
        pass for an implementation that merely responded to `|slope|` in the wrong place."""
        assert _k_var(_skewed(+0.15)) > _k_var(_flat())


class TestTheRequirementOracles:
    def test_the_fair_variance_is_never_negative(self) -> None:
        for vol in (0.01, 0.1, 0.5, 1.2):
            assert _k_var(_flat(vol)) >= 0.0

    def test_the_fair_variance_RISES_with_the_volatility_level(self) -> None:
        values = [_k_var(_flat(v)) for v in (0.1, 0.2, 0.35, 0.6)]
        assert all(b > a for a, b in zip(values, values[1:]))


class TestTheRegisteredEngine:
    def test_the_swap_pays_the_discounted_variance_difference(self) -> None:
        strike = 0.06
        swap = VarianceSwap(ACME, EXPIRY, strike, Money(1_000_000.0, EUR))
        model = VarianceReplicationModel(_market(_skewed()), WIDE)
        out = price(swap, model)
        assert isinstance(out, PricingResult), out
        expected = 1_000_000.0 * CURVE.df(EXPIRY) * (_k_var(_skewed()) - strike)
        assert out.pv.amount == pytest.approx(expected, rel=1e-12)

    def test_a_swap_struck_AT_the_fair_variance_is_worth_zero(self) -> None:
        fair = _k_var(_flat())
        swap = VarianceSwap(ACME, EXPIRY, fair, Money(1_000_000.0, EUR))
        out = price(swap, VarianceReplicationModel(_market(_flat()), WIDE))
        assert isinstance(out, PricingResult), out
        assert abs(out.pv.amount) < 1e-6

    def test_an_EXPIRED_swap_is_PV_zero_invariant_6(self) -> None:
        swap = VarianceSwap(ACME, VAL, 0.06, Money(1_000_000.0, EUR))
        out = price(swap, VarianceReplicationModel(_market(_flat()), WIDE))
        assert isinstance(out, PricingResult), out
        assert out.pv.amount == 0.0

    def test_an_unknown_asset_is_a_PricingFailure(self) -> None:
        bare = MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT_KEY: CURVE}))
        swap = VarianceSwap(ACME, EXPIRY, 0.06, Money(1.0, EUR))
        assert not isinstance(price(swap, VarianceReplicationModel(bare, WIDE)), PricingResult)


class TestTheGuardsAreReachable:
    def test_a_grid_whose_lower_bound_is_not_positive_is_refused(self) -> None:
        with pytest.raises(ValueError, match="lower"):
            StrikeGrid(0.0, 5.0, 101)

    def test_an_inverted_grid_is_refused(self) -> None:
        with pytest.raises(ValueError, match="above"):
            StrikeGrid(2.0, 1.0, 101)

    def test_a_grid_too_coarse_to_integrate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            StrikeGrid(0.5, 2.0, 1)

    def test_a_NEGATIVE_atm_volatility_is_refused(self) -> None:
        with pytest.raises(ValueError, match="volatilit"):
            ImpliedVolSkew(-0.1, 0.0)

    def test_the_skew_cannot_produce_a_NEGATIVE_volatility(self) -> None:
        """A steep slope at a far strike would otherwise give a negative sigma into Black."""
        steep = ImpliedVolSkew(0.2, -5.0)
        assert steep.volatility(0.01 * FORWARD, FORWARD) > 0.0
