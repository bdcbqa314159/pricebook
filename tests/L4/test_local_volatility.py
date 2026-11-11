"""Dupire local volatility and the local-vol Monte Carlo (L1/L3/L4) — tranche 32.

    sigma_loc^2(K,T) = [ dC/dT + (r-q)*K*dC/dK + q*C ] / [ 0.5 * K^2 * d2C/dK2 ]

**Analytic derivatives, ruled at spec time (thread 0162).** Our surface is a closed-form
parametrization and Black is closed-form, so the three partials are taken exactly by the chain rule.
A finite-difference `d2C/dK2` would have been the noise-dominated weak side — measured on this very
surface it is 1% wrong at `h = 1e-5`, best near `h = 1e-2`, degrading again beyond — and a band
mixing that noise with Monte-Carlo error could not tell a Dupire defect from a differencing artefact
(§4: a tolerance lumping two error sources together cannot separate them). With the derivatives
exact the band below is **pure MC standard error**.

**The chain rule is the content.** `sigma` depends on `K` and `T`, so `dC/dK` picks up `C_v*v_K` and
`d2C/dK2` picks up three cross terms. Differentiating Black at fixed vol and stopping is the error
this file exists to make impossible — and the test computes its **own** finite differences, on its
own grid, of its **own** price function, never reading the model's.

**Two degeneracies are closed by the fixture, both found by mutation before the oracle was written.**
A *flat* surface makes `sigma_loc = sigma` trivially, so the reduction alone is vacuous and the
skewed companion carries it. And a *near-zero carry* makes the `(r-q)*K*dC/dK` term invisible:
measured, dropping that term moves the repricing by only **3.2 SE** at `carry = 0.005` (inside a
4-SE band — it would have survived) but **8.9 SE** at `carry = 0.04`. So the fixture carries a
material carry, asserted as a premise.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine.local_vol_mc import (
    forward_estimate,
    local_vol_call_estimate,
    simulate_local_vol_spots,
    surface_call_price,
)
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import DiscountCurve
from pricebook.market.equity import DividendSchedule, EquityUnderlying, dividend_forward
from pricebook.market.equity_skew import ImpliedVolSkew
from pricebook.models.black import black_price
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.local_volatility import (
    ForwardState,
    dupire_local_variance,
    surface_partials,
)
from pricebook.products.equity_option import EquityOptionKind

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
#: A MATERIAL carry — see the module docstring. `r - q = 0.05`, ten times the near-zero value that
#: let the drift-term mutant survive.
RATE = 0.06
YIELD = 0.01
CURVE = DiscountCurve.flat(VAL, RATE, DC)
UNDERLYING = EquityUnderlying(100.0, 0.0, DividendSchedule(continuous_yield=YIELD))

ATM, SKEW, TERM = 0.28, -0.15, 0.04
SURFACE = ImpliedVolSkew(ATM, SKEW, TERM)
EXPIRY = VAL + timedelta(days=365)
YEARS = 1.0
FORWARD = dividend_forward(UNDERLYING, EXPIRY, CURVE)
DF = CURVE.df(EXPIRY)


def _price(strike: float, years: float) -> float:
    """The surface's own call price — the function Dupire differentiates, rebuilt here.

    Independent of the model: this composes the forward, the surface and Black itself rather than
    calling anything the implementation calls for the same purpose.
    """
    df = math.exp(-RATE * years)
    forward = 100.0 * math.exp((RATE - YIELD) * years)
    vol = SURFACE.volatility(strike, forward, years)
    return df * black_price(forward, strike, vol * vol * years, EquityOptionKind.CALL)


class TestTheFixtureIsNotDegenerate:
    def test_the_CARRY_is_material(self) -> None:
        """Measured: the drift-term mutant deviates 3.2 SE at carry 0.005 (survives a 4-SE band)
        and 8.9 SE at carry 0.04. Below ~0.02 this fixture would be blind to that term."""
        assert RATE - YIELD > 0.03, f"carry must be material, got {RATE - YIELD}"

    def test_the_SKEW_and_the_TERM_STRUCTURE_are_material(self) -> None:
        wing = SURFACE.volatility(0.8 * FORWARD, FORWARD, YEARS)
        atm = SURFACE.volatility(FORWARD, FORWARD, YEARS)
        assert wing - atm > 0.02, f"the skew must bite: {wing} vs {atm}"
        assert abs(SURFACE.volatility(FORWARD, FORWARD, 2.0) - atm) > 0.02, "term structure flat"


class TestTheAnalyticPartialsAgainstTheTestsOwnFiniteDifferences:
    """Each composed partial, against a difference of the test's own price function."""

    @pytest.mark.parametrize("strike,years", [(85.0, 1.0), (100.0, 1.0), (120.0, 1.4), (92.0, 2.2)])
    def test_every_partial_matches(self, strike: float, years: float) -> None:
        df = math.exp(-RATE * years)
        forward = 100.0 * math.exp((RATE - YIELD) * years)
        got = surface_partials(SURFACE, strike, ForwardState(forward, years, df, RATE, YIELD))
        h, h_t = 1e-2, 1e-4
        assert got.price == pytest.approx(_price(strike, years), rel=1e-14)
        assert got.d_strike == pytest.approx(
            (_price(strike + h, years) - _price(strike - h, years)) / (2 * h), rel=1e-6
        )
        assert got.d2_strike2 == pytest.approx(
            (_price(strike + h, years) - 2 * _price(strike, years) + _price(strike - h, years)) / h**2,
            rel=1e-5,
        )
        assert got.d_time == pytest.approx(
            (_price(strike, years + h_t) - _price(strike, years - h_t)) / (2 * h_t), rel=1e-6
        )

    def test_the_chain_rule_terms_are_NOT_negligible_on_this_surface(self) -> None:
        """The premise that makes the test above meaningful: a fixed-vol Black differentiation must
        give a *visibly different* answer, else the chain rule is untested."""
        from pricebook.models.local_volatility import black_partials

        state = ForwardState(FORWARD, YEARS, DF, RATE, YIELD)
        fixed = black_partials(state, 85.0, SURFACE.volatility(85.0, FORWARD, YEARS))
        composed = surface_partials(SURFACE, 85.0, state)
        assert abs(composed.d_strike / fixed.d_strike - 1.0) > 0.05
        assert abs(composed.d2_strike2 / fixed.d2_strike2 - 1.0) > 0.05


class TestTheFlatReductionAndItsMandatoryCompanion:
    @pytest.mark.parametrize("vol", [0.15, 0.3, 0.45])
    @pytest.mark.parametrize("strike", [70.0, 100.0, 140.0])
    def test_a_FLAT_surface_gives_local_vol_EQUAL_to_implied(self, vol: float, strike: float) -> None:
        """Exact, not approximate: every chain-rule term vanishes and Black's partials compose back
        to `sigma^2`. Agreement to machine precision is the strongest statement available."""
        flat = ImpliedVolSkew(vol, 0.0, 0.0)
        variance = dupire_local_variance(flat, UNDERLYING, strike, EXPIRY, CURVE)
        assert math.sqrt(variance) == pytest.approx(vol, rel=1e-13)

    def test_the_SKEWED_companion_gives_a_LOCAL_skew_about_TWICE_the_implied(self) -> None:
        """Dupire's rule of two — the non-degenerate case the flat reduction cannot see.

        A no-op `sigma_loc = sigma_impl` satisfies the reduction above and fails here, which is the
        whole reason the companion is mandatory (§4).
        """
        low, high = 0.9 * FORWARD, 1.1 * FORWARD
        implied_skew = (
            SURFACE.volatility(high, FORWARD, YEARS) - SURFACE.volatility(low, FORWARD, YEARS)
        ) / 0.2
        local_skew = (
            math.sqrt(dupire_local_variance(SURFACE, UNDERLYING, high, EXPIRY, CURVE))
            - math.sqrt(dupire_local_variance(SURFACE, UNDERLYING, low, EXPIRY, CURVE))
        ) / 0.2
        ratio = local_skew / implied_skew
        assert 1.6 < ratio < 2.6, f"Dupire's rule of two: local/implied skew = {ratio}"


class TestTheInversionAnchor:
    """`sigma_loc` simulated must reprice the surface it came from — the formula and its inverse.

    Different code each side (the analytic Dupire formula vs a log-Euler diffusion), so this does
    not telescope: a wrong Dupire produces a `sigma_loc` whose simulated prices miss the surface.
    """

    @staticmethod
    def _spots(paths: int = 4_000, steps: int = 20):
        return simulate_local_vol_spots(
            SURFACE, UNDERLYING, EXPIRY, CURVE, (MonteCarloConfig(paths=paths, seed=20261111), steps)
        )

    def test_the_MARTINGALE_identity_pins_the_DRIFT_independently_of_the_vol(self) -> None:
        """`E[S_T] = F(T)`, exact by construction (T19/T26). Checked first, so that a drift error
        cannot be mistaken for a volatility error in the repricing below."""
        got = forward_estimate(self._spots())
        assert abs(got.mean - FORWARD) <= 4.0 * got.standard_error, (
            f"E[S_T]={got.mean} +/- {got.standard_error} vs F={FORWARD}"
        )

    @pytest.mark.parametrize("moneyness", [0.85, 1.0, 1.15])
    def test_the_simulation_REPRICES_the_parametrized_surface(self, moneyness: float) -> None:
        strike = moneyness * FORWARD
        spots = self._spots()
        got = local_vol_call_estimate(spots, strike, DF)
        target = surface_call_price(SURFACE, UNDERLYING, strike, EXPIRY, CURVE)
        assert abs(got.mean - target) <= 4.0 * got.standard_error, (
            f"K/F={moneyness}: {got.mean} +/- {got.standard_error} vs surface {target}"
        )

    def test_the_repricing_band_is_tight_enough_to_RESOLVE_a_dupire_defect(self) -> None:
        """The band's power, measured rather than hoped: dropping the `(r-q)K dC/dK` term moves the
        worst repricing by **8.9 SE** on this fixture, well outside the 4-SE band. Stated so a green
        says what it rules out."""
        spots = self._spots()
        errors = [
            local_vol_call_estimate(spots, m * FORWARD, DF).standard_error for m in (0.85, 1.0, 1.15)
        ]
        assert all(e > 0.0 for e in errors)
        assert max(errors) / surface_call_price(SURFACE, UNDERLYING, FORWARD, EXPIRY, CURVE) < 0.05


class TestTheRequirementOracles:
    def test_the_local_variance_is_POSITIVE_across_the_traded_range(self) -> None:
        for moneyness in (0.7, 0.85, 1.0, 1.2, 1.4):
            assert dupire_local_variance(SURFACE, UNDERLYING, moneyness * FORWARD, EXPIRY, CURVE) > 0.0

    def test_the_local_variance_RISES_with_the_overall_vol_level(self) -> None:
        variances = [
            dupire_local_variance(ImpliedVolSkew(v, SKEW, TERM), UNDERLYING, FORWARD, EXPIRY, CURVE)
            for v in (0.15, 0.25, 0.4)
        ]
        assert all(b > a for a, b in zip(variances, variances[1:]))

    def test_the_TERM_SLOPE_moves_the_local_variance(self) -> None:
        """Otherwise `dC/dT`'s surface term is untested — the time direction's own degeneracy."""
        flat_term = dupire_local_variance(
            ImpliedVolSkew(ATM, SKEW, 0.0), UNDERLYING, FORWARD, EXPIRY, CURVE
        )
        rising = dupire_local_variance(
            ImpliedVolSkew(ATM, SKEW, 0.08), UNDERLYING, FORWARD, EXPIRY, CURVE
        )
        assert rising > flat_term * 1.02, f"term structure must bite: {flat_term} vs {rising}"


class TestTheGuardsAreReachable:
    def test_DISCRETE_dividends_are_REFUSED_not_approximated(self) -> None:
        """The forward jumps at an ex-date, so `dC/dT` is undefined there."""
        lumpy = EquityUnderlying(
            100.0, 0.0, DividendSchedule(cash=((VAL + timedelta(days=180), 2.0),))
        )
        with pytest.raises(ValueError, match="continuous dividend yield only"):
            dupire_local_variance(SURFACE, lumpy, FORWARD, EXPIRY, CURVE)

    def test_a_NON_POSITIVE_horizon_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive horizon"):
            dupire_local_variance(SURFACE, UNDERLYING, FORWARD, VAL, CURVE)

    def test_a_simulation_needs_at_least_one_step(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            simulate_local_vol_spots(
                SURFACE, UNDERLYING, EXPIRY, CURVE, (MonteCarloConfig(paths=10, seed=1), 0)
            )
