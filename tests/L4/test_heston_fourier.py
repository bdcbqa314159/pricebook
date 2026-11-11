"""The Heston Fourier vanilla pricer (L4) — tranche 33. **Pre-registered before the pricer exists.**

    C = df(T) * [ F*P1 - K*P2 ],   P_j = 1/2 + (1/pi) * integral_0^inf Re[...] du

Every oracle below can be stated without the inversion integral existing, which is why they are
written first (the tranche-32 fix, thread 0169):

* **the `sigma_v -> 0` reduction** compares against `black_price`, verified since tranche 5;
* **put-call parity** is model-free;
* **`P1, P2` are probabilities**, so they lie in `[0,1]` whatever the quadrature does;
* **the no-arbitrage bounds** hold for any call.

The one oracle that cannot be pre-registered is the non-degenerate anchor — Fourier against an
independent **Heston Monte-Carlo** — because it needs the simulation. The spec rules that the
reduction alone is *vacuous* (a deterministic-variance Black satisfies it), so the MC is the
companion that makes the stochastic case real, and it lives in `test_heston_mc.py`.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine.heston_fourier import FourierGrid, heston_call_price, heston_probabilities
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import DiscountCurve
from pricebook.market.equity import DividendSchedule, EquityUnderlying, dividend_forward
from pricebook.models.black import black_price
from pricebook.models.heston import HestonParams
from pricebook.products.equity_option import EquityOptionKind

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
RATE, YIELD = 0.04, 0.015
CURVE = DiscountCurve.flat(VAL, RATE, DC)
UNDERLYING = EquityUnderlying(100.0, 0.0, DividendSchedule(continuous_yield=YIELD))

PARAMS = HestonParams(kappa=1.5, theta=0.04, sigma_v=0.6, rho=-0.7, v0=0.05)
EXPIRY = VAL + timedelta(days=730)
GRID = FourierGrid(upper=200.0, count=4001)
FORWARD = dividend_forward(UNDERLYING, EXPIRY, CURVE)
DF = CURVE.df(EXPIRY)
YEARS = 730.0 / 365.0


def _call(strike: float, params: HestonParams = PARAMS, grid: FourierGrid = GRID) -> float:
    return heston_call_price(params, UNDERLYING, strike, EXPIRY, (CURVE, grid))


class TestTheProbabilitiesAreProbabilities:
    """`P1` and `P2` are measures of the same event under two numeraires — no quadrature artefact
    may push them outside `[0,1]`, and nothing about Heston is needed to say so."""

    @pytest.mark.parametrize("moneyness", [0.5, 0.8, 1.0, 1.3, 2.0])
    def test_both_lie_in_the_unit_interval(self, moneyness: float) -> None:
        p1, p2 = heston_probabilities(PARAMS, UNDERLYING, moneyness * FORWARD, EXPIRY, (CURVE, GRID))
        assert 0.0 <= p1 <= 1.0, f"P1={p1}"
        assert 0.0 <= p2 <= 1.0, f"P2={p2}"

    def test_both_FALL_as_the_strike_rises(self) -> None:
        """They are `Prob(S_T > K)` under two measures; a rising strike cannot raise either."""
        for which in (0, 1):
            values = [
                heston_probabilities(PARAMS, UNDERLYING, m * FORWARD, EXPIRY, (CURVE, GRID))[which]
                for m in (0.7, 0.9, 1.1, 1.4)
            ]
            assert all(b < a for a, b in zip(values, values[1:])), f"P{which + 1}: {values}"


class TestTheReductionToVerifiedBlack:
    """`sigma_v = 0` makes the variance deterministic, so Heston must be Black on the integrated
    variance. Exact in the limit — and **vacuous alone**, which is why the MC companion exists."""

    @pytest.mark.parametrize("moneyness", [0.8, 1.0, 1.25])
    def test_a_ZERO_vol_of_vol_reproduces_black_on_the_integrated_variance(self, moneyness: float) -> None:
        params = HestonParams(kappa=1.5, theta=0.04, sigma_v=0.0, rho=0.0, v0=0.05)
        strike = moneyness * FORWARD
        expected = DF * black_price(
            FORWARD, strike, params.integrated_variance(YEARS), EquityOptionKind.CALL
        )
        assert _call(strike, params) == pytest.approx(expected, rel=1e-8)

    def test_the_reduction_is_NOT_what_makes_the_stochastic_case_right(self) -> None:
        """The premise that forces the MC companion: at the tranche's real parameters the price is
        materially away from the deterministic-variance one, so agreeing with Black proves nothing
        about the model actually being used."""
        deterministic = HestonParams(1.5, 0.04, 0.0, 0.0, 0.05)
        assert abs(_call(FORWARD) / _call(FORWARD, deterministic) - 1.0) > 0.01


class TestTheModelFreeRequirements:
    def test_PUT_CALL_PARITY_holds(self) -> None:
        """`C - P = df*(F - K)`, true of any arbitrage-free model, so it survives a transcription
        error in the characteristic function that a value oracle would share."""
        for moneyness in (0.8, 1.0, 1.2):
            strike = moneyness * FORWARD
            call = _call(strike)
            p1, p2 = heston_probabilities(PARAMS, UNDERLYING, strike, EXPIRY, (CURVE, GRID))
            put = DF * (FORWARD * (p1 - 1.0) - strike * (p2 - 1.0))
            assert call - put == pytest.approx(DF * (FORWARD - strike), rel=1e-9)

    @pytest.mark.parametrize("moneyness", [0.6, 1.0, 1.6])
    def test_the_call_respects_the_NO_ARBITRAGE_bounds(self, moneyness: float) -> None:
        strike = moneyness * FORWARD
        call = _call(strike)
        assert DF * max(FORWARD - strike, 0.0) - 1e-10 <= call <= DF * FORWARD + 1e-10

    def test_the_call_FALLS_with_the_strike_and_RISES_with_the_initial_variance(self) -> None:
        strikes = [_call(m * FORWARD) for m in (0.8, 1.0, 1.2)]
        assert all(b < a for a, b in zip(strikes, strikes[1:]))
        variances = [
            _call(FORWARD, HestonParams(1.5, 0.04, 0.6, -0.7, v)) for v in (0.02, 0.05, 0.10)
        ]
        assert all(b > a for a, b in zip(variances, variances[1:]))

    def test_a_NEGATIVE_correlation_produces_a_SKEW_in_the_right_direction(self) -> None:
        """Heston's whole point: `rho < 0` must make downside strikes relatively dearer than the
        symmetric case. A sign error in `rho` inverts this and nothing else here would see it."""
        low, high = 0.8 * FORWARD, 1.2 * FORWARD
        skewed = HestonParams(1.5, 0.04, 0.6, -0.7, 0.05)
        flat = HestonParams(1.5, 0.04, 0.6, 0.0, 0.05)
        ratio_skewed = _call(low, skewed) / _call(high, skewed)
        ratio_flat = _call(low, flat) / _call(high, flat)
        assert ratio_skewed > ratio_flat, f"{ratio_skewed} vs {ratio_flat}"


class TestTheQuadraturesValidityRegime:
    """The inversion integral is truncated and discretised — both get stated, not assumed (§4)."""

    def test_REFINING_the_grid_converges(self) -> None:
        reference = _call(FORWARD, grid=FourierGrid(upper=400.0, count=16001))
        errors = [
            abs(_call(FORWARD, grid=FourierGrid(upper=200.0, count=n)) / reference - 1.0)
            for n in (251, 1001, 4001)
        ]
        assert all(b < a for a, b in zip(errors, errors[1:])), f"must converge: {errors}"
        assert errors[-1] < 1e-8

    def test_TRUNCATING_too_early_is_visible_rather_than_silent(self) -> None:
        """A too-low upper limit must measurably move the price — the premise that the committed
        limit is past where the integrand has decayed."""
        truncated = _call(FORWARD, grid=FourierGrid(upper=3.0, count=4001))
        assert abs(truncated / _call(FORWARD) - 1.0) > 1e-4


class TestTheGuardsAreReachable:
    def test_a_NON_POSITIVE_upper_limit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="upper"):
            FourierGrid(upper=0.0, count=101)

    def test_a_grid_too_coarse_to_integrate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            FourierGrid(upper=100.0, count=1)

    def test_an_EXPIRED_option_has_no_characteristic_function(self) -> None:
        with pytest.raises(ValueError, match="positive horizon"):
            heston_call_price(PARAMS, UNDERLYING, FORWARD, VAL, (CURVE, GRID))
