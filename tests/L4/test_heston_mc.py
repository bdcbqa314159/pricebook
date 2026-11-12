"""The Heston Monte-Carlo (L4) — tranche 33's non-degenerate anchor. **Pre-registered.**

The `sigma_v -> 0` reduction proves the Fourier pricer agrees with Black where the variance is
*deterministic*. The spec rules that vacuous on its own — a deterministic-variance implementation
satisfies it — so the stochastic case needs an **independent method**, and this is it:

    dv = kappa(theta - v)dt + sigma_v sqrt(v) dW2,   dlog S = (carry - v/2)dt + sqrt(v) dW1

simulated directly, with no characteristic function anywhere in it. Fourier and MC share no
expression: one inverts a complex integral, the other integrates a path. A transcription error in
the characteristic function — a wrong `rho` sign, a mis-set `g`, a dropped term — shows up as a
price gap, on parameters where the variance is genuinely stochastic and correlated.

**The drift is pinned first, separately.** `E[S_T] = F(T)` is exact by construction and says nothing
about the variance process, so checking it before any price comparison means a drift error cannot be
read as a volatility error (the T19/T26 identity, used the same way tranche 32 used it).

**The Feller condition is violated on purpose.** `2*kappa*theta = 0.12 < 0.36 = sigma_v^2`, so the
variance process genuinely reaches zero and the scheme's boundary handling is exercised rather than
assumed away. A fixture satisfying Feller would be the comfortable case and would leave
full-truncation untested — the degenerate-fixture trap, in the one place this model is known to be
delicate.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine.heston_fourier import FourierGrid, heston_call_price
from pricebook.engine.heston_mc import (
    heston_call_estimate,
    heston_forward_estimate,
    simulate_heston_spots,
)
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import DiscountCurve
from pricebook.market.equity import DividendSchedule, EquityUnderlying, dividend_forward
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.heston import HestonParams

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
RATE, YIELD = 0.04, 0.015
CURVE = DiscountCurve.flat(VAL, RATE, DC)
UNDERLYING = EquityUnderlying(100.0, 0.0, DividendSchedule(continuous_yield=YIELD))
PARAMS = HestonParams(kappa=1.5, theta=0.04, sigma_v=0.6, rho=-0.7, v0=0.05)
EXPIRY = VAL + timedelta(days=365)
FORWARD = dividend_forward(UNDERLYING, EXPIRY, CURVE)
DF = CURVE.df(EXPIRY)
GRID = FourierGrid(upper=200.0, count=4001)
SAMPLING = (MonteCarloConfig(paths=30_000, seed=20261112), 60)


#: The simulation is **deterministic** (seeded config), so every test below is reading the same
#: 30,000 paths. Computing them once is not a caching optimisation bolted on — it is the honest
#: shape: these tests examine *one* sample, and re-running it per test would also let two tests
#: silently disagree about which sample they were asserting on.
#:
#: It is also what keeps this file inside the `slow` ceiling. The first version called
#: `simulate_heston_spots` in every test and measured 2.8s locally — comfortably under the 10s
#: ceiling, and **10.0s on the Windows CI runner**, which failed the gate. The T19 lesson: local
#: headroom is not bankable, because the multiplier is the runner's, not ours.
_SPOTS = None


def _spots():
    global _SPOTS
    if _SPOTS is None:
        _SPOTS = simulate_heston_spots(PARAMS, UNDERLYING, EXPIRY, CURVE, SAMPLING)
    return _SPOTS


class TestTheFixtureExercisesTheDelicateCase:
    def test_the_FELLER_condition_is_VIOLATED_so_the_zero_boundary_is_reached(self) -> None:
        assert not PARAMS.satisfies_feller
        assert 2 * PARAMS.kappa * PARAMS.theta < PARAMS.sigma_v**2

    def test_the_variance_is_stochastic_and_the_correlation_material(self) -> None:
        assert PARAMS.sigma_v > 0.3 and abs(PARAMS.rho) > 0.5


class TestTheDriftIsPinnedBeforeAnyPrice:
    def test_the_MARTINGALE_identity_holds(self) -> None:
        """`E[S_T] = F(T)`, exact by construction and independent of the variance process. Checked
        first so a drift error cannot be mistaken for a volatility error below."""
        got = heston_forward_estimate(_spots())
        assert abs(got.mean - FORWARD) <= 4.0 * got.standard_error, (
            f"E[S_T]={got.mean} +/- {got.standard_error} vs F={FORWARD}"
        )


class TestFourierAgainstTheIndependentSimulation:
    """The anchor the reduction cannot be: genuinely stochastic, correlated variance."""

    @pytest.mark.parametrize("moneyness", [0.9, 1.0, 1.1])
    def test_the_two_methods_AGREE_within_the_simulation_error(self, moneyness: float) -> None:
        strike = moneyness * FORWARD
        simulated = heston_call_estimate(_spots(), strike, DF)
        analytic = heston_call_price(PARAMS, UNDERLYING, strike, EXPIRY, (CURVE, GRID))
        assert abs(simulated.mean - analytic) <= 4.0 * simulated.standard_error, (
            f"K/F={moneyness}: MC {simulated.mean} +/- {simulated.standard_error} vs "
            f"Fourier {analytic}"
        )

    def test_the_band_can_RESOLVE_a_wrong_correlation_sign(self) -> None:
        """The band's power, stated: flipping `rho` must move the Fourier price well outside the
        simulation's error bar, or agreement would prove nothing about `rho` at all."""
        strike = 0.9 * FORWARD
        simulated = heston_call_estimate(_spots(), strike, DF)
        flipped = HestonParams(
            PARAMS.kappa, PARAMS.theta, PARAMS.sigma_v, -PARAMS.rho, PARAMS.v0
        )
        wrong = heston_call_price(flipped, UNDERLYING, strike, EXPIRY, (CURVE, GRID))
        assert abs(wrong - simulated.mean) > 4.0 * simulated.standard_error


class TestTheVarianceProcessItself:
    def test_the_simulated_spots_are_STRICTLY_POSITIVE(self) -> None:
        """Log-Euler on the spot keeps `S > 0` even where `v` touches zero."""
        assert all(s > 0.0 for s in _spots())

    def test_a_ZERO_vol_of_vol_collapses_to_a_lognormal_with_the_integrated_variance(self) -> None:
        """The reduction on the simulation side, independent of the Fourier one."""
        params = HestonParams(1.5, 0.04, 0.0, 0.0, 0.05)
        spots = simulate_heston_spots(params, UNDERLYING, EXPIRY, CURVE, SAMPLING)
        realised = math.log(sum(s for s in spots) / len(spots) / UNDERLYING.spot)
        assert realised == pytest.approx(math.log(FORWARD / UNDERLYING.spot), abs=0.02)


class TestTheGuardsAreReachable:
    def test_a_simulation_needs_at_least_one_step(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            simulate_heston_spots(
                PARAMS, UNDERLYING, EXPIRY, CURVE, (MonteCarloConfig(paths=10, seed=1), 0)
            )

    def test_a_NON_POSITIVE_horizon_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive horizon"):
            simulate_heston_spots(PARAMS, UNDERLYING, VAL, CURVE, SAMPLING)
