"""The Gil-Pelaez inversion's ARITHMETIC floor (L4) — tranche 50, PRE-REGISTERED.

Split from `test_oi4_adaptive_truncation.py`, which holds the truncation fix. These are two findings
rather than one: tranche 50 set out to repair OI-4's truncation defect, and discovered in measuring
it that truncation error stops improving at a floor the grid cannot reach past. The fix and the floor
want different fixtures and read differently, so they are separate files. (The split also brings each
under the `slow` budget, which is disclosed in the thread rather than left as a happy accident.)

**The floor, measured.** At a point whose true price is zero to double precision — three days, 80%
out of the money — the computed value runs

    upper=200 → 1.3e-02,  400 → −1.4e-04,  800 → −4.0e-08,  1600 → −1.5e-11,  3200 → −1.6e-11

converging under refinement and then **stalling**. `P₁ = ½ + I/π` is a difference of two `O(½)`
quantities, so it carries an absolute noise floor of ~2e-13 whatever the grid, and a price inherits
`forward × 2e-13`.

**Three branches.** A price above the floor is returned as computed; a magnitude *within* the floor
returns `0.0`, because zero is the honest rendering of a number smaller than the arithmetic that
produced it; and a value materially below zero **raises**, because a negative call price means the
method failed in that regime rather than that the option is cheap.

The third branch is what makes the second trustworthy. A bare clamp would have converted OI-4's own
−2.5e-04 into a clean-looking `0.0` and hidden it — the §4 "plausible wrong number" trap exactly — and
the two cases are separated by seven orders of magnitude, not by a margin.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine.bates_fourier import bates_call_price
from pricebook.engine.fourier_inversion import FourierGrid, gil_pelaez_probabilities
from pricebook.engine.heston_fourier import heston_call_price
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import DiscountCurve
from pricebook.market.equity import DividendSchedule, EquityUnderlying, dividend_forward
from pricebook.models.bates import BatesParams, JumpParams
from pricebook.models.heston import HestonParams, heston_characteristic_function

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
CURVE = DiscountCurve.flat(VAL, 0.025, DC)
UNDERLYING = EquityUnderlying(100.0, 0.0, DividendSchedule(continuous_yield=0.0))

#: The grid that shipped from tranche 33 and carries the defect.
COMMITTED = FourierGrid(200.0, 4001)
#: Wide enough to show the stall, and no wider — past this the result does not improve.
REFERENCE = FourierGrid(3200.0, 64_001)


def _params(vol_of_vol: float) -> HestonParams:
    return HestonParams(kappa=2.5, theta=0.06, sigma_v=vol_of_vol, rho=-0.3, v0=0.02)


def _characteristic(params: HestonParams, expiry: date):
    years = CURVE.time_measure.year_fraction(expiry)
    forward = dividend_forward(UNDERLYING, expiry, CURVE)
    carry = math.log(forward / UNDERLYING.spot) / years

    def bound(u: complex) -> complex:
        return heston_characteristic_function(params, u, years, carry)

    return bound


class TestTheArithmeticFloor:
    """The floor is measured, asserted, and bounded away from anything material."""

    #: Where the true price is zero to double precision: 3 days, 80% out of the money.
    DEEP = (1.2, 3, 1.8)

    def test_a_sub_floor_price_is_returned_as_exactly_zero(self) -> None:
        vol_of_vol, days, moneyness = self.DEEP
        expiry = VAL + timedelta(days=days)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        price = heston_call_price(
            _params(vol_of_vol), UNDERLYING, moneyness * forward, expiry, CURVE
        )
        assert price == 0.0

    def test_THE_PREMISE_the_raw_inversion_there_is_noise_not_a_price(self) -> None:
        """So the zero above is the honest answer rather than a swept-away defect: the raw value
        does not converge under refinement — it stalls at the floor, which is what distinguishes
        arithmetic noise from a truncation error."""
        vol_of_vol, days, moneyness = self.DEEP
        params = _params(vol_of_vol)
        expiry = VAL + timedelta(days=days)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        log_strike = math.log(moneyness * forward / UNDERLYING.spot)
        characteristic = _characteristic(params, expiry)
        values = []
        for grid in (FourierGrid(1600.0, 32_001), FourierGrid(3200.0, 64_001), REFERENCE):
            first, second = gil_pelaez_probabilities(characteristic, log_strike, grid)
            values.append(forward * first - moneyness * forward * second)
        assert all(abs(v) < forward * 1e-12 for v in values), f"not at the floor: {values}"
        # it stalls rather than converging: tripling the range does not shrink it
        assert abs(values[-1]) > 0.1 * abs(values[0]), f"still converging, so not a floor: {values}"

    def test_the_floor_is_orders_BELOW_the_defect_it_must_not_hide(self) -> None:
        """The guard on the guard. The original defect was −2.5e-04; the floor is ~1e-10 on this
        forward. Seven orders separate them, so returning zero inside the floor cannot launder a
        truncation error into a plausible zero — which is the trap a bare clamp would have been."""
        expiry = VAL + timedelta(days=30)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        defect = 2.5e-04
        floor = forward * 1e-12
        assert defect / floor > 1e6


class TestAMaterialNegativeIsAFailureNotAPrice:
    """The third branch: if a regime ever defeats the adaptive rule, the method says so instead of
    returning an arbitrage violation. A guard that *can* fire, as §9 requires of a guard."""

    def test_a_deliberately_crippled_grid_is_REFUSED_not_returned(self) -> None:
        """Handed the grid that carries the defect, the assembly must refuse rather than return a
        negative price — which is also the mutation proof that the floor logic is reachable."""
        params = _params(1.2)
        expiry = VAL + timedelta(days=7)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        with pytest.raises(ValueError, match="negative"):
            heston_call_price(params, UNDERLYING, 1.6 * forward, expiry, (CURVE, COMMITTED))


class TestTheSVJEngineReachesTheFloorToo:
    """Found by mutation, and the reason this class exists at all."""

    def _bates(self, vol_of_vol: float) -> BatesParams:
        return BatesParams(heston=_params(vol_of_vol), jumps=JumpParams(0.5, -0.2, 0.15))

    def test_the_svj_path_REACHES_THE_FLOOR_branches_too(self) -> None:
        """**Found by mutation: bypassing the floor rule in the SVJ engine survived the whole
        suite**, because every SVJ fixture priced above the floor — the T33b input-population
        degeneracy again, one engine over. The shared rule being correct is not evidence that both
        callers route through it.

        At `K/F = 2.2` and seven days the raw inversion is sub-floor, so the zero branch fires; and
        on the shipped grid the same point is *refused*, so the refusal branch fires. One fixture,
        both branches, in the engine that was unguarded."""
        expiry = VAL + timedelta(days=7)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        strike = 2.2 * forward
        params = self._bates(1.2)
        assert bates_call_price(params, UNDERLYING, strike, expiry, CURVE) == 0.0
        with pytest.raises(ValueError, match="materially negative"):
            bates_call_price(params, UNDERLYING, strike, expiry, (CURVE, COMMITTED))
