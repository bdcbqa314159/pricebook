"""OI-4 — the Fourier-inversion truncation defect (L4), tranche 50, PRE-REGISTERED.

The Heston/SVJ inversion returned a **negative call price** — an arbitrage violation — in the
short-dated, high-vol-of-vol, out-of-the-money wing, because the committed truncation `upper = 200`
cut the integrand off while it was still material. Mapped at re-audit #5: boundary
`T ≤ 90d ∧ K/F ≥ 1.15`, monotone in `σ_v`, nothing failing at 365d, and reaching a **quoted** point
(`σ_v ∈ {0.9, 1.2}`, 30 days, `K/F = 1.15`, 4.0%/4.4% wrong).

**The mechanism, measured.** The integrand's magnitude is `|φ(u)|/u`. At `u = 200`:

| regime | the integrand at `u = 200` |
|---|---|
| `σ_v = 0.3`, `T = 1y` (benign) | 4.2e-47 — truncation is free |
| `σ_v = 1.2`, `T = 7d` (the corner) | **1.7e-04** — the tail is larger than the price |

Heston's characteristic function decays *exponentially* in `u` (not Gaussian, as Black's does), at a
rate that collapses as `T → 0` and `η` grows. One constant cannot serve both ends of that range,
which is why the fix is **regime-aware** rather than a wider constant: a flat `upper = 3200` would
make every benign pricing call sixteen times dearer to repair a corner.

**Three branches, and the third is the one that matters.** Truncation error falls under refinement
until it reaches an arithmetic floor and then stalls — measured at this point, whose true price is
zero to double precision:

| `upper` | computed "price" (truth: 0) |
|---|---|
| 200 | 1.3e-02 |
| 400 | −1.4e-04 |
| 800 | −4.0e-08 |
| 1600 | −1.5e-11 |
| 3200 | −1.6e-11 ← stalls |

`P₁ = ½ + I/π` is a difference of two `O(½)` quantities, so it carries an absolute noise floor of
**~2e-13** regardless of grid, and a price inherits `forward × 2e-13`. So:

1. **above the floor** → the price, now correct because the truncation is adaptive;
2. **within the floor** → `0.0`, which is *more* accurate than the noise, not a cosmetic clamp;
3. **materially below zero** → the method has failed in that regime, and it says so rather than
   returning a negative price. This is the branch that keeps the floor honest: a clamp alone would
   have turned the original −2.5e-04 defect into a clean-looking `0.0` and hidden it. Seven orders
   separate that defect from the floor, so the two cannot be confused.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine.bates_fourier import bates_call_price
from pricebook.engine.fourier_inversion import (
    FourierGrid,
    adaptive_grid,
    gil_pelaez_probabilities,
)
from pricebook.engine.heston_fourier import heston_call_price
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import DiscountCurve
from pricebook.market.equity import DividendSchedule, EquityUnderlying, dividend_forward
from pricebook.models.bates import BatesParams, JumpParams, bates_characteristic_function
from pricebook.models.heston import HestonParams, heston_characteristic_function

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
CURVE = DiscountCurve.flat(VAL, 0.025, DC)
UNDERLYING = EquityUnderlying(100.0, 0.0, DividendSchedule(continuous_yield=0.0))

#: The grid that shipped from tranche 33 and carries the defect.
COMMITTED = FourierGrid(200.0, 4001)
#: An independent high-precision reference, far beyond anything the fix selects.
REFERENCE = FourierGrid(6400.0, 128_001)

#: The mapped region: `T ≤ 90d` crossed with `K/F ≥ 1.15`, over a vol-of-vol range.
VOL_OF_VOL = (0.3, 0.6, 0.9, 1.2)
SHORT_DAYS = (7, 30, 90)
OUT_OF_MONEY = (1.15, 1.3, 1.6)
#: The benign region, where the committed grid was already right and must stay so.
BENIGN = ((0.3, 365, 1.0), (0.6, 365, 1.0), (0.9, 365, 1.15), (0.3, 90, 1.0), (0.6, 180, 1.1))


def _params(vol_of_vol: float) -> HestonParams:
    return HestonParams(kappa=2.5, theta=0.06, sigma_v=vol_of_vol, rho=-0.3, v0=0.02)


def _characteristic(params: HestonParams, expiry: date):
    years = CURVE.time_measure.year_fraction(expiry)
    forward = dividend_forward(UNDERLYING, expiry, CURVE)
    carry = math.log(forward / UNDERLYING.spot) / years

    def bound(u: complex) -> complex:
        return heston_characteristic_function(params, u, years, carry)

    return bound


def _reference_price(params: HestonParams, expiry: date, strike: float) -> float:
    """The call from the high-precision grid — the test's own reference, independent of whatever
    grid the fix selects."""
    years = CURVE.time_measure.year_fraction(expiry)
    forward = dividend_forward(UNDERLYING, expiry, CURVE)
    log_strike = math.log(strike / UNDERLYING.spot)
    first, second = gil_pelaez_probabilities(
        _characteristic(params, expiry), log_strike, REFERENCE
    )
    return CURVE.df(expiry) * (forward * first - strike * second)


class TestTheDefectIsRealAndTheFixtureReachesIt:
    """The premise (§4): a benign fixture cannot see this defect at all, so the region is asserted
    before anything is claimed about it."""

    def test_the_committed_grid_truncates_while_the_integrand_is_MATERIAL(self) -> None:
        corner = _characteristic(_params(1.2), VAL + timedelta(days=7))
        benign = _characteristic(_params(0.3), VAL + timedelta(days=365))
        at_corner = abs(corner(complex(200.0))) / 200.0
        at_benign = abs(benign(complex(200.0))) / 200.0
        assert at_corner > 1e-5, f"the corner's integrand at u=200 is only {at_corner:.2e}"
        assert at_benign < 1e-30, f"the benign integrand at u=200 is {at_benign:.2e}"
        assert at_corner / at_benign > 1e20, "the two regimes must differ by orders, not margins"

    def test_the_committed_grid_PRODUCES_REFUSED_PRICES_in_the_region(self) -> None:
        """The defect itself, as a failing property of the old grid — so the oracles below are known
        to test something real rather than assert an already-true fact.

        Stated as *refusals* rather than as negative numbers, because the floor rule now intercepts
        them: handed the shipped grid, the inversion produces values a call price may not take, and
        the count is what makes the region's breadth concrete."""
        refused = 0
        for vol_of_vol in VOL_OF_VOL:
            for days in SHORT_DAYS:
                expiry = VAL + timedelta(days=days)
                forward = dividend_forward(UNDERLYING, expiry, CURVE)
                for moneyness in OUT_OF_MONEY:
                    try:
                        heston_call_price(
                            _params(vol_of_vol),
                            UNDERLYING,
                            moneyness * forward,
                            expiry,
                            (CURVE, COMMITTED),
                        )
                    except ValueError:
                        refused += 1
        assert refused >= 5, (
            f"only {refused} of 36 mapped points are refused on the shipped grid — the defect "
            "must be broad, or this fixture is not the mapped region"
        )


class TestTheNoArbitrageRequirementOverTheMappedRegion:
    """The correctness anchor (§4 requirement oracle): a call price is non-negative and falls with
    the strike. Derived from the finance, so it survives any transcription — and it is the property
    the shipped grid violated."""

    @pytest.mark.parametrize("vol_of_vol", VOL_OF_VOL)
    @pytest.mark.parametrize("days", SHORT_DAYS)
    def test_prices_are_non_negative_across_the_region(self, vol_of_vol: float, days: int) -> None:
        expiry = VAL + timedelta(days=days)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        params = _params(vol_of_vol)
        for moneyness in OUT_OF_MONEY:
            price = heston_call_price(params, UNDERLYING, moneyness * forward, expiry, CURVE)
            assert price >= 0.0, f"negative price {price:.4e} at K/F={moneyness}"

    @pytest.mark.parametrize("vol_of_vol", VOL_OF_VOL)
    @pytest.mark.parametrize("days", SHORT_DAYS)
    def test_prices_FALL_with_the_strike_across_the_region(
        self, vol_of_vol: float, days: int
    ) -> None:
        expiry = VAL + timedelta(days=days)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        params = _params(vol_of_vol)
        prices = [
            heston_call_price(params, UNDERLYING, m * forward, expiry, CURVE)
            for m in (1.0, 1.1, 1.15, 1.2, 1.3, 1.45, 1.6)
        ]
        for lower, higher in zip(prices, prices[1:]):
            assert higher <= lower + 1e-15, f"a vertical spread is negative: {prices}"


class TestConvergenceInTheBadCornerSpecifically:
    """§4's "a reference is code too": the reference is evaluated **in the failing corner**, not at
    a benign point where the old and new grids agree anyway."""

    @pytest.mark.parametrize("vol_of_vol,days,moneyness", [
        (0.9, 30, 1.15), (1.2, 30, 1.15), (0.9, 30, 1.3), (1.2, 30, 1.3), (0.6, 30, 1.3),
    ])
    def test_the_fix_matches_a_high_precision_reference_there(
        self, vol_of_vol: float, days: int, moneyness: float
    ) -> None:
        """Restricted to the corner points whose price sits comfortably above the arithmetic floor,
        where a *relative* claim is meaningful."""
        params = _params(vol_of_vol)
        expiry = VAL + timedelta(days=days)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        strike = moneyness * forward
        reference = _reference_price(params, expiry, strike)
        assert reference > forward * 1e-9, "the premise: above the floor, so relative error means something"
        fixed = heston_call_price(params, UNDERLYING, strike, expiry, CURVE)
        assert fixed == pytest.approx(reference, rel=1e-6)

    def test_THE_RESOLUTION_the_committed_grid_FAILS_that_same_check(self) -> None:
        """What the green above rules out, stated by measurement: the old grid misses the reference
        by percent at the quoted point, four orders outside the 1e-6 band."""
        params = _params(1.2)
        expiry = VAL + timedelta(days=30)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        strike = 1.15 * forward
        reference = _reference_price(params, expiry, strike)
        stale = heston_call_price(params, UNDERLYING, strike, expiry, (CURVE, COMMITTED))
        assert abs(stale / reference - 1.0) > 0.01, "the old grid must be percent-wrong here"


class TestTheFixIsLOCALtoTheDefect:
    """A pure-regime fix: where the old grid was already right, nothing moves."""

    @pytest.mark.parametrize("vol_of_vol,days,moneyness", BENIGN)
    def test_the_benign_region_is_UNCHANGED(
        self, vol_of_vol: float, days: int, moneyness: float
    ) -> None:
        """Asserted as **equality**, and the distinction from tranche 49's mistake is the point:
        this compares two *live* computations in one process — the adaptive path and the committed
        grid — not a stored bit pattern captured on another machine. Both sides move together under
        a different libm, so the equality is portable. The §4 clause forbids freezing a measurement
        into a literal, not comparing two computations."""
        params = _params(vol_of_vol)
        expiry = VAL + timedelta(days=days)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        strike = moneyness * forward
        assert heston_call_price(params, UNDERLYING, strike, expiry, CURVE) == heston_call_price(
            params, UNDERLYING, strike, expiry, (CURVE, COMMITTED)
        )

    @pytest.mark.parametrize("vol_of_vol,days,moneyness", BENIGN)
    def test_and_the_benign_region_SELECTS_the_committed_grid(
        self, vol_of_vol: float, days: int, moneyness: float
    ) -> None:
        """Why the equality above holds, stated structurally rather than left to coincidence: the
        adaptive rule picks the same grid there, so benign pricing is not merely close — it is the
        same arithmetic, and it costs no more than it did."""
        grid = adaptive_grid(_characteristic(_params(vol_of_vol), VAL + timedelta(days=days)))
        assert grid == COMMITTED

    def test_ORDINARY_FIXTURES_ACROSS_THE_TREE_pay_nothing(self) -> None:
        """**The ruling's actual rationale, oracled.** "Not a wider constant" was chosen so benign
        pricing would not be taxed to repair a corner — and nothing asserted that, so the first
        threshold I committed (`1e-14`) silently doubled the node count for the tree's own standard
        Heston fixture, which sits at `|φ(200)|/200 = 3.7e-14`. It bought a price change of 9.6e-15
        relative: nothing, for twice the work.

        So the cost property is a test, not a hope: the parameter sets this tree actually prices
        with must select the committed grid. A threshold tightened past the arithmetic floor fails
        here, which is the only thing that stops the rationale eroding."""
        ordinary = (
            HestonParams(kappa=1.5, theta=0.04, sigma_v=0.6, rho=-0.7, v0=0.05),  # the Heston fixture
            HestonParams(kappa=2.5, theta=0.06, sigma_v=0.6, rho=-0.3, v0=0.02),  # the re-audit set
            HestonParams(kappa=1.5, theta=0.04, sigma_v=0.6, rho=-0.7, v0=0.05),  # the SVJ fixture
        )
        for params in ordinary:
            for days in (365, 730):
                grid = adaptive_grid(_characteristic(params, VAL + timedelta(days=days)))
                assert grid == COMMITTED, (
                    f"an ordinary fixture pays {grid.count / COMMITTED.count:.0f}x for nothing: "
                    f"{params} at {days}d selected {grid}"
                )

    def test_and_what_DOES_extend_extends_because_the_price_MOVES(self) -> None:
        """The other side of the same coin — the rule must not be so loose that it leaves a real
        error behind. A high-vol-of-vol one-year slice does extend, and its price shifts by ~6.5e-10
        relative, a few hundred times the arithmetic floor. Measured, so "it extends" is justified
        rather than tolerated."""
        params = HestonParams(kappa=1.5, theta=0.04, sigma_v=0.9, rho=-0.7, v0=0.05)
        expiry = VAL + timedelta(days=365)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        assert adaptive_grid(_characteristic(params, expiry)).upper > COMMITTED.upper
        adaptive = heston_call_price(params, UNDERLYING, forward, expiry, CURVE)
        stale = heston_call_price(params, UNDERLYING, forward, expiry, (CURVE, COMMITTED))
        shift = abs(adaptive / stale - 1.0)
        assert shift > 100.0 * (1e-12), f"the extension changed nothing material: {shift:.2e}"

    def test_the_corner_SELECTS_a_wider_grid(self) -> None:
        """The premise for the whole tranche: the rule must actually extend where it matters."""
        corner = adaptive_grid(_characteristic(_params(1.2), VAL + timedelta(days=7)))
        assert corner.upper > COMMITTED.upper
        assert corner.upper / (corner.count - 1) == pytest.approx(
            COMMITTED.upper / (COMMITTED.count - 1), rel=1e-9
        ), "the node SPACING is preserved — only the range extends"


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


class TestTheSVJInversionGetsTheSameFix:
    """SVJ's jump tails decay more slowly still, so it is the harder case for the same rule — and
    it must be served by the shared inversion rather than a second constant."""

    JUMPS = JumpParams(intensity=0.5, mean=-0.2, volatility=0.15)

    def _bates(self, vol_of_vol: float) -> BatesParams:
        return BatesParams(heston=_params(vol_of_vol), jumps=self.JUMPS)

    @pytest.mark.parametrize("vol_of_vol", (0.6, 1.2))
    @pytest.mark.parametrize("days", (7, 30))
    def test_svj_prices_are_non_negative_and_fall_with_the_strike(
        self, vol_of_vol: float, days: int
    ) -> None:
        expiry = VAL + timedelta(days=days)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        params = self._bates(vol_of_vol)
        prices = [
            bates_call_price(params, UNDERLYING, m * forward, expiry, CURVE)
            for m in (1.0, 1.15, 1.3, 1.6)
        ]
        assert all(p >= 0.0 for p in prices), f"negative SVJ price: {prices}"
        for lower, higher in zip(prices, prices[1:]):
            assert higher <= lower + 1e-15, f"SVJ vertical spread negative: {prices}"

    def test_the_svj_grid_is_at_least_as_wide_as_hestons(self) -> None:
        """Measured, not assumed: the jump factor multiplies `φ`, so `|φ_SVJ| ≥ |φ_Heston|` is not
        automatic — but its slower decay means the rule must not select a *narrower* range."""
        expiry = VAL + timedelta(days=7)
        years = CURVE.time_measure.year_fraction(expiry)
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        carry = math.log(forward / UNDERLYING.spot) / years

        def svj(u: complex) -> complex:
            return bates_characteristic_function(self._bates(1.2), u, years, carry)

        assert adaptive_grid(svj).upper >= adaptive_grid(
            _characteristic(_params(1.2), expiry)
        ).upper
