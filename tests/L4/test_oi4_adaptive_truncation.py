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

# slow-test-exempt: this file establishes a no-arbitrage REQUIREMENT over a measured
# two-dimensional region (twelve `(σ_v, T)` cells of OI-4's mapped boundary, four strikes each), and
# every one of those prices is a Fourier inversion at an extended grid — up to 32,001 nodes, two
# characteristic-function evaluations per node. The cost is the oracle's domain: §4 says an oracle
# must contain the failure, and the region IS what caught a defect that had shipped for sixteen
# tranches, so trimming it to fit a stopwatch would trade the thing that works for the thing that
# measures it. Ledgered in `OPEN.md` as OI-5, with the re-open trigger named there: the inversion
# re-evaluates `φ(u)` **per strike** although `φ` does not depend on the strike, so a smile costs
# 4-10x what it needs to. Fixing that (out of tranche 50's scope by ruling — "no reshaping of the
# inversion where it was already correct") would bring this file back under the budget on its own.
# Measured: 2.8s locally, ~11s on the Windows CI runner against a 10.0s ceiling.

import math
from datetime import date, timedelta
from functools import lru_cache

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
#: An independent high-precision reference. `upper = 3200` is **twice** the widest range the fix ever
#: selects over the mapped region (1600), so it is an independent check rather than a re-run of the
#: same truncation — and it is as far as it needs to go, because beyond this the result stalls on the
#: arithmetic floor rather than converging (asserted below). A wider one bought nothing but CI time:
#: at 128,001 nodes this file took **23.1s on the Windows runner against a 10s budget** while reading
#: 5.9s locally, which is §9's "a local re-time is not the measurement" for the fourth time.
REFERENCE = FourierGrid(3200.0, 64_001)

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


@lru_cache(maxsize=None)
def _adaptive_price(vol_of_vol: float, days: int, moneyness: float) -> float:
    """The fix's price, memoised on its inputs.

    Several oracles reach for the same `(σ_v, T, K/F)` point — the non-negativity sweep's strikes are
    a subset of the monotonicity sweep's — and each Heston price at an extended grid costs tens of
    thousands of characteristic-function evaluations. Memoising is lossless (the pricer is pure, §3.1)
    and is the difference between this file fitting the `slow` budget and not."""
    expiry = VAL + timedelta(days=days)
    forward = dividend_forward(UNDERLYING, expiry, CURVE)
    return heston_call_price(_params(vol_of_vol), UNDERLYING, moneyness * forward, expiry, CURVE)


@lru_cache(maxsize=None)
def _svj_price(vol_of_vol: float, days: int, moneyness: float) -> float:
    """The SVJ price, memoised for the same reason as `_adaptive_price`."""
    expiry = VAL + timedelta(days=days)
    forward = dividend_forward(UNDERLYING, expiry, CURVE)
    params = BatesParams(heston=_params(vol_of_vol), jumps=JumpParams(0.5, -0.2, 0.15))
    return bates_call_price(params, UNDERLYING, moneyness * forward, expiry, CURVE)


@lru_cache(maxsize=None)
def _reference_price_at(vol_of_vol: float, days: int, moneyness: float) -> float:
    """The high-precision reference, memoised for the same reason."""
    expiry = VAL + timedelta(days=days)
    forward = dividend_forward(UNDERLYING, expiry, CURVE)
    return _reference_price(_params(vol_of_vol), expiry, moneyness * forward)


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
        for vol_of_vol in (0.6, 0.9, 1.2):
            for days in (7, 30):
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
            f"only {refused} of 18 mapped points are refused on the shipped grid — the defect "
            "must be broad, or this fixture is not the mapped region"
        )


class TestTheNoArbitrageRequirementOverTheMappedRegion:
    """The correctness anchor (§4 requirement oracle): a call price is non-negative and falls with
    the strike. Derived from the finance, so it survives any transcription — and it is the property
    the shipped grid violated."""

    @pytest.mark.parametrize("vol_of_vol", VOL_OF_VOL)
    @pytest.mark.parametrize("days", SHORT_DAYS)
    def test_prices_are_non_negative_AND_fall_with_the_strike(
        self, vol_of_vol: float, days: int
    ) -> None:
        """Both halves in one pass over one strike ladder, because the second implies the first:
        prices that fall with the strike and are non-negative at the **highest** strike are
        non-negative throughout. Two separate sweeps priced the same points twice for no extra
        coverage — and this file's cost is not free (see `REFERENCE`)."""
        # Four strikes, not seven: the oracle's domain is the **region** (all twelve
        # `(σ_v, T)` cells of the mapped boundary are swept), and within a cell four strikes
        # already exercise ordering and the sign. The at-the-money anchor, the quoted 1.15, the
        # 1.3 where several mapped points failed, and the far wing.
        prices = [_adaptive_price(vol_of_vol, days, m) for m in (1.0, 1.15, 1.3, 1.6)]
        for lower, higher in zip(prices, prices[1:]):
            assert higher <= lower + 1e-15, f"a vertical spread is negative: {prices}"
        assert prices[-1] >= 0.0, f"negative at the highest strike: {prices}"
        # ... and therefore everywhere, which is the implication this test leans on
        assert all(p >= 0.0 for p in prices), f"negative price in {prices}"


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
        reference = _reference_price_at(vol_of_vol, days, moneyness)
        assert reference > forward * 1e-9, "the premise: above the floor, so relative error means something"
        assert _adaptive_price(vol_of_vol, days, moneyness) == pytest.approx(reference, rel=1e-6)

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

    def test_the_rule_STOPS_where_it_should_rather_than_running_to_the_cap(self) -> None:
        """**Found by mutation.** Probing the decay at the *base* instead of at the current upper
        leaves every price correct — and silently drives the search to its 25,600 cap, sixteen
        times the work the corner actually needs. Every oracle passed, because correctness was
        never the casualty; the cost was.

        So the corner's grid is bounded above as well as below: it must extend (the price is wrong
        otherwise) and it must stop (or "adaptive" means "always maximal"). Measured: the worst
        corner in the mapped region selects 1600, well inside the cap."""
        for vol_of_vol, days in ((1.2, 7), (0.9, 30), (1.2, 30)):
            grid = adaptive_grid(_characteristic(_params(vol_of_vol), VAL + timedelta(days=days)))
            assert grid.upper > COMMITTED.upper, f"must extend: {grid}"
            assert grid.upper <= 4_000.0, (
                f"σ_v={vol_of_vol} at {days}d selected upper={grid.upper:.0f} — the search is not "
                "stopping proportionately, it is walking to its cap"
            )

    def test_the_corner_SELECTS_a_wider_grid(self) -> None:
        """The premise for the whole tranche: the rule must actually extend where it matters."""
        corner = adaptive_grid(_characteristic(_params(1.2), VAL + timedelta(days=7)))
        assert corner.upper > COMMITTED.upper
        assert corner.upper / (corner.count - 1) == pytest.approx(
            COMMITTED.upper / (COMMITTED.count - 1), rel=1e-9
        ), "the node SPACING is preserved — only the range extends"


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
        prices = [_svj_price(vol_of_vol, days, m) for m in (1.0, 1.15, 1.3)]
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
