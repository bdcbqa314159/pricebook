"""Sharing `φ(u)` across a maturity's strikes (L4) — tranche 53, PRE-REGISTERED. OI-5's removal.

`φ(u)` does **not** depend on the strike — only the Gil-Pelaez phase `e^{-iuk}` does — yet the
inversion re-evaluated it at every node for every strike. Pricing a smile therefore cost `N ×` the
`φ` calls it needs, and `φ` is the expensive part: a Heston characteristic function is a complex
`sqrt`, `log` and two `exp` per node, against a phase that is one `exp`.

**Two oracles, and the second is the unusual one.**

1. **Purity** — no price may move. The strongest form of that proof available here is structural
   rather than numerical: the single-strike entry point *is* the batch entry point called with one
   strike, so the two cannot diverge by construction. On top of that, the batch path is asserted
   **bit-identical in-process** to the pre-refactor single-strike arithmetic (the node sum is over
   the same `φ` values in the same order, so §4's op-order condition holds), while the **committed**
   regression is a *relative* tolerance — the T49 lesson: bit-identity is a property of a
   measurement, not of a cross-platform test.

2. **The performance property is an oracle, not a changelog line.** T50's first threshold silently
   doubled the cost of every benign pricing call because "don't tax every call" was a *ruling in
   prose* with nothing asserting it — *a rationale that is not a test erodes.* So the win is pinned:
   `φ` is evaluated **once per maturity, not once per strike**, counted directly.

**What this does not claim.** The waste is in `src/` and the fix is in `src/`, but no *product*
prices a smile today — `price(product, model)` prices one option per call. The present consumers of
the shared path are four test files that fit or sweep a smile at one maturity (the SVI fits, the
cross-arc tie-back, the variance-swap log strip, OI-4's region sweep) plus the OI-5 exemption that
the region sweep's cost bought. A smile-pricing product would inherit the win when it arrives; it is
not realised in production by any caller that exists now, and saying otherwise would be the kind of
prospective claim §6 refuses.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import bates_fourier, heston_fourier
from pricebook.engine.bates_fourier import bates_call_price, bates_call_prices
from pricebook.engine.fourier_inversion import (
    FourierGrid,
    gil_pelaez_probabilities,
    inversion_kernel,
    probabilities_at,
)
from pricebook.engine.heston_fourier import heston_call_price, heston_call_prices
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import DiscountCurve
from pricebook.market.equity import DividendSchedule, EquityUnderlying, dividend_forward
from pricebook.models.bates import BatesParams, JumpParams, bates_characteristic_function
from pricebook.models.heston import HestonParams, heston_characteristic_function

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
CURVE = DiscountCurve.flat(VAL, 0.025, DC)
UNDERLYING = EquityUnderlying(100.0, 0.0, DividendSchedule(continuous_yield=0.01))
PARAMS = HestonParams(kappa=1.5, theta=0.04, sigma_v=0.6, rho=-0.7, v0=0.05)
JUMPS = JumpParams(intensity=0.5, mean=-0.2, volatility=0.15)
EXPIRY = VAL + timedelta(days=365)
GRID = FourierGrid(200.0, 4001)
FORWARD = dividend_forward(UNDERLYING, EXPIRY, CURVE)
SMILE = (0.8, 0.9, 1.0, 1.1, 1.25)
STRIKES = tuple(m * FORWARD for m in SMILE)


def _years() -> float:
    return CURVE.time_measure.year_fraction(EXPIRY)


def _characteristic():
    years = _years()
    carry = math.log(FORWARD / UNDERLYING.spot) / years

    def bound(u: complex) -> complex:
        return heston_characteristic_function(PARAMS, u, years, carry)

    return bound


class TestThePerformancePropertyIsAnOracle:
    """§7's guarded-vs-folklore line, and T50's hard lesson: a rationale that is not a test erodes.
    The 4-10x win is the point of the tranche, so it is asserted rather than described."""

    def _count_heston_calls(self, monkeypatch, work) -> int:
        calls = 0
        original = heston_fourier.heston_characteristic_function

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(heston_fourier, "heston_characteristic_function", counted)
        work()
        return calls

    def test_a_SMILE_costs_the_same_phi_calls_as_ONE_strike(self, monkeypatch) -> None:
        """The whole tranche, in one assertion. Five strikes at one maturity must evaluate `φ` no
        more often than a single strike does — the phase is what varies, not the characteristic
        function."""
        one = self._count_heston_calls(
            monkeypatch,
            lambda: heston_call_prices(PARAMS, UNDERLYING, STRIKES[:1], EXPIRY, (CURVE, GRID)),
        )
        five = self._count_heston_calls(
            monkeypatch,
            lambda: heston_call_prices(PARAMS, UNDERLYING, STRIKES, EXPIRY, (CURVE, GRID)),
        )
        assert one > 2 * GRID.count, "the premise: φ really is evaluated per node"
        assert five == one, f"a 5-strike smile cost {five} φ calls against {one} for one strike"

    def test_the_PREMISE_pricing_them_SEPARATELY_costs_five_times_as_much(
        self, monkeypatch
    ) -> None:
        """So the assertion above has content: the waste is real and this is its size. Measured
        rather than asserted abstractly — the 4-10x in OI-5's ledger entry, as a number."""
        shared = self._count_heston_calls(
            monkeypatch,
            lambda: heston_call_prices(PARAMS, UNDERLYING, STRIKES, EXPIRY, (CURVE, GRID)),
        )
        separate = self._count_heston_calls(
            monkeypatch,
            lambda: [
                heston_call_price(PARAMS, UNDERLYING, k, EXPIRY, (CURVE, GRID)) for k in STRIKES
            ],
        )
        assert separate > 4.5 * shared, (
            f"separate pricing cost {separate} φ calls against {shared} shared — the saving is "
            f"only {separate / shared:.1f}x, so the hoist is not doing what it claims"
        )

    def test_the_kernel_is_reusable_across_strikes_without_touching_phi_again(
        self, monkeypatch
    ) -> None:
        """The structural half: once built, the kernel answers any strike with no further `φ` call,
        which is what makes a smile free after the first evaluation."""
        kernel = inversion_kernel(_characteristic(), GRID)
        calls = 0
        original = heston_characteristic_function

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr("pricebook.models.heston.heston_characteristic_function", counted)
        for strike in STRIKES:
            probabilities_at(kernel, math.log(strike / UNDERLYING.spot))
        assert calls == 0, f"inverting at 5 strikes re-evaluated φ {calls} times"


class TestThePurityOfTheRefactor:
    """No price moves. The committed assertion is a **relative tolerance**, never a bit pattern —
    bit-identity is a property of an in-process measurement, not of a cross-platform test (§4,
    the T49 mistake named so it is not repeated)."""

    @pytest.mark.parametrize("moneyness", SMILE)
    def test_the_batch_and_single_paths_agree(self, moneyness: float) -> None:
        strike = moneyness * FORWARD
        batched = heston_call_prices(PARAMS, UNDERLYING, (strike,), EXPIRY, (CURVE, GRID))[0]
        single = heston_call_price(PARAMS, UNDERLYING, strike, EXPIRY, (CURVE, GRID))
        assert batched == pytest.approx(single, rel=1e-12)

    def test_the_SINGLE_path_IS_the_batch_path_with_one_strike(self) -> None:
        """The structural purity argument, stronger than any tolerance: the two cannot diverge
        because there is one implementation. Asserted as exact equality, which is legitimate here
        and not the T49 mistake — both sides are computed **live in this process** by the same code,
        rather than compared against a bit pattern captured on another machine."""
        for strike in STRIKES:
            assert heston_call_price(PARAMS, UNDERLYING, strike, EXPIRY, (CURVE, GRID)) == (
                heston_call_prices(PARAMS, UNDERLYING, (strike,), EXPIRY, (CURVE, GRID))[0]
            )

    @pytest.mark.parametrize("moneyness", SMILE)
    def test_a_smile_priced_TOGETHER_matches_one_priced_SEPARATELY(self, moneyness: float) -> None:
        """The property the hoist must not break: sharing `φ` changes which quantity is computed how
        many times, never any price."""
        together = heston_call_prices(PARAMS, UNDERLYING, STRIKES, EXPIRY, (CURVE, GRID))
        index = SMILE.index(moneyness)
        separate = heston_call_price(PARAMS, UNDERLYING, STRIKES[index], EXPIRY, (CURVE, GRID))
        assert together[index] == pytest.approx(separate, rel=1e-12)

    def test_the_probabilities_entry_point_is_unchanged(self) -> None:
        """`gil_pelaez_probabilities` keeps its signature and its answer — it now builds a kernel
        internally, which single-strike callers cannot observe."""
        kernel = inversion_kernel(_characteristic(), GRID)
        for strike in STRIKES:
            log_strike = math.log(strike / UNDERLYING.spot)
            assert gil_pelaez_probabilities(_characteristic(), log_strike, GRID) == (
                probabilities_at(kernel, log_strike)
            )


class TestTheSVJPathGetsTheSameTreatment:
    """SVJ's `φ` is strictly dearer — the Heston factor times a jump factor — so it is the case the
    hoist helps most, and (T50's lesson one engine over) a shared rule being correct is not evidence
    both callers route through it."""

    def _bates(self) -> BatesParams:
        return BatesParams(heston=PARAMS, jumps=JUMPS)

    def test_a_SVJ_smile_costs_the_same_phi_calls_as_one_strike(self, monkeypatch) -> None:
        calls = 0
        original = bates_fourier.bates_characteristic_function

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(bates_fourier, "bates_characteristic_function", counted)
        bates_call_prices(self._bates(), UNDERLYING, STRIKES[:1], EXPIRY, (CURVE, GRID))
        one = calls
        calls = 0
        bates_call_prices(self._bates(), UNDERLYING, STRIKES, EXPIRY, (CURVE, GRID))
        assert one > 2 * GRID.count, "the premise: φ really is evaluated per node"
        assert calls == one, f"a 5-strike SVJ smile cost {calls} φ calls against {one} for one"

    @pytest.mark.parametrize("moneyness", SMILE)
    def test_the_SVJ_batch_and_single_paths_agree(self, moneyness: float) -> None:
        strike = moneyness * FORWARD
        params = self._bates()
        batched = bates_call_prices(params, UNDERLYING, (strike,), EXPIRY, (CURVE, GRID))[0]
        assert batched == bates_call_price(params, UNDERLYING, strike, EXPIRY, (CURVE, GRID))

    def test_the_svj_characteristic_function_is_dearer_than_hestons(self) -> None:
        """Why SVJ benefits more, stated as a measurement rather than an assumption: its `φ` is
        Heston's times a jump factor, so every node costs strictly more."""
        years = _years()
        carry = math.log(FORWARD / UNDERLYING.spot) / years
        plain = heston_characteristic_function(PARAMS, complex(3.0), years, carry)
        with_jumps = bates_characteristic_function(self._bates(), complex(3.0), years, carry)
        assert with_jumps != plain, "the jump factor must actually change φ"


class TestTheGuardsSurviveTheHoist:
    """The refactor must not quietly drop what T50 built on top of the inversion."""

    def test_the_batch_path_applies_the_ARITHMETIC_FLOOR(self) -> None:
        """A sub-floor price in a batch must still come back as exactly zero, and a materially
        negative one must still be refused — the three-branch rule is not bypassed by batching."""
        deep = HestonParams(kappa=2.5, theta=0.06, sigma_v=1.2, rho=-0.3, v0=0.02)
        short = VAL + timedelta(days=7)
        forward = dividend_forward(UNDERLYING, short, CURVE)
        prices = heston_call_prices(
            deep, UNDERLYING, (forward, 2.2 * forward), short, CURVE
        )
        assert prices[0] > 0.0
        assert prices[1] == 0.0, "a sub-floor price must be exactly zero, not noise"
        with pytest.raises(ValueError, match="materially negative"):
            heston_call_prices(deep, UNDERLYING, (2.2 * forward,), short, (CURVE, GRID))

    def test_the_batch_path_adapts_the_grid_ONCE_for_the_maturity(self) -> None:
        """The adaptive truncation is a property of `(model, maturity)`, not of a strike, so the
        batch path chooses one grid for the smile — which is also why sharing `φ` is sound."""
        deep = HestonParams(kappa=2.5, theta=0.06, sigma_v=1.2, rho=-0.3, v0=0.02)
        short = VAL + timedelta(days=7)
        forward = dividend_forward(UNDERLYING, short, CURVE)
        together = heston_call_prices(deep, UNDERLYING, (forward, 1.3 * forward), short, CURVE)
        apart = tuple(
            heston_call_price(deep, UNDERLYING, k, short, CURVE)
            for k in (forward, 1.3 * forward)
        )
        assert together == apart, "the batch must select the same grid the singles do"

    def test_an_empty_strike_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one strike"):
            heston_call_prices(PARAMS, UNDERLYING, (), EXPIRY, (CURVE, GRID))
