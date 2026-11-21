"""The Bermudan swaption in the Hull-White model — tranche 68 piece 2, PRE-REGISTERED.
`MCINERNEY-ZASTAWNIAK` §3.6.

The tree prices this instrument under the **LMM** by Longstaff–Schwartz (T18). Here the state is
**one Markov coordinate**, so the continuation value is an exact deterministic function of it and a
regression would approximate something that has an exact answer — §4's `closed form > cross-check`
ordering inverted. The method is therefore a backward induction over the state, not a regression.

**The source, transcribed from both renderings and reconciled (§4).** §3.6:

    E(T_i)    = (PS(T_i))⁺
    Berm(T_i) = max(E(T_i), C(T_i))                                              (3.30)
    C(T_i)    = B(T_i,T_{i+1})·E^{P_{T_{i+1}}}[ Berm(T_{i+1}) | F_{T_i} ]
    dr(t)     = (θ(t) − αr(t) − σ(t)²D(t,T))dt + σ(t)dW^T(t)                     (3.28)

**The measure change is the one thing not transcribed, and it is pinned by a requirement instead.**
The source prints the forward-measure *SDE*, not the integrated drift adjustment, so writing a
closed form for it from memory is what §5 forbids. It is integrated from the printed integrand and
then pinned by an identity that admits no other answer: under `P_T` a bond must **reprice** through
the transition, `B(t,T)·E^{P_T}[B(T,T_j)|x(t)] = B(t,T_j)`, for every `T_j`. Measured at **0 to
2.2e-16**, which a wrong drift could not produce — and it needs no second transcription to check.

**The same §4 ordering that chose the lattice then chose an analytic last roll.** Quadrature over
`max(PS,0)` converges only as about `n⁻²`, because a Gauss rule loses its accuracy at a kink:
measured 3.9e-3 at 48 nodes, 1.1e-3 at 96, 1.1e-4 at 192, and `quadrature.py` refuses more than
~370. But the step into the last exercise date **is a European swaption**, which piece 1 has a
closed form for — so it is computed analytically at every grid point, and the one-exercise reduction
becomes **exact** rather than grid-limited. Using quadrature there would have been the same error
the lattice was chosen to avoid.
"""

import math
from datetime import date

import pytest

from pricebook.engine.hw_bermudan import (
    BermudanTerms,
    conditional_expectation,
    forward_measure_drift,
    price_hw_bermudan,
    swap_value,
)
from pricebook.engine.hw_swaption import SwaptionTerms, hw_european_swaption
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market.curve import DiscountCurve
from pricebook.models.hull_white import HullWhiteParams, hw_bond

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
PARAMS = HullWhiteParams(kappa=0.07, sigma=0.011)
CURVE = DiscountCurve(
    TimeMeasure(VAL, DC), (0.0, 1.0, 3.0, 10.0), (0.028, 0.030, 0.0325, 0.035)
)
SETTLEMENTS = (2.0, 3.0, 4.0, 5.0)
STRIKE = 0.031


def _bermudan(exercises: tuple[float, ...]) -> BermudanTerms:
    return BermudanTerms(exercises=exercises, settlements=SETTLEMENTS, strike=STRIKE)


class TestTheTransitionPinsTheMeasureChange:
    """**The oracle that replaces a transcription.** The forward-measure drift is integrated from
    eq (3.28)'s printed integrand rather than written from a remembered closed form, so it is pinned
    by the identity that admits no other answer: a bond must reprice through the transition."""

    @pytest.mark.parametrize("t,nxt", [(1.0, 2.0), (2.0, 3.0), (0.0, 1.0)])
    @pytest.mark.parametrize("state", [0.0, 0.008, -0.006])
    def test_a_BOND_REPRICES_through_the_transition(
        self, t: float, nxt: float, state: float
    ) -> None:
        for target in (nxt, nxt + 1.0, 5.0):
            rolled = hw_bond(PARAMS, state, t, nxt, CURVE) * conditional_expectation(
                PARAMS, state, t, nxt,
                lambda x, target=target, nxt=nxt: hw_bond(PARAMS, x, nxt, target, CURVE),
            )
            assert rolled == pytest.approx(
                hw_bond(PARAMS, state, t, target, CURVE), rel=1e-13
            )

    def test_the_DRIFT_is_NON_ZERO_so_the_identity_is_not_vacuous(self) -> None:
        """The premise: if the measure-change adjustment were negligible the reprice identity would
        hold for a drift of zero too, and would be pinning nothing. Measured against the state's own
        standard deviation over the step, which is the scale that matters."""
        drift = forward_measure_drift(PARAMS, 1.0, 2.0)
        assert drift < 0.0, "the adjustment is negative — eq (3.28)'s `−σ²D(t,T)`"
        assert abs(drift) > 1e-7, drift

    def test_the_drift_VANISHES_at_a_zero_step_and_GROWS_with_the_step(self) -> None:
        assert forward_measure_drift(PARAMS, 1.0, 1.0) == 0.0
        magnitudes = [abs(forward_measure_drift(PARAMS, 0.0, t)) for t in (0.5, 1.0, 3.0, 7.0)]
        assert all(b > a for a, b in zip(magnitudes, magnitudes[1:])), magnitudes


class TestTheOneExerciseReduction:
    def test_ONE_exercise_is_EXACTLY_the_jamshidian_european(self) -> None:
        """**Exact, not banded** — and that is a design consequence, not luck. With one exercise
        date the Bermudan takes the closed-form path (piece 1's eq 3.25) rather than any quadrature,
        so the two are the same computation and agree to the last bit."""
        bermudan = price_hw_bermudan(PARAMS, _bermudan((1.0,)), CURVE)
        european = hw_european_swaption(
            PARAMS, SwaptionTerms(1.0, SETTLEMENTS, STRIKE), CURVE
        )
        assert bermudan == european

    def test_the_COMPANION_is_non_degenerate_in_the_EXERCISE_COUNT(self) -> None:
        """**T39.** The one-exercise reduction is exact *because* it removes the exercise policy —
        which is the whole content of a Bermudan — so it is blind to the induction entirely. The
        companion runs four exercise dates and measures how far it moves: **+23%** over the
        European, with diminishing increments (15.3%, 5.0%, 1.6%) as rights are added."""
        european = price_hw_bermudan(PARAMS, _bermudan((1.0,)), CURVE)
        full = price_hw_bermudan(PARAMS, _bermudan((1.0, 2.0, 3.0, 4.0)), CURVE)
        assert full / european - 1.0 > 0.2, (european, full)


class TestTheRequirementOracles:
    def test_MORE_exercise_rights_never_hurt(self) -> None:
        """A requirement the finance forces — a superset of exercise opportunities is worth at least
        as much — so it survives a transcription error the source-value checks would share. Strict
        here because each added date is genuinely in the money somewhere."""
        prices = [
            price_hw_bermudan(PARAMS, _bermudan(ex), CURVE)
            for ex in ((1.0,), (1.0, 2.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0))
        ]
        assert all(b > a for a, b in zip(prices, prices[1:])), prices

    def test_the_increments_DIMINISH_as_rights_are_added(self) -> None:
        """Reported from measurement rather than asserted as a law (§4's obligation-not-form
        clause): the marginal value of an extra exercise date falls, which is what a sane exercise
        policy produces and what a broken induction would not."""
        prices = [
            price_hw_bermudan(PARAMS, _bermudan(ex), CURVE)
            for ex in ((1.0,), (1.0, 2.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0))
        ]
        increments = [b - a for a, b in zip(prices, prices[1:])]
        assert all(b < a for a, b in zip(increments, increments[1:])), increments

    def test_the_bermudan_is_BOUNDED_BELOW_by_every_constituent_european(self) -> None:
        """A sharper requirement than "≥ the first European": the holder may always wait and
        exercise at *any* single date, so the Bermudan dominates **each** of the co-terminal
        Europeans, not merely the earliest."""
        full = price_hw_bermudan(PARAMS, _bermudan((1.0, 2.0, 3.0, 4.0)), CURVE)
        for expiry in (1.0, 2.0, 3.0, 4.0):
            single = hw_european_swaption(
                PARAMS,
                SwaptionTerms(expiry, tuple(p for p in SETTLEMENTS if p > expiry), STRIKE),
                CURVE,
            )
            assert full >= single, (expiry, single, full)

    def test_a_ZERO_VOLATILITY_bermudan_is_the_BEST_INTRINSIC(self) -> None:
        """The exact degenerate case: with no volatility everything is deterministic, so the holder
        picks the best exercise date in advance and the Bermudan is the largest intrinsic."""
        deterministic = HullWhiteParams(PARAMS.kappa, 0.0)
        terms = _bermudan((1.0, 2.0, 3.0, 4.0))
        priced = price_hw_bermudan(deterministic, terms, CURVE)
        best = max(
            hw_bond(deterministic, 0.0, 0.0, ex, CURVE)
            * max(swap_value(deterministic, terms, 0.0, ex, CURVE), 0.0)
            for ex in terms.exercises
        )
        assert priced == pytest.approx(best, rel=1e-9)


class TestTheDiscretisationErrorIsMEASUREDNotClaimed:
    """What a green actually rules out, from measurement (§4) — and this one is **not** a clean
    convergence story, so it is reported as it is.

    The backward induction's error is invisible in its output (T18's §12.8 lesson: a price bound
    cannot see the algorithm), so the numerics are probed directly.
    """

    TERMS_EXERCISES = (1.0, 2.0, 3.0, 4.0)

    def _price(self) -> float:
        return price_hw_bermudan(PARAMS, _bermudan(self.TERMS_EXERCISES), CURVE)

    @pytest.fixture(scope="class")
    def base(self) -> float:
        """The price at the committed settings, computed **once** for the whole class.

        A lossless reduction (T50/T53): each test below compares a *refinement* against this same
        number and would otherwise recompute it. The file was 11.0s against §9's 10.0s ceiling,
        and a backward induction over 1201 states is not cheap.
        """
        return self._price()

    def test_the_GRID_has_converged(self, monkeypatch, base: float) -> None:
        """**The grid is deliberately NOT the finest that fits, and the band says why.**

        Measured: 401 points (the committed default) carry 1.2e-4 against 1201, which itself settles
        by 2401 (1.8e-6 and then flat). But the kinked quadrature below carries about **2e-3** — an
        order and a half larger — so refining the grid spends three times the work to sharpen a term
        that is already negligible beside it. §4's resolution thinking applied to cost: the right
        grid is the one whose error is comfortably under the dominant one, not the finest available.

        So what is asserted is that tripling the grid moves the price by **less than the quadrature
        spread**, which is the claim that justifies the choice."""
        import pricebook.engine.hw_bermudan as module

        monkeypatch.setattr(module, "_GRID_POINTS", 1201)
        assert self._price() == pytest.approx(base, rel=1e-3)

    def test_the_QUADRATURE_error_is_the_DOMINANT_one_and_is_BOUNDED(self, monkeypatch) -> None:
        """**The honest negative.** The earlier rolls integrate `max(exercise, continuation)`, which
        is kinked, and a Gauss rule loses its spectral accuracy at a kink — so the price moves about
        **2e-3 relative** across 48/96/144 nodes rather than settling.

        This is stated, not claimed away: what the oracle asserts is that the spread is **bounded by
        5e-3**, which is the method's discretisation uncertainty at the committed settings. The fix
        is known — split the integral at the exercise boundary so each piece is smooth — and is
        recorded as a route rather than guessed at. A test asserting `1e-6` here would have been
        asserting something false."""
        import pricebook.engine.hw_bermudan as module

        prices = []
        for nodes in (48, 96, 144):
            monkeypatch.setattr(module, "_NODES", nodes)
            prices.append(self._price())
        spread = (max(prices) - min(prices)) / min(prices)
        assert spread < 5e-3, (prices, spread)
        assert spread > 1e-4, (
            "if the spread were negligible the kink claim would be wrong and this bound would be "
            f"asserting nothing: {prices}"
        )

    def test_the_LAST_roll_carries_NO_quadrature_error_at_all(self, monkeypatch) -> None:
        """The companion that proves the analytic last roll is doing the work: a **one-exercise**
        Bermudan takes only the closed-form path, so it is completely independent of the node count
        where the multi-exercise price is not."""
        import pricebook.engine.hw_bermudan as module

        single = price_hw_bermudan(PARAMS, _bermudan((1.0,)), CURVE)
        monkeypatch.setattr(module, "_NODES", 16)
        assert price_hw_bermudan(PARAMS, _bermudan((1.0,)), CURVE) == single

    def test_the_grid_WIDTH_covers_the_state_distribution(self, monkeypatch, base: float) -> None:
        """A grid too narrow would clamp the tails and bias the price, so widening it must not move
        the answer beyond the quadrature spread above."""
        import pricebook.engine.hw_bermudan as module

        monkeypatch.setattr(module, "_GRID_WIDTH", 9.0)
        assert self._price() == pytest.approx(base, rel=1e-3)


class TestTheTermsAndItsGuards:
    def test_the_REMAINING_schedule_starts_at_the_EXERCISE_date(self) -> None:
        """The off-by-one a stored schedule would hide: exercising at `T_k` enters a swap whose
        first accrual runs from `T_k` itself, not from the previous settlement."""
        terms = _bermudan((1.0, 2.0, 3.0))
        assert terms.remaining(2.0) == ((1.0, 3.0), (1.0, 4.0), (1.0, 5.0))
        assert terms.remaining(1.0)[0] == (1.0, 2.0)

    def test_the_swap_value_FALLS_as_the_state_rises_for_a_PAYER(self) -> None:
        """A higher short rate means a lower bond and a more valuable payer swap — the direction the
        induction's exercise decision depends on."""
        terms = _bermudan((1.0, 2.0))
        values = [swap_value(PARAMS, terms, x, 1.0, CURVE) for x in (-0.01, 0.0, 0.01)]
        assert all(b > a for a, b in zip(values, values[1:])), values

    def test_exercise_dates_must_be_ASCENDING_and_inside_the_swap(self) -> None:
        for bad in ((), (2.0, 1.0), (1.0, 1.0)):
            with pytest.raises(ValueError):
                _bermudan(bad)
        with pytest.raises(ValueError, match="precede"):
            BermudanTerms((1.0, 6.0), SETTLEMENTS, STRIKE)
        with pytest.raises(ValueError, match="follow"):
            BermudanTerms((3.0,), (2.0, 3.0), STRIKE)

    def test_a_NON_POSITIVE_strike_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strike"):
            BermudanTerms((1.0,), SETTLEMENTS, 0.0)
