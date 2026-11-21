"""The Hull-White European swaption by Jamshidian's trick — tranche 68 piece 1, PRE-REGISTERED.
`MCINERNEY-ZASTAWNIAK` §3.5 (eqs 3.23–3.25).

The tree has had Hull-White's caplet since T10 but never its **swaption**, which the step-1 grep
found and the spec flagged: the Bermudan's one-exercise reduction anchor had to be built before the
Bermudan could have one.

**The source, transcribed from both renderings and reconciled (§4).** §3.5, "Formula for swaptions":

    payoff = (1 − K·Σᵢ τᵢ B(T₀,Tᵢ) − B(T₀,Tₙ))⁺                                   (3.23)

    the critical rate `r̃`:  K·Σᵢ τᵢ F(T₀,r̃;Tᵢ) + F(T₀,r̃;Tₙ) = 1                   (3.24)
    with Kᵢ = F(T₀,r̃;Tᵢ)

    PSwpt(0) = Σᵢ K(Tᵢ−Tᵢ₋₁)·BP(0;T₀,Tᵢ,Kᵢ) + BP(0;T₀,Tₙ,Kₙ)                      (3.25)

**Jamshidian's trick rests on one stated premise, and it is asserted here rather than assumed.** The
source's argument is that `B(t,T)` is a *decreasing* function of `r(t)`, so the coupon bond is too
and the critical rate exists and is unique. That monotonicity is what makes the decomposition valid
at all, so it is a test, not a remark.

**`BP` is `zero_bond_option` with `FLOOR`** — eq (3.17), which tranche 10 already holds and oracled.
So the only genuinely-new code is the `r̃` root-find and the portfolio assembly; the option valuation
is reused whole.

**And the new HW bond has an oracle against already-verified code.** `hw_bond` is the one-factor
case of tranche 67's `g2pp_bond` — identical at `η = 0` for *any* second speed, to 1e-15 — so it is
cross-checked against a function that carried its own anchors one tranche ago rather than against a
second transcription. The tree verifying itself.
"""

import math
from datetime import date

import pytest

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market.curve import DiscountCurve
from pricebook.models.g2pp import G2ppParams, G2ppState, g2pp_bond
from pricebook.models.hull_white import (
    HullWhiteParams,
    bond_option_variance,
    hw_beta,
    hw_bond,
    hw_state_variance,
    zero_bond_option,
)
from pricebook.engine.hw_swaption import (
    SwaptionTerms,
    hw_european_swaption,
    jamshidian_critical_state,
)

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
PARAMS = HullWhiteParams(kappa=0.07, sigma=0.011)
CURVE = DiscountCurve(
    TimeMeasure(VAL, DC), (0.0, 1.0, 3.0, 10.0), (0.028, 0.030, 0.0325, 0.035)
)
#: A 1-into-4 annual payer: expiry at 1.0, settlements at 2…5. Non-unit accruals would be better
#: still, but the source's own example is annual and the accruals are asserted non-degenerate below.
EXPIRY = 1.0
SETTLEMENTS = (2.0, 3.0, 4.0, 5.0)
STRIKE = 0.031


def _terms(strike: float = STRIKE, settlements=SETTLEMENTS) -> SwaptionTerms:
    return SwaptionTerms(expiry=EXPIRY, settlements=settlements, strike=strike)


class TestTheNewBondAgainstVERIFIEDCode:
    """`hw_bond` is the one-factor case of tranche 67's `g2pp_bond`, so it is checked against code
    that carried its own anchors one tranche ago — not against a second transcription. A
    **consistency guard** in the §4 sense (same object), but of the strongest kind: the two were
    written from different equations in different chapters and must agree exactly."""

    @pytest.mark.parametrize("second_speed", [0.11, 0.9])
    @pytest.mark.parametrize("state", [0.0, 0.006, -0.004])
    @pytest.mark.parametrize("t,maturity", [(0.0, 5.0), (1.0, 5.0), (2.0, 9.0)])
    def test_it_equals_the_two_factor_bond_at_zero_second_vol(
        self, second_speed: float, state: float, t: float, maturity: float
    ) -> None:
        assert hw_bond(PARAMS, state, t, maturity, CURVE) == pytest.approx(
            g2pp_bond(
                G2ppParams(PARAMS.kappa, second_speed, PARAMS.sigma, 0.0, 0.0),
                G2ppState(state, 0.0),
                t,
                maturity,
                CURVE,
            ),
            rel=1e-14,
        )

    def test_the_bond_is_the_CURVE_at_inception_and_ONE_at_maturity(self) -> None:
        for maturity in (1.0, 5.0, 9.0):
            assert hw_bond(PARAMS, 0.0, 0.0, maturity, CURVE) == pytest.approx(
                CURVE.df_at(maturity), rel=1e-14
            )
            assert hw_bond(PARAMS, 0.0, maturity, maturity, CURVE) == 1.0

    def test_the_bond_DECREASES_in_the_state_which_is_JAMSHIDIANS_PREMISE(self) -> None:
        """**Not a remark — the thing the decomposition rests on.** The source argues that `B(t,T)`
        is decreasing in `r(t)`, hence so is the coupon bond, hence a unique critical rate exists.
        If it did not hold, (3.24) would have no solution or several and (3.25) would be invalid.
        Asserted across a wide state range and at several maturities."""
        for t, maturity in ((1.0, 5.0), (2.0, 9.0)):
            values = [
                hw_bond(PARAMS, x, t, maturity, CURVE)
                for x in (-0.02, -0.01, 0.0, 0.01, 0.02)
            ]
            assert all(b < a for a, b in zip(values, values[1:])), (t, maturity, values)


class TestTheCriticalState:
    """Eq (3.24): the state at which the coupon bond is worth exactly 1."""

    def test_the_coupon_bond_at_the_critical_state_is_EXACTLY_ONE(self) -> None:
        """A **requirement** oracle, and the defining property — so it survives a transcription
        error that a source-value check would share. The root-find is only correct if this holds."""
        terms = _terms()
        critical = jamshidian_critical_state(PARAMS, terms, CURVE)
        total = sum(
            terms.strike * accrual * hw_bond(PARAMS, critical, terms.expiry, pay, CURVE)
            for accrual, pay in zip(terms.accruals, terms.settlements)
        ) + hw_bond(PARAMS, critical, terms.expiry, terms.settlements[-1], CURVE)
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_the_critical_state_MOVES_with_the_strike(self) -> None:
        """The companion: a higher strike makes the coupon bond worth more at any state, so the
        critical state must rise to offset it. A root-find that ignored its inputs would fail here
        and pass the identity above."""
        criticals = [
            jamshidian_critical_state(PARAMS, _terms(strike=k), CURVE)
            for k in (0.01, 0.031, 0.06)
        ]
        assert all(b > a for a, b in zip(criticals, criticals[1:])), criticals

    def test_the_bond_price_ORDERING_matches_the_state_ORDERING(self) -> None:
        """The source's "`B(T₀,Tᵢ) < Kᵢ` if and only if `r(T₀) > r̃`" — the step that lets the
        coupon-bond payoff be split into per-bond payoffs. Asserted directly, because it is the
        hinge of the decomposition and a sign error there would still produce a number."""
        terms = _terms()
        critical = jamshidian_critical_state(PARAMS, terms, CURVE)
        strikes = [
            hw_bond(PARAMS, critical, terms.expiry, pay, CURVE) for pay in terms.settlements
        ]
        above = critical + 0.01
        below = critical - 0.01
        for pay, strike in zip(terms.settlements, strikes):
            assert hw_bond(PARAMS, above, terms.expiry, pay, CURVE) < strike
            assert hw_bond(PARAMS, below, terms.expiry, pay, CURVE) > strike


class TestTheSwaptionPortfolio:
    def test_the_price_matches_eq_3_25_POINTWISE(self) -> None:
        """The portfolio assembled here from the source's own formula, with `BP` reused from
        tranche 10 — so what is transcribed is the decomposition, not the option maths."""
        terms = _terms()
        critical = jamshidian_critical_state(PARAMS, terms, CURVE)
        expected = 0.0
        for accrual, pay in zip(terms.accruals, terms.settlements):
            strike = hw_bond(PARAMS, critical, terms.expiry, pay, CURVE)
            expected += (
                terms.strike
                * accrual
                * zero_bond_option(
                    CURVE.df_at(terms.expiry),
                    CURVE.df_at(pay),
                    strike,
                    bond_option_variance(PARAMS, terms.expiry, pay),
                    CapFloorKind.FLOOR,
                )
            )
        final = terms.settlements[-1]
        expected += zero_bond_option(
            CURVE.df_at(terms.expiry),
            CURVE.df_at(final),
            hw_bond(PARAMS, critical, terms.expiry, final, CURVE),
            bond_option_variance(PARAMS, terms.expiry, final),
            CapFloorKind.FLOOR,
        )
        assert hw_european_swaption(PARAMS, terms, CURVE) == pytest.approx(expected, rel=1e-14)

    def test_the_swaption_is_POSITIVE_and_RISES_with_volatility(self) -> None:
        """Requirement oracles: an option is worth something, and more vol is worth more. They
        survive a transcription error the pointwise check would share with the source."""
        base = hw_european_swaption(PARAMS, _terms(), CURVE)
        assert base > 0.0
        higher = hw_european_swaption(HullWhiteParams(PARAMS.kappa, 0.02), _terms(), CURVE)
        assert higher > base

    def test_the_price_FALLS_MONOTONICALLY_with_the_strike_and_tends_to_zero(self) -> None:
        """A payer swaption pays `(S − K)⁺` on the swap rate, so it is worth **more** at a lower
        strike and tends to zero as the strike rises — the direction being the point, since this
        file's first draft asserted the opposite and the portfolio was right.

        Measured on this curve (forward swap rate 3.458%): 0.1217 at `K = 0.05%`, 0.0210 at the
        3.1% fixture strike, 8.2e-4 at 5%, 1e-8 at 8%. Monotone decrease plus the vanishing tail is
        a stronger statement than either end alone, and it is what a mis-signed leg in the
        portfolio would break."""
        strikes = (0.0005, 0.01, 0.02, 0.031, 0.05, 0.08)
        prices = [hw_european_swaption(PARAMS, _terms(strike=k), CURVE) for k in strikes]
        assert all(p >= 0.0 for p in prices), prices
        assert all(b < a for a, b in zip(prices, prices[1:])), prices
        assert prices[-1] < 1e-6, prices[-1]
        assert prices[0] > 5.0 * prices[3], prices

    def test_the_fixture_strike_is_NEAR_but_not_AT_the_forward_swap_rate(self) -> None:
        """The premise behind the sweep above and the intrinsic check below: at `K` exactly equal
        to the forward swap rate the zero-vol intrinsic is exactly zero, which would make that
        reduction vacuous. 3.1% against a 3.458% forward — close enough to be a realistic
        at-the-money-ish fixture, far enough that the intrinsic is non-zero."""
        annuity = sum(
            accrual * CURVE.df_at(pay)
            for accrual, pay in zip(_terms().accruals, SETTLEMENTS)
        )
        forward_rate = (CURVE.df_at(EXPIRY) - CURVE.df_at(SETTLEMENTS[-1])) / annuity
        assert abs(STRIKE - forward_rate) > 0.002, (STRIKE, forward_rate)
        assert abs(STRIKE / forward_rate - 1.0) < 0.2

    def test_a_ZERO_VOLATILITY_swaption_is_the_INTRINSIC_value(self) -> None:
        """The exact degenerate case: with no volatility the swap is deterministic, so the swaption
        is worth today's forward-starting swap value when positive and nothing otherwise. An exact
        reduction, and the companion that stops the vol-monotonicity check standing alone."""
        terms = _terms()
        deterministic = HullWhiteParams(PARAMS.kappa, 0.0)
        annuity = sum(
            accrual * CURVE.df_at(pay)
            for accrual, pay in zip(terms.accruals, terms.settlements)
        )
        swap_value = (
            CURVE.df_at(terms.expiry)
            - CURVE.df_at(terms.settlements[-1])
            - terms.strike * annuity
        )
        assert hw_european_swaption(deterministic, terms, CURVE) == pytest.approx(
            max(swap_value, 0.0), abs=1e-12
        )


class TestTheTermsAndItsGuards:
    def test_the_ACCRUALS_are_derived_and_NON_DEGENERATE(self) -> None:
        """The premise behind every `τᵢ` above: the accruals must be real and, ideally, not all
        equal — a uniform schedule cannot tell a dropped `τ` from a dropped constant."""
        terms = _terms(settlements=(2.0, 2.5, 4.0, 5.0))
        assert terms.accruals == (1.0, 0.5, 1.5, 1.0)
        assert len(set(terms.accruals)) > 1

    def test_the_first_accrual_runs_from_the_EXPIRY(self) -> None:
        """`T₀` is the expiry and the first reset, so the first accrual is `T₁ − T₀` — an off-by-one
        here would shift every accrual by one period."""
        assert _terms().accruals[0] == SETTLEMENTS[0] - EXPIRY

    def test_settlements_must_be_ASCENDING_and_AFTER_the_expiry(self) -> None:
        for bad in ((2.0, 1.5), (0.5, 2.0), ()):
            with pytest.raises(ValueError):
                _terms(settlements=bad)

    def test_a_NON_POSITIVE_strike_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strike"):
            _terms(strike=0.0)

    def test_the_STATE_VARIANCE_is_the_one_factor_case_of_the_two(self) -> None:
        """`hw_state_variance` is G2++'s first term, so it inherits that tranche's quadrature
        oracle rather than needing its own transcription."""
        from pricebook.models.g2pp import g2pp_bond_variance

        for t, maturity in ((0.0, 3.0), (1.0, 7.0)):
            assert hw_state_variance(PARAMS, t, maturity) == pytest.approx(
                g2pp_bond_variance(
                    G2ppParams(PARAMS.kappa, 0.5, PARAMS.sigma, 0.0, 0.0), t, maturity
                ),
                rel=1e-14,
            )

    def test_hw_beta_is_the_affine_coefficient(self) -> None:
        assert hw_beta(PARAMS.kappa, 1.0, 5.0) == pytest.approx(
            (1.0 - math.exp(-PARAMS.kappa * 4.0)) / PARAMS.kappa, rel=1e-15
        )
