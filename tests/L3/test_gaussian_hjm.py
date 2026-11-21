"""One-factor Gaussian HJM — tranche 69, PRE-REGISTERED. `MCINERNEY-ZASTAWNIAK` §4.1–4.2.

The framework the short-rate models are special cases of: HJM models the **instantaneous forward
rate** directly, and the whole chapter turns on one equation.

**The source, transcribed from both renderings and reconciled (§4).** §4.1 derives it rather than
quoting it, which gives **two printed forms of one condition** — the integrated statement and its
derivative:

    df(t,T) = α(t,T)dt + σ(t,T)dW(t)                                             (4.1)
    ∫ₜᵀ α(t,u)du = ½(∫ₜᵀ σ(t,u)du)²              the integrated no-arbitrage condition
    α(t,T)       = σ(t,T)·∫ₜᵀ σ(t,u)du           its derivative in `T`        (Theorem 4.1)

**The drift condition is a REQUIREMENT, not a source value** — it is forced by no-arbitrage (the
discounted bond must be driftless), so it holds for *every* `σ(t,T)` and survives a transcription
error a source-value oracle would share (§4). It is asserted both ways: against the integrated form
the source also prints, and at a **humped, non-Hull-White** vol where nothing about the HW structure
can be doing the work.

**And the reduction is the arc's strongest anchor: already-verified code.** At
`σ(t,T) = σ·e^{−κ(T−t)}` the Gaussian HJM bond-option variance

    ν(0,S) = ∫₀ˢ (∫ₛᵀ σ(s,u)du)² ds

**is** tranche 10's `bond_option_variance` — measured identical to **1.3e-15** — so this tranche's
one new closed form is checked against code that has carried its own oracles for fifty-nine
tranches. Not a transcription; the tree verifying itself (the T67 form, and the pair Cowork named at
T68: *reduce to what is already proved*, or *pin by an identity the answer must satisfy*). Flagged
as a consistency guard, not cross-source independence (§4's same-object caveat).
"""

import math
from datetime import date

import pytest
from scipy.integrate import quad

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market.curve import DiscountCurve
from pricebook.models.gaussian_hjm import (
    GaussianHjmModel,
    hjm_bond_option_variance,
    hjm_drift,
    humped_volatility,
    hull_white_volatility,
)
from pricebook.models.hull_white import (
    HullWhiteParams,
    bond_option_variance,
    zero_bond_option,
)

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
HW = HullWhiteParams(kappa=0.07, sigma=0.011)
CURVE = DiscountCurve(
    TimeMeasure(VAL, DC), (0.0, 1.0, 3.0, 10.0), (0.028, 0.030, 0.0325, 0.035)
)
#: A **humped** vol — genuinely not Hull-White, which is the T39 companion's whole job.
HUMP = humped_volatility(level=0.006, slope=0.012, decay=0.45)
HW_VOL = hull_white_volatility(HW)


class TestTheDriftConditionIsAREQUIREMENT:
    """§4's requirement-oracle rule, in its cleanest available form: the condition is forced by
    no-arbitrage, so it holds for **every** `σ(t,T)` and a transcription error cannot hide behind a
    matching source value. Asserted against the integrated form the source *also* prints — so the
    two printed forms check each other (the derivation-step rule)."""

    @pytest.mark.parametrize("volatility,name", [(HW_VOL, "hull-white"), (HUMP, "humped")])
    @pytest.mark.parametrize("t,maturity", [(0.5, 3.0), (1.0, 7.0), (0.0, 10.0)])
    def test_the_INTEGRATED_and_DIFFERENTIAL_forms_agree(
        self, volatility, name: str, t: float, maturity: float
    ) -> None:
        """`∫ₜᵀ α(t,u)du = ½(∫ₜᵀ σ(t,u)du)²` against `α = σ·∫σ` integrated numerically. Two
        printed forms, two computation paths, and it must hold for the humped vol as well as the
        Hull-White one — which is what makes it a requirement rather than a coincidence."""
        integrated = quad(
            lambda u: hjm_drift(volatility, t, u), t, maturity, epsabs=1e-14
        )[0]
        total_vol = quad(lambda u: volatility(t, u), t, maturity, epsabs=1e-15)[0]
        assert integrated == pytest.approx(0.5 * total_vol * total_vol, rel=1e-12), name

    @pytest.mark.parametrize("volatility", [HW_VOL, HUMP])
    def test_the_drift_is_TRANSCRIBED_pointwise_as_well(self, volatility) -> None:
        """T12(b): a requirement oracle never *replaces* a pointwise transcription — the two catch
        different failures. The reference forms its own `∫σ` by quadrature."""
        for t, maturity in ((0.5, 3.0), (1.0, 7.0)):
            expected = volatility(t, maturity) * quad(
                lambda u: volatility(t, u), t, maturity, epsabs=1e-15
            )[0]
            assert hjm_drift(volatility, t, maturity) == pytest.approx(expected, rel=1e-13)

    def test_the_drift_VANISHES_at_the_maturity_and_is_POSITIVE_before_it(self) -> None:
        """Requirement companions: the inner integral is empty at `T = t`, and both factors are
        positive for a positive vol — so a sign error in either is visible."""
        for volatility in (HW_VOL, HUMP):
            assert hjm_drift(volatility, 1.0, 1.0) == 0.0
            assert hjm_drift(volatility, 1.0, 5.0) > 0.0

    def test_the_two_VOLS_are_genuinely_DIFFERENT_structures(self) -> None:
        """The premise behind every parametrisation above: if the humped vol were near the Hull-White
        one, the requirement checks would all be exercising one structure twice. Measured at the
        peak of the hump."""
        assert HUMP(0.0, 2.0) / HW_VOL(0.0, 2.0) > 1.25
        # And the hump genuinely humps: it rises then falls, where the HW vol only decays.
        humped = [HUMP(0.0, T) for T in (0.5, 2.0, 5.0, 10.0)]
        assert humped[1] > humped[0] and humped[-1] < humped[1], humped
        decaying = [HW_VOL(0.0, T) for T in (0.5, 2.0, 5.0, 10.0)]
        assert all(b < a for a, b in zip(decaying, decaying[1:])), decaying


class TestTheReductionToALREADYVERIFIEDCode:
    """**The arc's strongest anchor.** At the Hull-White vol the HJM bond-option variance *is*
    tranche 10's `bond_option_variance` — so the one new closed form here is checked against code
    that has carried its own oracles for fifty-nine tranches, rather than against a second
    transcription of itself.

    A **consistency guard, not cross-source independence** (§4's same-object caveat): they are one
    quantity reached two ways. What it catches is this tranche's integral being wrong, which is
    exactly what it is for."""

    @pytest.mark.parametrize("expiry,maturity", [(1.0, 3.0), (2.0, 7.0), (0.5, 10.0)])
    def test_at_the_HULL_WHITE_vol_it_IS_the_held_variance(
        self, expiry: float, maturity: float
    ) -> None:
        assert hjm_bond_option_variance(HW_VOL, expiry, maturity) == pytest.approx(
            bond_option_variance(HW, expiry, maturity), rel=1e-13
        )

    def test_the_COMPANION_is_non_degenerate_in_the_VOL_STRUCTURE(self) -> None:
        """**T39.** The Hull-White reduction is exact *because* it collapses `σ(t,T)` to one
        exponential — which is everything the framework generalises — so the anchor is blind to the
        whole vol structure. The humped companion differs measurably: 2.6% at `(1,3)` and rising
        with maturity, so the general integral is doing real work."""
        gaps = []
        for expiry, maturity in ((1.0, 3.0), (2.0, 7.0), (0.5, 10.0)):
            humped = hjm_bond_option_variance(HUMP, expiry, maturity)
            held = bond_option_variance(HW, expiry, maturity)
            gaps.append(abs(humped / held - 1.0))
        assert all(gap > 0.02 for gap in gaps), gaps

    def test_the_variance_VANISHES_at_a_zero_expiry_and_a_zero_span(self) -> None:
        """Both degenerate directions, which bracket the double integral: no time to accumulate
        variance, and no bond span to accumulate it over."""
        for volatility in (HW_VOL, HUMP):
            assert hjm_bond_option_variance(volatility, 0.0, 5.0) == pytest.approx(0.0, abs=1e-18)
            assert hjm_bond_option_variance(volatility, 2.0, 2.0) == pytest.approx(0.0, abs=1e-18)

    def test_the_variance_RISES_with_the_SPAN_at_a_fixed_expiry(self) -> None:
        """A requirement the double integral forces: a longer bond has more exposure to the factor
        at every `s`, so the inner integral grows and so does its square."""
        for volatility in (HW_VOL, HUMP):
            by_span = [hjm_bond_option_variance(volatility, 1.0, T) for T in (2.0, 5.0, 10.0)]
            assert all(b > a for a, b in zip(by_span, by_span[1:])), by_span

    def test_the_variance_is_NOT_MONOTONE_in_the_expiry_and_that_is_the_mathematics(self) -> None:
        """**Stated because the obvious claim is false, and asserting it would have been wrong.**

        "More time to accumulate" suggests the variance rises with the option expiry. It does not:
        two effects compete, since moving `S` later lengthens the outer accumulation but **shortens
        the bond span `T − S`**, which shrinks the inner integral. Measured at `T = 10` for the
        Hull-White vol: 2.81e-3 at `S = 0.5`, **7.92e-3 at `S = 2`**, then 7.74e-3 at `S = 5` — it
turns over.

        Measured at `T = 10`, the peak sits near `S = 3` for **both** vols — Hull-White: 2.81e-3,
        5.03e-3, 7.92e-3, **9.08e-3**, 7.74e-3, 3.95e-3, 5.78e-4 across `S = 0.5…9`; humped the same
        shape one notch lower. So the sample straddles the measured turning point rather than
        guessing where it is: an earlier draft sampled `S = 2` as the peak and passed for Hull-White
        while failing for the humped vol, which is the fixture-chosen-by-assumption trap in
        miniature.

        What is asserted is the turnover itself, which is stronger than either direction alone and
        is what a one-sided monotonicity claim would have hidden. (§4's obligation-not-form clause,
        met by measuring rather than by assuming a shape.)"""
        for volatility in (HW_VOL, HUMP):
            early, peak, late = (
                hjm_bond_option_variance(volatility, S, 10.0) for S in (0.5, 3.0, 7.0)
            )
            assert peak > early, (early, peak, late)
            assert late < peak, (early, peak, late)

    def test_the_variance_VANISHES_as_the_expiry_REACHES_the_maturity(self) -> None:
        """The limit the turnover points at: with no bond left to be uncertain about, the option has
        no variance however long it accumulated for."""
        for volatility in (HW_VOL, HUMP):
            closing = [
                hjm_bond_option_variance(volatility, 10.0 - gap, 10.0)
                for gap in (2.0, 0.5, 0.05)
            ]
            assert all(b < a for a, b in zip(closing, closing[1:])), closing
            assert closing[-1] < 0.01 * closing[0], closing


class TestTheBondOptionReusesTheHeldPricer:
    def test_the_option_is_zero_bond_option_on_the_HJM_variance(self) -> None:
        """Gaussian forwards make the bond lognormal, so the option is the held closed form on this
        tranche's variance — the only new input. `zero_bond_option` is reused, not re-derived."""
        from pricebook.market.curve_set import CurveKey, CurveRole, CurveSet
        from pricebook.market.snapshot import MarketSnapshot
        from pricebook.foundation.money import Currency

        market = MarketSnapshot(
            valuation_date=VAL,
            curves=CurveSet({CurveKey(CurveRole.DISCOUNT, Currency("EUR")): CURVE}),
        )
        model = GaussianHjmModel(market=market, volatility=HUMP)
        expiry, maturity, strike = 1.0, 5.0, 0.86
        variance = hjm_bond_option_variance(HUMP, expiry, maturity)
        call = zero_bond_option(
            CURVE.df_at(expiry), CURVE.df_at(maturity), strike, variance, CapFloorKind.CAP
        )
        put = zero_bond_option(
            CURVE.df_at(expiry), CURVE.df_at(maturity), strike, variance, CapFloorKind.FLOOR
        )
        assert call > 0.0
        # Put-call parity, which the variance cancels out of entirely — so it survives a
        # transcription error in the variance and checks the discounting instead.
        assert call - put == pytest.approx(
            CURVE.df_at(maturity) - strike * CURVE.df_at(expiry), rel=1e-12
        )
        assert model.volatility is HUMP

    def test_a_HUMPED_vol_prices_DIFFERENTLY_from_the_hull_white_one(self) -> None:
        """The companion that makes the framework worth having: if the two vols priced the same, HJM
        would add nothing over the short-rate model it reduces to."""
        expiry, maturity, strike = 1.0, 5.0, 0.86
        prices = [
            zero_bond_option(
                CURVE.df_at(expiry),
                CURVE.df_at(maturity),
                strike,
                hjm_bond_option_variance(volatility, expiry, maturity),
                CapFloorKind.CAP,
            )
            for volatility in (HW_VOL, HUMP)
        ]
        assert abs(prices[1] / prices[0] - 1.0) > 0.01, prices


class TestTheVolatilitiesAndTheirGuards:
    def test_the_HULL_WHITE_vol_is_the_printed_exponential(self) -> None:
        for t, maturity in ((0.0, 3.0), (1.0, 7.0)):
            assert HW_VOL(t, maturity) == pytest.approx(
                HW.sigma * math.exp(-HW.kappa * (maturity - t)), rel=1e-15
            )

    def test_the_HULL_WHITE_vol_depends_only_on_the_SPAN(self) -> None:
        """A time-homogeneous vol — `σ(t,T) = σ(0,T−t)` — which is what makes the short-rate model a
        *special* case rather than merely a possible one. A mutation reading `T` instead of `T−t`
        dies here and would survive a single-`t` fixture."""
        for span in (1.0, 4.0):
            assert HW_VOL(0.0, span) == pytest.approx(HW_VOL(2.0, 2.0 + span), rel=1e-15)

    def test_a_NEGATIVE_span_is_refused_by_both_vols(self) -> None:
        for volatility in (HW_VOL, HUMP):
            with pytest.raises(ValueError, match="before"):
                volatility(5.0, 2.0)

    def test_the_humped_vol_refuses_a_NEGATIVE_level_or_decay(self) -> None:
        for kwargs in ({"level": -0.01}, {"decay": 0.0}):
            with pytest.raises(ValueError):
                humped_volatility(**{"level": 0.006, "slope": 0.012, "decay": 0.45, **kwargs})

    def test_the_drift_and_the_variance_REFUSE_an_inverted_interval_IN_THEIR_OWN_WORDS(
        self,
    ) -> None:
        """**The T44b reachability shape, found by the sweep.** Dropping either outer guard still
        raises — the volatility callable refuses the inverted interval itself once `quad` reaches
        it — so a test matching only `"before"` passes either way and pins nothing. Both survived
        the whole suite until this test named the message.

        The guards are kept rather than deleted (they are not the §4 third-survivor case of dead
        code): each refuses at the level the caller asked at, naming the *operation* rather than the
        volatility, and each fires before any integration is wasted. But a guard whose removal still
        raises has to be pinned by its message, which is the only thing that distinguishes it."""
        with pytest.raises(ValueError, match="a drift cannot be asked"):
            hjm_drift(HW_VOL, 5.0, 2.0)
        with pytest.raises(ValueError, match="a bond cannot mature"):
            hjm_bond_option_variance(HW_VOL, 5.0, 2.0)
        # And the volatility's own refusal is a *different* message, which is what makes the two
        # above discriminating rather than incidental.
        with pytest.raises(ValueError, match="a forward volatility cannot be asked"):
            HW_VOL(5.0, 2.0)
