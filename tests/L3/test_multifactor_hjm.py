"""Multi-factor Gaussian HJM — tranche 70, PRE-REGISTERED. `MCINERNEY-ZASTAWNIAK` §4.4.

T69 with **n driving factors**. Transcribed from both renderings and reconciled (§4), eq (4.10) and
Theorem 4.6:

    df(t,T) = α(t,T)dt + Σᵢ σᵢ(t,T)dWᵢ(t)                                       (4.10)
    α(t,T)  = Σᵢ σᵢ(t,T)·∫ₜᵀ σᵢ(t,u)du                                          (4.11)

**Two corrections to the spec, both from the source, and the second removes an oracle.** The spec
wrote per-factor drifts `αₖ` and an explicit factor correlation `ρₖⱼ`. The source has neither:

1. There is **one** drift `α(t,T)` — the forward rate has a single drift and `n` diffusion terms,
   not `n` drifts.
2. The Brownian motions are **independent** ("the forward-rate process is driven by `n`
   *independent* Brownian motions", §4.4). So there is no `ρₖⱼ` parameter at all.

The second matters beyond notation: the spec anticipated the T17/T43 **correlation discipline**
returning here, and it does not, because there is no correlation parameter to pin. Correlation
between forward rates of *different maturities* is an **output** of the vol vector, not an input —
which is the framework's actual content. Where G2++ had a `ρ` to pin two ways, this model has a
structure to check instead.

**And with independent factors every quantity is a plain SUM of T69's**, which is why this tranche
is short:

    α(t,T)  = Σᵢ (T69's drift for σᵢ)
    ν(0,S)  = Σᵢ (T69's variance for σᵢ)      — no cross terms; independence removes them

So the implementation **calls T69's functions and sums them**, and the `N = 1` reduction is
**bit-identical by construction** — a one-element sum is the summand — rather than exact to a
tolerance. That is the arc's reduce-to-already-verified-code anchor in its strongest possible form:
not agreement, identity.
"""

import math

import pytest
from scipy.integrate import quad

from pricebook.models.gaussian_hjm import (
    hjm_bond_option_variance,
    hjm_drift,
    hull_white_volatility,
    humped_volatility,
)
from pricebook.models.hull_white import HullWhiteParams, bond_option_variance
from pricebook.models.multifactor_hjm import (
    MultiFactorHjmModel,
    multifactor_bond_option_variance,
    multifactor_drift,
)

HW = HullWhiteParams(kappa=0.07, sigma=0.011)
#: Three genuinely different structures: a decaying exponential, a hump, and a second hump with a
#: different decay — so no two factors are interchangeable and a dropped term cannot hide.
FIRST = hull_white_volatility(HW)
SECOND = humped_volatility(level=0.004, slope=0.009, decay=0.60)
THIRD = humped_volatility(level=0.002, slope=0.004, decay=0.18)
FACTORS = (FIRST, SECOND, THIRD)


class TestThePremises:
    def test_the_THREE_factors_are_genuinely_DIFFERENT_structures(self) -> None:
        """The premise behind every sum below: equal factors would make a dropped term look like a
        scaling, and a swapped pair invisible. Measured at three spans, all three differ."""
        for span in (0.5, 2.0, 7.0):
            values = [volatility(0.0, span) for volatility in FACTORS]
            assert len(set(round(v, 12) for v in values)) == 3, (span, values)

    def test_the_SECOND_and_THIRD_factors_are_not_scalar_multiples(self) -> None:
        """Stronger than "different values": two factors proportional to each other would behave as
        one factor with a rescaled vol, so the fixture would be `N = 2` in disguise."""
        ratios = [SECOND(0.0, span) / THIRD(0.0, span) for span in (0.5, 2.0, 7.0)]
        assert max(ratios) / min(ratios) > 1.5, ratios


class TestTheDriftConditionIsAREQUIREMENT:
    """Theorem 4.6's factor sum. Still forced by no-arbitrage, so it holds for **every** set of
    vols — §4's requirement-oracle case, inherited from T69 and generalised."""

    @pytest.mark.parametrize("t,maturity", [(0.5, 3.0), (1.0, 7.0), (0.0, 10.0)])
    def test_the_INTEGRATED_and_FACTOR_SUM_forms_agree(self, t: float, maturity: float) -> None:
        """The source's §4.1 derivation generalises: `∫ₜᵀ α(t,u)du = ½Σᵢ(∫ₜᵀ σᵢ(t,u)du)²`, which is
        the integrated condition **per factor** because the factors are independent. Two forms, two
        computation paths — and it is the independence that lets the squares be summed rather than
        a full quadratic form appearing."""
        integrated = quad(
            lambda u: multifactor_drift(FACTORS, t, u), t, maturity, epsabs=1e-14
        )[0]
        per_factor = sum(
            quad(lambda u, v=v: v(t, u), t, maturity, epsabs=1e-15)[0] ** 2 for v in FACTORS
        )
        assert integrated == pytest.approx(0.5 * per_factor, rel=1e-11)

    def test_the_drift_is_TRANSCRIBED_pointwise_as_the_factor_sum(self) -> None:
        """T12(b): the requirement never replaces the transcription. The reference forms each
        factor's `σᵢ(t,T)·∫σᵢ` independently and sums — eq (4.11) as printed."""
        for t, maturity in ((0.5, 3.0), (1.0, 7.0)):
            expected = sum(
                v(t, maturity) * quad(lambda u, v=v: v(t, u), t, maturity, epsabs=1e-15)[0]
                for v in FACTORS
            )
            assert multifactor_drift(FACTORS, t, maturity) == pytest.approx(expected, rel=1e-13)

    def test_DROPPING_a_factor_from_the_sum_CHANGES_the_drift(self) -> None:
        """The companion that gives the sum its teeth: each of the three terms must be worth
        something, or a mutation dropping one would be invisible. Measured per factor."""
        full = multifactor_drift(FACTORS, 0.5, 5.0)
        for index in range(3):
            without = multifactor_drift(
                tuple(v for i, v in enumerate(FACTORS) if i != index), 0.5, 5.0
            )
            assert abs(without / full - 1.0) > 0.01, (index, full, without)

    def test_the_drift_VANISHES_at_the_maturity(self) -> None:
        assert multifactor_drift(FACTORS, 1.0, 1.0) == 0.0


class TestTheReductionIsBITIDENTICAL:
    """**The strongest form the arc's reduce-to-verified-code anchor takes: identity, not
    agreement.** With independent factors every quantity is a plain sum of T69's, so a one-element
    sum *is* its summand — the `N = 1` model returns the same floats T69 does, bit for bit.

    A **consistency guard, not independence** (§4's same-object caveat) — and here that caveat is
    exact rather than approximate, since it is literally the same call."""

    @pytest.mark.parametrize("t,maturity", [(0.5, 3.0), (1.0, 7.0)])
    def test_ONE_factor_drift_is_BIT_IDENTICAL_to_T69(self, t: float, maturity: float) -> None:
        assert multifactor_drift((FIRST,), t, maturity) == hjm_drift(FIRST, t, maturity)

    @pytest.mark.parametrize("expiry,maturity", [(1.0, 5.0), (2.0, 10.0)])
    def test_ONE_factor_variance_is_BIT_IDENTICAL_to_T69(
        self, expiry: float, maturity: float
    ) -> None:
        assert multifactor_bond_option_variance(
            (FIRST,), expiry, maturity
        ) == hjm_bond_option_variance(FIRST, expiry, maturity)

    def test_ONE_factor_at_the_HULL_WHITE_vol_reaches_T10s_held_variance(self) -> None:
        """The chain in full, which is worth asserting end to end: multi-factor at `N = 1` is T69,
        and T69 at the Hull-White vol is tranche 10's `bond_option_variance`. Three tranches, one
        number — so a defect anywhere along the chain shows here."""
        for expiry, maturity in ((1.0, 3.0), (2.0, 7.0)):
            assert multifactor_bond_option_variance(
                (hull_white_volatility(HW),), expiry, maturity
            ) == pytest.approx(bond_option_variance(HW, expiry, maturity), rel=1e-13)

    def test_the_COMPANION_is_non_degenerate_in_the_FACTOR_COUNT(self) -> None:
        """**T39.** The `N = 1` reduction is exact *because* it removes every factor but one —
        which is everything this tranche adds — so the anchor is blind to the sum entirely. The
        three-factor variance is measurably larger: +43% at `(1,5)`, and each factor contributes."""
        one = multifactor_bond_option_variance((FIRST,), 1.0, 5.0)
        three = multifactor_bond_option_variance(FACTORS, 1.0, 5.0)
        assert three / one - 1.0 > 0.2, (one, three)
        for index in range(3):
            without = multifactor_bond_option_variance(
                tuple(v for i, v in enumerate(FACTORS) if i != index), 1.0, 5.0
            )
            assert without < three, index


class TestTheVarianceIsASUMWithNOCrossTerms:
    """The structural consequence of the source's **independent** Brownians, and the reason this
    tranche has no correlation oracle: independence removes the cross terms a correlated model (like
    G2++) would carry, so the variance is a sum of squares rather than a quadratic form."""

    @pytest.mark.parametrize("expiry,maturity", [(1.0, 5.0), (2.0, 10.0)])
    def test_it_equals_the_sum_of_the_PER_FACTOR_variances(
        self, expiry: float, maturity: float
    ) -> None:
        assert multifactor_bond_option_variance(FACTORS, expiry, maturity) == pytest.approx(
            sum(hjm_bond_option_variance(v, expiry, maturity) for v in FACTORS), rel=1e-13
        )

    def test_it_equals_a_SINGLE_integral_of_the_summed_squares(self) -> None:
        """The same quantity by a different computation path: one outer integral over the sum of
        squared exposures, rather than a sum of outer integrals. Equal because integration is
        linear — and it is the check that the per-factor decomposition was not an approximation."""
        expiry, maturity = 1.0, 5.0

        def exposure(s: float, volatility) -> float:
            return quad(lambda u: volatility(s, u), expiry, maturity, epsabs=1e-15)[0]

        direct = quad(
            lambda s: sum(exposure(s, v) ** 2 for v in FACTORS), 0.0, expiry, epsabs=1e-16
        )[0]
        assert multifactor_bond_option_variance(FACTORS, expiry, maturity) == pytest.approx(
            direct, rel=1e-11
        )

    def test_the_factors_are_ORDER_INDEPENDENT_which_a_cross_term_would_break(self) -> None:
        """A requirement the independence forces, and a sharp probe: a sum of squares is symmetric
        in the factors, where a cross term `ρᵢⱼσᵢσⱼ` would not be unless `ρ` were symmetric too. So
        a mutation introducing a spurious asymmetric cross term dies here."""
        import itertools

        values = [
            multifactor_bond_option_variance(order, 1.0, 5.0)
            for order in itertools.permutations(FACTORS)
        ]
        assert max(values) == pytest.approx(min(values), rel=1e-14), values

    def test_the_variance_RISES_with_EVERY_factor_added(self) -> None:
        """Each factor contributes a non-negative square, so adding one can only increase the
        variance — strictly here, since none of the three is identically zero."""
        running = [
            multifactor_bond_option_variance(FACTORS[:n], 1.0, 5.0) for n in (1, 2, 3)
        ]
        assert all(b > a for a, b in zip(running, running[1:])), running


class TestTheModelAndItsGuards:
    def test_the_model_carries_the_FACTORS_as_an_ordered_tuple(self) -> None:
        from datetime import date

        from pricebook.foundation.day_count import DayCountConvention
        from pricebook.foundation.money import Currency
        from pricebook.foundation.time_measure import TimeMeasure
        from pricebook.market.curve import DiscountCurve
        from pricebook.market.curve_set import CurveKey, CurveRole, CurveSet
        from pricebook.market.snapshot import MarketSnapshot

        valuation = date(2026, 1, 15)
        curve = DiscountCurve.flat(valuation, 0.03, DayCountConvention.ACT_365_FIXED)
        market = MarketSnapshot(
            valuation_date=valuation,
            curves=CurveSet({CurveKey(CurveRole.DISCOUNT, Currency("EUR")): curve}),
        )
        model = MultiFactorHjmModel(market=market, volatilities=FACTORS)
        assert len(model.volatilities) == 3
        assert model.volatilities[0] is FIRST

    def test_an_EMPTY_factor_set_is_refused(self) -> None:
        """A zero-factor HJM is a deterministic forward curve, not a model — and the drift and
        variance would both be zero, which is a plausible wrong number rather than a failure."""
        with pytest.raises(ValueError, match="at least one factor"):
            multifactor_drift((), 0.0, 5.0)
        with pytest.raises(ValueError, match="at least one factor"):
            multifactor_bond_option_variance((), 1.0, 5.0)

    def test_an_INVERTED_interval_is_refused_IN_THE_OPERATIONS_OWN_WORDS(self) -> None:
        """The T69 lesson carried forward: dropping these guards still raises (the per-factor
        function refuses), so the message is what discriminates. Pinned here rather than
        rediscovered by the sweep."""
        with pytest.raises(ValueError, match="a multi-factor drift cannot be asked"):
            multifactor_drift(FACTORS, 5.0, 2.0)
        with pytest.raises(ValueError, match="a multi-factor bond variance cannot be asked"):
            multifactor_bond_option_variance(FACTORS, 5.0, 2.0)
        # And tranche 69's per-factor function refuses with a DIFFERENT message, which is what makes
        # the two above discriminating. The sweep found this: the multi-factor variance originally
        # reused T69's wording verbatim, so bypassing its guard still raised the same text and no
        # test could tell. The drift escaped only because its message already differed.
        from pricebook.models.gaussian_hjm import hjm_bond_option_variance as per_factor

        with pytest.raises(ValueError, match="a bond cannot mature"):
            per_factor(FIRST, 5.0, 2.0)
