"""The Vasicek short-rate model — tranche 66, PRE-REGISTERED. `MCINERNEY-ZASTAWNIAK` §3.3, §3.4.

First tranche of the single-curve short-rate arc, and the deliberate counterpoint to the
collateralised arc: this model **makes** the discount curve out of the short rate, so the
single-curve mode of the design invariant is exercised by construction rather than by a flag.

**The source, transcribed from both renderings and reconciled (§4).** The book's notation is its
own and is not the spec's — worth stating because the two differ in a way that matters:

    Vasicek  (§3.4):  dr(t) = (θ − α·r(t))dt + σ dW(t)
    Merton   (§3.3):  dr(t) = α dt + σ dW(t)

So **`α` is the mean-reversion speed in Vasicek and a plain drift in Merton** — the same letter for
two different things, three pages apart — and Vasicek's long-run mean is `θ/α`, not `θ`. The spec
wrote `dr = κ(θ − r)dt + σdW`, which is the *other* common parameterisation; this file follows the
source (§5), so `theta` here is the **drift constant**, not the reversion level.

    D(t,T) = ∫ₜᵀ e^{−α(s−t)} ds = (1 − e^{−α(T−t)})/α                              (3.5)
    E[r(s)] = r(0)e^{−αs} + θ(1 − e^{−αs})/α
    Var[r(s)] = σ²(1 − e^{−2αs})/(2α)            → σ²/(2α) as s → ∞
    B(t,T) = exp( −r(t)D(t,T) − θ∫ₜᵀ D(u,T)du + ½σ²∫ₜᵀ D(u,T)² du )                (3.6)
    Merton: B(t,T) = exp( −r(t)(T−t) − ½α(T−t)² + (1/6)σ²(T−t)³ )                   (3.4)

**`D(t,T)` IS `hw_beta`**, which is the reuse the arc is built on: `(1−e^{−ατ})/α` is the same affine
function tranche 10 defined for Hull-White, and every model in the book's ch 3 shares it. The
agreement is therefore a **consistency guard, not cross-source independence** — one object read
twice (§4's same-formula caveat) — and is labelled as such below.

**The source prints (3.6) as INTEGRALS and leaves the closed form as Exercise 3.2**, which is a gift:
the two integrals are evaluated here by **numerical quadrature** of the printed integrands, which is
a genuinely different computation path from the worked closed form rather than a second
transcription of it. That is §4's derivation-step rule in its strongest available form — the source's
own expression, computed another way.

**The κ→0 reduction is exact in ℝ and NUMERICALLY UNREACHABLE in the closed form**, which the spec's
"exact by construction" did not anticipate. `∫D du = (τ − D)/α` is `0/0` as `α → 0` and the variance
integral is worse (a `0/0` over `α²`), so the closed form suffers catastrophic cancellation long
before the limit: measured against quadrature it is machine-exact at `α = 0.1`, loses precision from
`α ≈ 1e-3` (5e-11) and is **1.4% wrong at `α = 1e-6`**. So the reduction cannot be asserted by
evaluating at a tiny `α`. It is asserted instead as a **convergence rate** over the regime where the
closed form is valid — the T25 rate-detector, and §4's "a reference is code too: it needs its own
accuracy argument in the regime where it is used", here applied to the code rather than the
reference. The approach is **O(α)**, measured.
"""

import math
from datetime import date

import pytest
from scipy.integrate import quad

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market.curve import DiscountCurve
from pricebook.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.hull_white import HullWhiteParams, hw_beta
from pricebook.models.vasicek import (
    MertonParams,
    VasicekModel,
    VasicekParams,
    merton_bond,
    vasicek_bond,
    vasicek_drift_integral,
    vasicek_short_rate_moments,
    vasicek_variance_integral,
)

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
#: **α materially non-zero** — the model's whole point is the parameter the reduction zeroes (T39).
ALPHA, THETA, SIGMA = 0.18, 0.0135, 0.014
SHORT_RATE = 0.031
#: Not one year, so `α·τ` cannot be mistaken for `α`.
HORIZON = 2.4986


def _params(alpha: float = ALPHA, theta: float = THETA, sigma: float = SIGMA) -> VasicekParams:
    return VasicekParams(alpha=alpha, theta=theta, sigma=sigma)


def _model(alpha: float = ALPHA) -> VasicekModel:
    market = MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet(
            {
                CurveKey(CurveRole.DISCOUNT, __import__(
                    "pricebook.foundation.money", fromlist=["Currency"]
                ).Currency("EUR")): DiscountCurve.flat(VAL, 0.03, DC)
            }
        ),
    )
    return VasicekModel(market=market, params=_params(alpha), short_rate=SHORT_RATE)


# ── references, formed here from the printed integrands rather than from the closed forms ──────


def _printed_d(alpha: float, u: float, maturity: float) -> float:
    """Eq (3.5)'s integrand antiderivative, written from the print."""
    return (1.0 - math.exp(-alpha * (maturity - u))) / alpha


def _quadrature_drift_integral(alpha: float, t: float, maturity: float) -> float:
    """`∫ₜᵀ D(u,T) du` by quadrature of the PRINTED integrand — a different computation path."""
    return quad(lambda u: _printed_d(alpha, u, maturity), t, maturity, epsabs=1e-13)[0]


def _quadrature_variance_integral(alpha: float, t: float, maturity: float) -> float:
    return quad(
        lambda u: _printed_d(alpha, u, maturity) ** 2, t, maturity, epsabs=1e-13
    )[0]


def _quadrature_bond(
    params: VasicekParams, short_rate: float, t: float, maturity: float
) -> float:
    """Eq (3.6) exactly as printed, with both integrals done numerically."""
    return math.exp(
        -short_rate * _printed_d(params.alpha, t, maturity)
        - params.theta * _quadrature_drift_integral(params.alpha, t, maturity)
        + 0.5
        * params.sigma
        * params.sigma
        * _quadrature_variance_integral(params.alpha, t, maturity)
    )


class TestThePremisesTheOraclesRestOn:
    def test_the_mean_reversion_is_MATERIALLY_non_zero(self) -> None:
        """T39: the α→0 anchor is exact *because* it kills the mean reversion, so it is blind to the
        whole OU structure. Measured: at α = 0.18 the bond differs from Merton's by 17%."""
        assert ALPHA > 0.05
        vasicek = vasicek_bond(_params(), SHORT_RATE, 0.0, HORIZON)
        merton = merton_bond(MertonParams(drift=THETA, sigma=SIGMA), SHORT_RATE, 0.0, HORIZON)
        assert abs(math.log(vasicek) / math.log(merton) - 1.0) > 0.1

    def test_the_horizon_is_NOT_one_year(self) -> None:
        assert abs(HORIZON - 1.0) > 1.4

    def test_the_LONG_RUN_MEAN_differs_from_the_current_short_rate(self) -> None:
        """`θ/α` is the level the rate reverts to; if it coincided with `r₀` the reversion would be
        invisible. Here 7.5% against a 3.1% spot — a mutation swapping them cannot hide."""
        assert abs(THETA / ALPHA - SHORT_RATE) > 0.02


class TestDIsTheSameObjectAsHwBeta:
    """A **consistency guard, not cross-source independence** — §4's same-formula caveat. Eq (3.5)
    and tranche 10's `hw_beta` are one function, so their agreement is guaranteed by construction
    and proves nothing about either transcription. What it does catch is a future drift between the
    arc's models and the Hull-White machinery they reuse."""

    @pytest.mark.parametrize("alpha", [0.03, 0.18, 0.9])
    @pytest.mark.parametrize("maturity", [0.5, 2.4986, 10.0])
    def test_the_affine_B_agrees_with_hw_beta_BY_CONSTRUCTION(
        self, alpha: float, maturity: float
    ) -> None:
        assert _printed_d(alpha, 0.0, maturity) == pytest.approx(
            hw_beta(alpha, 0.0, maturity), rel=1e-15
        )


class TestTheTwoIntegralsPointwise:
    """§4: the genuinely-new closed forms, each transcribed pointwise — and against **quadrature of
    the source's own printed integrand**, not against a second reading of a closed form the source
    never prints. Exercise 3.2 is the reader's to work; this is the check that it was worked right.

    Both are named surface for the F-6.1 reason: a formula living only inside the bond is never
    actually pinned."""

    @pytest.mark.parametrize("alpha", [0.03, 0.18, 0.45, 1.2])
    @pytest.mark.parametrize("maturity", [0.5, 2.4986, 7.0, 20.0])
    def test_the_DRIFT_integral_matches_quadrature(self, alpha: float, maturity: float) -> None:
        assert vasicek_drift_integral(_params(alpha=alpha), 0.0, maturity) == pytest.approx(
            _quadrature_drift_integral(alpha, 0.0, maturity), rel=1e-11
        )

    @pytest.mark.parametrize("alpha", [0.03, 0.18, 0.45, 1.2])
    @pytest.mark.parametrize("maturity", [0.5, 2.4986, 7.0, 20.0])
    def test_the_VARIANCE_integral_matches_quadrature(
        self, alpha: float, maturity: float
    ) -> None:
        assert vasicek_variance_integral(_params(alpha=alpha), 0.0, maturity) == pytest.approx(
            _quadrature_variance_integral(alpha, 0.0, maturity), rel=1e-10
        )

    def test_the_QUADRATURE_reference_is_a_DIFFERENT_computation_path(self) -> None:
        """The premise that makes the checks above independent rather than circular: the reference
        integrates the printed integrand numerically and shares no algebra with the closed forms —
        it does not even know `α` appears in a denominator. So a slip in Exercise 3.2's working
        would show, which is exactly what the oracle is for."""
        alpha, maturity = 0.45, 7.0
        # The closed forms are rational in `1/α`; the reference is a Gauss rule over `u`.
        closed = vasicek_drift_integral(_params(alpha=alpha), 0.0, maturity)
        assert closed == pytest.approx(
            _quadrature_drift_integral(alpha, 0.0, maturity), rel=1e-11
        )
        # And it would NOT agree with a plausible mis-working — the missing `1/α`.
        assert closed != pytest.approx(maturity - hw_beta(alpha, 0.0, maturity), rel=1e-6)


class TestTheBondPointwise:
    @pytest.mark.parametrize("alpha", [0.03, 0.18, 0.45, 1.2])
    @pytest.mark.parametrize("maturity", [0.5, 2.4986, 7.0, 20.0])
    def test_the_bond_matches_eq_3_6_computed_by_QUADRATURE(
        self, alpha: float, maturity: float
    ) -> None:
        assert vasicek_bond(_params(alpha=alpha), SHORT_RATE, 0.0, maturity) == pytest.approx(
            _quadrature_bond(_params(alpha=alpha), SHORT_RATE, 0.0, maturity), rel=1e-11
        )

    def test_the_bond_is_EXACTLY_ONE_at_its_own_maturity(self) -> None:
        """`B(T,T) = 1` — a requirement the finance forces, so it survives a transcription error the
        quadrature reference would share with the closed form (both read eq 3.6)."""
        for maturity in (0.5, 2.4986, 20.0):
            assert vasicek_bond(_params(), SHORT_RATE, maturity, maturity) == 1.0

    def test_the_bond_FALLS_with_maturity_at_these_parameters(self) -> None:
        """Not asserted as a law — a Gaussian short rate can produce a rising bond curve at extreme
        parameters — but measured here, and it is what makes the fixture a sane one to oracle on."""
        bonds = [vasicek_bond(_params(), SHORT_RATE, 0.0, t) for t in (0.5, 2.0, 5.0, 20.0)]
        assert all(b < a for a, b in zip(bonds, bonds[1:])), bonds


class TestTheShortRateMoments:
    """Pinned against the printed closed forms, and the stationary limit **measured** rather than
    pre-asserted as a shape (§4's obligation-not-form clause)."""

    @pytest.mark.parametrize("horizon", [0.25, 1.0, 2.4986, 10.0, 60.0])
    def test_the_mean_and_variance_match_the_print(self, horizon: float) -> None:
        mean, variance = vasicek_short_rate_moments(_params(), SHORT_RATE, horizon)
        decay = math.exp(-ALPHA * horizon)
        assert mean == pytest.approx(
            SHORT_RATE * decay + THETA * (1.0 - decay) / ALPHA, rel=1e-14
        )
        assert variance == pytest.approx(
            SIGMA * SIGMA * (1.0 - math.exp(-2.0 * ALPHA * horizon)) / (2.0 * ALPHA),
            rel=1e-14,
        )

    def test_the_STATIONARY_variance_is_measured_to_SATURATE(self) -> None:
        """The structural fact separating a mean-reverting short rate from a driftless one: the
        variance is bounded by `σ²/(2α)` where Merton's grows linearly. Measured across a sweep and
        checked against the closed form; the approach *shape* is reported, not asserted."""
        stationary = SIGMA * SIGMA / (2.0 * ALPHA)
        measured = [
            vasicek_short_rate_moments(_params(), SHORT_RATE, t)[1] / stationary
            for t in (1.0, 5.0, 20.0, 100.0)
        ]
        assert measured[-1] == pytest.approx(1.0, rel=1e-12)
        # Reported from the measurement: a quarter of the stationary variance is reached inside a
        # year and 99.9% inside twenty, which is what "mean reverting" buys.
        assert 0.2 < measured[0] < 0.4, measured
        assert measured[2] > 0.999, measured

    def test_the_mean_tends_to_THETA_OVER_ALPHA_not_to_theta(self) -> None:
        """The notational trap this file's header names: the long-run mean is `θ/α`. A reader (or a
        mutation) carrying the other parameterisation would expect `θ`, and the two differ by a
        factor of `α` — 5.6× here."""
        mean, _ = vasicek_short_rate_moments(_params(), SHORT_RATE, 500.0)
        assert mean == pytest.approx(THETA / ALPHA, rel=1e-12)
        assert abs(THETA / ALPHA - THETA) > 0.05


class TestTheMertonReduction:
    """The anchor — and the correction to the spec, which called it "exact by construction"."""

    def test_MERTON_matches_its_own_printed_closed_form(self) -> None:
        """Eq (3.4), transcribed pointwise. Merton is its own model in the source (§3.3), not a
        limit to be evaluated, which is the whole reason it gets its own function."""
        for maturity in (0.5, 2.4986, 7.0):
            expected = math.exp(
                -SHORT_RATE * maturity
                - 0.5 * THETA * maturity**2
                + SIGMA * SIGMA * maturity**3 / 6.0
            )
            assert merton_bond(
                MertonParams(drift=THETA, sigma=SIGMA), SHORT_RATE, 0.0, maturity
            ) == pytest.approx(expected, rel=1e-14)

    def test_the_reduction_is_a_CONVERGENCE_RATE_not_an_evaluation_at_zero(self) -> None:
        """**The spec said "exact by construction"; it is exact in ℝ and numerically unreachable.**

        `∫D du = (τ − D)/α` is `0/0` as `α → 0`, and the variance integral is a `0/0` over `α²`, so
        the closed form suffers catastrophic cancellation long before the limit. Measured against
        quadrature: machine-exact at `α = 0.1`, 5e-11 at `α = 1e-3`, and **1.4% wrong at `α = 1e-6`**
        — the formula diverges from the limit rather than approaching it to within a few ULP, so
        this is not the T61 "band it" case but a stated-regime case.

        So the reduction is asserted as the **rate** of approach over the regime where the closed
        form is valid (the T25 rate-detector): the gap to Merton is **O(α)**, falling by ~3.3× for
        each 3.3× in `α`, measured across `α` from 0.1 down to 1e-3 where the closed form's own
        error stays below 1e-10 — three orders of separation between signal and noise.
        """
        merton = math.log(
            merton_bond(MertonParams(drift=THETA, sigma=SIGMA), SHORT_RATE, 0.0, HORIZON)
        )
        gaps = []
        for alpha in (0.1, 0.03, 0.01, 3e-3, 1e-3):
            vasicek = math.log(vasicek_bond(_params(alpha=alpha), SHORT_RATE, 0.0, HORIZON))
            # The closed form must still be accurate here, or the gap is measuring arithmetic.
            quadrature = math.log(
                _quadrature_bond(_params(alpha=alpha), SHORT_RATE, 0.0, HORIZON)
            )
            assert abs(vasicek / quadrature - 1.0) < 1e-9, (alpha, vasicek, quadrature)
            gaps.append((alpha, abs(vasicek - merton)))
        # O(alpha): each ~3.3x drop in alpha must cut the gap by ~3.3x, never by 1x or 10x.
        for (coarse, wide), (fine, narrow) in zip(gaps, gaps[1:]):
            ratio = (wide / narrow) / (coarse / fine)
            assert 0.8 < ratio < 1.25, (coarse, fine, wide, narrow, ratio)
        assert gaps[-1][1] < 0.01 * gaps[0][1]

    def test_the_CLOSED_FORM_is_asserted_to_DEGRADE_so_the_regime_is_stated(self) -> None:
        """The companion that makes the regime claim checkable rather than rhetorical, and the
        reason the test above stops at `α = 1e-3`. At `α = 1e-6` the closed form disagrees with
        quadrature by more than 1%, so an oracle that had evaluated the limit there would have been
        measuring cancellation and calling it a reduction."""
        alpha = 1e-6
        closed = math.log(vasicek_bond(_params(alpha=alpha), SHORT_RATE, 0.0, HORIZON))
        quadrature = math.log(_quadrature_bond(_params(alpha=alpha), SHORT_RATE, 0.0, HORIZON))
        assert abs(closed / quadrature - 1.0) > 0.005, (closed, quadrature)
        # And at the regime's edge it is still sound, which is what makes the edge an edge.
        edge = math.log(vasicek_bond(_params(alpha=1e-3), SHORT_RATE, 0.0, HORIZON))
        edge_reference = math.log(
            _quadrature_bond(_params(alpha=1e-3), SHORT_RATE, 0.0, HORIZON)
        )
        assert abs(edge / edge_reference - 1.0) < 1e-9


class TestTheGuards:
    def test_a_NON_POSITIVE_mean_reversion_is_refused(self) -> None:
        """Every closed form divides by `α`, so zero is not a slow model but an undefined one — and
        the α→0 limit is approached analytically, never evaluated. `HullWhiteParams` sets the
        precedent."""
        for alpha in (0.0, -0.1):
            with pytest.raises(ValueError, match="mean reversion"):
                _params(alpha=alpha)

    def test_a_NEGATIVE_volatility_and_NON_FINITE_parameters_are_refused(self) -> None:
        with pytest.raises(ValueError, match="volatility"):
            _params(sigma=-0.1)
        with pytest.raises(ValueError, match="finite"):
            _params(theta=float("nan"))

    def test_MERTON_refuses_a_negative_volatility_too(self) -> None:
        with pytest.raises(ValueError, match="volatility"):
            MertonParams(drift=THETA, sigma=-0.1)

    def test_a_NEGATIVE_horizon_is_refused_rather_than_extrapolated(self) -> None:
        with pytest.raises(ValueError, match="before"):
            vasicek_bond(_params(), SHORT_RATE, 2.0, 1.0)

    def test_the_PARAMS_are_their_own_type_not_HullWhiteParams_plus_theta(self) -> None:
        """§6, and the ruling the spec left to Code. Vasicek's `α` and `σ` play the same algebraic
        role as Hull-White's, and `D` *is* `hw_beta` — but Vasicek's `θ` is a **constant** while
        Hull-White's `θ(t)` is fitted to the initial curve, which is the difference between a model
        that produces a curve and one that reproduces it. Same SDE family, different idea; the LGM
        contrast is the line (there the numbers literally *are* Hull-White's)."""
        assert not isinstance(_params(), HullWhiteParams)
        assert _model().params.alpha == ALPHA
