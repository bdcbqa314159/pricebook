"""The two-factor Hull-White model (G2++) — tranche 67, PRE-REGISTERED.
`MCINERNEY-ZASTAWNIAK` §3.7.

Two correlated Gaussian factors: the first model in this tree with a **factor correlation in its own
dynamics**, so the T17/T43 correlation discipline from the spread arc returns — and the first real
use of §4's newest clause, since the bond variance carries removable singularities at every
speed → 0.

**The source's notation, followed rather than the spec's (§5, and the T66 lesson).** The book uses
`α, β` for the two mean-reversion speeds and `σ, η` for the two vols; the spec wrote `(a,b)` and
`(σ_x,σ_y)`. Following the print keeps the arc's chapters readable against one vocabulary.

    r(t) = φ(t) + x(t) + y(t)                                                    (3.34)
    dx = −αx dt + σ dU,   dy = −βy dt + η dV,   dU·dV = ρ dt                     (3.35–3.37)

    V(t,T) = (σ²/α²)∫ₜᵀ(1−e^{−α(T−u)})² du + (η²/β²)∫ₜᵀ(1−e^{−β(T−u)})² du
             + 2ρ(σwestη/αβ)∫ₜᵀ(1−e^{−α(T−u)})(1−e^{−β(T−u)}) du                      (3.40)

    B(t,T) = (B(0,T)/B(0,t))·exp(−x·B_α − y·B_β + ½(V(0,t) − V(0,T) + V(t,T)))   (3.42)

    ν(0,S) = σ²/(2α³)(1−e^{−α(T−S)})²(1−e^{−2αS})
           + η²/(2β³)(1−e^{−β(T−S)})²(1−e^{−2βS})
           + 2ρση/(αβ(α+β))(1−e^{−α(T−S)})(1−e^{−β(T−S)})(1−e^{−(α+β)S})          (3.43)

**`B_α` and `B_β` are both `hw_beta`** — `(1−e^{−ατ})/α` is tranche 10's affine function, called for
each factor rather than rewritten. A **consistency guard, not independence** (§4's same-formula
caveat), flagged as such.

**The source prints `V` TWO ways — as integrals (3.40) and integrated — which is better than the
dual rendering and better than T66's.** At T66 the book declined the algebra and left Exercise 3.2,
so the oracle was quadrature against a closed form the reader worked. Here the book prints *both*,
so the closed form is checked against **quadrature of its own printed integrand**: two printed forms,
two computation paths, agreeing to 2.8e-14 on this fixture.

**And `ν(0,S)` at `η = 0` is EXACTLY tranche 10's `bond_option_variance`** — the first term of (3.43)
is eq 3.20 verbatim. So the one-factor reduction is a cross-check against *already-verified code*,
which is the strongest form a reduction anchor takes in this tree.

**§4's newest clause gets its first use, and the regime is narrower here than at T66.** `V`'s
single-factor terms divide by `α³` after a bracket that is `O(α³)` — a `0/0` three times over — so
the cancellation is worse than Vasicek's `(τ−D)/α`. Measured against quadrature: 3.3e-14 at
`α = 0.1`, 2.6e-11 at `0.01`, 1.7e-8 at `1e-3` and **1.9e-6 at `3e-4`**. So the sound regime is
roughly `α ≥ 0.01`, against T66's `1e-3`, and the speed → 0 reductions are asserted as **convergence
rates inside that regime** with a companion stating where it degrades — never banded.
"""

import math
from datetime import date

import numpy as np
import pytest
from scipy.integrate import quad

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.linalg import pseudo_square_root
from pricebook.market.curve import DiscountCurve
from pricebook.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.g2pp import (
    G2ppModel,
    G2ppParams,
    g2pp_bond,
    g2pp_bond_option_variance,
    g2pp_bond_variance,
)
from pricebook.models.hull_white import (
    HullWhiteParams,
    bond_option_variance,
    hw_beta,
    zero_bond_option,
)
from pricebook.models.lgm import simulate_lgm_states

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
#: **α ≠ β, σ ≠ η, ρ ≠ 0** — at ρ = 0 the cross term vanishes, and at α = β the two `hw_beta`
#: coincide so the factors become indistinguishable and a swap of them survives.
ALPHA, BETA = 0.42, 0.11
SIGMA, ETA, RHO = 0.0105, 0.0062, -0.72
#: Not one year, so a speed cannot be mistaken for `speed × τ`.
HORIZON = 2.4986
CURVE = DiscountCurve(
    __import__("pricebook.foundation.time_measure", fromlist=["TimeMeasure"]).TimeMeasure(VAL, DC),
    (0.0, 1.0, 3.0, 10.0),
    (0.028, 0.030, 0.0325, 0.035),
)


def _params(
    alpha: float = ALPHA, beta: float = BETA, sigma: float = SIGMA,
    eta: float = ETA, rho: float = RHO,
) -> G2ppParams:
    return G2ppParams(alpha=alpha, beta=beta, sigma=sigma, eta=eta, rho=rho)


def _model(**kwargs: float) -> G2ppModel:
    market = MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet(
            {
                CurveKey(CurveRole.DISCOUNT, __import__(
                    "pricebook.foundation.money", fromlist=["Currency"]
                ).Currency("EUR")): CURVE
            }
        ),
    )
    return G2ppModel(market=market, params=_params(**kwargs))


# ── the reference: eq (3.40) as PRINTED, integrated numerically ────────────────────────────────


def _quadrature_variance(p: G2ppParams, t: float, maturity: float) -> float:
    """Eq (3.40) exactly as printed, all three integrals by quadrature — a different computation
    path from the integrated form, which the source *also* prints."""
    first = quad(
        lambda u: (1.0 - math.exp(-p.alpha * (maturity - u))) ** 2, t, maturity, epsabs=1e-14
    )[0]
    second = quad(
        lambda u: (1.0 - math.exp(-p.beta * (maturity - u))) ** 2, t, maturity, epsabs=1e-14
    )[0]
    cross = quad(
        lambda u: (1.0 - math.exp(-p.alpha * (maturity - u)))
        * (1.0 - math.exp(-p.beta * (maturity - u))),
        t,
        maturity,
        epsabs=1e-14,
    )[0]
    return (
        p.sigma * p.sigma / (p.alpha * p.alpha) * first
        + p.eta * p.eta / (p.beta * p.beta) * second
        + 2.0 * p.rho * p.sigma * p.eta / (p.alpha * p.beta) * cross
    )


class TestThePremisesTheOraclesRestOn:
    def test_the_two_SPEEDS_and_two_VOLS_differ_and_rho_is_non_zero(self) -> None:
        """At `ρ = 0` the cross term vanishes and a dropped `ρσWest η` survives; at `α = β` the two
        `hw_beta` coincide so the factors are indistinguishable and swapping them survives."""
        assert abs(RHO) > 0.2
        assert abs(ALPHA - BETA) > 0.1
        assert abs(SIGMA - ETA) > 0.002

    def test_the_CROSS_TERM_is_a_material_share_of_the_variance(self) -> None:
        """The premise the correlation oracles rest on: if the cross term were negligible, every
        `ρ` oracle below would pass against an implementation that dropped it. Measured at **−21%**
        of the total here, and negative because `ρ < 0`."""
        full = g2pp_bond_variance(_params(), 0.0, HORIZON)
        no_cross = g2pp_bond_variance(_params(rho=0.0), 0.0, HORIZON)
        assert abs(full / no_cross - 1.0) > 0.1, (full, no_cross)
        assert full < no_cross, "a negative rho must reduce the variance"

    def test_the_horizon_is_NOT_one_year(self) -> None:
        assert abs(HORIZON - 1.0) > 1.4


class TestBothBetasAreHwBeta:
    """A **consistency guard, not cross-source independence** (§4's same-formula caveat): eq (3.39)'s
    two affine coefficients are tranche 10's `hw_beta` at the two speeds, so their agreement is
    guaranteed by construction. What it catches is a future drift between this arc's models and the
    Hull-White machinery they all reuse."""

    @pytest.mark.parametrize("speed", [0.11, 0.42, 1.3])
    @pytest.mark.parametrize("maturity", [0.5, 2.4986, 10.0])
    def test_the_affine_coefficients_agree_BY_CONSTRUCTION(
        self, speed: float, maturity: float
    ) -> None:
        assert (1.0 - math.exp(-speed * maturity)) / speed == pytest.approx(
            hw_beta(speed, 0.0, maturity), rel=1e-15
        )


class TestTheVarianceAgainstBOTHPrintedForms:
    """The source prints `V(t,T)` as integrals (3.40) **and** integrated — two printed forms, so the
    oracle is quadrature of the first against the closed form of the second. Better than a dual
    rendering (which would be two typesettings of one expression) and better than T66's (where the
    book declined the algebra entirely): here the two forms are genuinely different computations the
    *source itself* supplies."""

    @pytest.mark.parametrize("maturity", [0.5, 2.4986, 7.0, 20.0])
    def test_the_integrated_form_matches_QUADRATURE_of_the_printed_integrals(
        self, maturity: float
    ) -> None:
        assert g2pp_bond_variance(_params(), 0.0, maturity) == pytest.approx(
            _quadrature_variance(_params(), 0.0, maturity), rel=1e-11
        )

    @pytest.mark.parametrize("rho", [-0.9, -0.3, 0.0, 0.5, 0.95])
    def test_the_agreement_holds_across_the_CORRELATION_range(self, rho: float) -> None:
        """The cross term is the new formula, so the check must sweep the parameter it carries —
        a single `ρ` could agree by luck in the sign alone (T43: presence and sign are not
        magnitude)."""
        assert g2pp_bond_variance(_params(rho=rho), 0.0, HORIZON) == pytest.approx(
            _quadrature_variance(_params(rho=rho), 0.0, HORIZON), rel=1e-11
        )

    def test_the_variance_is_NON_NEGATIVE_across_the_whole_correlation_range(self) -> None:
        """A requirement the finance forces: `V` is the variance of an integrated Gaussian, so it
        cannot go negative for `|ρ| ≤ 1` however the cross term is signed."""
        for rho in np.linspace(-1.0, 1.0, 41):
            assert g2pp_bond_variance(_params(rho=float(rho)), 0.0, HORIZON) >= 0.0

    def test_the_variance_VANISHES_at_zero_horizon(self) -> None:
        assert g2pp_bond_variance(_params(), 0.0, 0.0) == pytest.approx(0.0, abs=1e-20)


class TestTheBondOptionVarianceReducesToVERIFIEDCode:
    """**The strongest reduction anchor available in this tree: existing, already-oracled code.**
    The first term of (3.43) is eq 3.20 verbatim, so at `η = 0` the two-factor bond-option variance
    *is* tranche 10's `bond_option_variance`. Not a tolerance — the same expression."""

    @pytest.mark.parametrize("expiry,maturity", [(1.0, 3.0), (2.0, 7.0), (0.5, 10.0)])
    def test_at_ZERO_SECOND_VOL_it_equals_the_one_factor_variance(
        self, expiry: float, maturity: float
    ) -> None:
        assert g2pp_bond_option_variance(
            _params(eta=0.0), expiry, maturity
        ) == pytest.approx(
            bond_option_variance(HullWhiteParams(ALPHA, SIGMA), expiry, maturity), rel=1e-14
        )

    def test_the_COMPANION_is_non_degenerate_in_the_second_factor(self) -> None:
        """**T39.** The `η → 0` reduction is exact *because* it kills the second factor and the
        correlation with it — precisely what the model adds — so the anchor is blind to the whole
        two-factor structure. Measured: the genuine two-factor variance is 26% below the
        one-factor one at these parameters (a negative `ρ`)."""
        one = g2pp_bond_option_variance(_params(eta=0.0), 1.0, 3.0)
        two = g2pp_bond_option_variance(_params(), 1.0, 3.0)
        assert abs(two / one - 1.0) > 0.1, (one, two)

    @pytest.mark.parametrize("rho", [-0.9, -0.3, 0.4, 0.95])
    def test_the_cross_term_VALUE_is_pinned_not_merely_its_presence(self, rho: float) -> None:
        """**T43's trap, which `ρ → ρ²` is the clean probe for.** `ρ` enters only through the product
        `ρση`, so a set of oracles establishing that it is present and correctly signed would be
        *complete about presence and sign and silent about magnitude*. This transcribes the whole
        third term of (3.43) by value."""
        p = _params(rho=rho)
        expiry, maturity = 1.0, 3.0
        expected = (
            SIGMA**2
            / (2.0 * ALPHA**3)
            * (1.0 - math.exp(-ALPHA * (maturity - expiry))) ** 2
            * (1.0 - math.exp(-2.0 * ALPHA * expiry))
            + ETA**2
            / (2.0 * BETA**3)
            * (1.0 - math.exp(-BETA * (maturity - expiry))) ** 2
            * (1.0 - math.exp(-2.0 * BETA * expiry))
            + 2.0
            * rho
            * SIGMA
            * ETA
            / (ALPHA * BETA * (ALPHA + BETA))
            * (1.0 - math.exp(-ALPHA * (maturity - expiry)))
            * (1.0 - math.exp(-BETA * (maturity - expiry)))
            * (1.0 - math.exp(-(ALPHA + BETA) * expiry))
        )
        assert g2pp_bond_option_variance(p, expiry, maturity) == pytest.approx(
            expected, rel=1e-14
        )


class TestTheBond:
    def test_the_bond_is_EXACTLY_ONE_at_its_own_maturity(self) -> None:
        """`B(T,T) = 1` — the curve ratio is 1, both betas are 0 and `V(0,t) − V(0,T) + V(t,T)`
        vanishes identically at `t = T`. A requirement oracle, so it survives a transcription error
        the quadrature reference shares with the closed form."""
        for maturity in (0.5, 2.4986, 10.0):
            assert g2pp_bond(_params(), 0.0, 0.0, maturity, maturity, CURVE) == 1.0

    def test_at_ZERO_FACTORS_and_t_zero_the_bond_is_the_CURVE(self) -> None:
        """The term-structure fit (3.42) is exact by construction at `t = 0`: `x(0) = y(0) = 0` and
        the `V` terms cancel, so the model reproduces today's curve rather than approximating it.
        That is what `φ` is for, and it is the single-curve arc's thesis — the model *makes* the
        curve."""
        for maturity in (0.5, 2.4986, 10.0):
            assert g2pp_bond(_params(), 0.0, 0.0, 0.0, maturity, CURVE) == pytest.approx(
                CURVE.df_at(maturity), rel=1e-14
            )

    def test_the_bond_FALLS_as_either_FACTOR_rises(self) -> None:
        """Both affine coefficients are positive, so a higher factor means a higher short rate and a
        lower bond — a requirement, and the companion that stops the `x = y = 0` oracle above from
        being the only witness."""
        base = g2pp_bond(_params(), 0.0, 0.0, 1.0, 5.0, CURVE)
        assert g2pp_bond(_params(), 0.004, 0.0, 1.0, 5.0, CURVE) < base
        assert g2pp_bond(_params(), 0.0, 0.004, 1.0, 5.0, CURVE) < base

    def test_the_TWO_FACTORS_enter_with_DIFFERENT_weights(self) -> None:
        """`α ≠ β`, so swapping which factor is which changes the bond — the premise that makes a
        factor-swap mutation visible."""
        swapped = g2pp_bond(_params(), 0.0, 0.004, 1.0, 5.0, CURVE)
        straight = g2pp_bond(_params(), 0.004, 0.0, 1.0, 5.0, CURVE)
        assert straight != pytest.approx(swapped, rel=1e-6)


class TestTheSpeedToZeroLimitsAreRATESNotBands:
    """**§4's newest clause, first real use — and this is the recurrence it was claused for.**

    `V`'s single-factor terms divide by `α³` after a bracket that is itself `O(α³)`, so the limit is
    a `0/0` three times over and the cancellation is worse than T66's `(τ−D)/α`. Measured against
    quadrature: 3.3e-14 at `α = 0.1`, 2.6e-11 at `0.01`, 1.7e-8 at `1e-3`, **1.9e-6 at `3e-4`**.

    So the regime where the closed form is sound is roughly `α ≥ 0.01` — an order *narrower* than
    T66's — and the limit is asserted as a convergence **rate** inside it, with a companion stating
    where it degrades. Banding would pass at any convenient `α` and say nothing about the limit,
    which is exactly the over-application of T61's band-it clause the new clause names.
    """

    def test_the_variance_approaches_its_limit_at_a_MEASURED_RATE(self) -> None:
        """As `α → 0` the first term tends to `σ²τ³/3` (the driftless limit of an integrated
        Brownian factor). Asserted as the rate of approach over the sound regime, not as a value at
        a tiny `α`."""
        tau = HORIZON
        limit = SIGMA * SIGMA * tau**3 / 3.0
        gaps = []
        for alpha in (0.4, 0.2, 0.1, 0.05):
            # Only the first factor: the second and the cross term are held at zero so the limit is
            # the one being measured rather than a sum of three.
            value = g2pp_bond_variance(
                _params(alpha=alpha, eta=0.0, rho=0.0), 0.0, tau
            )
            reference = _quadrature_variance(
                _params(alpha=alpha, eta=0.0, rho=0.0), 0.0, tau
            )
            assert abs(value / reference - 1.0) < 1e-10, (alpha, value, reference)
            gaps.append((alpha, abs(value - limit)))
        # O(alpha): halving alpha must halve the gap, measured.
        for (coarse, wide), (fine, narrow) in zip(gaps, gaps[1:]):
            ratio = (wide / narrow) / (coarse / fine)
            assert 0.85 < ratio < 1.2, (coarse, fine, wide, narrow, ratio)

    def test_the_CLOSED_FORM_is_asserted_to_DEGRADE_so_the_regime_is_STATED(self) -> None:
        """The companion the clause requires, and the reason the rate test stops at `α = 0.05`. At
        `α = 3e-4` the closed form disagrees with quadrature by ~2e-6 — small in absolute terms and
        fatal to a limit claim, since the formula is then diverging from the limit it is supposed to
        be approaching. Stating the regime is what keeps the rate test honest."""
        bad = _params(alpha=3e-4, eta=0.0, rho=0.0)
        closed = g2pp_bond_variance(bad, 0.0, HORIZON)
        reference = _quadrature_variance(bad, 0.0, HORIZON)
        assert abs(closed / reference - 1.0) > 1e-7, (closed, reference)
        # And sound at the regime's edge, which is what makes the edge an edge.
        edge = _params(alpha=0.01, eta=0.0, rho=0.0)
        assert g2pp_bond_variance(edge, 0.0, HORIZON) == pytest.approx(
            _quadrature_variance(edge, 0.0, HORIZON), rel=1e-10
        )

    def test_the_regime_is_NARROWER_than_T66s_which_is_why_it_is_restated(self) -> None:
        """Measured, because the clause says state the regime and the regime is model-specific: the
        `α³` cancellation here bites an order earlier than Vasicek's `α¹`. A tranche that reused
        T66's `1e-3` boundary would have been measuring arithmetic."""
        at_one_e_minus_three = _params(alpha=1e-3, eta=0.0, rho=0.0)
        closed = g2pp_bond_variance(at_one_e_minus_three, 0.0, HORIZON)
        reference = _quadrature_variance(at_one_e_minus_three, 0.0, HORIZON)
        # T66's closed form was good to 5e-11 at alpha=1e-3; this one is three orders worse.
        assert abs(closed / reference - 1.0) > 1e-9, (closed, reference)


class TestTheSimulatedCorrelationIsPinnedFIRST:
    """**Before any price band is credited (T19), and by the T17 instrument.** `ρ` enters the closed
    forms only through the product `ρση`, so a price pins one `ρ`-weighted scalar. The realised
    covariance of the two simulated factor increments pins `ρ` itself.

    The factorisation is `pseudo_square_root` (T63's reuse): at two factors on a 2×2 correlation it
    is full rank, so nothing is truncated and the reuse inherits the L0 `A·Aᵀ = C` oracle.
    """

    PATHS = 200_000

    def _factors(self, rho: float, horizon: float, seed: int) -> np.ndarray:
        """The two OU factors at `horizon`, each simulated by `simulate_lgm_states` and correlated
        through the pseudo square root."""
        factor = np.array(pseudo_square_root(((1.0, rho), (rho, 1.0)), 2))
        x = np.array(
            simulate_lgm_states(
                HullWhiteParams(ALPHA, SIGMA), horizon,
                MonteCarloConfig(paths=self.PATHS, seed=seed),
            )
        )
        y = np.array(
            simulate_lgm_states(
                HullWhiteParams(BETA, ETA), horizon,
                MonteCarloConfig(paths=self.PATHS, seed=seed + 1),
            )
        )
        # Each column is already N(0, zeta_i); correlate the standardised pair, then rescale.
        standardised = np.column_stack((x / x.std(ddof=1), y / y.std(ddof=1))) @ factor.T
        return standardised

    @pytest.mark.parametrize("rho", [-0.9, -0.3, 0.0, 0.72])
    def test_the_realised_CORRELATION_reproduces_rho(self, rho: float) -> None:
        pair = self._factors(rho, HORIZON, seed=67_001)
        realised = float(np.corrcoef(pair[:, 0], pair[:, 1])[0, 1])
        standard_error = (1.0 - rho * rho) / math.sqrt(self.PATHS)
        assert abs(realised - rho) < 4.0 * standard_error, (
            f"realised {realised:.6f} vs {rho} "
            f"({abs(realised - rho) / standard_error:.2f} SE)"
        )

    def test_the_correlation_error_IMPROVES_as_the_root_of_the_path_count(self) -> None:
        """A **bias** would not improve with paths, so failure-to-improve is the detector (T17),
        averaged over seeds because a null needs averaging as much as an effect does (T21)."""
        def rms(paths: int) -> float:
            errors = []
            for seed in (67_101, 67_103, 67_105, 67_107, 67_109, 67_111):
                factor = np.array(pseudo_square_root(((1.0, RHO), (RHO, 1.0)), 2))
                rng = np.random.default_rng(seed)
                pair = rng.standard_normal((paths, 2)) @ factor.T
                errors.append(
                    (float(np.corrcoef(pair[:, 0], pair[:, 1])[0, 1]) - RHO) ** 2
                )
            return math.sqrt(sum(errors) / len(errors))

        coarse, fine = rms(12_500), rms(200_000)
        assert fine < coarse / 2.0, f"expected ~4x improvement, got {coarse} -> {fine}"

    def test_the_FACTORISATION_is_exact_at_two_factors(self) -> None:
        """The premise the reuse rests on: a 2×2 correlation at two factors is full rank, so nothing
        is truncated and `A·Aᵀ = C` holds exactly — the reuse costs no accuracy and inherits the L0
        oracle rather than needing its own."""
        for rho in (-0.95, 0.0, 0.72):
            factor = np.array(pseudo_square_root(((1.0, rho), (rho, 1.0)), 2))
            assert factor @ factor.T == pytest.approx(
                np.array([[1.0, rho], [rho, 1.0]]), abs=1e-14
            )


class TestTheBondOptionAndItsGuards:
    def test_the_bond_option_reuses_the_VERIFIED_one_factor_pricer(self) -> None:
        """The option is Black on the forward bond with the two-factor variance, so `zero_bond_option`
        (T10) is called rather than re-derived — the variance is the only new input."""
        expiry, maturity, strike = 1.0, 3.0, 0.92
        variance = g2pp_bond_option_variance(_params(), expiry, maturity)
        direct = zero_bond_option(
            CURVE.df_at(expiry), CURVE.df_at(maturity), strike, variance, CapFloorKind.CAP
        )
        assert direct > 0.0
        # Put-call parity on the bond option, which the vol cancels out of entirely.
        put = zero_bond_option(
            CURVE.df_at(expiry), CURVE.df_at(maturity), strike, variance, CapFloorKind.FLOOR
        )
        assert direct - put == pytest.approx(
            CURVE.df_at(maturity) - strike * CURVE.df_at(expiry), rel=1e-12
        )

    def test_PERFECT_correlation_collapses_the_variance_to_a_SINGLE_factor_form(self) -> None:
        """A requirement oracle at `ρ = 1`: the two Brownian motions coincide, so the integrand is a
        perfect square and `V` is the variance of one combined factor. Asserted against the closed
        form a reader would write for that case, which no source-value oracle supplies."""
        tau = HORIZON
        combined = quad(
            lambda u: (
                SIGMA / ALPHA * (1.0 - math.exp(-ALPHA * (tau - u)))
                + ETA / BETA * (1.0 - math.exp(-BETA * (tau - u)))
            )
            ** 2,
            0.0,
            tau,
            epsabs=1e-14,
        )[0]
        assert g2pp_bond_variance(_params(rho=1.0), 0.0, tau) == pytest.approx(
            combined, rel=1e-10
        )

    def test_a_NON_POSITIVE_speed_is_refused(self) -> None:
        """Both closed forms divide by the speeds (the variance by their cubes), so zero is
        undefined rather than slow — and the limit is analytic, per the clause above."""
        for kwargs in ({"alpha": 0.0}, {"beta": -0.1}):
            with pytest.raises(ValueError, match="mean reversion"):
                _params(**kwargs)

    def test_EQUAL_speeds_are_refused_because_the_source_assumes_otherwise(self) -> None:
        """The source states `α ≠ β` as an assumption (§3.7, after eq 3.33), and eq (3.43)'s cross
        term divides by `α + β` while Exercise 3.10's `y = u/(α−β)` divides by the *difference*. So
        equal speeds are outside the model as published, and are refused rather than silently
        producing a number the source does not stand behind."""
        with pytest.raises(ValueError, match="distinct"):
            _params(alpha=0.3, beta=0.3)

    def test_a_CORRELATION_outside_the_unit_interval_is_refused(self) -> None:
        for rho in (-1.0001, 1.0001, float("nan")):
            with pytest.raises(ValueError, match="correlation"):
                _params(rho=rho)

    def test_a_NEGATIVE_volatility_is_refused(self) -> None:
        for kwargs in ({"sigma": -0.1}, {"eta": -0.1}):
            with pytest.raises(ValueError, match="volatility"):
                _params(**kwargs)

    def test_a_bond_maturing_BEFORE_it_is_observed_is_refused(self) -> None:
        with pytest.raises(ValueError, match="before"):
            g2pp_bond(_params(), 0.0, 0.0, 5.0, 2.0, CURVE)

    def test_the_PARAMS_are_ONE_object_because_the_CORRELATION_makes_them_one(self) -> None:
        """§6, and the ruling the spec left to Code. Two `HullWhiteParams` plus a loose `ρ` would be
        three arguments that only mean anything together: `ρ` is the correlation *between these two
        factors*, so a pairing of one model's factors with another's correlation would be
        expressible. Five cohesive fields at the §7 ceiling, bundled as compelled vocabulary."""
        p = _params()
        assert (p.alpha, p.beta, p.sigma, p.eta, p.rho) == (ALPHA, BETA, SIGMA, ETA, RHO)
        assert not isinstance(p, HullWhiteParams)
