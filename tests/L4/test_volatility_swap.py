"""The volatility swap's oracles (L4) — tranche 48, PRE-REGISTERED. `GATHERAL-2006` ch 11.

**Written before a line of the engine.** The volatility swap pays `√V_T` where
`V_T = (1/T)∫₀^T v_t dt` is the *annualised* realized variance (ch 11: "Realized volatility `Σ_T`
is the square root of realized variance `V_T := x_T/T`"), and `√` has no static replication — so
`K_vol = E[√V_T]` is model-dependent and the **convexity adjustment** `√(K_var) − K_vol ≥ 0` is the
quantity worth having.

The engine is eq 11.6 — `E[√W_T] = (1/2√π)∫₀^∞ (1 − E[e^{−ψW_T}]) ψ^{−3/2} dψ` on the **total**
variance `W_T = ∫₀^T v_t dt` — closed by the CIR Laplace transform `E[e^{−ψW_T}] = A·e^{−ψ v₀ B}`.
Note the two different time normalisations: 11.6 is in `W_T`, the swap pays in `V_T = W_T/T`, so
`K_vol = E[√W_T]/√T`. That `1/√T` is the tranche's `τ=1` hazard, which is why every oracle below
runs at **two maturities neither of which is one year**.

**Both renderings lose the OVERBAR on `v̄`, exactly as T46's lost the radical.** `-raw` and
`-layout` both render `A`'s exponent as `2λv/η²` and the mean as `(1−e^{−λT})/λ·(v − v) + vT`, so
the long-run and initial variances are indistinguishable in the extraction — and §4's
dual-rendering rule is powerless, both being faithful to a dropped glyph. What resolves it is a
**requirement oracle against code we already hold**: `−∂_ψ E[e^{−ψW_T}]|_{ψ=0}` must equal
`HestonParams.integrated_variance(T)` = `θT + (v₀−θ)(1−e^{−κT})/κ`, built and oracled at T33, which
is precisely the text's printed mean. That pins which `v` is which — **but only on a fixture where
`v₀ ≠ θ`**, and the source's own illustrative parameters (Figures 11.2 and 11.3, Heston-Nandi and
Bakshi-Cao-Chen) both set `v = v̄ = 0.04`. The source hands us the degenerate fixture; the
degeneracy is asserted below rather than avoided by luck.

**The results in ch 11 are PLOTS** (Figures 11.2, 11.3), so §4's "no number is fitted to a figure"
binds: no magnitude here is read off a chart. The absolute level comes from the transform
cross-checked against an independent Monte-Carlo of `√V_T`, and the only *number* the text states —
that BCC parameters give a convexity "roughly 3.39 times greater" than Heston-Nandi — is **not**
asserted, because it is wrong by a power. See `TestTheVolOfVolScaling`.
"""

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.engine import price
from pricebook.engine.heston_fourier import FourierGrid, heston_call_price
from pricebook.engine.volatility_swap import (
    convexity_adjustment,
    fair_volatility,
    integrated_variance_transform,
)
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import (
    DividendSchedule,
    EquityAsset,
    EquityUnderlying,
    dividend_forward,
)
from pricebook.models.heston import HestonModel, HestonParams
from pricebook.products.volatility_swap import VolatilitySwap

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
EUR = Currency("EUR")
CURVE = DiscountCurve.flat(VAL, 0.0, DC)
ASSET = EquityAsset("ACME")
UNDERLYING = EquityUnderlying(100.0, 0.0, DividendSchedule(continuous_yield=0.0))

#: The working fixture. `v0 != theta` (so the overbar reading is observable), `sigma_v` large
#: relative to `kappa` (so the convexity exceeds the MC band — see `TestAgainstMonteCarlo`).
PARAMS = HestonParams(kappa=1.5, theta=0.04, sigma_v=0.6, rho=-0.7, v0=0.05)

#: Two maturities, **neither one year**, because `K_vol = E[√W_T]/√T` and at `T = 1` a dropped
#: `1/√T` is invisible — the τ=1 trap, here living in the time normalisation.
SHORT_YEARS = 0.75
LONG_YEARS = 2.0
SHORT_EXPIRY = VAL + timedelta(days=round(365 * SHORT_YEARS))
LONG_EXPIRY = VAL + timedelta(days=round(365 * LONG_YEARS))

#: The source's own figure parameters. **Degenerate on purpose** — kept only where the text's own
#: derivation assumes `v = v̄`, and never used for the transcription oracles.
HESTON_NANDI = HestonParams(kappa=10.0, theta=0.04, sigma_v=1.0, rho=0.0, v0=0.04)
BCC = HestonParams(kappa=1.15, theta=0.04, sigma_v=0.39, rho=0.0, v0=0.04)


def _market(valuation: date = VAL) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=valuation,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): DiscountCurve.flat(valuation, 0.0, DC)}),
        equities={ASSET: UNDERLYING},
    )


def _model(valuation: date = VAL) -> HestonModel:
    return HestonModel(_market(valuation), PARAMS)


def _fair_variance(params: HestonParams, years: float) -> float:
    """`K_var = E[V_T] = E[W_T]/T`, from T33's exact integrated variance — the test's own
    reference, not read back from the engine."""
    return params.integrated_variance(years) / years


class TestTheFixturesAreNotDegenerate:
    """The premises, asserted rather than believed — the T44b lesson that a premise merely
    *reported* is content, not an artifact."""

    def test_the_working_fixture_separates_the_initial_and_long_run_variances(self) -> None:
        assert PARAMS.v0 != PARAMS.theta, "v0 == theta makes the dropped overbar unobservable"
        assert abs(PARAMS.v0 - PARAMS.theta) > 0.2 * PARAMS.theta, "and not merely unequal"

    def test_the_sources_own_figure_parameters_ARE_degenerate(self) -> None:
        """Stated so the fixture choice is a decision and not an accident: Gatheral's Figures 11.2
        and 11.3 both set `v = v̄`, so building on them would have hidden the overbar."""
        assert HESTON_NANDI.v0 == HESTON_NANDI.theta
        assert BCC.v0 == BCC.theta

    def test_neither_maturity_is_one_year(self) -> None:
        for years in (SHORT_YEARS, LONG_YEARS):
            assert abs(years - 1.0) > 0.2, f"T={years} cannot see a dropped 1/sqrt(T)"

    def test_the_vol_of_vol_is_live(self) -> None:
        assert PARAMS.sigma_v > 0.0, "a zero vol-of-vol makes every convexity oracle vacuous"


class TestTheTransformIsALaplaceTransform:
    """Requirements true of `E[e^{−ψW_T}]` whatever its transcription — so they survive a reading
    error the code and the source might share."""

    @pytest.mark.parametrize("years", [SHORT_YEARS, LONG_YEARS])
    def test_it_is_one_at_zero(self, years: float) -> None:
        assert integrated_variance_transform(PARAMS, 0.0, years) == 1.0

    @pytest.mark.parametrize("years", [SHORT_YEARS, LONG_YEARS])
    def test_it_lies_in_the_unit_interval_and_decreases(self, years: float) -> None:
        previous = 1.0
        for i in range(1, 60):
            psi = 0.5 * i
            value = integrated_variance_transform(PARAMS, psi, years)
            assert 0.0 < value <= 1.0, f"psi={psi} gave {value}"
            assert value < previous, f"not decreasing at psi={psi}"
            previous = value

    @pytest.mark.parametrize("years", [SHORT_YEARS, LONG_YEARS])
    def test_its_slope_at_zero_IS_the_integrated_variance(self, years: float) -> None:
        """**The oracle that resolves the dropped overbar.** `−∂_ψ E[e^{−ψW_T}]|_0 = E[W_T]`, and
        `E[W_T]` is T33's `integrated_variance` — which is exactly the mean the text prints. A
        cross-tranche requirement oracle, and stronger than a second rendering would have been."""
        step = 1e-7
        slope = -(
            integrated_variance_transform(PARAMS, step, years)
            - integrated_variance_transform(PARAMS, 0.0, years)
        ) / step
        assert slope == pytest.approx(PARAMS.integrated_variance(years), rel=1e-6)

    def test_swapping_the_initial_and_long_run_variances_is_VISIBLE_here(self) -> None:
        """Proof the oracle above has power over the overbar, and that the source's own fixture
        would have destroyed it."""
        step = 1e-7

        def slope(params: HestonParams) -> float:
            return -(
                integrated_variance_transform(params, step, LONG_YEARS) - 1.0
            ) / step

        swapped = HestonParams(
            kappa=PARAMS.kappa, theta=PARAMS.v0, sigma_v=PARAMS.sigma_v,
            rho=PARAMS.rho, v0=PARAMS.theta,
        )
        assert abs(slope(PARAMS) - slope(swapped)) > 1e-3, "the swap must move the mean"
        # ... and on the source's degenerate parameters it does NOT
        degenerate_swap = HestonParams(
            kappa=HESTON_NANDI.kappa, theta=HESTON_NANDI.v0, sigma_v=HESTON_NANDI.sigma_v,
            rho=HESTON_NANDI.rho, v0=HESTON_NANDI.theta,
        )
        assert slope(HESTON_NANDI) == pytest.approx(slope(degenerate_swap), rel=1e-12)

    @pytest.mark.parametrize("years", [SHORT_YEARS, LONG_YEARS])
    def test_it_does_not_depend_on_the_spot_vol_correlation(self, years: float) -> None:
        """`W_T` is a functional of the variance path alone, and the variance SDE carries no `ρ` —
        so the transform, `K_vol` and the convexity are all `ρ`-independent. A requirement that
        kills any mutant smuggling `ρ` into the transform."""
        for rho in (-0.9, -0.3, 0.0, 0.5):
            shifted = HestonParams(
                kappa=PARAMS.kappa, theta=PARAMS.theta, sigma_v=PARAMS.sigma_v,
                rho=rho, v0=PARAMS.v0,
            )
            assert integrated_variance_transform(shifted, 3.0, years) == pytest.approx(
                integrated_variance_transform(PARAMS, 3.0, years), rel=0.0, abs=1e-15
            )
            assert fair_volatility(shifted, years) == pytest.approx(
                fair_volatility(PARAMS, years), rel=0.0, abs=1e-15
            )


class TestTheJensenRequirement:
    """`E[√V_T] ≤ √E[V_T]` by Jensen, so the convexity adjustment is non-negative — derived from
    the finance, hence immune to a transcription error (§4). Asserted across a sweep, not a point."""

    @pytest.mark.parametrize("years", [0.25, SHORT_YEARS, LONG_YEARS, 5.0, 10.0])
    @pytest.mark.parametrize("sigma_v", [0.05, 0.3, 0.6, 0.95])
    @pytest.mark.parametrize("v0,theta", [(0.05, 0.04), (0.02, 0.07), (0.09, 0.03)])
    def test_the_convexity_adjustment_is_never_negative(
        self, years: float, sigma_v: float, v0: float, theta: float
    ) -> None:
        params = HestonParams(kappa=1.5, theta=theta, sigma_v=sigma_v, rho=-0.5, v0=v0)
        root_variance = math.sqrt(_fair_variance(params, years))
        vol = fair_volatility(params, years)
        assert vol <= root_variance + 1e-12, f"Jensen violated: {vol} > {root_variance}"
        assert convexity_adjustment(params, years) >= -1e-12

    @pytest.mark.parametrize("years", [SHORT_YEARS, LONG_YEARS])
    def test_the_convexity_is_the_gap_it_claims_to_be(self, years: float) -> None:
        expected = math.sqrt(_fair_variance(PARAMS, years)) - fair_volatility(PARAMS, years)
        assert convexity_adjustment(PARAMS, years) == pytest.approx(expected, abs=1e-15)
        assert expected > 0.0, "the premise: a live convexity, or the equality is vacuous"


class TestTheZeroVolOfVolReduction:
    """The exact anchor — and T39's blind spot stated in full, because this reduction is exact
    *precisely because* it zeroes the parameter the convexity lives in."""

    @pytest.mark.parametrize("years", [SHORT_YEARS, LONG_YEARS, 7.0])
    def test_zero_vol_of_vol_collapses_the_convexity_exactly(self, years: float) -> None:
        """`η = 0` makes `W_T` deterministic, so `E[√W_T] = √E[W_T]` and the convexity is 0."""
        flat = HestonParams(kappa=1.5, theta=0.04, sigma_v=0.0, rho=-0.5, v0=0.05)
        assert fair_volatility(flat, years) == pytest.approx(
            math.sqrt(_fair_variance(flat, years)), rel=1e-10
        )
        assert convexity_adjustment(flat, years) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("years", [SHORT_YEARS, LONG_YEARS])
    def test_THE_COMPANION_a_live_vol_of_vol_gives_a_NON_ZERO_convexity(self, years: float) -> None:
        """The reduction above is satisfied by an implementation that always returns `√K_var` —
        i.e. one with no convexity at all. T39's rule is that the companion must move the very
        parameter the reduction zeroes, so this asserts `η ≠ 0` produces a convexity that is
        *materially* non-zero, measured as a fraction of `K_vol`."""
        assert PARAMS.sigma_v > 0.0
        adjustment = convexity_adjustment(PARAMS, years)
        relative = adjustment / fair_volatility(PARAMS, years)
        assert relative > 0.01, f"the convexity is only {relative:.3%} of K_vol — too small to see"

    def test_the_convexity_grows_with_the_vol_of_vol(self) -> None:
        """Monotonicity in `η` — the direction the finance requires, and a mutant that mis-signs
        the vol-of-vol in `φ = √(λ² + 2ψη²)` would have to preserve it."""
        previous = -1.0
        for sigma_v in (0.0, 0.2, 0.4, 0.6, 0.8):
            params = HestonParams(
                kappa=PARAMS.kappa, theta=PARAMS.theta, sigma_v=sigma_v,
                rho=PARAMS.rho, v0=PARAMS.v0,
            )
            current = convexity_adjustment(params, LONG_YEARS)
            assert current > previous, f"not increasing at sigma_v={sigma_v}"
            previous = current


class TestTheSquareRootIdentityItself:
    """Eq 11.6 rests on `√x = (1/2√π)∫₀^∞(1 − e^{−sx})s^{−3/2} ds`, which holds for a *number*.
    Checking it on a deterministic variance validates the quadrature and its tail handling
    independently of anything Heston — the `η = 0` case does this implicitly, but asserting the
    identity directly separates a quadrature defect from a transform defect."""

    @pytest.mark.parametrize("years", [0.3, SHORT_YEARS, LONG_YEARS, 12.0])
    def test_a_deterministic_variance_integrates_to_its_own_square_root(self, years: float) -> None:
        flat = HestonParams(kappa=2.0, theta=0.06, sigma_v=0.0, rho=0.0, v0=0.06)
        # with sigma_v = 0 and v0 = theta the total variance is exactly theta*T
        assert flat.integrated_variance(years) == pytest.approx(0.06 * years, rel=1e-14)
        assert fair_volatility(flat, years) == pytest.approx(math.sqrt(0.06), rel=1e-10)


class TestAgainstMonteCarlo:
    """The independent absolute-level anchor (§4, T43 — the Jensen bound is sign-complete and
    magnitude-silent, so a level check is mandatory). The MC simulates the CIR variance path and
    averages `√V_T`; it shares no line with the transform.

    **Banded per T17, with the two error sources separated rather than lumped.** The band is
    `4·SE` plus the quadrature's own error, and the quadrature's share is *measured* where sampling
    noise is absent: at the committed settings it is **below 1e-12 relative** (asserted in
    `TestTheQuadratureIsConverged`), so the band is effectively `4·SE` and saying so is honest
    rather than padding one flat percentage.

    **The band's resolution, stated:** at 40,000 paths the SE is ~0.23%, so `4·SE` ≈ 0.9% — and the
    fixture's convexity is ~9% of `K_vol`, a ~10× margin. That margin is the point: on the source's
    Heston-Nandi parameters the convexity is ~0.6%, *inside* the band, so an MC check there could
    not see a dropped convexity at all.
    """

    PATHS = 40_000
    STEPS = 500

    def _mc(self, params: HestonParams, years: float, seed: int) -> tuple[float, float]:
        """`E[√V_T]` by full-truncation Euler on the variance, accumulating the **floored**
        variance.

        Flooring matters and was measured: accumulating the *signed* state instead showed a 6.6-SE
        "bias" that is not there — realized variance is the quadratic variation of the actual
        non-negative process, so the floored state is the one that accrues.

        **Driven by numpy's generator rather than the tree's `UniformSource`, deliberately.** Two
        reasons, and independence is the first: an oracle built on the same RNG the production MC
        engines use shares a component with them, while a different stream shares nothing. The
        second is cost — `standard_normals(source.uniforms(n))` round-trips through Python lists,
        which at 500 steps x 20,000 paths is ~10M Python-level conversions and measured ~19x the
        vectorised draw, enough on its own to push this file past the `slow` ceiling. The seed is
        fixed, so determinism is unaffected (the L5 exposure tests set the same precedent)."""
        generator = np.random.default_rng(seed)
        dt = years / self.STEPS
        root_dt = math.sqrt(dt)
        variance = np.full(self.PATHS, params.v0, dtype=float)
        total = np.zeros(self.PATHS, dtype=float)
        for _ in range(self.STEPS):
            used = np.maximum(variance, 0.0)
            total += used * dt
            shocks = generator.standard_normal(self.PATHS)
            variance = (
                variance
                + params.kappa * (params.theta - used) * dt
                + params.sigma_v * np.sqrt(used) * root_dt * shocks
            )
        realized = np.sqrt(total / years)
        mean = float(realized.mean())
        return mean, float(realized.std(ddof=1) / math.sqrt(self.PATHS))

    @pytest.mark.parametrize("years", [SHORT_YEARS, LONG_YEARS])
    def test_the_transform_agrees_with_an_independent_simulation(self, years: float) -> None:
        closed = fair_volatility(PARAMS, years)
        mean, se = self._mc(PARAMS, years, seed=20_481)
        band = 4.0 * se
        assert abs(mean - closed) < band, (
            f"K_vol={closed:.6f} vs MC={mean:.6f} +- {se:.6f} "
            f"(gap {abs(mean - closed) / se:.2f} SE, band 4 SE)"
        )

    def test_THE_RESOLUTION_the_band_is_sharp_enough_to_see_the_convexity(self) -> None:
        """A green above is evidence only about what the band can resolve (§4). Here: the band is
        ~0.9% and the convexity is ~9% of `K_vol`, so dropping the convexity entirely — returning
        `√K_var` — would fail the MC test by an order of magnitude."""
        closed = fair_volatility(PARAMS, LONG_YEARS)
        _, se = self._mc(PARAMS, LONG_YEARS, seed=20_481)
        band = 4.0 * se / closed
        convexity = convexity_adjustment(PARAMS, LONG_YEARS) / closed
        assert band < 0.02, f"the band is {band:.3%}, too coarse to be useful"
        assert convexity > 5.0 * band, (
            f"the convexity is {convexity:.3%} against a {band:.3%} band — "
            "the MC check cannot discriminate a dropped convexity"
        )

    def test_THE_NULL_the_simulation_shows_no_bias_AVERAGED_over_seeds(self) -> None:
        """An oracle whose expected answer is "no difference" is the one a single seed most easily
        fakes (T21), so the discretisation null is averaged. Measured: the mean relative gap over
        six seeds sits inside one SE of the mean."""
        closed = fair_volatility(PARAMS, LONG_YEARS)
        gaps, errors = [], []
        for seed in range(1, 7):
            mean, se = self._mc(PARAMS, LONG_YEARS, seed=seed * 977)
            gaps.append((mean - closed) / closed)
            errors.append(se / closed)
        average = sum(gaps) / len(gaps)
        se_of_mean = (sum(errors) / len(errors)) / math.sqrt(len(gaps))
        assert abs(average) < 3.0 * se_of_mean, (
            f"a systematic bias: mean gap {average:+.4%} against SE_mean {se_of_mean:.4%}"
        )


class TestTheQuadratureIsConverged:
    """The quadrature is the *reference* half of every band above, so it needs its own accuracy
    argument in the regime where it is used (§4, "a reference is code too")."""

    @pytest.mark.parametrize(
        "v0,theta,kappa,sigma_v,years",
        [
            (0.05, 0.04, 1.5, 0.6, LONG_YEARS),
            (0.09, 0.03, 2.2, 0.5, SHORT_YEARS),
            (0.02, 0.06, 0.8, 0.25, 0.25),
            (0.06, 0.05, 3.0, 0.9, 10.0),
        ],
    )
    def test_the_committed_settings_match_a_far_finer_reference(
        self, v0: float, theta: float, kappa: float, sigma_v: float, years: float
    ) -> None:
        """Measured below 1e-12 relative across the parameter range the oracles use — which is why
        the MC band may be quoted as `4·SE` with the quadrature's share named as negligible rather
        than folded into a flat percentage."""
        params = HestonParams(kappa=kappa, theta=theta, sigma_v=sigma_v, rho=-0.5, v0=v0)
        committed = fair_volatility(params, years)
        reference = _reference_fair_volatility(params, years)
        assert committed == pytest.approx(reference, rel=1e-10)


def _reference_fair_volatility(params: HestonParams, years: float) -> float:
    """A deliberately over-resolved Simpson evaluation of eq 11.6, written here so the convergence
    test compares the engine against a reference it does not share settings with.

    Substitution `ψ = u²` tames the `ψ^{−3/2}` singularity (the integrand tends to `E[W_T]` at
    `u = 0`), and the `1` in `(1 − F)` is integrated **analytically** beyond the truncation —
    `∫_U^∞ ψ^{−3/2} dψ = 2/√U` — which is what makes a modest upper limit enough."""
    upper, nodes = 800.0, 80_000
    mean = params.integrated_variance(years)

    def integrand(u: float) -> float:
        if u <= 0.0:
            return mean
        return (1.0 - integrated_variance_transform(params, u * u, years)) / (u * u)

    step = upper / nodes
    total = integrand(0.0) + integrand(upper)
    for i in range(1, nodes):
        total += (4.0 if i % 2 else 2.0) * integrand(i * step)
    e_sqrt_total = (2.0 * (total * step / 3.0 + 1.0 / upper)) / (2.0 * math.sqrt(math.pi))
    return e_sqrt_total / math.sqrt(years)


class TestTheLogStripCrossProduct:
    """The cross-product anchor (§4, T3/T5): `K_var` two ways, with **different code each side**.

    One side is `E[W_T]/T` from the Heston transform's mean; the other is the variance swap's
    log-strip `(2/T)∫ Õ(K)/K² dK` over Heston option prices (T33's Fourier engine). A shared
    misreading of the Heston parameters cannot survive both, because the strip never touches the
    integrated-variance transform.

    The strip's error is **one-directional** — truncating the strike range omits real value, so it
    underprices — which is asserted as a signature rather than a tolerance."""

    FOURIER = FourierGrid(upper=200.0, count=801)

    def _strip(self, years: float, expiry: date, lower: float, upper: float, count: int) -> float:
        forward = dividend_forward(UNDERLYING, expiry, CURVE)
        discount = CURVE.df(expiry)
        strikes = [forward * (lower + (upper - lower) * i / (count - 1)) for i in range(count)]
        weighted = []
        for strike in strikes:
            call = heston_call_price(PARAMS, UNDERLYING, strike, expiry, (CURVE, self.FOURIER))
            undiscounted = call / discount
            # out-of-the-money throughout: puts below the forward, by parity
            otm = undiscounted if strike >= forward else undiscounted - (forward - strike)
            weighted.append(max(otm, 0.0) / (strike * strike))
        area = sum(
            0.5 * (weighted[i] + weighted[i + 1]) * (strikes[i + 1] - strikes[i])
            for i in range(count - 1)
        )
        return 2.0 / years * area

    def test_the_strip_converges_UP_to_the_transforms_variance(self) -> None:
        exact = _fair_variance(PARAMS, LONG_YEARS)
        narrow = self._strip(LONG_YEARS, LONG_EXPIRY, 0.3, 3.0, 121)
        wide = self._strip(LONG_YEARS, LONG_EXPIRY, 0.15, 6.0, 241)
        assert narrow < wide < exact, (
            f"truncation must underprice monotonically: {narrow} < {wide} < {exact}"
        )
        assert abs(wide - exact) / exact < 0.01, f"wide strip is {(wide - exact) / exact:+.4%} off"
        assert abs(narrow - exact) / exact > abs(wide - exact) / exact


class TestTheVolOfVolScaling:
    """**The source's own order-of-magnitude claim is wrong by a power, and this test records the
    corrected one.** Ch 11 p.146 derives `var[W_T] = v̄T·η²/λ² + O(T⁰)` as `T → ∞` with `v = v̄`,
    concludes that the standard deviation of annualised *volatility* scales as `η/λ`, and then says
    the convexity adjustment "should be of the order of" that — predicting BCC to be "roughly 3.39
    times" Heston-Nandi, `(η/λ)_BCC/(η/λ)_HN = 0.3391/0.1`.

    But Taylor-expanding the square root gives `√E[V] − E[√V] ≈ var[V]/(8·E[V]^{3/2})`, which is
    **quadratic** in `η/λ`, not linear: the predicted ratio is `3.3913² = 11.50`. Measured here at
    long maturity: **11.49**. So the convexity is second order in the vol-of-vol, the text's
    heuristic drops a power, and §4's "no number is fitted to a figure" forbids settling it against
    Figures 11.2/11.3 — the derivation settles it instead.

    This is the `T → ∞` limit, *not* small-`T`; the spec said short-maturity and that is the other
    half of the same correction.
    """

    def test_the_convexity_is_SECOND_order_in_the_vol_of_vol(self) -> None:
        ratio_of_ratios = (BCC.sigma_v / BCC.kappa) / (HESTON_NANDI.sigma_v / HESTON_NANDI.kappa)
        assert ratio_of_ratios == pytest.approx(3.3913, abs=1e-3), "the text's own 3.39"
        measured = convexity_adjustment(BCC, 2_000.0) / convexity_adjustment(
            HESTON_NANDI, 2_000.0
        )
        assert measured == pytest.approx(ratio_of_ratios**2, rel=0.01), (
            f"measured {measured:.4f}; second order predicts {ratio_of_ratios**2:.4f}, "
            f"the text's first-order heuristic predicts {ratio_of_ratios:.4f}"
        )
        # and the first-order reading is decisively excluded
        assert measured > 3.0 * ratio_of_ratios

    def test_the_taylor_approximation_is_the_leading_order(self) -> None:
        """`√E[V] − E[√V] ≈ var[V]/(8 E[V]^{3/2})` with `var[V_T] → v̄η²/(Tλ²)`, independently
        transcribed here — the derivation that overturns the text's heuristic, asserted rather
        than asserted-about."""
        years = 2_000.0
        for params in (HESTON_NANDI, BCC):
            variance = params.theta * params.sigma_v**2 / (years * params.kappa**2)
            predicted = variance / (8.0 * params.theta**1.5)
            assert convexity_adjustment(params, years) == pytest.approx(predicted, rel=0.02)


class TestTheRegisteredPricer:
    """`price(VolatilitySwap, HestonModel)` — the §3 contract."""

    NOTIONAL = Money(1_000_000.0, EUR)

    def test_it_prices_the_discounted_volatility_difference(self) -> None:
        strike = 0.18
        swap = VolatilitySwap(ASSET, LONG_EXPIRY, strike, self.NOTIONAL)
        out = price(swap, _model())
        assert isinstance(out, PricingResult), out
        expected = 1_000_000.0 * (fair_volatility(PARAMS, LONG_YEARS) - strike)
        assert out.pv.amount == pytest.approx(expected, rel=1e-12)
        assert out.pv.currency == EUR

    def test_a_swap_struck_at_its_own_fair_volatility_is_worth_nothing(self) -> None:
        fair = fair_volatility(PARAMS, LONG_YEARS)
        swap = VolatilitySwap(ASSET, LONG_EXPIRY, fair, self.NOTIONAL)
        out = price(swap, _model())
        assert isinstance(out, PricingResult), out
        assert out.pv.amount == pytest.approx(0.0, abs=1e-6)

    def test_an_EXPIRED_swap_is_PV_zero_invariant_6(self) -> None:
        """The tenth-shape premise: the fixture must be **in the money**, or the fall-through
        would return 0 anyway and the guard would be invisible (re-audit #4's F-4.1)."""
        strike = 0.05
        swap = VolatilitySwap(ASSET, VAL, strike, self.NOTIONAL)
        out = price(swap, _model())
        assert isinstance(out, PricingResult), out
        assert out.pv.amount == 0.0
        # the number a dropped guard would produce instead, so the assertion is not vacuous
        assert 1_000_000.0 * (fair_volatility(PARAMS, LONG_YEARS) - strike) > 100_000.0

    def test_THE_PREMISE_a_live_swap_prices_through_this_engine(self) -> None:
        swap = VolatilitySwap(ASSET, LONG_EXPIRY, 0.05, self.NOTIONAL)
        assert isinstance(price(swap, _model()), PricingResult)

    def test_a_missing_underlying_is_a_failure_not_an_exception(self) -> None:
        bare = MarketSnapshot(
            valuation_date=VAL,
            curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): CURVE}),
        )
        swap = VolatilitySwap(ASSET, LONG_EXPIRY, 0.2, self.NOTIONAL)
        assert isinstance(price(swap, HestonModel(bare, PARAMS)), PricingFailure)


class TestTheProductGuards:
    @pytest.mark.parametrize("strike", [-0.01, -1.0])
    def test_a_negative_volatility_strike_is_refused(self, strike: float) -> None:
        with pytest.raises(ValueError, match="negative"):
            VolatilitySwap(ASSET, LONG_EXPIRY, strike, Money(1.0, EUR))

    def test_a_zero_strike_is_allowed(self) -> None:
        """Zero is a legitimate strike — the swap then pays realized volatility outright."""
        assert VolatilitySwap(ASSET, LONG_EXPIRY, 0.0, Money(1.0, EUR)).volatility_strike == 0.0

    @pytest.mark.parametrize("years", [0.0, -1.0])
    def test_a_non_positive_horizon_is_refused_by_the_engine(self, years: float) -> None:
        with pytest.raises(ValueError, match="positive horizon"):
            fair_volatility(PARAMS, years)
