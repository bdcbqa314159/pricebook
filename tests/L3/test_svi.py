"""SVI's oracles (L3) — tranche 46, PRE-REGISTERED. `GATHERAL-2006` eq 3.20 + ch 7 (Lee).

**Written before a line of `models/svi.py`, deliberately**, and every oracle here is statable from
the source alone — none can be reverse-engineered from the code it tests.

**The quantity is TOTAL variance `w(k) = σ²_BS(k)·T`** (spec ruling 1). The spec first wrote the
wing bound as `b(1+|ρ|) ≤ 2` in *annualised* variance, which is right only at `T = 1`: Lee bounds
`σ²_BS(k,T)·T/|k|`, so the annualised form carries a `T`. That is the `τ=1` degeneracy living in a
**formula** rather than a fixture, and a one-year fixture would have passed it silently. In total
variance the bound is clean and the calendar condition is monotonicity of `w` in `T`.

**How eq 3.20's radical is pinned, since neither rendering shows it.** `pdftotext` drops the glyph
in both `-raw` and `-layout` (it survives 123 times elsewhere in the book, so this is not a generic
loss — what sits in its place is a large-math *vinculum*). Measuring the vinculum's span on the
layout line pins the **scope**: it begins at column 47 and covers the radicand at columns 51–66,
while `ρ (k − m)` sits at columns 33–45, *outside* it. So the term grouping — the v0.4.0 mis-nesting
hazard — is confirmed by the extraction. The radical's **identity** is then forced by ch 7: with it
the wings are *linear* in `k`; without it `w` is *quadratic*, which violates Lee's model-independent
`β*, α* ∈ [0,2]` unboundedly as `|k| → ∞`. §4's derivation-step independence, across chapters.

**The butterfly condition is derived, not transcribed** (spec ruling 3). The 2006 text carries only
the calendar condition; the rigorous `g(k) ≥ 0` is Gatheral–Jacquier 2014, which we do not hold. So
the density is derived here from first principles and cross-checked numerically. For the
undiscounted call per unit forward `c(k) = N(d₊) − e^k N(d₋)`, the payoff `(e^X − e^k)⁺` gives

    ∂c/∂k = −e^k (1 − P(k))   ⇒   e^k p(k) = c''(k) − **c'(k)**

— the first-derivative term is the trap, and `p = e^{−k} c''` would be wrong. Carrying the algebra
through with Black's identity `n(d₊) = e^k n(d₋)` collapses it to `p(k) = g(k)·n(d₋)/√w` with
`g = (1 − k w'/(2w))² − (w'²/4)(1/w + 1/4) + w''/2`, whose three independent confirmations are
asserted below: the numerical `e^{−k}(c'' − c')` from Black prices, the unit integral, and the
`b = 0` collapse onto the Black–Scholes normal density.

**Premises asserted up front** (§4, the arc's hard lesson — the fixture must break the symmetry each
mutation exploits):

* `ρ ≠ 0` **and** the smile asymmetric — else the wings are mirror images and a `ρ` sign flip lives.
* `m ≠ 0` — else the minimum sits at `k = 0` and dropping `m` is invisible.
* `b ≠ 0` for every companion to the `b → 0` reduction (T39: the companion must move the very
  parameter the reduction zeroes).
* strikes spanning **both** wings, since one wing alone cannot see `ρ`'s sign.
* an **arbitrageable** parameter set supplied so each refusal's reject branch is exercised (T33b:
  a uniformly well-formed input population leaves a guard's negative branch untested).
"""

import math
from datetime import date, timedelta
from statistics import NormalDist

import pytest

from pricebook.engine.heston_fourier import FourierGrid, heston_call_price
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.inflation import CapFloorKind
from pricebook.market import DiscountCurve
from pricebook.market.equity import DividendSchedule, EquityUnderlying, dividend_forward
from pricebook.models.black import black_implied_vol, black_price
from pricebook.models.heston import HestonParams
from pricebook.models.svi import (
    SviParams,
    calendar_arbitrage_free,
    fit_svi_slice,
    risk_neutral_density,
    satisfies_lee_bound,
)

# --- the reference slice: asymmetric (rho != 0), off-centre (m != 0), both wings live -------------
A, B, RHO, M, SIGMA = 0.04, 0.4, -0.3, 0.05, 0.15
SLICE = SviParams(a=A, b=B, rho=RHO, m=M, sigma=SIGMA)

#: Log-strikes spanning both wings. Deliberately not symmetric about `m`.
STRIKES = (-0.6, -0.4, -0.25, -0.1, 0.0, 0.05, 0.15, 0.3, 0.5)

_PDF = NormalDist().pdf
_CDF = NormalDist().cdf


def test_the_fixture_is_not_degenerate() -> None:
    """The premises, asserted rather than believed — §4, and the T44b lesson that a premise only
    *reported* is content, not an artifact."""
    assert RHO != 0.0, "a zero rho makes the wings mirror images and a sign flip invisible"
    assert M != 0.0, "a zero m puts the minimum at k=0 and makes dropping m invisible"
    assert B != 0.0, "a zero b flattens the smile and makes every wing oracle vacuous"
    left = [k for k in STRIKES if k < M]
    right = [k for k in STRIKES if k > M]
    assert len(left) >= 3 and len(right) >= 3, "both wings must be exercised"
    # the smile must actually be asymmetric about its own minimum, not merely have rho != 0
    k_min, _ = SLICE.minimum()
    lo = SLICE.total_variance(k_min - 0.4)
    hi = SLICE.total_variance(k_min + 0.4)
    assert abs(lo - hi) > 0.01 * max(lo, hi), f"the smile is near-symmetric: {lo} vs {hi}"


class TestTheTranscriptionOfEq320:
    """Pointwise, against a reference computed in the test — §4 demands a pointwise transcription
    *and* requirement oracles, since neither substitutes for the other."""

    @pytest.mark.parametrize("k", STRIKES)
    def test_it_reproduces_an_independent_transcription(self, k: float) -> None:
        y = k - M
        expected = A + B * (RHO * y + math.sqrt(y * y + SIGMA * SIGMA))
        assert SLICE.total_variance(k) == pytest.approx(expected, rel=0.0, abs=1e-15)

    def test_the_vertex_value_is_a_plus_b_sigma(self) -> None:
        """At `k = m` the radicand is `σ²` exactly, so `w(m) = a + bσ` — an **absolute level**,
        which the T43 lesson requires: a set of purely scale-invariant oracles pins the shape and
        says nothing about the level."""
        assert SLICE.total_variance(M) == pytest.approx(A + B * SIGMA, rel=0.0, abs=1e-15)
        assert A + B * SIGMA == pytest.approx(0.1, abs=1e-15), "the independent absolute number"

    def test_the_minimum_is_where_the_closed_form_says(self) -> None:
        """`w' = 0` solves exactly: `k* = m − ρσ/√(1−ρ²)`, `w* = a + bσ√(1−ρ²)`. A closed form,
        so it beats any numerical search — and it is non-vacuous only because `ρ ≠ 0` moves `k*`
        off `m`."""
        root = math.sqrt(1.0 - RHO * RHO)
        k_star, w_star = SLICE.minimum()
        assert k_star == pytest.approx(M - RHO * SIGMA / root, abs=1e-14)
        assert w_star == pytest.approx(A + B * SIGMA * root, abs=1e-14)
        assert k_star != pytest.approx(M, abs=1e-6), "rho must move the vertex off m"
        # and it really is the minimum
        for delta in (-0.3, -0.05, 0.05, 0.3):
            assert SLICE.total_variance(k_star + delta) > w_star


class TestTheRadicalIsPinnedByTheWings:
    """Neither rendering shows `√`; ch 7's Lee bound forces it. A non-radical 3.20 is quadratic in
    `k` and violates `β*, α* ∈ [0,2]` unboundedly, so the *linear* asymptote is the pin."""

    @pytest.mark.parametrize("sign,expected", [(+1, B * (1 + RHO)), (-1, B * (1 - RHO))])
    def test_each_wing_is_linear_with_the_transcribed_slope(self, sign: int, expected: float) -> None:
        """`w(k) → a + b(ρ ± 1)(k − m)`, so the local slope tends to `b(1±ρ)` — distinct per wing
        precisely because `ρ ≠ 0`."""
        far = sign * 400.0
        h = 1e-3
        slope = (SLICE.total_variance(far + h) - SLICE.total_variance(far - h)) / (2 * h)
        assert slope == pytest.approx(expected, rel=1e-5)

    def test_the_two_wing_slopes_differ_so_rhos_sign_is_observable(self) -> None:
        up, down = SLICE.wing_slopes()
        assert up == pytest.approx(B * (1 + RHO), abs=1e-15)
        assert down == pytest.approx(B * (1 - RHO), abs=1e-15)
        assert abs(up - down) > 0.1, "mirror-image wings would hide a rho sign flip"

    def test_growth_is_linear_not_quadratic(self) -> None:
        """The decisive discrimination: `w(k)/|k|` **converges** under the radical and would
        **diverge** without it. This is what makes the extraction's missing glyph recoverable."""
        ratios = [SLICE.total_variance(k) / k for k in (100.0, 1_000.0, 10_000.0)]
        for ratio in ratios:
            assert ratio == pytest.approx(B * (1 + RHO), rel=1e-3)
        assert abs(ratios[-1] - ratios[0]) < 1e-3, f"not converging: {ratios}"

    def test_lee_bound_holds_for_the_reference_slice_and_refuses_a_steep_one(self) -> None:
        """`b(1+|ρ|) ≤ 2` in total variance (ruling 1 — no `T`, which is the whole point)."""
        assert B * (1 + abs(RHO)) <= 2.0
        assert satisfies_lee_bound(SLICE)
        steep = SviParams(a=A, b=1.8, rho=RHO, m=M, sigma=SIGMA)
        assert steep.b * (1 + abs(steep.rho)) > 2.0, "the fixture must actually violate the bound"
        assert not satisfies_lee_bound(steep), "the reject branch must be exercised (T33b)"


class TestTheBZeroReduction:
    """The exact anchor, plus the companion T39 requires: a reduction that a flat-smile fixture also
    satisfies is vacuous, and the companion must be non-degenerate in `b` — the very parameter the
    reduction zeroes."""

    @pytest.mark.parametrize("k", STRIKES)
    def test_b_zero_is_exactly_flat(self, k: float) -> None:
        flat = SviParams(a=A, b=0.0, rho=RHO, m=M, sigma=SIGMA)
        assert flat.total_variance(k) == A, "exact by construction, not approximately"

    def test_the_companion_is_not_flat(self) -> None:
        values = [SLICE.total_variance(k) for k in STRIKES]
        assert max(values) - min(values) > 0.05, "the companion must exercise b != 0 (T39)"
        assert SLICE.b != 0.0


class TestTheRiskNeutralDensity:
    """Derived, not transcribed (ruling 3), and confirmed three independent ways."""

    def _call(self, params: SviParams, k: float) -> float:
        """`c(k) = N(d₊) − e^k N(d₋)`, the undiscounted call per unit forward — computed here, so
        it is not read back from the implementation."""
        w = params.total_variance(k)
        root = math.sqrt(w)
        return _CDF(-k / root + root / 2) - math.exp(k) * _CDF(-k / root - root / 2)

    @pytest.mark.parametrize("k", [-0.6, -0.3, -0.1, 0.0, 0.05, 0.2, 0.5])
    def test_it_matches_the_numerical_breeden_litzenberger(self, k: float) -> None:
        """`e^k p = c'' − c'`. The `− c'` is the trap: `p = e^{−k}c''` is wrong for a
        `(e^X − e^k)⁺` payoff, and dropping it changes the answer by `O(1)`."""
        h = 1e-4
        up, mid, down = (self._call(SLICE, k + h), self._call(SLICE, k), self._call(SLICE, k - h))
        second = (up - 2 * mid + down) / (h * h)
        first = (up - down) / (2 * h)
        numerical = math.exp(-k) * (second - first)
        assert risk_neutral_density(SLICE, k) == pytest.approx(numerical, rel=2e-5)

    def test_dropping_the_first_derivative_term_would_be_visible(self) -> None:
        """Proof the companion above has power: the wrong form is not a near-miss."""
        k, h = 0.0, 1e-4
        up, mid, down = (self._call(SLICE, k + h), self._call(SLICE, k), self._call(SLICE, k - h))
        wrong = math.exp(-k) * (up - 2 * mid + down) / (h * h)
        right = risk_neutral_density(SLICE, k)
        assert abs(wrong - right) > 0.3 * right, f"indistinguishable: {wrong} vs {right}"

    def test_it_integrates_to_one(self) -> None:
        """A density's defining requirement — it survives any transcription error the source and the
        code might share. **Simpson over `±40`:** SVI's linear-in-`|k|` total variance gives
        genuinely fat tails, and `±10` truncates 4e-4 of the mass, so the range is part of the
        oracle's validity regime (§4 — a reference is code too)."""
        lo, hi, count = -40.0, 40.0, 8_000
        step = (hi - lo) / count
        total = risk_neutral_density(SLICE, lo) + risk_neutral_density(SLICE, hi)
        for i in range(1, count):
            weight = 4 if i % 2 else 2
            total += weight * risk_neutral_density(SLICE, lo + i * step)
        assert total * step / 3 == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("k", [-0.4, 0.0, 0.3])
    def test_at_b_zero_it_collapses_onto_the_black_scholes_normal(self, k: float) -> None:
        """The flat limit: `g ≡ 1`, so `p(k) = n(d₋)/√w` — the normal density of `log(S_T/F)` with
        mean `−w/2` and variance `w`. Exact, which is a stronger confirmation than any tolerance."""
        flat = SviParams(a=A, b=0.0, rho=RHO, m=M, sigma=SIGMA)
        root = math.sqrt(A)
        expected = _PDF((k + A / 2) / root) / root
        assert risk_neutral_density(flat, k) == pytest.approx(expected, abs=1e-15)

    def test_the_reference_slice_is_butterfly_free(self) -> None:
        for i in range(-300, 301):
            k = i / 100.0
            assert risk_neutral_density(SLICE, k) >= 0.0, f"negative density at k={k}"

    def test_an_arbitrageable_slice_is_refused(self) -> None:
        """The T33b premise: supply an input the guard rejects, or its negative branch is never in
        the test population. A large `b` with a tight vertex buckles the density."""
        bad = SviParams(a=0.01, b=0.9, rho=-0.9, m=0.0, sigma=0.02)
        worst = min(risk_neutral_density(bad, i / 100.0) for i in range(-100, 101))
        assert worst < 0.0, f"the fixture must actually be arbitrageable, got min density {worst}"
        with pytest.raises(ValueError, match="butterfly"):
            fit_svi_slice(STRIKES, tuple(bad.total_variance(k) for k in STRIKES))


class TestTheCalendarCondition:
    """Total variance is non-decreasing in `T` at every `k` — slices do not cross (ruling 1: in
    total variance this *is* the condition, with no `T` factor to get wrong)."""

    def test_two_nested_slices_are_calendar_free(self) -> None:
        later = SviParams(a=A * 2.1, b=B * 1.05, rho=RHO, m=M, sigma=SIGMA)
        gaps = [later.total_variance(k) - SLICE.total_variance(k) for k in STRIKES]
        assert min(gaps) > 0.0, "the fixture must genuinely separate the slices"
        assert max(gaps) - min(gaps) > 1e-3, "a parallel shift would not exercise the k-dependence"
        assert calendar_arbitrage_free(SLICE, later, STRIKES)

    def test_crossing_slices_are_refused(self) -> None:
        crossing = SviParams(a=A * 1.2, b=B * 0.3, rho=RHO, m=M, sigma=SIGMA)
        gaps = [crossing.total_variance(k) - SLICE.total_variance(k) for k in STRIKES]
        assert min(gaps) < 0.0 < max(gaps), f"the fixture must actually cross: {gaps}"
        assert not calendar_arbitrage_free(SLICE, crossing, STRIKES)

    def test_the_order_of_the_arguments_matters(self) -> None:
        """A mutation swapping the two slices must die — so the check is asserted asymmetric."""
        later = SviParams(a=A * 2.1, b=B * 1.05, rho=RHO, m=M, sigma=SIGMA)
        assert calendar_arbitrage_free(SLICE, later, STRIKES)
        assert not calendar_arbitrage_free(later, SLICE, STRIKES)


class TestTheSyntheticRoundTrip:
    """The correctness anchor: generate from known parameters, fit back, recover them **and**
    reprice. Non-telescoping — the generation above is the test's own transcription."""

    def test_the_fit_recovers_the_generating_parameters(self) -> None:
        quotes = tuple(SLICE.total_variance(k) for k in STRIKES)
        fitted = fit_svi_slice(STRIKES, quotes)
        for name in ("a", "b", "rho", "m", "sigma"):
            assert getattr(fitted, name) == pytest.approx(getattr(SLICE, name), abs=5e-4), name

    def test_the_fit_reprices_the_smile(self) -> None:
        quotes = tuple(SLICE.total_variance(k) for k in STRIKES)
        fitted = fit_svi_slice(STRIKES, quotes)
        for k, quote in zip(STRIKES, quotes, strict=True):
            assert fitted.total_variance(k) == pytest.approx(quote, rel=1e-6)

    def test_identifiability_by_multi_start_spread_and_distance_from_truth(self) -> None:
        """T11's discipline, both halves. **Multi-start agreement is necessary, not sufficient** —
        all starts can land in the same *wrong* basin — so the synthetic truth is among the starts
        and distance-from-truth is asserted beside spread. A condition number is not the oracle."""
        quotes = tuple(SLICE.total_variance(k) for k in STRIKES)
        starts = (
            (A, B, RHO, M, SIGMA),  # the truth: an optimum no other start can beat
            (0.01, 0.1, 0.0, 0.0, 0.1),
            (0.2, 1.0, -0.8, -0.2, 0.5),
            (0.08, 0.6, 0.5, 0.2, 0.3),
        )
        fits = [fit_svi_slice(STRIKES, quotes, start) for start in starts]
        for name in ("a", "b", "rho", "m", "sigma"):
            values = [getattr(f, name) for f in fits]
            scale = max(abs(v) for v in values) or 1.0
            spread = (max(values) - min(values)) / scale
            assert spread < 5e-3, f"{name} is not identified: spread {spread:.3%} over {values}"
            truth = getattr(SLICE, name)
            assert max(abs(v - truth) for v in values) < 5e-3 * max(abs(truth), 0.1), name


class TestHestonAgainstSviThroughBlack:
    """The second-source cross-check — and the structure that makes **C14 guarded** rather than
    per-module (Cowork's ruling-5 nudge). The path is end-to-end: Heston prices → `black_implied_vol`
    → **×T into total variance** → SVI fit → `black_price` straight off `w` → back to prices. A
    dimension slip anywhere (a missing `×T` going in, a missing `÷T` coming out) fails this test,
    which is what a *guarded* convention means.

    They are different objects — an SVI fit of a Heston smile is an approximation — so the gap is
    asserted small **and non-zero**, never treated as cross-source identity (§4's same-formula
    caveat).
    """

    VAL = date(2026, 1, 1)
    DC = DayCountConvention.ACT_365_FIXED
    CURVE = DiscountCurve.flat(VAL, 0.02, DC)
    UNDERLYING = EquityUnderlying(100.0, 0.0, DividendSchedule(continuous_yield=0.0))
    PARAMS = HestonParams(kappa=1.5, theta=0.04, sigma_v=0.6, rho=-0.7, v0=0.05)
    EXPIRY = VAL + timedelta(days=730)
    YEARS = 730.0 / 365.0
    GRID = FourierGrid(upper=200.0, count=1_001)
    MONEYNESS = (0.80, 0.90, 0.95, 1.00, 1.05, 1.15, 1.30)

    def _heston_smile(self) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
        forward = dividend_forward(self.UNDERLYING, self.EXPIRY, self.CURVE)
        discount = self.CURVE.df(self.EXPIRY)
        strikes = tuple(x * forward for x in self.MONEYNESS)
        prices = tuple(
            heston_call_price(self.PARAMS, self.UNDERLYING, k, self.EXPIRY, (self.CURVE, self.GRID))
            for k in strikes
        )
        # undiscounted, per unit forward -- what Black's total variance speaks to
        return forward, strikes, tuple(p / discount for p in prices)

    def test_the_expiry_is_not_one_year(self) -> None:
        """The premise that gives this test power over the `T` factor: at `T = 1` the annualised
        and total-variance conventions coincide and a missing `×T` is invisible — the `τ=1`
        degeneracy that ruling 1 corrected in the spec's own formula."""
        assert abs(self.YEARS - 1.0) > 0.5, f"T={self.YEARS} cannot discriminate the convention"

    def test_svi_fits_the_heston_smile_and_reprices_it_through_black(self) -> None:
        forward, strikes, undiscounted = self._heston_smile()
        log_strikes = tuple(math.log(k / forward) for k in strikes)
        vols = tuple(
            black_implied_vol(p * forward, forward, k, self.YEARS, CapFloorKind.CAP)
            for p, k in zip(undiscounted, strikes, strict=True)
        )
        # the ONE conversion, at the input boundary (ruling 1): annualised quote -> total variance
        total_variances = tuple(v * v * self.YEARS for v in vols)
        fitted = fit_svi_slice(log_strikes, total_variances)

        gaps = []
        for k, strike, target in zip(log_strikes, strikes, undiscounted, strict=True):
            # w feeds black_price with NO further conversion -- the other end of C14
            repriced = black_price(forward, strike, fitted.total_variance(k), CapFloorKind.CAP)
            gaps.append(abs(repriced / forward - target) / target)
        worst = max(gaps)
        assert worst < 0.02, f"SVI does not fit the Heston smile: worst relative gap {worst:.3%}"
        assert worst > 1e-8, "an exact match would mean the two objects are not independent"

    def test_a_missing_time_factor_would_fail_this_test(self) -> None:
        """Proof the test above has power over the convention, rather than merely passing: feeding
        **annualised** variance where `w` belongs moves the reprice far outside its band."""
        forward, strikes, undiscounted = self._heston_smile()
        log_strikes = tuple(math.log(k / forward) for k in strikes)
        vols = tuple(
            black_implied_vol(p * forward, forward, k, self.YEARS, CapFloorKind.CAP)
            for p, k in zip(undiscounted, strikes, strict=True)
        )
        annualised = tuple(v * v for v in vols)  # the mistake: no x T
        fitted = fit_svi_slice(log_strikes, annualised)
        worst = max(
            abs(
                black_price(forward, strike, fitted.total_variance(k), CapFloorKind.CAP) / forward
                - target
            )
            / target
            for k, strike, target in zip(log_strikes, strikes, undiscounted, strict=True)
        )
        assert worst > 0.02, f"the convention slip is invisible here: {worst:.3%}"


class TestTheParameterGuards:
    """Definitional constraints refuse at construction; *admissibility* (Lee, butterfly) does not,
    so an arbitrageable fixture can still be built for the guards above to reject."""

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"b": -0.1}, "wing"),
            ({"rho": 1.0}, "correlation"),
            ({"rho": -1.2}, "correlation"),
            ({"sigma": 0.0}, "vertex"),
            ({"sigma": -0.2}, "vertex"),
            ({"a": -0.5}, "negative"),
        ],
    )
    def test_a_definitionally_impossible_slice_is_refused(
        self, kwargs: dict[str, float], match: str
    ) -> None:
        fields = {"a": A, "b": B, "rho": RHO, "m": M, "sigma": SIGMA} | kwargs
        with pytest.raises(ValueError, match=match):
            SviParams(**fields)

    def test_an_arbitrageable_but_well_formed_slice_still_constructs(self) -> None:
        """Else the refusal tests above could not build their fixtures — the guards must live at
        the fit, not at construction."""
        assert SviParams(a=0.01, b=0.9, rho=-0.9, m=0.0, sigma=0.02).b == 0.9
        assert SviParams(a=A, b=1.8, rho=RHO, m=M, sigma=SIGMA).b == 1.8

    def test_the_minimum_variance_guard_is_tight_not_loose(self) -> None:
        """`a + bσ√(1−ρ²) ≥ 0` is the exact condition for `w ≥ 0` everywhere, derived from the
        closed-form minimum — so a slice just inside it must construct and just outside must not."""
        root = math.sqrt(1.0 - RHO * RHO)
        floor = -B * SIGMA * root
        SviParams(a=floor + 1e-9, b=B, rho=RHO, m=M, sigma=SIGMA)
        with pytest.raises(ValueError, match="negative"):
            SviParams(a=floor - 1e-6, b=B, rho=RHO, m=M, sigma=SIGMA)
