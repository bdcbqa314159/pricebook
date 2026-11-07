"""Black-76 on a forward, and its inverse (L3). BERHAN-2012 §2.6 / §5.2, standard Black-76.

    Black(F, K, V^2, w) = w * [ F*Phi(w*d1) - K*Phi(w*d2) ]
    d1 = [ln(F/K) + V^2/2] / V ,   d2 = d1 - V ,   V = sqrt(V^2)

**Parameterised by TOTAL VARIANCE, not (sigma, T), and that is the point.** Jarrow-Yildirim's
`V^2` (eq 2.6.4) is a total variance assembled from eight terms -- it is NOT `sigma^2 * T`, so a
`(sigma, T)` signature would force every caller to invent a sigma that does not exist in the model.
The annualised vol appears only at the inverse's boundary, where the surface of Figures 5.3/5.4
wants `sigma_I` per (expiry, strike).

**On oracle power, stated so the round trip is not mistaken for the anchor.** The inversion
round-trip test below *telescopes*: pricing at `sigma` and implying it back cannot catch an error
shared by the forward and the inverse, since the inverse solves the forward. It earns its keep only
because **tranche 5 already oracled this exact forward against eq 2.6.5 independently**
(`tests/L4/test_inflation_capfloor.py::test_the_capfloor_matches_eq_2_6_5`, whose reference value is
transcribed from the PDF, not computed by the engine). With that prior, the round trip isolates the
root-find. Monotonicity and the refusal tests do not telescope and stand on their own.
"""

import math

import pytest
from scipy.stats import norm

from pricebook.foundation import CapFloorKind
from pricebook.foundation.distributions import norm_cdf
from pricebook.models.black import (
    black_delta,
    black_gamma,
    black_implied_vol,
    black_price,
    black_vega,
    digital_delta,
    digital_price,
)

F = 1.025          # a forward index ratio, as the cap/floor sees it
K = 1.020          # 1 + kappa


def _reference_black(forward, strike, variance, kind):
    """Black-76 written independently here, from the formula -- not imported from the module."""
    w = 1.0 if kind is CapFloorKind.CAP else -1.0
    if variance <= 0.0:
        return max(w * (forward - strike), 0.0)
    v = math.sqrt(variance)
    d1 = (math.log(forward / strike) + 0.5 * variance) / v
    d2 = d1 - v
    return w * (forward * norm.cdf(w * d1) - strike * norm.cdf(w * d2))


# ---- the forward, against an independent transcription ----

@pytest.mark.parametrize("kind", list(CapFloorKind))
@pytest.mark.parametrize("variance", [1e-8, 1e-4, 0.0025, 0.04, 1.0])
def test_the_forward_matches_an_independent_transcription(kind, variance):
    assert black_price(F, K, variance, kind) == pytest.approx(
        _reference_black(F, K, variance, kind), rel=1e-14
    )


@pytest.mark.parametrize("kind", list(CapFloorKind))
def test_at_zero_variance_the_price_IS_the_intrinsic(kind):
    """The degenerate branch, exact rather than a limit: no uncertainty means the option is worth
    its intrinsic on the deterministic forward."""
    w = 1.0 if kind is CapFloorKind.CAP else -1.0
    assert black_price(F, K, 0.0, kind) == max(w * (F - K), 0.0)


def test_put_call_parity_holds_to_the_ULP():
    """`cap - floor = F - K` for every variance -- a REQUIREMENT, derived from the payoff, so it
    survives a transcription error that both the code and the reference above could share."""
    for variance in (0.0, 1e-6, 0.01, 0.25, 4.0):
        cap = black_price(F, K, variance, CapFloorKind.CAP)
        floor = black_price(F, K, variance, CapFloorKind.FLOOR)
        assert cap - floor == pytest.approx(F - K, abs=1e-15), variance


@pytest.mark.parametrize("kind", list(CapFloorKind))
def test_the_price_is_bounded_by_its_no_arbitrage_limits(kind):
    """As `V -> inf` a cap tends to `F` and a floor to `K`; neither may EXCEED its bound at any
    variance. The bound the inverse's bracket relies on.

    **`<=`, not `<`, and the difference is floating point rather than finance.** Mathematically the
    limit is approached and never reached, so the first version of this test asserted `< limit`
    and failed at `V^2 = 1e4`: there `Phi(w*d1)` saturates to exactly 1.0 and `Phi(w*d2)` to 0.0, so
    the price is `F*1 - K*0 = F` exactly, as a double. The test was wrong, not the code -- and the
    saturation is asserted below rather than hidden, because it is precisely WHY the inverse
    refuses a price at the limit instead of returning whatever its expansion reached."""
    limit = F if kind is CapFloorKind.CAP else K
    for variance in (1e-6, 1.0, 100.0, 10_000.0):
        assert 0.0 <= black_price(F, K, variance, kind) <= limit, variance
    assert black_price(F, K, 1e4, kind) == limit, "the saturation point moved -- the inverse's refusal boundary depends on it"
    assert black_price(F, K, 100.0, kind) < limit, "the bound must still be STRICT at a realistic variance"


# ---- the inverse: round trip (telescopes -- see the module docstring), monotonicity, refusal ----

@pytest.mark.parametrize("kind", list(CapFloorKind))
@pytest.mark.parametrize("sigma", [0.002, 0.01, 0.05, 0.20, 0.80])
@pytest.mark.parametrize("expiry", [0.5, 1.0, 7.0])
def test_the_implied_vol_round_trips(kind, sigma, expiry):
    price = black_price(F, K, sigma * sigma * expiry, kind)
    assert black_implied_vol(price, F, K, expiry, kind) == pytest.approx(sigma, rel=1e-9)


@pytest.mark.parametrize("kind", list(CapFloorKind))
def test_the_implied_vol_INCREASES_with_the_price(kind):
    """Does not telescope: a wrong root or a sign error gives a non-monotone map even when a
    single round trip happens to close."""
    prices = [black_price(F, K, s * s, kind) for s in (0.01, 0.05, 0.10, 0.30, 0.60)]
    vols = [black_implied_vol(p, F, K, 1.0, kind) for p in prices]
    assert vols == sorted(vols), vols
    assert len(set(round(v, 12) for v in vols)) == len(vols), f"the map is flat: {vols}"


@pytest.mark.parametrize("kind", list(CapFloorKind))
def test_a_price_BELOW_intrinsic_is_REFUSED_not_guessed(kind):
    """No Black vol exists, so there is nothing to return. A constructor-side raise, per the
    tranche 8 ruling: `PricingFailure` is the engine contract, not this boundary."""
    w = 1.0 if kind is CapFloorKind.CAP else -1.0
    intrinsic = max(w * (F - K), 0.0)
    with pytest.raises(ValueError, match="below its intrinsic"):
        black_implied_vol(intrinsic - 1e-6, F, K, 1.0, kind)


@pytest.mark.parametrize(
    ("kind", "limit"), [(CapFloorKind.CAP, F), (CapFloorKind.FLOOR, K)]
)
def test_a_price_AT_OR_ABOVE_the_no_arbitrage_limit_is_REFUSED(kind, limit):
    """The price is approached only as `V -> inf`, so there is no finite vol. Refuse rather than
    return whatever the bracket expansion stopped at."""
    with pytest.raises(ValueError, match="no-arbitrage"):
        black_implied_vol(limit, F, K, 1.0, kind)
    with pytest.raises(ValueError, match="no-arbitrage"):
        black_implied_vol(limit * 1.5, F, K, 1.0, kind)


def test_a_non_positive_expiry_is_REFUSED():
    """An annualised vol needs a positive horizon to divide by; `sigma = sqrt(V^2/T)` is undefined
    at `T = 0`, and the historical case is the shell's business (invariant 6)."""
    price = black_price(F, K, 0.04, CapFloorKind.CAP)
    with pytest.raises(ValueError, match="positive expiry"):
        black_implied_vol(price, F, K, 0.0, CapFloorKind.CAP)


def test_a_price_AT_intrinsic_implies_a_vol_of_ZERO():
    """The boundary that must NOT raise: exactly intrinsic is attainable, at `V = 0`."""
    intrinsic = max(F - K, 0.0)
    assert black_implied_vol(intrinsic, F, K, 1.0, CapFloorKind.CAP) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("kind", list(CapFloorKind))
def test_a_DEEP_vol_still_inverts_so_the_bracket_EXPANDS(kind):
    """The bracket is structural, not a magic constant: a vol far outside any plausible default
    must still be recovered, which is what proves the expansion runs rather than a hardcoded cap."""
    sigma = 3.0
    price = black_price(F, K, sigma * sigma, kind)
    assert black_implied_vol(price, F, K, 1.0, kind) == pytest.approx(sigma, rel=1e-7)


# ---------------------------------------------------------------------------------------------
# Tranche 22 — Black's analytic Greeks, which did NOT exist before this tranche.
#
# The T22 spec names them as its anchor ("we have Black's analytic Greeks (T9)"). **We did not.**
# T9 built the implied-vol *surface* (`FloorQuote`, `ImpliedVolSurface`, `implied_vol_surface`);
# a grep of `src/` for any `delta`/`gamma`/`vega` definition returned nothing. So the anchor the
# Monte Carlo Greeks are measured against is built here first, and it needs its own oracle rather
# than being assumed — which is the point of noticing.
#
# The oracle is a FINITE DIFFERENCE of `black_price`, which has been verified since tranche 5. That
# is a genuinely independent check: one side is a closed-form derivative, the other a numerical
# derivative of a different closed form. A wrong `d₊`/`d₋` in the Greek cannot survive it.
# ---------------------------------------------------------------------------------------------

#: A step small enough that the finite-difference truncation error is below the comparison bound and
#: large enough that cancellation in `price(F+h) − price(F−h)` does not dominate. Central differences,
#: so the error is `O(h²)` ≈ 1e-10 at this step, against a 1e-6 relative comparison.
_FD_STEP = 1e-5


def _central(f, x: float, step: float) -> float:
    return (f(x + step) - f(x - step)) / (2.0 * step)


def _converges_quadratically(
    exact: float, difference, steps=(1e-3, 1e-4, 1e-5, 1e-6), floor: float = 1e-6
) -> None:
    """Assert a central difference converges to `exact` at the `O(h²)` rate central differences have.

    **Stronger than comparing at one step, and it had to be.** A first version of these oracles
    compared the analytic Greek to a single central difference at `h = 1e-5` and failed on the deep
    out-of-the-money cases by ~4e-6 relative. The *finite difference* was the weaker side: varying
    the step showed its gap falling 5.6e-2 → 5.6e-4 → 5.6e-6 → 5.6e-8 as `h` fell tenfold each time
    — textbook `O(h²)`, converging **onto** the analytic value. So the analytic formula was right and
    the reference was truncating (§4: a reference is code too, and here its error was a hundred times
    the comparison bound).

    Asserting the *rate* proves the closed form is the **limit** of the numerical derivative, which
    is a different and better claim than being near one approximation of it. A wrong `d₊` would not
    produce a clean quadratic approach to the wrong number."""
    gaps = [abs(difference(h) - exact) / abs(exact) for h in steps]
    for coarse, fine in zip(gaps, gaps[1:]):
        # Tenfold smaller step, hundredfold smaller error — generously banded, since the last steps
        # begin to feel floating-point cancellation. Measured ratios run 87 to 102.
        assert fine < coarse / 30.0, f"gaps {gaps} do not fall at the O(h^2) rate"

    # **`floor` is the REFERENCE's achievable accuracy, and it differs by instrument.** A first
    # difference reaches ~1e-8; a second divides by `h²`, so cancellation bites sooner and it
    # plateaus a couple of orders higher. Asserting one floor for both would be asserting the second
    # difference is as good as the first — which it is not, and which is the reference's property
    # rather than the formula's.
    assert gaps[-1] < floor, f"the finest gap {gaps[-1]:.2e} is above the reference's floor"


@pytest.mark.parametrize("kind", [CapFloorKind.CAP, CapFloorKind.FLOOR])
@pytest.mark.parametrize(
    ("forward", "strike", "variance"),
    [(0.04, 0.04, 0.04), (0.04, 0.05, 0.04), (0.05, 0.04, 0.01), (0.02, 0.06, 0.09)],
)
def test_T22_black_DELTA_is_the_LIMIT_of_a_finite_difference_of_the_verified_price(
    forward, strike, variance, kind
):
    """`∂price/∂F = ω·N(ω d₊)`, proved as the limit of a central difference of `black_price`.

    Swept across the money (at, in, out and deep out) and both directions, because `d₊`'s sign
    handling is where a delta goes wrong and at-the-money alone would hide it. The deep case
    `(0.02, 0.06, 0.09)` is the one that exposed the reference's own truncation."""
    _converges_quadratically(
        black_delta(forward, strike, variance, kind),
        lambda h: (
            black_price(forward + h, strike, variance, kind)
            - black_price(forward - h, strike, variance, kind)
        ) / (2.0 * h),
    )


@pytest.mark.parametrize(
    ("forward", "strike", "variance"),
    [(0.04, 0.04, 0.04), (0.04, 0.05, 0.04), (0.05, 0.04, 0.01)],
)
def test_T22_black_GAMMA_is_the_LIMIT_of_a_SECOND_finite_difference(forward, strike, variance):
    """`∂²price/∂F² = φ(d₊)/(F·√V²)`, again as a limit rather than a point comparison.

    A second difference divides by `h²`, so cancellation bites sooner and its achievable floor is a
    couple of orders higher than a first difference's — `1e-5` rather than `1e-6`, with the steps
    starting coarser. That is the reference's limit, stated rather than worked around; the measured
    convergence ratios are 87 to 102 per tenfold step, which is the claim that matters."""
    _converges_quadratically(
        black_gamma(forward, strike, variance, CapFloorKind.CAP),
        lambda h: (
            black_price(forward + h, strike, variance, CapFloorKind.CAP)
            - 2.0 * black_price(forward, strike, variance, CapFloorKind.CAP)
            + black_price(forward - h, strike, variance, CapFloorKind.CAP)
        ) / h**2,
        steps=(1e-2, 1e-3, 1e-4, 1e-5),
        floor=1e-5,
    )


@pytest.mark.parametrize(
    ("forward", "strike", "variance"),
    [(0.04, 0.04, 0.04), (0.04, 0.05, 0.04), (0.05, 0.04, 0.01)],
)
def test_T22_black_VEGA_matches_a_finite_difference_in_ROOT_VARIANCE(forward, strike, variance):
    """`∂price/∂√(V²) = F·φ(d₊)` — vega per unit of `σ√T`, not per unit of `σ`.

    **The parameterisation is ours, and it is the honest one here.** `black_price` takes a *total
    variance*, never `σ` and `T` separately, so its natural vega is with respect to `√V²`; the chain
    rule to `∂/∂σ = F·φ(d₊)·√T` belongs at a call site that knows `T`. Returning a `∂/∂σ` from a
    function that is not told `T` would mean inventing one."""
    root = math.sqrt(variance)
    numeric = _central(
        lambda r: black_price(forward, strike, r * r, CapFloorKind.CAP), root, _FD_STEP
    )
    assert black_vega(forward, strike, variance, CapFloorKind.CAP) == pytest.approx(
        numeric, rel=1e-6
    )


def test_T22_GAMMA_and_VEGA_are_the_SAME_for_a_CAP_and_a_FLOOR():
    """A requirement oracle from put-call parity, so it survives a transcription error.

    `cap − floor = F − K`, which is linear in `F` and independent of volatility. So the second
    derivative in `F` and the first in `σ` must be identical for the two — while **delta differs by
    exactly 1**, which is the same identity read once. A formula that got `d₊` wrong would be
    unlikely to preserve both."""
    forward, strike, variance = 0.045, 0.04, 0.0225
    assert black_gamma(forward, strike, variance, CapFloorKind.CAP) == pytest.approx(
        black_gamma(forward, strike, variance, CapFloorKind.FLOOR), rel=1e-15
    )
    assert black_vega(forward, strike, variance, CapFloorKind.CAP) == pytest.approx(
        black_vega(forward, strike, variance, CapFloorKind.FLOOR), rel=1e-15
    )
    cap_delta = black_delta(forward, strike, variance, CapFloorKind.CAP)
    floor_delta = black_delta(forward, strike, variance, CapFloorKind.FLOOR)
    assert cap_delta - floor_delta == pytest.approx(1.0, rel=1e-14)


def test_T22_the_DETERMINISTIC_limit_is_the_intrinsic_slope():
    """At zero variance the price is the intrinsic, so delta is the step function and vega vanishes.

    Exact by construction, matching `black_price`'s own zero-variance branch — which returns the
    intrinsic *exactly* rather than approaching it, so the Greeks must do the same rather than
    dividing by a zero `√V²`."""
    assert black_delta(0.05, 0.04, 0.0, CapFloorKind.CAP) == 1.0
    assert black_delta(0.03, 0.04, 0.0, CapFloorKind.CAP) == 0.0
    assert black_vega(0.05, 0.04, 0.0, CapFloorKind.CAP) == 0.0
    assert black_gamma(0.05, 0.04, 0.0, CapFloorKind.CAP) == 0.0


def test_T22_the_DIGITAL_price_and_delta_anchor_the_discontinuous_case():
    """A cash-or-nothing digital — `1{ωF_T > ωK}`, price `N(ω d₋)` — and its delta.

    Built because the pathwise/likelihood-ratio split needs a payoff where **pathwise genuinely
    breaks**, and a digital is the canonical one: its payoff is flat almost everywhere, so a pathwise
    derivative is zero where the true delta is not. Without an analytic digital delta that companion
    would have nothing to be measured against."""
    forward, strike, variance = 0.04, 0.042, 0.04
    assert digital_price(forward, strike, variance, CapFloorKind.CAP) == pytest.approx(
        _digital_reference(forward, strike, variance), rel=1e-14
    )
    numeric = _central(
        lambda f: digital_price(f, strike, variance, CapFloorKind.CAP), forward, _FD_STEP
    )
    assert digital_delta(forward, strike, variance, CapFloorKind.CAP) == pytest.approx(
        numeric, rel=1e-6
    )
    # A digital's delta is strictly positive and its price lies in (0,1) — it is a probability.
    assert 0.0 < digital_price(forward, strike, variance, CapFloorKind.CAP) < 1.0
    assert digital_delta(forward, strike, variance, CapFloorKind.CAP) > 0.0


def _digital_reference(forward: float, strike: float, variance: float) -> float:
    """`P(F_T > K) = N(d₋)`, transcribed independently of the module."""
    root = math.sqrt(variance)
    d_minus = (math.log(forward / strike) - 0.5 * variance) / root
    return norm_cdf(d_minus)
