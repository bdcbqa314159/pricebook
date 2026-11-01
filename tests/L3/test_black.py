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
from pricebook.models.black import black_implied_vol, black_price

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
