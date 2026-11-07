"""Black-76 on a forward, and its inverse (L3). BERHAN-2012 §2.6, §5.2; standard Black-76.

    Black(F, K, V², ω) = ω·[ F·Φ(ω·d₁) − K·Φ(ω·d₂) ]
    d₁ = [ln(F/K) + V²/2] / V ,   d₂ = d₁ − V ,   V = √(V²)

**Parameterised by TOTAL VARIANCE, not `(σ, T)`, and that is the point.** Jarrow–Yildirim's `V²`
(eq 2.6.4) is a total variance assembled from eight terms — it is **not** `σ²·T`, so a `(σ, T)`
signature would force every caller to invent a `σ` the model does not have. The annualised vol
appears only at the inverse's boundary, where §5.2's surface wants `σ_I` per (expiry, strike).

**Why L3 and not L0.** Strikes, call/put and an option payoff are finance, and L0 is finance-free
(§2, machine-checked by `verify.py layers`). The migration tree's own `foundation/black.py` is the
named precedent for exactly this drift, so this lives in `models/`.

**Extracted from tranche 5's inline caplet, at its second consumer.** T5 computed Black inside
`engine/capfloor.py`, which was right with one consumer; tranche 9's inverter is the second, so §3d
and the rule of two say extract it now — one definition the pricer and the inverter both compose,
so a Black convention cannot diverge between pricing and implying. The extraction is a **pure
refactor**: T5's oracles are the guard, and the caplet's numbers must not move by a bit.

**Oracle power, so the round trip is not mistaken for the anchor.** Pricing at `σ` and implying it
back *telescopes* — the inverse solves the forward, so it cannot catch an error they share. It earns
its keep only because **T5 oracled this forward against eq 2.6.5 independently**, from a value
transcribed out of the PDF. Parity (`cap − floor = F − K`), monotonicity and the refusals do not
telescope and stand on their own.

Provenance:
  stock:  _stock/quarry/pricebook/options/ — mined for the stripping/implied-vol SHAPE only; the
          formula is read from standard Black-76 and from T5's transcription of eq 2.6.5, not from
          the stock's code, which may carry a different parameterisation
  source: BERHAN-2012 (docs/references.md) §2.6 eq 2.6.5 (the forward), §5.2 (the inverse builds
          the implied-vol surface); standard Black-76 for the inverse's bounds
  oracle: tests/L3/test_black.py — the forward against an independent transcription; put-call
          parity to 1e-15 as a REQUIREMENT; the no-arbitrage bounds; the inverse round-tripping,
          monotone, and REFUSING a sub-intrinsic or at-limit price rather than guessing a vol
  crosscheck: quantlib blackFormula / blackFormulaImpliedStdDev — deferred to the optional
          QuantLib tier
  slice:  tranche 9(a) — Black extracted to L3, and its inverse
"""

from __future__ import annotations

import math

from pricebook.foundation.distributions import norm_cdf
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.solvers import bracketed_root

#: Total variance at which the expansion gives up. `V² = 1e6` is `σ = 1000` at one year — far past
#: any price a no-arbitrage check admits, so reaching it means the target is unattainable rather
#: than merely large. It is a termination bound, not a modelling assumption.
_MAX_VARIANCE = 1.0e6


def black_price(forward: float, strike: float, total_variance: float, kind: CapFloorKind) -> float:
    """Black-76 per unit of notional and undiscounted — eq 2.6.5's bracket.

    `total_variance` is `V²`, not `σ`. Zero (or less) is the deterministic case and returns the
    intrinsic exactly, rather than approaching it as a limit."""
    omega = kind.omega
    if total_variance <= 0.0:
        return max(omega * (forward - strike), 0.0)
    v = math.sqrt(total_variance)
    d1 = (math.log(forward / strike) + 0.5 * total_variance) / v
    d2 = d1 - v
    return omega * (forward * norm_cdf(omega * d1) - strike * norm_cdf(omega * d2))


def _d_plus(forward: float, strike: float, total_variance: float) -> tuple[float, float]:
    """`(d₊, √V²)` — the pair every Greek below needs, computed once in one place.

    Factored out rather than repeated in four functions: it is the expression a transcription error
    would most likely land in, so it is written once and the Greeks compose it."""
    root = math.sqrt(total_variance)
    return (math.log(forward / strike) + 0.5 * total_variance) / root, root


def black_delta(forward: float, strike: float, total_variance: float, kind: CapFloorKind) -> float:
    """`∂price/∂F = ω·N(ω d₊)` — Black-76's delta, undiscounted, per unit of notional.

    **Built at tranche 22, and the T22 spec was wrong that we already had it.** The spec names
    "Black's analytic Greeks (T9)" as the anchor for the Monte Carlo Greeks; tranche 9 built the
    implied-vol *surface*, and no analytic Greek existed anywhere in `src/`. Noticing mattered
    because an anchor cannot be assumed: these are oracled against central differences of
    `black_price`, which has been verified since tranche 5 — a closed-form derivative against a
    numerical derivative of a different closed form.

    At zero variance the price is the intrinsic exactly (see `black_price`), so the delta is the step
    function rather than a limit approached — and `√V² = 0` is never divided by."""
    if total_variance <= 0.0:
        omega = kind.omega
        return 1.0 if omega * (forward - strike) > 0.0 else 0.0
    d_plus, _ = _d_plus(forward, strike, total_variance)
    omega = kind.omega
    return omega * norm_cdf(omega * d_plus)


def black_gamma(forward: float, strike: float, total_variance: float, kind: CapFloorKind) -> float:
    """`∂²price/∂F² = φ(d₊)/(F·√V²)` — the same for a cap and a floor.

    `kind` is accepted and unused, which §6 normally forbids — kept because gamma IS a property of
    the option and a caller holding a `kind` should not have to know it cancels. That it cancels is
    put-call parity (`cap − floor = F − K` is linear in `F`), and the oracle asserts the two agree
    rather than leaving the reader to trust the comment."""
    if total_variance <= 0.0:
        return 0.0
    d_plus, root = _d_plus(forward, strike, total_variance)
    return _norm_pdf(d_plus) / (forward * root)


def black_vega(forward: float, strike: float, total_variance: float, kind: CapFloorKind) -> float:
    """`∂price/∂√(V²) = F·φ(d₊)` — vega per unit of **root variance**, not per unit of `σ`.

    **The parameterisation is forced by ours and it is the honest one.** `black_price` takes a total
    variance and is never told `σ` and `T` separately, so the only vega it can report is with respect
    to `√V² = σ√T`. A caller that knows `T` multiplies by `√T` to get `∂/∂σ`; a function not told `T`
    returning a `∂/∂σ` would be inventing one. Like gamma, identical for a cap and a floor."""
    if total_variance <= 0.0:
        return 0.0
    d_plus, _ = _d_plus(forward, strike, total_variance)
    return forward * _norm_pdf(d_plus)


def digital_price(forward: float, strike: float, total_variance: float, kind: CapFloorKind) -> float:
    """A cash-or-nothing digital: `1{ω·F_T > ω·K}`, worth `N(ω d₋)` undiscounted.

    Here because the Greeks need a payoff on which **pathwise differentiation genuinely breaks** —
    a digital is flat almost everywhere, so its pathwise derivative is zero where its true delta is
    not. Without it, "pathwise agrees with the closed form" would be a vacuous claim and the
    likelihood-ratio method would have nothing to justify it."""
    if total_variance <= 0.0:
        return 1.0 if kind.omega * (forward - strike) > 0.0 else 0.0
    d_plus, root = _d_plus(forward, strike, total_variance)
    return norm_cdf(kind.omega * (d_plus - root))


def digital_delta(forward: float, strike: float, total_variance: float, kind: CapFloorKind) -> float:
    """`∂/∂F N(ω d₋) = ω·φ(d₋)/(F·√V²)` — strictly positive for a digital call."""
    if total_variance <= 0.0:
        return 0.0
    d_plus, root = _d_plus(forward, strike, total_variance)
    return kind.omega * _norm_pdf(d_plus - root) / (forward * root)


def _norm_pdf(x: float) -> float:
    """The standard normal density. Local to this module because the Greeks are its only consumer —
    §6's rule-of-two has not fired, so it has not been promoted to `foundation.distributions`."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _no_arbitrage_limit(forward: float, strike: float, kind: CapFloorKind) -> float:
    """The price approached as `V² → ∞`: the forward for a cap, the strike for a floor.

    Approached and never reached, so a price AT the limit has no finite vol — which is why the
    inverse refuses it rather than returning whatever the expansion stopped at."""
    return forward if kind is CapFloorKind.CAP else strike


def black_implied_vol(
    price: float, forward: float, strike: float, variance_years: float, kind: CapFloorKind
) -> float:
    """The annualised `σ` for which Black reproduces `price` — `σ = √(V² / variance_years)`.

    **`variance_years` is the horizon the variance accumulates over, and it is NOT always the time
    to expiry — which is why it is not called `expiry`.** The first cut used that name, and tranche
    9(b) showed it wrong at its second consumer: a year-on-year floorlet on `[2y, 3y]` expires at 3y
    but accumulates one year of variance, since `V² = σ_I²·(Tᵢ − Tᵢ₋₁)` under the thesis's
    simplified JY. A caller passing the expiry there would under-state `σ_I` by `√(Tᵢ/(Tᵢ−Tᵢ₋₁))` —
    a factor of 1.7 at the 3y node — with every oracle green, because the forward and inverse
    would agree on the wrong horizon. If a name needs a comment to explain it, the name is wrong.

    Black is strictly increasing in `V²` from the intrinsic to the no-arbitrage limit, so the
    inverse is a monotone 1-D root find with a **structural** bracket: the lower end is `V² = 0`
    (the intrinsic) and the upper end is found by doubling until the price is exceeded. No magic
    constant decides the range; the monotonicity does.

    **Refuses rather than guessing.** A price below the intrinsic or at/above the no-arbitrage limit
    has no Black vol at all, and inventing one would be a number that looks like a measurement. The
    raise follows tranche 8's ruling: `PricingFailure` is the *engine* contract (§3), while an L1/L3
    constructor validates at its trust boundary by raising, as `DiscountCurve`, `Money` and
    `nominal_curve_from_par_swaps` do."""
    if variance_years <= 0.0:
        raise ValueError(
            f"an annualised vol needs a positive expiry (variance horizon) to divide by, "
            f"got {variance_years}"
        )

    intrinsic = black_price(forward, strike, 0.0, kind)
    if price < intrinsic - 1e-14:
        raise ValueError(
            f"a {kind.value} price of {price} is below its intrinsic {intrinsic}; no Black "
            f"volatility reproduces it"
        )
    limit = _no_arbitrage_limit(forward, strike, kind)
    if price >= limit:
        raise ValueError(
            f"a {kind.value} price of {price} is at or above its no-arbitrage limit {limit}, "
            f"which Black approaches only as the variance diverges; no finite volatility "
            f"reproduces it"
        )
    if price <= intrinsic:
        # Exactly intrinsic is attainable, at V² = 0 — a boundary that must not raise and must not
        # be handed to the root-find, whose bracket would be degenerate there.
        return 0.0

    def excess(total_variance: float) -> float:
        return black_price(forward, strike, total_variance, kind) - price

    hi = max(1e-8, 4.0 * variance_years)
    while excess(hi) < 0.0:
        hi *= 2.0
        if hi > _MAX_VARIANCE:
            # Unreachable for a price the checks above admit (it is strictly inside the attainable
            # range, and Black is continuous and increasing to the limit). Kept as a loud refusal
            # rather than an infinite loop: if it IS reached, the invariant is broken and silence
            # would be the worst outcome.
            raise ValueError(
                f"no Black volatility reproduces a {kind.value} price of {price} below total "
                f"variance {_MAX_VARIANCE}; the no-arbitrage check should have excluded it"
            )
    return math.sqrt(bracketed_root(excess, 0.0, hi) / variance_years)
