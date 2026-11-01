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


def _no_arbitrage_limit(forward: float, strike: float, kind: CapFloorKind) -> float:
    """The price approached as `V² → ∞`: the forward for a cap, the strike for a floor.

    Approached and never reached, so a price AT the limit has no finite vol — which is why the
    inverse refuses it rather than returning whatever the expansion stopped at."""
    return forward if kind is CapFloorKind.CAP else strike


def black_implied_vol(
    price: float, forward: float, strike: float, expiry: float, kind: CapFloorKind
) -> float:
    """The annualised `σ` for which Black reproduces `price` — `σ = √(V²/expiry)`.

    Black is strictly increasing in `V²` from the intrinsic to the no-arbitrage limit, so the
    inverse is a monotone 1-D root find with a **structural** bracket: the lower end is `V² = 0`
    (the intrinsic) and the upper end is found by doubling until the price is exceeded. No magic
    constant decides the range; the monotonicity does.

    **Refuses rather than guessing.** A price below the intrinsic or at/above the no-arbitrage limit
    has no Black vol at all, and inventing one would be a number that looks like a measurement. The
    raise follows tranche 8's ruling: `PricingFailure` is the *engine* contract (§3), while an L1/L3
    constructor validates at its trust boundary by raising, as `DiscountCurve`, `Money` and
    `nominal_curve_from_par_swaps` do."""
    if expiry <= 0.0:
        raise ValueError(
            f"an annualised vol needs a positive expiry to divide by, got {expiry}"
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

    hi = max(1e-8, 4.0 * expiry)
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
    return math.sqrt(bracketed_root(excess, 0.0, hi) / expiry)
