"""SVI — the stochastic-volatility-inspired smile parameterization (L3). `GATHERAL-2006` eq 3.20.

One expiration's slice of **total** implied variance `w(k) = σ²_BS(k)·T` as a closed form in five
parameters, plus the no-arbitrage conditions that decide whether a fitted slice is admissible.

**Total variance, not annualised, and the choice is load-bearing.** The spec first wrote Lee's wing
bound as `b(1+|ρ|) ≤ 2` against *annualised* variance, which is right only at `T = 1`: ch 7 bounds
`σ²_BS(k,T)·T/|k|`, so the annualised form carries a `T`. In total variance every condition is
clean — the wing slopes are `b(1±ρ)` bounded by 2 directly, and the calendar condition is plain
monotonicity of `w` in `T` — and the output feeds `models/black.py`, which is itself
total-variance-parameterised, with **no conversion at all**. The single conversion
(annualised quote → `w`, a `×T`) sits at the fit's *input* boundary, which is why this module holds
no clock and `SviParams` has no `T` field. Convention row **C14**.

**The archived predecessor got exactly this wrong**, which is the §5 argument for reading the source
rather than the stock: its `SVIParams` declares `w = σ²_BS × T` in the docstring, carries `T` as a
sixth field, and then tests `b(1+|ρ|) ≤ 4/T` — a `T`-dependent bound that cannot be right for a
quantity already multiplied by `T`. Same defect as the spec's, shipped.

**How eq 3.20's radical is pinned, given neither rendering shows it.** `pdftotext` drops the glyph
in both `-raw` and `-layout`, leaving a large-math vinculum (`√` survives 123 times elsewhere in the
book, so this is not a generic loss). The vinculum's measured span settles the *scope* — it covers
the radicand and excludes `ρ(k−m)`, which is the v0.4.0 mis-nesting hazard — and ch 7's
model-independent `β*, α* ∈ [0,2]` forces the *identity*, because a non-radical 3.20 grows
quadratically in `k` and violates the bound unboundedly. §4's derivation-step independence, across
chapters. (The stock's transcription agrees, but §5 makes that corroboration, not the anchor.)

**The butterfly condition is derived here, not transcribed.** The 2006 text gives only the calendar
condition; the rigorous `g(k) ≥ 0` is Gatheral–Jacquier 2014, which we do not hold (an
acquire-source note sits in `docs/to_go_deeper.md`). So `risk_neutral_density` is derived from
Breeden–Litzenberger on the fitted smile — see its docstring for the algebra and the trap in it.

Provenance:
  stock:  _stock/quarry/pricebook/options/vol_derivatives_advanced.py `SVIParams` — mined for
          CONTENT only, and chiefly as a counter-example: it confirms the radical and the
          total-variance reading, and its `b(1+|ρ|) ≤ 4/T` is the dimensional defect described
          above. Its `T` field and its `to_dict` shape are not carried.
  source: GATHERAL-2006 (docs/references.md) ch 3 eq 3.20 ("Another Digression: The SVI
          Parameterization", p. 37) for the slice; ch 7 pp. 97–98 (Lee's moment formula,
          `β*, α* ∈ [0,2]`) for the wing bound that pins the radical and bounds `b(1+|ρ|)`
  oracle: tests/L3/test_svi.py — eq 3.20 transcribed pointwise, the closed-form vertex and
          minimum, the linear wings with slopes `b(1±ρ)`, the `b→0` reduction with a
          non-degenerate companion, the derived density against a numerical
          Breeden–Litzenberger second difference plus its unit integral and its exact
          Black–Scholes collapse at `b=0`, the synthetic round-trip with T11 multi-start spread
          AND distance-from-truth, and the Heston↔SVI reprice through `black_price`
  crosscheck: quantlib SviInterpolation / SviSmileSection — deferred to the optional QuantLib tier
  slice:  tranche 46 — the SVI surface parameterization
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pricebook.foundation.distributions import norm_pdf
from pricebook.foundation.solvers import least_squares_fit

#: Lee's bound on a total-variance wing slope (ch 7, `β*, α* ∈ [0,2]`). Model-independent — it
#: assumes only that a martingale measure exists — so it bounds any admissible slice.
_LEE_WING_BOUND = 2.0

#: How many points the butterfly certificate is checked at across the quoted span.
#:
#: **There is no margin beyond the span, and that is a derived decision rather than a default.** An
#: earlier version widened the scan by 50% of the span; the mutation sweep then killed nothing by
#: narrowing it back to zero, which is the reachability rule firing on a *constant* — an unexercised
#: widening is dead config. Working out why no fixture could exercise it produced the better answer:
#: as `|k| -> infinity` the slice's `w' -> b(1±rho)`, `w'' -> 0` and the skew term -> 1/2, so
#:
#:     g(±infinity) = 1/4 - (b(1±rho))^2 / 16   >=  0   <==>   b(1±rho) <= 2
#:
#: — the asymptotic no-butterfly condition **is Lee's bound** (verified to 8 decimals on both wings
#: of three slices; `tests/L3/test_svi.py` asserts it). So the tails are certified analytically by
#: `satisfies_lee_bound` and the interior numerically here, and the two together are why a margin
#: bought nothing.
_BUTTERFLY_POINTS = 401

#: `|ρ| < 1` and `σ > 0` are strict, so the solver is kept just inside them: at `|ρ| = 1` the slice
#: degenerates to one straight wing and at `σ = 0` the vertex is a corner with no second derivative.
_RHO_LIMIT = 1.0 - 1e-9
_SIGMA_FLOOR = 1e-9


@dataclass(frozen=True)
class SviParams:
    """One expiration's slice, in total variance. Five parameters — exactly §7's limit, and no `T`.

    `a` the overall level, `b` the wing angle, `rho` the skew, `m` the horizontal shift of the
    vertex, `sigma` the vertex smoothness. Total variance needs no clock, so adding `T` here (as the
    stock did) would be a sixth field buying nothing but the chance to disagree with the caller's.

    **The guards are definitional only.** `b ≥ 0`, `|ρ| < 1`, `σ > 0` and a non-negative minimum
    variance are what it takes for the expression to *be* a variance slice. Lee's bound and the
    butterfly condition are **admissibility**, checked at the fit — deliberately not here, because
    an arbitrageable-but-well-formed slice must be constructible or the guards that reject one could
    never be given a fixture to reject (T33b: a uniformly well-formed input population leaves a
    guard's negative branch untested).
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if self.b < 0.0:
            raise ValueError(f"the wing angle b must be non-negative, got {self.b}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"the skew correlation rho must lie strictly in (-1, 1), got {self.rho}")
        if self.sigma <= 0.0:
            raise ValueError(f"the vertex smoothness sigma must be positive, got {self.sigma}")
        floor = self.a + self.b * self.sigma * math.sqrt(1.0 - self.rho * self.rho)
        if floor < 0.0:
            raise ValueError(
                f"the slice has negative total variance: its minimum is "
                f"a + b*sigma*sqrt(1-rho^2) = {floor}, which must be non-negative"
            )

    def total_variance(self, log_strike: float) -> float:
        """`w(k) = a + b[ρ(k−m) + √((k−m)² + σ²)]` — eq 3.20."""
        shifted = log_strike - self.m
        radical = math.sqrt(shifted * shifted + self.sigma * self.sigma)
        return self.a + self.b * (self.rho * shifted + radical)

    def slope(self, log_strike: float) -> float:
        """`w'(k) = b[ρ + (k−m)/√((k−m)² + σ²)]`."""
        shifted = log_strike - self.m
        radical = math.sqrt(shifted * shifted + self.sigma * self.sigma)
        return self.b * (self.rho + shifted / radical)

    def curvature(self, log_strike: float) -> float:
        """`w''(k) = bσ²/((k−m)² + σ²)^{3/2}` — strictly positive for `b > 0`, so `w` is convex.

        The quotient rule collapses: `d/dk[(k−m)/r] = (r² − (k−m)²)/r³ = σ²/r³`."""
        shifted = log_strike - self.m
        radical = math.sqrt(shifted * shifted + self.sigma * self.sigma)
        return self.b * self.sigma * self.sigma / radical**3

    def wing_slopes(self) -> tuple[float, float]:
        """The right and left asymptotic slopes, `b(1+ρ)` and `b(1−ρ)`.

        As `k → ±∞` the radical tends to `±(k−m)`, so `w → a + b(ρ ± 1)(k − m)`: two straight
        lines. They differ precisely when `ρ ≠ 0`, which is what makes `ρ`'s sign observable."""
        return self.b * (1.0 + self.rho), self.b * (1.0 - self.rho)

    def minimum(self) -> tuple[float, float]:
        """Where `w' = 0`, and the value there — both closed form, so no search is involved.

        `w' = 0` needs `(k−m)/r = −ρ`, which squares to `(k−m)²(1−ρ²) = ρ²σ²`, giving
        `k* = m − ρσ/√(1−ρ²)` (the sign of `k*−m` must oppose `ρ`) and then `r = σ/√(1−ρ²)`, so
        `w* = a + bσ√(1−ρ²)` — the same quantity `__post_init__` requires non-negative."""
        root = math.sqrt(1.0 - self.rho * self.rho)
        return self.m - self.rho * self.sigma / root, self.a + self.b * self.sigma * root


def annualised_vol(params: SviParams, log_strike: float, years: float) -> float:
    """`σ_BS(k) = √(w(k)/T)` — the slice read back in the market's own units.

    The clock appears here and nowhere else in this module, which is the honest shape of C14: the
    conversion is a boundary concern, not a property of the slice."""
    if years <= 0.0:
        raise ValueError(f"an annualised vol needs a positive horizon, got {years}")
    return math.sqrt(params.total_variance(log_strike) / years)


def risk_neutral_density(params: SviParams, log_strike: float) -> float:
    """The Breeden–Litzenberger density of `log(S_T/F)` implied by the slice. Derived, not cited.

    For the undiscounted call per unit forward `c(k) = N(d₊) − e^k N(d₋)` on the payoff
    `(e^X − e^k)⁺`, differentiating under the integral gives `∂c/∂k = −e^k(1 − P(k))`, hence

        e^k p(k) = c''(k) − c'(k)

    — **and the `− c'` is the trap**: `p = e^{−k}c''` is the reflex and it is wrong by `O(1)`.
    Carrying it through with Black's identity `n(d₊) = e^k n(d₋)` leaves `c' = e^k·S` for
    `S = n(d₋)w'/(2√w) − N(d₋)`, so `p = S'`, and expanding `S'` collapses to

        p(k) = g(k)·n(d₋)/√w,
        g(k) = (1 − k w'/(2w))² − (w'²/4)(1/w + 1/4) + w''/2

    Since `n(d₋)/√w > 0` always, the no-butterfly condition is exactly `g ≥ 0`; the density is
    returned rather than `g` because `g` is only a reparameterisation of it and the density carries
    the two extra requirement oracles — it integrates to 1, and at `b = 0` it collapses *exactly*
    onto the Black–Scholes normal, where `g ≡ 1`.
    """
    variance = params.total_variance(log_strike)
    first = params.slope(log_strike)
    second = params.curvature(log_strike)
    skew_term = 1.0 - log_strike * first / (2.0 * variance)
    g = skew_term * skew_term - (first * first / 4.0) * (1.0 / variance + 0.25) + second / 2.0
    root = math.sqrt(variance)
    d_minus = -log_strike / root - root / 2.0
    return g * norm_pdf(d_minus) / root


def satisfies_lee_bound(params: SviParams) -> bool:
    """`b(1+|ρ|) ≤ 2` — ch 7's bound on the steeper total-variance wing.

    Lee's `β*` and `α*` are the two wing slopes of `w` (`b(1−ρ)` and `b(1+ρ)`), and both lie in
    `[0,2]`; the larger is `b(1+|ρ|)`, so one comparison covers both. No `T` appears, which is the
    whole reason the slice is held in total variance."""
    return params.b * (1.0 + abs(params.rho)) <= _LEE_WING_BOUND


def calendar_arbitrage_free(
    earlier: SviParams, later: SviParams, log_strikes: Sequence[float]
) -> bool:
    """Total variance must not decrease with expiry at any `k` — the slices do not cross.

    Asymmetric in its arguments on purpose: `earlier` is the shorter expiry, and swapping them is a
    different question with a different answer. Checked on the strikes supplied rather than
    asymptotically, because that is where the quotes are and a certificate is a statement about the
    domain it examined (§4)."""
    return all(
        later.total_variance(k) >= earlier.total_variance(k) - 1e-14 for k in log_strikes
    )


def _butterfly_breach(params: SviParams, log_strikes: Sequence[float]) -> float | None:
    """The log-strike of the worst negative density across the quoted span, or `None`.

    **What this does NOT cover, stated rather than implied (§4).** Together with
    `satisfies_lee_bound` it certifies the quoted span (numerically, here) and the two tails
    (analytically — see `_BUTTERFLY_POINTS`). It does **not** certify the intermediate region
    between the far edge of the quotes and the regime where the asymptote dominates. Closing that
    gap is exactly what Gatheral–Jacquier (2014)'s closed-form conditions would buy, and it is why
    the acquire-source note in `docs/to_go_deeper.md` exists: a real gap, not a decorative one."""
    lo, hi = min(log_strikes), max(log_strikes)
    step = (hi - lo) / (_BUTTERFLY_POINTS - 1)
    worst: tuple[float, float] | None = None
    for i in range(_BUTTERFLY_POINTS):
        k = lo + i * step
        density = risk_neutral_density(params, k)
        if density < 0.0 and (worst is None or density < worst[1]):
            worst = (k, density)
    return None if worst is None else worst[0]


def _starting_point(
    log_strikes: Sequence[float], total_variances: Sequence[float]
) -> tuple[float, float, float, float, float]:
    """A start read off the data, with **no special case** — which is the second version.

    The first estimated each wing's slope from the vertex to the corresponding endpoint. That has a
    degenerate case the obvious fixtures never show: a **one-sided** smile puts the smallest quote
    *at* an endpoint, so one side has no data at all, its estimated slope is exactly zero, `rho0`
    pins to its clip, and the solve walks out of evaluations at a cost of 2e-10 — a good fit
    reported as non-convergence. Flooring the span did not help, because the defect is missing data
    rather than a small denominator. So the slope estimate is **one-sided on purpose**: the total
    rise over the span, with `rho0 = 0` leaving the skew entirely to the solver.

    Deliberately start-neutral rather than tuned: a `rho0` guessing the usual equity skew would be
    an assumption about the market smuggled into a fit that is meant to read it off the quotes. The
    identifiability oracle drives the same fit from wild starts precisely to show the answer does
    not depend on this.
    """
    vertex = min(range(len(total_variances)), key=lambda i: total_variances[i])
    floor = min(total_variances)
    span = max(log_strikes[-1] - log_strikes[0], 1e-3)
    rise = max(max(total_variances) - floor, 0.0)
    return floor * 0.5, max(rise / span, 1e-3), 0.0, log_strikes[vertex], 0.1 * span


def fit_svi_slice(
    log_strikes: Sequence[float],
    total_variances: Sequence[float],
    start: Sequence[float] | None = None,
) -> SviParams:
    """Fit eq 3.20 to a slice of **total** variances by least squares, refusing an inadmissible fit.

    The quotes are total variances `w = σ²_BS·T`, not annualised — the `×T` is the caller's, at the
    boundary where the market's units stop (C14).

    A fit that converges on an arbitrageable slice is refused rather than returned, because §4's
    standing lesson is that a plausible wrong number is the dangerous output: a surface that prices
    smoothly while admitting a negative density would pass every eyeball test. The three refusals
    are distinct — non-convergence, Lee's wing bound, and the butterfly density — and each names
    what it saw.
    """
    if len(log_strikes) != len(total_variances):
        raise ValueError(
            f"a slice needs one total variance per log-strike, got "
            f"{len(log_strikes)} and {len(total_variances)}"
        )
    if len(log_strikes) < 5:
        raise ValueError(
            f"eq 3.20 has five parameters, so a fit needs at least five quotes, got {len(log_strikes)}"
        )

    def residuals(x: Sequence[float]) -> Sequence[float]:
        a, b, rho, m, sigma = x
        out = []
        for k, target in zip(log_strikes, total_variances, strict=True):
            shifted = k - m
            radical = math.sqrt(shifted * shifted + sigma * sigma)
            out.append(a + b * (rho * shifted + radical) - target)
        return out

    x0 = tuple(start) if start is not None else _starting_point(log_strikes, total_variances)
    bounds = (
        (-10.0, 0.0, -_RHO_LIMIT, -10.0, _SIGMA_FLOOR),
        (10.0, 10.0, _RHO_LIMIT, 10.0, 10.0),
    )
    outcome = least_squares_fit(residuals, x0, bounds)
    if not outcome.converged:
        raise ValueError(
            f"the SVI fit did not converge from {tuple(x0)} after {outcome.iterations} "
            f"evaluations (cost {outcome.cost})"
        )
    a, b, rho, m, sigma = outcome.x
    fitted = SviParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
    if not satisfies_lee_bound(fitted):
        raise ValueError(
            f"the fitted slice violates Lee's wing bound: b(1+|rho|) = "
            f"{fitted.b * (1.0 + abs(fitted.rho))} exceeds {_LEE_WING_BOUND}"
        )
    breach = _butterfly_breach(fitted, log_strikes)
    if breach is not None:
        raise ValueError(
            f"the fitted slice admits butterfly arbitrage: its risk-neutral density is negative "
            f"at log-strike {breach}, checked at {_BUTTERFLY_POINTS} points across the quoted "
            f"span [{min(log_strikes)}, {max(log_strikes)}]"
        )
    return fitted
