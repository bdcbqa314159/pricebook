"""Variance swaps under time-changed Lévy processes (L4). KWOK-ZHENG §4.4.

Two deliverables, one built on the other:

- the **continuous** fair strike, `Kv(0,T;∞) = var(L₁)·E[Z_T]/T` (Prop 4.2 with Prop 4.1);
- the **discretely-monitored** one, `Kv(0,T;N) = Kv(0,T;∞) + adjustment` (Prop 4.3), where the
  adjustment is `O(1/N)` (Prop 4.8) and always positive, so the discrete swap is worth strictly
  more.

**Two readings of the source had to be ruled, both pinned by oracles rather than left to the reader
(§4).** They are recorded here because the code embodies the ruling:

1. **Prop 4.8 contradicts Prop 4.2.** It prints `Kv(∞) = −(1/T)φ_L⁽²⁾(0)E[Z_T−Z₀]`, which under the
   book's own (4.84) (`φ_{L_t}(u) = e^{−tψ_L(u)}`) is the **second moment** `E[L₁²]`. Prop 4.2 with
   Prop 4.1 gives `Q_L·(κ_L(1) − κ_L′(0)) = κ_L″(0) = var(L₁)`. They differ by `(E[L₁])²` — 20% and
   more on ordinary parameters, and agreeing only for a centred `L`. **We compute `var(L₁)`**: a
   fair strike is the expected quadratic variation, `[L,L]` is **drift-free**, and Prop 4.1's own
   proof states `E[L,L]₁ = var(L₁)`. Props 4.9/4.10 use `ψ_L″`, not `φ`, which makes `φ⁽²⁾` in 4.8
   look like the misprint.
2. **The multiplier buys no independence.** `Q_L·(κ_L(1) − κ_L′(0)) = κ_L″(0)` is an algebraic
   identity, so computing the strike "both ways" is one computation written twice (T70). `Q_L` is
   built because Table 4.3 tabulates it — it is the transcription target — and the strike's
   independent check is the simulation, not the second formula.

Provenance:
  stock:  none — the migration tree carried no Lévy variance-swap engine
  source: KWOK-ZHENG (docs/references.md) §4.4.1 Prop 4.1 eq (4.79) — the multiplier; Prop 4.2
          eq (4.83) — the quadratic variation as a multiple of the log contract; §4.4.2 Prop 4.3
          eqs (4.96)/(4.97) — the discrete decomposition; §4.4.4 Prop 4.8 — the O(1/N) rate and
          the contradicting continuous strike, ruled above; Table 4.3 — the CGMY/Brownian
          multipliers, transcribed in `-raw` and `-layout` and reconciled
  oracle: tests/L4/test_time_changed_levy.py — the multiplier against Table 4.3's own row and its
          exact reduction to the Brownian row's 2; the pure-diffusion adjustment against its exact
          closed form `(r−q−σ²/2)²T/N`; the O(1/N) rate measured as `adjustment × N → constant`;
          and an exact-increment Lévy simulation as the one independent check
  crosscheck: none — QuantLib has no time-changed-Lévy variance-swap engine
  slice:  tranche 78 — variance swaps under time-changed Lévy processes
"""

from __future__ import annotations

import math

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.equity import dividend_forward
from pricebook.models.time_changed_levy import (
    ActivityRateParams,
    LevyParams,
    TimeChangedLevyModel,
)
from pricebook.products.discrete_variance_swap import DiscreteVarianceSwap
from pricebook.products.variance_swap import VarianceSwap


def levy_cumulant(params: LevyParams, u: float) -> float:
    """`κ_L(u) = ln E[e^{u·L₁}]` — the cumulant exponent, elementary for this process.

    `σ²u²/2` from the diffusion part and `C[ln(G/(G+u)) + ln(M/(M−u))]` from the Variance-Gamma
    jumps. Everything the engine needs is a derivative of this at 0 or its value at 1, which is
    why no characteristic function or Fourier inversion appears anywhere in this module.
    """
    if u >= params.m:
        raise ValueError(f"the cumulant exponent needs u < M = {params.m}, got {u}")
    if u <= -params.g:
        raise ValueError(f"the cumulant exponent needs u > -G = {-params.g}, got {u}")
    jumps = 0.0
    if params.c > 0.0:
        jumps = params.c * (
            math.log(params.g / (params.g + u)) + math.log(params.m / (params.m - u))
        )
    return params.sigma * params.sigma * u * u / 2.0 + jumps


def levy_variance(params: LevyParams) -> float:
    """`var(L₁) = κ_L″(0) = σ² + C(G⁻² + M⁻²)`.

    **This is the continuous fair strike per unit of clock time** (Prop 4.2), and the quantity
    Prop 4.8's printed form gets wrong by `(E[L₁])²` — see the module docstring.
    """
    return params.sigma * params.sigma + params.c * (params.g**-2 + params.m**-2)


def levy_mean(params: LevyParams) -> float:
    """`E[L₁] = κ_L′(0) = C(1/M − 1/G)`. The diffusion part is centred and contributes nothing."""
    return params.c * (1.0 / params.m - 1.0 / params.g)


def levy_multiplier(params: LevyParams) -> float:
    """`Q_L = var(L₁) / (ln E[e^{L₁}] − E[L₁]) = κ″(0)/(κ(1) − κ′(0))` — eq (4.79).

    Table 4.3's tabulated quantity, and **2 exactly** with no jumps. It depends only on the Lévy
    process, never on the clock, which is Prop 4.2's headline: the quadratic variation is
    replicable by a log contract whatever the time change does.
    """
    gap = levy_cumulant(params, 1.0) - levy_mean(params)
    if not gap > 0.0:
        # Convexity of the exponential makes this positive for any genuine Levy process; a
        # non-positive gap means the parameters are not one, and a silent division would hand
        # back a negative "variance".
        raise ValueError(f"the multiplier needs a positive log-moment gap, got {gap}")
    return levy_variance(params) / gap


def clock_increment_variance(clock: ActivityRateParams, step: float) -> float:
    """`var(ΔZ) = 2s²(κΔ − 1 + e^{−κΔ})/κ²` for the stationary CIR activity rate, `s² = σ_v²/(2κ)`.

    From `cov(v_u, v_w) = s²e^{−κ|u−w|}` integrated twice over the interval. Two things it must get
    right, both oracled:

    - it is `O(Δ²)`, which is exactly §4.4.4's `E[(ΔZ)²] ~ O(N⁻²)` convergence condition;
    - the bracket `κΔ − 1 + e^{−κΔ}` is `O((κΔ)²/2)` and **cancels catastrophically** — written
      naively it is 29% wrong at `κ = 1e-7` and returns **exactly 0.0** by `κ = 1e-9`. `expm1`
      computes it to full relative precision, so no small-`κ` branch is needed. This is a
      production path, not a reference: the regime was measured for this model rather than
      inherited from a prior tranche's boundary (T67).
    """
    if clock.vol_of_vol == 0.0:
        return 0.0
    stationary = clock.vol_of_vol * clock.vol_of_vol / (2.0 * clock.kappa)
    return 2.0 * stationary * _decay_bracket(clock.kappa * step) / (clock.kappa * clock.kappa)


def _decay_bracket(x: float) -> float:
    """`x − 1 + e^{−x}`, which is `x²/2 + O(x³)` and cancels to nothing if written that way.

    Two regimes, with the crossover **measured** rather than chosen by eye (T67 — a prior
    tranche's boundary is not this one's):

    - `x ≥ 1e-3`: `x + expm1(−x)`. `expm1` is relative-accurate in its own result, but the sum
      still cancels, losing about `2ε/x` — 4.4e-13 at the crossover and worse below it, which is
      why there is a crossover at all rather than `expm1` everywhere.
    - `x < 1e-3`: the series `x²/2 − x³/6 + x⁴/24 − x⁵/120`, whose truncation is `x⁴/360`
      relative — 2.8e-15 at the crossover. The two errors meet there, which is what picks it.

    Written naively (`x - 1.0 + exp(-x)`) this is 29% wrong at `x = 2.5e-8` and returns **exactly
    0.0** by `x = 2.5e-10`; both forms here stay accurate to 1e-12 relative throughout.
    """
    if x >= 1e-3:
        return x + math.expm1(-x)
    return x * x * (1.0 / 2.0 + x * (-1.0 / 6.0 + x * (1.0 / 24.0 - x / 120.0)))


def levy_fair_variance(params: LevyParams, clock: ActivityRateParams, years: float) -> float:
    """`Kv(0,T;∞) = var(L₁)·E[Z_T]/T = var(L₁)` — Prop 4.2 under the `E[Z_T] = T` normalisation.

    The clock is an argument though it does not appear in the result, because that **is** the
    result: Prop 4.2 says the continuous strike is independent of the time change, and a signature
    that could not express the clock could not express the claim either.
    """
    if years <= 0.0:
        raise ValueError(f"the fair variance needs a positive horizon, got {years}")
    del clock  # Prop 4.2: the continuous strike does not depend on the time change.
    return levy_variance(params)


def discrete_monitoring_adjustment(
    params: LevyParams,
    clock: ActivityRateParams,
    years: float,
    monitoring_count: int,
    carry: float,
) -> float:
    """`Kv(0,T;N) − Kv(0,T;∞)` — Prop 4.3 eqs (4.96)/(4.97), always positive, `O(1/N)`.

    Two terms over equally-spaced monitorings, both manifestly non-negative:

    - `(1/T)Σ(E[Δln S])²`, the squared mean log-return. **This term survives the vanishing of the
      jumps**, which is why the adjustment does not disappear in the pure-diffusion limit: there it
      is `(r − q − σ²/2)²·T/N` exactly.
    - `(c²/T)Σvar(ΔZ_k)` with `c = E[L₁] − κ_L(1)`, the time change's own contribution, which is
      `O(Δ²)` per interval and so `O(1/N)` in total — the same order as the first.
    """
    if years <= 0.0:
        raise ValueError(f"the adjustment needs a positive horizon, got {years}")
    if monitoring_count <= 0:
        raise ValueError(f"the adjustment needs a positive monitoring count, got {monitoring_count}")
    step = years / monitoring_count
    gap = levy_mean(params) - levy_cumulant(params, 1.0)
    mean_return = (carry + gap) * step
    squared_means = monitoring_count * mean_return * mean_return
    clock_term = gap * gap * monitoring_count * clock_increment_variance(clock, step)
    return (squared_means + clock_term) / years


def _price_variance(
    model: TimeChangedLevyModel,
    product: VarianceSwap | DiscreteVarianceSwap,
    monitoring_count: int | None,
) -> PricingResult | PricingFailure:
    """The shared body: both swaps are a discounted variance difference, one with an adjustment.

    The **continuous** arm needs no equity data at all — Prop 4.2 makes its strike a property of
    the Lévy process alone — so the underlying is looked up only when an adjustment is wanted, and
    a snapshot carrying no equity still prices the continuous swap.
    """
    market = model.market
    expiry = product.expiry
    notional = product.variance_notional
    currency = notional.currency

    # Invariant 6: a matured swap is the shell's business, PV 0 before any market lookup.
    if is_historical(expiry, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))

    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the time-changed Levy variance swap: {exc}")

    years = discount.time_measure.year_fraction(expiry)
    fair = levy_fair_variance(model.levy, model.clock, years)
    if monitoring_count is not None:
        underlying = market.equities.get(product.asset)
        if underlying is None:
            return PricingFailure(f"no equity data for {product.asset.ticker} in the snapshot")
        # r - q, from the dividend forward -- the same carry every equity engine in the tree uses,
        # rather than a second definition of the same quantity (§6).
        carry = math.log(dividend_forward(underlying, expiry, discount) / underlying.spot) / years
        fair += discrete_monitoring_adjustment(
            model.levy, model.clock, years, monitoring_count, carry
        )
    payoff = notional.amount * (fair - product.variance_strike)
    return PricingResult(pv=Money(discount.df(expiry) * payoff, currency))


@register(VarianceSwap, TimeChangedLevyModel)
def price_levy_variance_swap(
    product: VarianceSwap, model: TimeChangedLevyModel
) -> PricingResult | PricingFailure:
    """The continuous-realized-variance swap, Prop 4.2 — T31's product against a fourth model."""
    return _price_variance(model, product, None)


@register(DiscreteVarianceSwap, TimeChangedLevyModel)
def price_discrete_levy_variance_swap(
    product: DiscreteVarianceSwap, model: TimeChangedLevyModel
) -> PricingResult | PricingFailure:
    """The discretely-monitored swap, Prop 4.3 — the continuous strike **plus** the adjustment."""
    return _price_variance(model, product, product.monitoring_count)
