"""The perpetual timer option under Heston, by first passage on accumulated variance (L4).

`KWOK-ZHENG` §6.2. A timer exercises when `∫₀^τ V_s ds` first reaches the budget `B`, so pricing is
a **first-passage** problem: simulate the Heston pair, accumulate the variance clock, and stop each
path when its clock reaches `B`.

**Why the Monte Carlo is the production pricer, and why that is still in-method.** §6.2.2 carries a
closed-form integral price, and it is *not* built here: it needs a **complex-argument Bessel `I_ν`**
(the Bromwich contour in eq (6.25) makes `λ` complex, so the real-argument `noncentral_chisquare`
ladder cannot supply it), a **Laplace inversion to a density** (distinct from the Gil-Pelaez CDF
inversion the tree has), and a **2-D `(x,t)` quadrature`. The book itself opens §6.2.3 by calling
that route *"quite daunting"*. Those three primitives belong to the numerical track, which §6.2.2
becomes the first customer for (`to_go_deeper`, ruled 0424); when they land they retro-anchor this
MC exactly. Meanwhile the MC is **not the sole oracle** — it is pinned by two independent non-MC
results, which is what keeps the tranche inside §1's oracle gate rather than over the
out-of-method line:

1. the **exact reduction**: at `ρ = 0` *and* `r = q = 0` the drift `µ_τB = (r−q)τ_B − B/2` collapses
   to the deterministic `−B/2`, so `ln(S_τB/S₀) ~ N(−B/2, B)` with no model dependence left and eq
   (6.10) is exactly Black–Scholes at total variance `B`. Both conditions bind: `(r−q)τ_B` is random
   because `τ_B` is;
2. the **§6.2.3 perturbation** at small vol-of-vol — which, unlike the reduction, is
   non-degenerate in `ρ`, so it covers exactly what the reduction is blind to.

Provenance:
  stock:  none — the migration tree carried no timer option
  source: KWOK-ZHENG (docs/references.md) §1.3.3 — the payoff; §6.2.1 eq (6.13)/(6.14) — the
          lognormal law of `S_τB` given the variance path, and eq (6.10) the `ρ=0`/zero-carry
          closed form; §6.2.3 eqs (6.27)/(6.47)/(6.54)/(6.56)/(6.57) — the perturbation reference,
          transcribed in the ORACLE only (it needs Lambert W; production needs no special
          function). All read in `-raw` and `-layout` and reconciled
  oracle: tests/L4/test_timer_option.py — the exact `ρ=0`/zero-carry Black–Scholes reduction, the
          §6.2.3 perturbation with its own error measured and its regime stated, the source's two
          printed forms of `T₀` agreeing, pathwise put-call parity, budget monotonicity, and the
          first-passage clock pinned against `T₀` before any payoff
  crosscheck: none — QuantLib has no timer-option engine
  slice:  tranche 79 — perpetual timer options
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np

from pricebook.engine.registry import register
from pricebook.foundation.mc_estimator import MonteCarloEstimate, estimate
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.market.equity import EquityUnderlying, dividend_forward, implied_carry
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.heston import HestonModel, HestonParams
from pricebook.products.equity_option import EquityOptionKind
from pricebook.products.timer_option import TimerOption
from pricebook.foundation.rng import UniformSource
from pricebook.foundation.variates import standard_normals

#: How many expected hitting times to allow before refusing. `E[τ_B] ≈ B/θ`, and the hitting-time
#: tail is thin under mean reversion, so 64× is ample — but it is a REFUSAL rather than a
#: truncation, because a prototype that truncated at `3·B/θ` scored 5.5% of paths at zero and that
#: reads as a price rather than as a failure.
_MAX_HORIZONS = 64


def simulate_timer_first_passage(
    params: HestonParams,
    underlying: EquityUnderlying,
    budget: float,
    sampling: tuple[MonteCarloConfig, int],
) -> tuple[np.ndarray, np.ndarray]:
    """First-passage times of the variance clock, and the spot at each.

    `sampling` is `(config, steps)` where `steps` is the number of steps per **expected** hitting
    time `B/θ` — the natural clock here, since a perpetual option has no horizon to divide. The two
    are one decision about cost against discretisation bias, bundled for the same reason
    `simulate_heston_spots` bundles them.

    The hitting time is **interpolated within the step** it occurs in rather than rounded up to the
    step boundary: the clock is accumulated as `Σ V·dt`, so the fraction of the final step needed to
    reach the budget is known exactly, and rounding it away would be an `O(dt)` upward bias in `τ`
    on every path — a bias in the same direction for all of them, which is the kind that does not
    average out.
    """
    config, steps = sampling
    if steps < 1:
        raise ValueError(f"the simulation needs at least one step, got {steps}")
    if not budget > 0.0:
        raise ValueError(f"the simulation needs a positive variance budget, got {budget}")

    # The clock's natural scale: the variance averages `theta` in the long run, so the budget takes
    # about `B/theta` to accumulate. `v0` guards the case where the process starts far above it.
    scale = budget / max(params.theta, params.v0)
    if not scale > 0.0 or not math.isfinite(scale):
        raise ValueError(
            f"the variance budget {budget} is unreachable for this variance process "
            f"(theta={params.theta}, v0={params.v0})"
        )
    step = scale / steps
    root_step = math.sqrt(step)
    orthogonal = math.sqrt(1.0 - params.rho * params.rho)

    source = UniformSource(config.seed)
    log_spots = np.full(config.paths, math.log(underlying.spot), dtype=float)
    variances = np.full(config.paths, params.v0, dtype=float)
    clocks = np.zeros(config.paths, dtype=float)
    hitting = np.full(config.paths, np.nan, dtype=float)

    # The carry is zero in the payoff's own measure: the discounting is applied at `tau` by the
    # caller, and the drift the lognormal law (eq 6.13) carries is `(r-q)t - (1/2)int V`. The
    # `(r-q)` part is restored by the caller through `dividend_forward`, which is where the tree
    # defines that quantity (C17) -- this function simulates the driftless martingale part.
    for index in range(_MAX_HORIZONS * steps):
        live = np.isnan(hitting)
        if not live.any():
            break
        first = np.array(standard_normals(source.uniforms(config.paths)))
        second = np.array(standard_normals(source.uniforms(config.paths)))
        correlated = params.rho * first + orthogonal * second

        # Full truncation, as `simulate_heston_spots` does: `v` is carried SIGNED and every read is
        # floored, so the drift still pulls a negative state back toward `theta`.
        used = np.maximum(variances, 0.0)
        root_v = np.sqrt(used)
        advanced = clocks + used * step

        # The budget is reached inside this step wherever `advanced` crosses it. The crossing
        # fraction is exact because the clock is piecewise linear in the step.
        crossing = live & (advanced >= budget)
        fraction = np.where(
            crossing, (budget - clocks) / np.where(advanced > clocks, advanced - clocks, 1.0), 0.0
        )
        hitting = np.where(crossing, (index + fraction) * step, hitting)

        # The spot advances by the FRACTION of the step on a crossing path and the whole step
        # otherwise; a crossing path then stops moving, since `live` excludes it next round.
        taken = np.where(crossing, fraction, 1.0) * np.where(live, 1.0, 0.0)
        log_spots = log_spots + taken * (-0.5 * used * step) + np.sqrt(taken) * root_v * root_step * first
        clocks = np.where(live, advanced, clocks)
        variances = np.where(
            live,
            variances
            + params.kappa * (params.theta - used) * step
            + params.sigma_v * root_v * root_step * correlated,
            variances,
        )

    unreached = int(np.count_nonzero(np.isnan(hitting)))
    if unreached:
        raise ValueError(
            f"{unreached} of {config.paths} paths did not reach the variance budget {budget} "
            f"within {_MAX_HORIZONS} expected hitting times — refusing rather than scoring them "
            f"at zero, which would read as a price"
        )
    return hitting, np.exp(log_spots)


def timer_option_estimate(
    product: TimerOption,
    model: HestonModel,
    sampling: tuple[MonteCarloConfig, int],
) -> MonteCarloEstimate:
    """`E[e^{−rτ}(S_τ − K)⁺]` per unit notional, with its standard error.

    The L0 statistical primitive (§3.7's two layers): this returns the estimate, and the registered
    pricer below projects it into a `PricingResult`.
    """
    market = model.market
    underlying = market.equities[product.asset]
    discount = market.curves.discount(product.notional.currency)

    times, martingale_spots = simulate_timer_first_passage(
        model.params, underlying, product.variance_budget, sampling
    )

    # Restore the `(r-q)` drift at each path's own exercise time. The carry comes from the
    # DIVIDEND FORWARD -- the tree's one definition of `r-q` (C17) -- read over a one-year
    # reference horizon, because a perpetual option has no expiry to read it at. That is exact for
    # a continuous yield, which is what eq (6.13)'s constant `(r-q)` assumes; a DISCRETE dividend
    # schedule is outside the source's assumption and would make this a one-year average, so it is
    # refused rather than silently averaged.
    if underlying.dividends.cash or underlying.dividends.proportional:
        raise ValueError(
            "the perpetual timer prices under eq (6.13)'s constant continuous carry, so a discrete "
            "or proportional dividend schedule is outside its assumption; use a repo/borrow yield"
        )
    reference = market.valuation_date + timedelta(days=365)
    years = discount.time_measure.year_fraction(reference)
    carry = implied_carry(
        dividend_forward(underlying, reference, discount), underlying.spot, years
    )
    spots = martingale_spots * np.exp(carry * times)

    intrinsic = (
        np.maximum(spots - product.strike, 0.0)
        if product.kind is EquityOptionKind.CALL
        else np.maximum(product.strike - spots, 0.0)
    )
    factors = np.array([discount.df_at(float(t)) for t in times])
    return estimate((factors * intrinsic).tolist())


@register(TimerOption, HestonModel)
def price_timer_option(
    product: TimerOption, model: HestonModel
) -> PricingResult | PricingFailure:
    """The perpetual timer option under Heston.

    **No invariant-6 filter, and the absence is deliberate rather than an omission:** invariant 6
    excludes flows on or before the valuation date, and a perpetual timer has no flow date at all —
    its exercise is a future random time. There is nothing to filter, so a filter here would be
    unreachable code (§4's third survivor kind).
    """
    currency = product.notional.currency
    market = model.market
    if product.asset not in market.equities:
        return PricingFailure(f"no equity data for {product.asset.ticker} in the snapshot")
    try:
        market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the timer option: {exc}")

    try:
        unit = timer_option_estimate(product, model, (MonteCarloConfig(paths=20_000, seed=7), 200))
    except ValueError as exc:
        return PricingFailure(f"cannot price the timer option: {exc}")

    scale = product.notional.amount
    return PricingResult(
        pv=Money(scale * unit.mean, currency),
        standard_error=scale * unit.standard_error,
    )
