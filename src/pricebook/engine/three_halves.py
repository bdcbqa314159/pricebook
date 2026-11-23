"""The 3/2-model variance swap (L4). KWOK-ZHENG §4.2.2 eq (4.35); ZHAO-CUI-FLORESCU-2018 eq (19).

**The fair variance is priced WITHOUT a confluent hypergeometric, and that is the tranche's point.**
The obvious route is the Laplace transform of `∫V ds` — eq (19) of the held paper, the canonical
Carr–Sun form — differentiated at zero. That transform *is* hypergeometric (`M(α, γ, ·)`), which
would put a new special function on the **production** path and scipy inside `src/`, against the
tranche-20a arrangement. It is not needed:

**the reciprocal is CIR, and the inverse moment is a ladder of elementary terms.** §4.2.2's eq (4.35)
gives `Y = 1/V` as a CIR process, `dY = [κ + ε² − κθ·Y]dt − ε√Y dW`, so `Y_t` is a scaled non-central
chi-squared and

    E[V_t] = E[1/Y_t] = (1/scale) · sum_j Poisson(j; λ_t/2) · 1 / (dof + 2j − 2)

— because `χ'²_d(λ) = χ²_{d+2j}` mixed over `j ~ Poisson(λ/2)` and `E[1/χ²_m] = 1/(m−2)`. Elementary
summands on the **same Poisson ladder** tranche 72 already walks, which is why that ladder was
extracted to L0 here (§6, second consumer). The fair variance is then `∫₀^T E[V_t]dt / T`.

**`E[1/Y]` always exists.** It needs `dof > 2`, and `dof = 4·speed·level/ε² = 4(κ + ε²)/ε² > 4` for
any positive `κ` — so the Feller-type condition §4.2.2 discusses is automatic in this
parameterisation. Asserted in the oracle rather than guarded for.

**Why the oracle is the other route, and why the book's own equation could not be it.** The book's
eq (3.30) prints the transform as `₁F₁(−2/σ²U; α, γ)` — **argument first** — and read that way it
**diverges**: `L(0)` comes out 4.32 or 4.19 depending on which rendering's grouping of `α` you take,
and the two renderings disagree about that grouping. A Laplace transform of a non-negative quantity
must satisfy `L(0) = 1`, so the printed form is not transcribable at all (tranche 77 step-1, thread
0407). Eq (3.30) attributes itself to Zhao et al. (2018), which was then acquired; that paper's
**Proposition 3 / eq (19)** writes `M(α, γ, −2/(σ²y_T))` — **argument last** — and `L(0) = 1`
*exactly*, since α's grouping makes `α(0) = 0` and then `M(0,γ,z) = 1`, `(·)^0 = 1`,
`Γ(γ)/Γ(γ) = 1`. So the requirement oracle does not merely check the transcription, it **pins the
grouping the renderings could not settle**. Measured, the two routes agree to **8.1e-10**.

Provenance:
  stock:  none — the migration tree carried no 3/2 model
  source: KWOK-ZHENG (docs/references.md) §4.2.2 eq (4.35) — the reciprocal-CIR map this prices on,
          transcribed in `-raw` and `-layout` and reconciled; ZHAO-CUI-FLORESCU-2018 Proposition 3
          eq (19) — the Carr–Sun Laplace transform, transcribed in the ORACLE only (it is
          hypergeometric; production stays elementary). The book's own eq (3.30) is a mis-rendering
          of eq (19) and is NOT the source for anything here — see `references.md`'s defect log
  oracle: tests/L4/test_three_halves.py — the ladder against eq (19)'s transform differentiated at
          zero (scipy's `hyp1f1`, test-only — two routes, one number, 8.1e-10); `L(0) = 1` exactly
          as the requirement oracle that pins α; the `ε → 0` **logistic** reduction with its
          convergence rate and a material-`ε` companion (T39); monotonicity in `v0`, the long-horizon
          limit to `θ` and the vanishing-horizon limit to `v0` as requirements
  crosscheck: none — QuantLib implements no non-affine 3/2 model
  slice:  tranche 77 — the 3/2-model variance swap
"""

from __future__ import annotations

import math

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.noncentral_chisquare import poisson_weighted_sum
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.models.three_halves import ThreeHalvesModel, ThreeHalvesParams
from pricebook.products.variance_swap import VarianceSwap

#: Noncentrality past which `E[1/Y]` is taken as `1/E[Y]` — the **concentrated** limit.
#:
#: As the noncentrality grows `Y_t`'s relative dispersion vanishes, so the inverse moment tends to
#: the inverse of the mean. **Measured, and the premise checked in both directions it can be reached**
#: (tranche 72's lesson, where "a large parameter means a negligible dispersion" was false): the
#: noncentrality here is `4·speed·e^{−speed·t} / (ε²·spread·v0)`, which diverges only as `t → 0` or
#: `ε → 0`, and the variance is near-deterministic in both — at `t → 0` because `V_t → v0`, and at
#: `ε → 0` because the whole process collapses to the logistic ODE. The limit converges as `1/nc`:
#: 2.5e-02 at nc=77, 1.4e-03 at 2.8e3, 1.4e-04 at 2.8e4, 4.3e-05 at 9.3e4 — so 1e8 puts it near
#: 4e-8, far inside every band here, while the ladder it replaces would cost O(sqrt(nc)) terms.
_MAX_NONCENTRALITY = 1.0e8

#: Midpoint nodes for `∫₀^T E[V_t] dt`. `E[V_t]` is smooth and monotone in `t`, so the midpoint rule
#: converges at second order; 400 holds the two routes' agreement at ~8e-10, two orders tighter than
#: the one-sided finite difference the oracle compares against can itself resolve.
_NODES = 400


def reciprocal_cir_law(params: ThreeHalvesParams) -> tuple[float, float, float]:
    """`(speed, level, dof)` of the CIR process `Y = 1/v` — §4.2.2 eq (4.35).

    `dY = [κ + ε² − κθ·Y]dt − ε√Y dW`, i.e. a CIR reverting at **`κθ`** to `(κ + ε²)/(κθ)` with
    vol-of-vol `ε`. (The sign on the diffusion is immaterial to the law.) `dof = 4·speed·level/ε²`
    is returned with them because every caller needs all three and computing them apart invites the
    `κθ` to be formed twice.
    """
    speed = params.kappa * params.theta
    level = (params.kappa + params.epsilon * params.epsilon) / speed
    dof = 4.0 * speed * level / (params.epsilon * params.epsilon)
    return speed, level, dof


def expected_variance(params: ThreeHalvesParams, years: float) -> float:
    """`E[V_t]` — the inverse first moment of the reciprocal's non-central chi-squared law.

    At `t = 0` the variance is known, so this returns `v0` exactly rather than approaching it: the
    chi-squared law degenerates there (its scale vanishes and the noncentrality diverges).
    """
    if years < 0.0:
        raise ValueError(f"the expected variance needs a non-negative horizon, got {years}")
    if years == 0.0:
        return params.v0
    speed, level, dof = reciprocal_cir_law(params)
    sigma_squared = params.epsilon * params.epsilon
    spread = 1.0 - math.exp(-speed * years)
    scale = sigma_squared * spread / (4.0 * speed)
    noncentrality = (
        4.0 * speed * math.exp(-speed * years) / (sigma_squared * spread * params.v0)
    )
    if noncentrality > _MAX_NONCENTRALITY:
        # `Y_t` has concentrated: its inverse moment is the inverse of its mean. See the constant.
        mean_reciprocal = level + (1.0 / params.v0 - level) * math.exp(-speed * years)
        return 1.0 / mean_reciprocal
    if noncentrality <= 0.0:
        # The law has become CENTRAL (`t` long enough that the initial condition has decayed out of
        # representable range), and `E[1/chi2_d] = 1/(d−2)` exactly — no ladder, no approximation.
        return 1.0 / (scale * (dof - 2.0))
    inverse = poisson_weighted_sum(
        noncentrality, lambda index: 1.0 / (dof + 2.0 * index - 2.0)
    )
    return inverse / scale


def stationary_variance(params: ThreeHalvesParams) -> float:
    """`E[V_∞] = θ / (1 + ε²/(2κ))` — the long-run mean variance, which is **not `θ`**.

    Worth its own name because it is a result, not a restatement: the vol-of-vol pulls the stationary
    mean of a 3/2 variance **below** its reversion level, by a factor `1/(1 + ε²/2κ)` — 3.8% at
    `ε = 0.4, κ = 2` and **26% at `ε = 1.2`**. It falls out of the reciprocal-CIR law: `Y`'s
    stationary law is a central chi-squared whose inverse mean is `1/(scale·(dof − 2))`, and
    substituting `dof = 4(κ+ε²)/ε²` and `scale → ε²/(4κθ)` gives exactly this — verified to machine
    precision against the ladder at four parameter sets.

    T77's spec asserted the long-horizon limit was `θ`. That is the `ε → 0` case of this.
    """
    return params.theta / (1.0 + params.epsilon * params.epsilon / (2.0 * params.kappa))


def three_halves_fair_variance(params: ThreeHalvesParams, years: float) -> float:
    """`E[(1/T)∫₀^T V_t dt]` — the fair strike of a variance swap under the 3/2 model.

    **Annualised**, so it is directly comparable to `fair_variance` (the model-free strip) and to
    `heston_fair_variance`: the same quantity under a third route, which is what lets the same
    `VarianceSwap` price against any of them.
    """
    if years <= 0.0:
        raise ValueError(f"the fair variance needs a positive horizon, got {years}")
    step = years / _NODES
    total = math.fsum(
        expected_variance(params, (index + 0.5) * step) for index in range(_NODES)
    )
    return total * step / years


@register(VarianceSwap, ThreeHalvesModel)
def price_three_halves_variance_swap(
    product: VarianceSwap, model: ThreeHalvesModel
) -> PricingResult | PricingFailure:
    """`N · df(T) · (K_var − K)` — the same discounted variance difference tranche 31 pays.

    The **third** model this one product prices against (the model-free strip, and now Heston's
    realized-variance machinery and the 3/2 model): the registry's `(product, model)` key is what
    makes that expressible without the product knowing anything about dynamics.
    """
    market = model.market
    currency = product.variance_notional.currency

    # Invariant 6: a matured swap is the shell's business, PV 0 before any market lookup.
    if is_historical(product.expiry, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))

    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the 3/2 variance swap: {exc}")

    years = discount.time_measure.year_fraction(product.expiry)
    fair = three_halves_fair_variance(model.params, years)
    payoff = product.variance_notional.amount * (fair - product.variance_strike)
    return PricingResult(pv=Money(discount.df(product.expiry) * payoff, currency))
