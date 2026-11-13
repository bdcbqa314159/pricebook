"""CVA of an equity option under independent default (L5) - Brockhaus 14.2.1.

    CVA = (1 - R) * (1 - p0(T)) * V0

**Where the integral went.** 14.2.1 states the general form
`CVA = (1-R) N0 E[(1/N_tau) 1{0<tau<=T} V_tau]`, with `V_tau` the *remaining* option value at the
default time - a weighted sum over default dates, not a single number. It collapses because, under
independence of `tau` and `(N,S)`, the **discounted option value is a martingale**: `E[V_t/N_t] =
V0/N0` for every fixed `t`, so the expectation does not depend on *when* default happens.
Integrating against the default density leaves `(V0/N0) P[0<tau<=T] = (V0/N0)(1 - p0(T))`.

**The book's forward contract does NOT collapse, and the contrast is the point.** Its exposure is
`(F_t - K)^+`, which is *not* the contract's value - a forward can be worth less than nothing - so
the martingale argument does not reach it, and the maturity-`t` option inside its integral is a
genuinely different object at each `t`. An option's exposure *is* its value, and that is the whole
reason this one is a closed form.

**This is the arc capstone's first seam: it is a consumer of two arcs and new arithmetic in neither.**
The option value comes from the equity arc's registry (so it works for *any* equity model registered
there - Black, Heston, the Hull-White hybrid, defaultable), and the survival probability from the XVA
arc's `SurvivalCurve`. The oracle reconciles it against tranche 25's `unilateral_cva`, which reaches
the same number by integrating `DF(t) EPE(t) dPD(t)` over a grid - different code, different source,
different arc.

Provenance:
  stock:  none - the archived predecessor's `risk/xva.py` was a phantom (tranche 24 pruned its rows)
  source: BROCKHAUS-2016 (docs/references.md) ch 14.2.1 - the CVA of standard products under
          independence, read in `-raw` and `-layout` and reconciled; the martingale collapse is
          DERIVED here rather than transcribed, which §4 rates higher
  oracle: tests/L5/test_equity_cva.py (PRE-REGISTERED) - the **cross-arc reduction** to tranche 25's
          `unilateral_cva` on the equity exposure, with its convergence rate under grid refinement
          (so the two are the same quantity, not two close numbers); the two zeros (`h -> 0`,
          `R -> 1`); the full-loss bound at `p -> 0`; monotonicity in the hazard and the recovery;
          and a NON-FLAT survival curve read beyond its last pillar
  crosscheck: quantlib (CVA on an equity option) - deferred to the optional QuantLib tier
  slice:  tranche 36 - equity CVA, the arc capstone's independent-default seam
"""

from __future__ import annotations

from pricebook.engine.registry import dispatch
from pricebook.foundation.results import PricingResult
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.products.equity_option import EuropeanEquityOption


def equity_option_cva(
    option: EuropeanEquityOption,
    model: object,
    survival: SurvivalCurve,
    recovery: float,
) -> float:
    """`(1-R)(1-p0(T)) V0` - the CVA of a long equity-option position.

    `model` is any model the equity option is registered against, so this works for Black, Heston,
    the Hull-White hybrid or the defaultable-equity model without knowing which: the collapse above
    needs only that `V_t/N_t` is a martingale, which is a property of the *contract*, not of the
    volatility model pricing it. Taking the value through `dispatch` rather than naming a pricer is
    what makes that reuse real rather than claimed.
    """
    if not 0.0 <= recovery <= 1.0:
        raise ValueError(f"the recovery rate must lie in [0, 1], got {recovery}")
    market = getattr(model, "market", None)
    if market is None:
        raise ValueError("the model must carry the market snapshot it was built against")
    if survival.time_measure.anchor != market.valuation_date:
        raise ValueError(
            "the survival curve and the snapshot must share one time measure anchor: "
            f"{survival.time_measure.anchor} vs {market.valuation_date}"
        )

    priced = dispatch(option, model)
    if not isinstance(priced, PricingResult):
        raise ValueError(f"cannot price the option the CVA is for: {priced.reason}")

    discount = market.curves.discount(option.notional.currency)
    years = discount.time_measure.year_fraction(option.expiry)
    default_probability = 1.0 - survival.survival(years)
    return (1.0 - recovery) * default_probability * priced.pv.amount
