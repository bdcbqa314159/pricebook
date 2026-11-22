"""The Normal/Bachelier and displaced-diffusion caplet pricers (L4). MCINERNEY-ZASTAWNIAK §8.2, §8.4.

Both price the same `NominalCap` strip the Hull-White engine does — the product is unchanged, and
that is the point of a smile model: a different *measure* on the same contract. So nothing new lands
at L2, and the two pricers differ from `engine/nominal_cap.py` only in the one line that values a
period.

**Where the forward comes from, and why it is formed here at all.** Eq 3.21 never writes `F` down —
Hull-White's caplet is expressed in discount factors throughout. These two formulas are stated in
terms of the forward rate, so the period's simple forward is read off the curve explicitly
(`CapletPeriod.forward_rate()`).

**The variance horizon is the FIXING date, not the payment date**, and the source is explicit:
`s_i^2 = int_t^{T_{i-1}} sigma_i^2(s) ds`. The forward stops diffusing when it fixes at `T_{i-1}`;
the payment at `T_i` only discounts. Passing the payment horizon would over-state every caplet, and
with a flat vol it would do so by a factor of `sqrt(T_i/T_{i-1})` with every requirement oracle
still green — parity, monotonicity and the intrinsic limit are all blind to the horizon.

Provenance:
  stock:  none — the migration tree carried neither pricer
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §8.2 (the Bachelier caplet, p.147), §8.4 (the
          displaced-diffusion caplet and its `d+/d-`, pp.151-152), Example 8.1's printed pair
  oracle: tests/L4/test_smile_caplet.py — both formulas against independent pointwise transcriptions
          read in `-raw` and `-layout`; the chapter's own printed 0.003789 and 0.003793; put-call
          parity, the vanishing-variance intrinsic limit and monotonicity in vol as requirements;
          and the implied-Black-vol smile MEASURED rather than asserted
  crosscheck: quantlib BachelierCapFloorEngine / shifted BlackCapFloorEngine — deferred to the
          optional QuantLib tier
  slice:  tranche 71 — the smile models
"""

from __future__ import annotations

import math

from pricebook.engine.cap_strip import price_cap_strip
from pricebook.engine.registry import register
from pricebook.foundation.distributions import normal_positive_part
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.models.black import black_price
from pricebook.models.hull_white import CapletPeriod
from pricebook.models.smile import BachelierModel, DisplacedDiffusionModel
from pricebook.products.nominal_cap import NominalCap


def bachelier_caplet(
    unit: CapletPeriod, strike: float, variance: float, kind: CapFloorKind
) -> float:
    """The §8.2 caplet, per unit notional: `tau B(t,T_i) [ (F-K) N(d1) + s phi(d1) ]`, `d1 = (F-K)/s`.

    **The bracket is `E[(omega(F-K) + s Z)+]` and that is already in the tree**, at L0 as
    `normal_positive_part` — the body tranche 25 wrote for a CVA's expected positive exposure. A
    Bachelier call and an expected positive exposure are the same closed form, so this is §6's rule
    of two rather than a coincidence of shape. The source prints the second term as
    `(s/sqrt(2 pi)) exp(-d1^2/2)`, which is `s phi(d1)`.

    `omega` turns the cap into the floor: a normal variable's positive part is symmetric about its
    mean, so `omega(F-K)` is the whole of the difference between the two — no second formula.

    **`math.sqrt` and not `variance**0.5`, which is a correctness choice rather than a style one.**
    A negative total variance means the horizon ran backwards — the caller passed a period that has
    already fixed — and `(-x)**0.5` returns a *complex number* in Python, silently, which then
    propagates into arithmetic that will not notice. `math.sqrt` refuses it. That makes the
    invariant-6 filter in `_price_strip` self-enforcing: drop it and this raises rather than
    returning a plausible wrong number, which is §4's "fail fast and loud at a trust boundary"
    buying a guard against the degeneracy the charter's tenth shape describes.
    """
    return unit.accrual * unit.df_pay * normal_positive_part(
        kind.omega * (unit.forward_rate() - strike), math.sqrt(variance)
    )


def displaced_caplet(
    unit: CapletPeriod,
    strike: float,
    variance: float,
    displacement: float,
    kind: CapFloorKind,
) -> float:
    """The §8.4 caplet: Black on the SHIFTED forward and strike — `(F+a) N(d+) - (K+a) N(d-)`.

    Shifting both sides is the whole model, and shifting only the forward is the classic bug — it
    breaks put-call parity by exactly the displacement, which is what the parity oracle is for.
    `F + a` is lognormal by eq (8.7), so `black_price` applies unchanged and nothing re-derives
    `d+/d-` here.
    """
    return unit.accrual * unit.df_pay * black_price(
        unit.forward_rate() + displacement, strike + displacement, variance, kind
    )


def _price_strip(
    cap: NominalCap, model: BachelierModel | DisplacedDiffusionModel
) -> PricingResult | PricingFailure:
    """Value each live period under whichever smile model this is, over the shared strip walk.

    One walk for both models, and from tranche 73 the **same** walk the CEV engine uses
    (`engine/cap_strip.py`) — the duplication re-audit #8's pass 2 measured, and the shared
    fixing-date horizon that makes convention C16 guarded rather than per-module. What is local here
    is only the per-period valuation, which is the one thing that genuinely differs."""
    if isinstance(model, BachelierModel):
        variance_rate = model.normal_vol * model.normal_vol

        def value(unit: CapletPeriod, years: float) -> float:
            return bachelier_caplet(unit, cap.strike, variance_rate * years, cap.kind)

    else:
        variance_rate = model.vol * model.vol
        displacement = model.displacement

        def value(unit: CapletPeriod, years: float) -> float:
            return displaced_caplet(
                unit, cap.strike, variance_rate * years, displacement, cap.kind
            )

    # Only `KeyError` — a missing curve. Neither closed form raises anything else it expects.
    return price_cap_strip(cap, model.market, value, (KeyError,))


@register(NominalCap, BachelierModel)
def price_bachelier_cap(cap: NominalCap, model: BachelierModel) -> PricingResult | PricingFailure:
    """The strip under a normal forward (§8.2)."""
    return _price_strip(cap, model)


@register(NominalCap, DisplacedDiffusionModel)
def price_displaced_cap(
    cap: NominalCap, model: DisplacedDiffusionModel
) -> PricingResult | PricingFailure:
    """The strip under a shifted-lognormal forward (§8.4)."""
    return _price_strip(cap, model)
