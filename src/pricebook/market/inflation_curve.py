"""Constructing the real curve from break-even quotes (L1).

**There is exactly ONE `Pr`, and it is a curve.** BERHAN gives two views of the real bond — used
directly in the leg value (1.2.13/1.2.14) and related to the nominal curve by the break-even
(1.2.16) — and the second is **how the curve is built**, not a second way to price. This module is
that constructor; the pricer never recomputes `Pr` from `b`.

That distinction is the `rpv01` lesson applied before it can bite: if construction and pricing each
rolled their own real discount factor and differed in *any* detail (compounding, day count), a swap
struck at the break-even would still reprice to zero — because a par reprice telescopes — while
every off-par trade priced wrong. So the guard is two oracles, not one: the par identity proves
construction and pricing share a `Pr`, and a **pointwise** check proves this constructor satisfies
1.2.16 on its own, which a par reprice cannot see.

  crosscheck: quantlib ZeroCouponInflationTermStructure / piecewise ZC inflation curve —
  deferred to the optional QuantLib tier (thread 0013/0014).

Provenance:
  stock:  _stock/quarry/pricebook/curves/inflation_curve.py (the real/nominal pairing CONCEPT;
          its `build_real_nominal_curves` is a joint bootstrap for a later calibration tranche)
  source: BERHAN-2012 (docs/references.md) §1.2.2 eq 1.2.16, `Pr(0,T) = Pn(0,T)(1+b(0,T))^T`
  oracle: pointwise 1.2.16 to 1e-12 at every quoted horizon (non-telescoping); and the par
          identity — a swap struck at `b` reprices to zero
  slice:  tranche 2 — the inflation spine
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date

from pricebook.market.curve import DiscountCurve


def real_curve_from_breakeven(
    nominal: DiscountCurve, breakevens: Mapping[date, float]
) -> DiscountCurve:
    """The real curve implied by `nominal` and break-even inflation quotes, by 1.2.16.

    `breakevens` maps a quote **date** to its break-even rate `b(0,T)`. Returns an ordinary
    `DiscountCurve` that every consumer reads through `df` — no inflation-specific type and no
    second discounting path.

    **Dates, not year fractions, and the oracle is why.** The first cut took `Mapping[float, float]`
    and the pointwise 1.2.16 check failed at the 2.5-year pillar: a caller saying "2.5 years"
    converts to 912 whole days, which is 2.4986 years on the curve's own measure, so the caller's
    `T` and the curve's `T` disagreed by a day and the identity missed by 6.6e-6. Taking dates
    removes the disagreement structurally — there is one `date → t` map, the curve's, and the
    caller cannot hold a second one. (Convention C2 in the register is this same lesson at a
    larger scale.)

    The conversion is exact rather than solved: `Pr = Pn(1+b)^T` with `Pn = e^{−rn·T}` gives
    `rr = rn − ln(1+b)`, the Fisher relation in continuous form. A root-find would be solving a
    closed form."""
    if not breakevens:
        raise ValueError("a real curve needs at least one break-even quote")
    measure = nominal.time_measure
    dates = tuple(sorted(breakevens))
    horizons = tuple(measure.year_fraction(d) for d in dates)
    if any(t <= 0.0 for t in horizons):
        raise ValueError(f"break-even quote dates must be after the curve anchor, got {dates}")

    # `rr(T) = rn(T) − ln(1 + b(T))`: the continuous Fisher relation that 1.2.16 is, rearranged.
    rates = tuple(
        nominal.zero_rate(t) - math.log1p(breakevens[d]) for d, t in zip(dates, horizons)
    )
    # t=0 is required of every curve; the shortest quote's rate extends back to the anchor, which
    # is flat extrapolation at the short end and is what a single-quote curve must do anyway.
    return DiscountCurve(measure, (0.0, *horizons), (rates[0], *rates))
