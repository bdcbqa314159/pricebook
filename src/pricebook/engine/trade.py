"""`trade_value` — a trade priced through the registry (L4). The structural track's first function.

`trade_value(trade, model) = direction · units · price(trade.product, model)`.

**What this tranche demonstrates, and the reason the structural track is worth a second arc type:** it
adds a capability over **46 existing engines without touching one of them**. The registry's
`(product, model)` dispatch is what makes that possible — a trade does not know what it holds, and the
engines do not know they are held.

**The oracle is the catalogue, not a source.** `trade_value` is *defined* by its reduction to `price`,
which makes the headline identity close to a tautology (§4's self-consistency-telescopes shape): a
wrapper that delegates and multiplies satisfies it by construction, and it cannot see a wrong direction
sign or a notional applied to the wrong side. The companions carry the test — see the oracle module.

Provenance:
  stock:  none — architecture-first
  source: CLAUDE.md §0/§1 — the representation contract and the purpose-defined 1.0
  oracle: tests/L4/test_trade.py — the bit-identical reduction for a long unit trade, the long/short
          asymmetry on a non-symmetric payoff, the notional's absolute level against an
          independently composed reference (T43), and the §3 failure contract riding through
  crosscheck: none — no library prices an abstract trade wrapper; the underlying engines carry theirs
  slice:  tranche 82 — R1.1, the Trade type
"""

from __future__ import annotations

from pricebook.engine.registry import dispatch
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.products.trade import Trade


def trade_value(trade: Trade, model: object) -> PricingResult | PricingFailure:
    """The trade's value, or the failure the underlying pricer returned — faithfully.

    **The `PricingFailure` rides through untouched** (§3.4). A wrapper that turned a failure into a
    raise would break the contract one layer up, and one that scored it zero would be worse: a zero is
    a *price*, and an unpriceable trade is not worth nothing.

    The standard error scales with the **units** where the underlying pricer supplies one (§3.7): for
    an estimate the band is part of the result, so a trade ten times the size has a band ten times as
    wide. The direction does **not** scale it — an error bar has no sign, hence `abs(scale)`.

    **`accrued` scales SIGNED, and that is forced rather than chosen** (T85): `clean = pv − accrued`
    must commute with scaling, so `scale·pv − scale·accrued = scale·clean` and accrued takes the same
    signed `scale` as the PV. A short owes its accrued interest, so the sign is the point — which is
    exactly why it cannot borrow the band's `abs`. `basis` (the discounting currency) is carried
    through unchanged, being a property of the pricing and not of the size of the holding.
    """
    priced = dispatch(trade.product, model)
    if isinstance(priced, PricingFailure):
        return priced

    # No currency guard, and its absence is the point: `units` is dimensionless (ruled 0446), so the
    # trade has no currency of its own to disagree with the product's. The earlier `Money` notional
    # needed such a guard precisely because the field was carrying two jobs at once.
    scale = trade.direction.sign * trade.units
    standard_error = (
        None if priced.standard_error is None else abs(scale) * priced.standard_error
    )
    accrued = (
        None
        if priced.accrued is None
        else Money(scale * priced.accrued.amount, priced.accrued.currency)
    )
    return PricingResult(
        pv=Money(scale * priced.pv.amount, priced.pv.currency),
        accrued=accrued,
        basis=priced.basis,
        standard_error=standard_error,
    )
