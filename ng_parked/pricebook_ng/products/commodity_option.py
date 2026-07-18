"""European commodity option as pure data (L2).

The right to buy (call) or sell (put) `quantity` units of `ticker` at `strike` per
unit (in `currency`) at `maturity`. Same shape as an equity option; the carry
curve (convenience yield net of storage) plays the dividend curve's role. Pure
data — no `pv` method (CLAUDE.md 2); pricing is L4 (shared spot-option engine).

Provenance:
  quarry: python/pricebook/commodity/ (commodity option)
  source: Black-Scholes on the commodity forward (carry)
  oracle: put-call parity ties to the commodity forward (commodity-option slice)
  slice:  commodity-option
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.money import Money


@dataclass(frozen=True)
class CommodityOption:
    """A European option on `quantity` units of `ticker`, struck at `strike` — the
    strike price per unit as `Money` (amount + currency)."""

    ticker: str
    quantity: float
    strike: Money
    maturity: date
    is_call: bool
