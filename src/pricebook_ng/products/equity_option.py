"""European equity option as pure data (L2).

An `EquityOption` is the right to buy (call) or sell (put) `quantity` shares of
`ticker` at `strike` per share (in `currency`) at `maturity`. Pure data — no `pv`
method (CLAUDE.md 2); pricing is L4 (Black-Scholes).

Provenance:
  quarry: python/pricebook/equity/ (equity option)
  source: Black-Scholes-Merton (dividends)
  oracle: put-call parity ties to the equity forward (equity-option slice)
  slice:  equity-option
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.money import Currency


@dataclass(frozen=True)
class EquityOption:
    """A European option on `quantity` shares of `ticker`, struck at `strike` per
    share in `currency`."""

    ticker: str
    quantity: float
    strike: float
    maturity: date
    currency: Currency
    is_call: bool
