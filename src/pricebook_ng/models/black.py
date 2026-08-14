"""Black-76 closed form — the analytic block of the lognormal model (L3).

`black(forward, strike, vol, t, option_type)` is the UNDISCOUNTED Black-76 optionlet value under
the T-forward measure: `E[(F − K)+]` (CALL) / `E[(K − F)+]` (PUT), with the forward a martingale.
Discounting is kept OUT (the L4 engine multiplies `df · N · τ`) so the signature stays ≤5 args and
the block composes cleanly. Lives at L3, NOT L0 — an analytic block of a dynamics is finance
(the `foundation/black.py`-at-L0 precedent, CLAUDE.md §1). Uses the `norm_cdf` foundation adapter
(§7bb), never scipy directly.

Provenance:
  quarry: python/pricebook/models/black76.py (black76_price)
  source: Black (1976); CLAUDE.md §1 (analytic block at L3); §7bb (distributions adapter)
  oracle: caplet reprices to this closed form to <1e-12; put-call parity; vol→0 intrinsic
  slice:  black-caplet (C2 slice 1)
"""

from __future__ import annotations

import math

from pricebook_ng.foundation import norm_cdf
from pricebook_ng.products.option import OptionType


def black(forward: float, strike: float, vol: float, t: float, option_type: OptionType) -> float:
    """Undiscounted Black-76 optionlet value under the T-forward measure. Degenerate
    (`vol ≤ 0`, `t ≤ 0`, or non-positive forward/strike where the lognormal is undefined)
    returns the forward intrinsic `(F − K)+` / `(K − F)+`."""
    call = option_type is OptionType.CALL
    if vol <= 0.0 or t <= 0.0 or forward <= 0.0 or strike <= 0.0:
        return max(forward - strike, 0.0) if call else max(strike - forward, 0.0)
    s = vol * math.sqrt(t)
    d1 = (math.log(forward / strike) + 0.5 * s * s) / s
    d2 = d1 - s
    if call:
        return forward * norm_cdf(d1) - strike * norm_cdf(d2)
    return strike * norm_cdf(-d2) - forward * norm_cdf(-d1)
