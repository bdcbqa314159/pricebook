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
from datetime import date

from pricebook_ng.foundation import DayCountConvention, TimeMeasure, norm_cdf, norm_pdf
from pricebook_ng.products.option import OptionType


def vol_time_measure(valuation_date: date) -> TimeMeasure:
    """The ONE canonical option-vol clock: ACT/365F from the valuation date. Defined once and composed
    by BOTH the pricer's expiry-`t` AND the vol surface's `T`-axis, so a vol read at an expiry is always
    consistent with the `t` it is priced at (§3d — one source, no drift, #4)."""
    return TimeMeasure(valuation_date, DayCountConvention.ACT_365_FIXED)


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


def black_vega(forward: float, strike: float, vol: float, t: float) -> float:
    """The UNDISCOUNTED Black-76 vega `∂black/∂σ = F·φ(d₁)·√t` (call and put share it — put-call
    parity is vol-independent). The engine scales by `df·N·τ`. Degenerate (`vol ≤ 0`, `t ≤ 0`, or a
    non-positive forward/strike) has no vol sensitivity → 0."""
    if vol <= 0.0 or t <= 0.0 or forward <= 0.0 or strike <= 0.0:
        return 0.0
    s = vol * math.sqrt(t)
    d1 = (math.log(forward / strike) + 0.5 * s * s) / s
    return forward * norm_pdf(d1) * math.sqrt(t)
