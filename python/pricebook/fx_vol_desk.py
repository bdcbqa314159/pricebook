"""FX vol desk: strategies, vol book management, smile monitor.

Builds on :mod:`pricebook.fx_option` (Garman-Kohlhagen, deltas) and
:mod:`pricebook.fx_vol_surface` (ATM/RR/BF surface).

* :class:`FXStraddle` / :class:`FXStrangle` / :class:`FXRiskReversal`
  / :class:`FXButterfly` — standard vol structures.
* :func:`fx_vol_rv` — implied vs realised vol z-score.
* :func:`fx_skew_monitor` — 25-delta RR level vs history.
* :func:`fx_vega_ladder` — aggregate vega by pair and expiry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.fx_option import fx_option_price, fx_vega
from pricebook.black76 import OptionType


# ---- Z-score helper ----

def _zscore(current, history, threshold=2.0):
    if not history or len(history) < 2:
        return None, "fair"
    mean = sum(history) / len(history)
    var = sum((h - mean) ** 2 for h in history) / len(history)
    std = math.sqrt(var) if var > 0 else 0.0
    if std < 1e-12:
        return None, "fair"
    z = (current - mean) / std
    signal = "rich" if z >= threshold else ("cheap" if z <= -threshold else "fair")
    return z, signal


# ---- Vol structures ----

@dataclass
class FXStraddle:
    """ATM straddle: long call + long put at ATM strike."""
    pair: str
    expiry_days: int
    strike: float
    notional: float = 1.0

    def premium(self, spot, r_d, r_f, vol, T):
        c = fx_option_price(spot, self.strike, r_d, r_f, vol, T, OptionType.CALL)
        p = fx_option_price(spot, self.strike, r_d, r_f, vol, T, OptionType.PUT)
        return self.notional * (c + p)

    def vega(self, spot, r_d, r_f, vol, T):
        return self.notional * 2 * fx_vega(spot, self.strike, r_d, r_f, vol, T)


@dataclass
class FXStrangle:
    """OTM strangle: long OTM call + long OTM put."""
    pair: str
    call_strike: float
    put_strike: float
    notional: float = 1.0

    def premium(self, spot, r_d, r_f, vol, T):
        c = fx_option_price(spot, self.call_strike, r_d, r_f, vol, T, OptionType.CALL)
        p = fx_option_price(spot, self.put_strike, r_d, r_f, vol, T, OptionType.PUT)
        return self.notional * (c + p)


@dataclass
class FXRiskReversal:
    """Long OTM call, short OTM put (or reverse)."""
    pair: str
    call_strike: float
    put_strike: float
    direction: int = 1
    notional: float = 1.0

    def premium(self, spot, r_d, r_f, vol_call, vol_put, T):
        c = fx_option_price(spot, self.call_strike, r_d, r_f, vol_call, T, OptionType.CALL)
        p = fx_option_price(spot, self.put_strike, r_d, r_f, vol_put, T, OptionType.PUT)
        return self.direction * self.notional * (c - p)


@dataclass
class FXButterfly:
    """Long strangle + short straddle (or reverse)."""
    pair: str
    atm_strike: float
    call_strike: float
    put_strike: float
    notional: float = 1.0

    def premium(self, spot, r_d, r_f, vol_atm, vol_call, vol_put, T):
        strangle = (
            fx_option_price(spot, self.call_strike, r_d, r_f, vol_call, T, OptionType.CALL)
            + fx_option_price(spot, self.put_strike, r_d, r_f, vol_put, T, OptionType.PUT)
        )
        straddle = (
            fx_option_price(spot, self.atm_strike, r_d, r_f, vol_atm, T, OptionType.CALL)
            + fx_option_price(spot, self.atm_strike, r_d, r_f, vol_atm, T, OptionType.PUT)
        )
        return self.notional * (strangle - straddle)


# ---- Vol RV ----

@dataclass
class FXVolRV:
    pair: str
    implied_vol: float
    z_score: float | None
    signal: str


def fx_vol_rv(
    pair: str,
    implied_vol: float,
    historical_vols: list[float],
    threshold: float = 2.0,
) -> FXVolRV:
    """Implied vs realised vol z-score for an FX pair."""
    z, signal = _zscore(implied_vol, historical_vols, threshold)
    return FXVolRV(pair, implied_vol, z, signal)


# ---- Skew monitor ----

@dataclass
class FXSkewSignal:
    pair: str
    rr_level: float
    z_score: float | None
    signal: str


def fx_skew_monitor(
    pair: str,
    rr_level: float,
    history: list[float],
    threshold: float = 2.0,
) -> FXSkewSignal:
    """25-delta risk reversal level vs history."""
    z, signal = _zscore(rr_level, history, threshold)
    return FXSkewSignal(pair, rr_level, z, signal)


# ---- Vega ladder ----

@dataclass
class FXVegaBucket:
    pair: str
    expiry: str
    vega: float


def fx_vega_ladder(
    positions: list[tuple[str, str, float]],
) -> list[FXVegaBucket]:
    """Aggregate FX vega by (pair, expiry).

    Args:
        positions: list of ``(pair, expiry_label, vega)``.
    """
    agg: dict[tuple[str, str], float] = {}
    for pair, expiry, v in positions:
        key = (pair, expiry)
        agg[key] = agg.get(key, 0.0) + v
    return [
        FXVegaBucket(k[0], k[1], v)
        for k, v in sorted(agg.items())
    ]


def total_fx_vega(buckets: list[FXVegaBucket]) -> float:
    return sum(b.vega for b in buckets)
