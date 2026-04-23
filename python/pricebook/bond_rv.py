"""Bond rich/cheap analysis and spread strategies.

* :func:`fitted_curve_rv` — rich/cheap vs a fitted yield curve, z-score.
* :func:`cross_market_rv` — UST vs Bunds vs Gilts (FX-adjusted spread).
* :func:`asw_spread_monitor` — asset swap spread z-score.
* :func:`zspread_monitor` — Z-spread z-score.
* :class:`CreditSpreadCurveTrade` — 2s10s credit spread curve trade.
* :func:`crossover_monitor` — BBB/BB boundary spread.
* :func:`new_issue_premium` — new issue vs secondary spread.
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.zscore import zscore as _zscore_impl


def _zscore(current: float, history: list[float], threshold: float = 2.0):
    r = _zscore_impl(current, history, threshold)
    return r.z_score, r.percentile, r.signal


# ---- Fitted curve RV ----

@dataclass
class FittedCurveRV:
    """Rich/cheap of a bond vs a fitted yield curve."""
    issuer: str
    market_yield: float
    fitted_yield: float
    spread_bps: float
    z_score: float | None
    percentile: float | None
    signal: str


def fitted_curve_rv(
    issuer: str,
    market_yield: float,
    fitted_yield: float,
    history_bps: list[float] | None = None,
    threshold: float = 2.0,
) -> FittedCurveRV:
    """Rich/cheap vs fitted curve: spread = market − fitted (in bps).

    Positive spread = bond is cheap (higher yield than model).
    """
    spread = (market_yield - fitted_yield) * 10_000
    z, pctile, signal = _zscore(spread, history_bps or [], threshold)
    return FittedCurveRV(
        issuer, market_yield, fitted_yield, spread, z, pctile, signal,
    )


# ---- Cross-market RV ----

@dataclass
class CrossMarketRV:
    """Cross-market relative value (e.g. UST vs Bunds)."""
    market_a: str
    market_b: str
    yield_a: float
    yield_b: float
    spread_bps: float
    fx_adjusted_spread_bps: float
    z_score: float | None
    signal: str


def cross_market_rv(
    market_a: str,
    yield_a: float,
    market_b: str,
    yield_b: float,
    fx_hedge_cost_bps: float = 0.0,
    history_bps: list[float] | None = None,
    threshold: float = 2.0,
) -> CrossMarketRV:
    """Cross-market spread: yield_A − yield_B − FX hedge cost.

    E.g. UST 10Y − Bund 10Y − xccy basis (FX-adjusted spread).
    """
    raw = (yield_a - yield_b) * 10_000
    adjusted = raw - fx_hedge_cost_bps
    z, _, signal = _zscore(adjusted, history_bps or [], threshold)
    return CrossMarketRV(
        market_a, market_b, yield_a, yield_b, raw, adjusted, z, signal,
    )


# ---- ASW / Z-spread monitors ----

@dataclass
class SpreadMonitorResult:
    """Spread monitor with z-score."""
    issuer: str
    spread_type: str
    spread_bps: float
    z_score: float | None
    percentile: float | None
    signal: str


def asw_spread_monitor(
    issuer: str,
    asw_bps: float,
    history_bps: list[float],
    threshold: float = 2.0,
) -> SpreadMonitorResult:
    """Z-score an asset swap spread vs history."""
    z, pctile, signal = _zscore(asw_bps, history_bps, threshold)
    return SpreadMonitorResult(issuer, "ASW", asw_bps, z, pctile, signal)


def zspread_monitor(
    issuer: str,
    zspread_bps: float,
    history_bps: list[float],
    threshold: float = 2.0,
) -> SpreadMonitorResult:
    """Z-score a Z-spread vs history."""
    z, pctile, signal = _zscore(zspread_bps, history_bps, threshold)
    return SpreadMonitorResult(issuer, "Z-spread", zspread_bps, z, pctile, signal)


# ---- Spread strategies ----

@dataclass
class CreditSpreadCurveTrade:
    """A DV01-neutral credit spread curve trade (e.g. 2s10s credit).

    Long the short-tenor credit bond, short the long-tenor credit bond,
    with DV01-matched notionals so the trade is rate-neutral.

    direction = +1: steepener (long short-tenor spread, short long-tenor).
    direction = -1: flattener.
    """
    short_tenor: str
    long_tenor: str
    short_spread_bps: float
    long_spread_bps: float
    short_dv01: float
    long_dv01: float
    short_face: float
    long_face: float
    direction: int = 1

    @property
    def curve_spread_bps(self) -> float:
        """Spread curve level = long_spread − short_spread."""
        return self.long_spread_bps - self.short_spread_bps

    @property
    def is_dv01_neutral(self) -> bool:
        """Check if the trade is approximately DV01-neutral."""
        net = abs(self.short_face * self.short_dv01 - self.long_face * self.long_dv01)
        gross = self.short_face * self.short_dv01 + self.long_face * self.long_dv01
        return net / gross < 0.01 if gross > 0 else True


def build_credit_curve_trade(
    short_tenor: str,
    long_tenor: str,
    short_spread_bps: float,
    long_spread_bps: float,
    short_dv01_per_million: float,
    long_dv01_per_million: float,
    notional: float = 10_000_000,
    direction: int = 1,
) -> CreditSpreadCurveTrade:
    """Build a DV01-neutral credit spread curve trade.

    Sets the long-tenor face to match the short-tenor DV01:
        long_face = notional × short_dv01 / long_dv01.
    """
    if long_dv01_per_million <= 0:
        long_face = notional
    else:
        long_face = notional * short_dv01_per_million / long_dv01_per_million

    return CreditSpreadCurveTrade(
        short_tenor=short_tenor,
        long_tenor=long_tenor,
        short_spread_bps=short_spread_bps,
        long_spread_bps=long_spread_bps,
        short_dv01=short_dv01_per_million,
        long_dv01=long_dv01_per_million,
        short_face=notional,
        long_face=long_face,
        direction=direction,
    )


# ---- Crossover monitor ----

@dataclass
class CrossoverSignal:
    """BBB/BB boundary spread monitor."""
    bbb_spread_bps: float
    bb_spread_bps: float
    crossover_spread_bps: float
    z_score: float | None
    signal: str


def crossover_monitor(
    bbb_spread_bps: float,
    bb_spread_bps: float,
    history_bps: list[float] | None = None,
    threshold: float = 2.0,
) -> CrossoverSignal:
    """Monitor the BBB/BB crossover spread.

    crossover = BB − BBB. Wide = HY cheap vs IG; tight = HY rich.
    """
    crossover = bb_spread_bps - bbb_spread_bps
    z, _, signal = _zscore(crossover, history_bps or [], threshold)
    return CrossoverSignal(bbb_spread_bps, bb_spread_bps, crossover, z, signal)


# ---- New issue premium ----

@dataclass
class NewIssuePremium:
    """New issue vs secondary market spread."""
    issuer: str
    new_issue_spread_bps: float
    secondary_spread_bps: float
    premium_bps: float


def new_issue_premium(
    issuer: str,
    new_issue_spread_bps: float,
    secondary_spread_bps: float,
) -> NewIssuePremium:
    """New issue premium = new_issue_spread − secondary_spread.

    Positive premium means the new issue is cheaper (wider) than
    existing bonds, attracting investor interest.
    """
    return NewIssuePremium(
        issuer, new_issue_spread_bps, secondary_spread_bps,
        new_issue_spread_bps - secondary_spread_bps,
    )
