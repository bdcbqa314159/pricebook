"""Inflation trading: breakeven, carry, RV, risk decomposition, regulatory.

Combines slices 167-170 into one module:
* Breakeven monitor + strategies (DV01-neutral construction).
* Carry analysis (real yield, breakeven roll-down, seasonal).
* RV analysis (breakeven vs CPI swap basis, cross-market, seasonality).
* Risk decomposition (IE01 + real DV01 = nominal DV01).
* Regulatory: GIRR inflation risk weight (1.6%).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.regulatory.market_risk_sa import (
    GIRR_INFLATION_RW,
    calculate_frtb_sa,
)


# ---- Z-score helper ----

def _zscore(current, history, threshold=2.0):
    if not history or len(history) < 2:
        return None, "fair"
    mean = sum(history) / len(history)
    var = sum((h - mean) ** 2 for h in history) / len(history)
    std = math.sqrt(var) if var > 0 else 0.0
    if std < 1e-12: return None, "fair"
    z = (current - mean) / std
    signal = "rich" if z >= threshold else ("cheap" if z <= -threshold else "fair")
    return z, signal


# ---- Breakeven monitor (slice 167 step 1) ----

@dataclass
class BreakevenLevel:
    tenor: str
    nominal_yield: float
    real_yield: float
    breakeven: float


def breakeven_term_structure(
    tenors: list[tuple[str, float, float]],
) -> list[BreakevenLevel]:
    """Build breakeven term structure.

    Args:
        tenors: list of ``(tenor_label, nominal_yield, real_yield)``.
    """
    return [
        BreakevenLevel(t, nom, real, nom - real)
        for t, nom, real in tenors
    ]


@dataclass
class BreakevenBasisSignal:
    tenor: str
    breakeven: float
    cpi_swap_rate: float
    basis_bps: float
    z_score: float | None
    signal: str


def breakeven_basis_monitor(
    tenor: str,
    breakeven: float,
    cpi_swap_rate: float,
    history_bps: list[float] | None = None,
    threshold: float = 2.0,
) -> BreakevenBasisSignal:
    """Monitor breakeven vs CPI swap basis."""
    basis = (breakeven - cpi_swap_rate) * 10_000
    z, signal = _zscore(basis, history_bps or [], threshold)
    return BreakevenBasisSignal(tenor, breakeven, cpi_swap_rate, basis, z, signal)


# ---- Breakeven strategies (slice 167 step 2) ----

@dataclass
class BreakevenTrade:
    """DV01-neutral breakeven trade: buy linker + sell nominal (or reverse)."""
    tenor: str
    linker_notional: float
    nominal_notional: float
    linker_dv01: float
    nominal_dv01: float
    direction: int = 1  # +1 = long breakeven (long inflation)

    @property
    def is_dv01_neutral(self) -> bool:
        net = abs(self.linker_notional * self.linker_dv01
                  - self.nominal_notional * self.nominal_dv01)
        gross = (self.linker_notional * self.linker_dv01
                 + self.nominal_notional * self.nominal_dv01)
        return net / gross < 0.01 if gross > 0 else True

    @property
    def net_rate_dv01(self) -> float:
        return (self.linker_notional * self.linker_dv01
                - self.nominal_notional * self.nominal_dv01)


def build_breakeven_trade(
    tenor: str,
    notional: float,
    linker_dv01_per_million: float,
    nominal_dv01_per_million: float,
    direction: int = 1,
) -> BreakevenTrade:
    """Build a DV01-neutral breakeven trade.

    Sets nominal notional to match the linker's rate DV01.
    """
    if nominal_dv01_per_million <= 0:
        nom_notional = notional
    else:
        nom_notional = notional * linker_dv01_per_million / nominal_dv01_per_million
    return BreakevenTrade(
        tenor, notional, nom_notional,
        linker_dv01_per_million, nominal_dv01_per_million, direction,
    )


# ---- Carry analysis (slice 168) ----

@dataclass
class InflationCarry:
    real_yield_carry: float
    breakeven_rolldown: float
    financing_cost: float
    net_carry: float


def inflation_carry(
    notional: float,
    real_yield: float,
    breakeven_rolldown_bps: float,
    financing_rate: float = 0.0,
    linker_price: float = 100.0,
    days: int = 1,
) -> InflationCarry:
    """Carry analysis for a linker position.

    real_yield_carry = notional × real_yield × dt
    breakeven_rolldown = notional × rolldown / 10000 × dt
    financing = notional × (price/100) × financing_rate × dt
    """
    dt = days / 365.0
    ryc = notional * real_yield * dt
    brd = notional * breakeven_rolldown_bps / 10_000 * dt
    fin = notional * (linker_price / 100.0) * financing_rate * dt
    return InflationCarry(ryc, brd, fin, ryc + brd - fin)


@dataclass
class SeasonalCarrySignal:
    month: int
    seasonal_cpi_change: float
    annualised: float
    signal: str


def seasonal_carry(
    month: int,
    seasonal_factors: dict[int, float],
    threshold: float = 0.003,
) -> SeasonalCarrySignal:
    """Seasonal CPI carry signal.

    Args:
        seasonal_factors: {month → expected m/m CPI change} (e.g. 0.004 = +0.4%).
        threshold: absolute level above which the signal is "strong".
    """
    change = seasonal_factors.get(month, 0.0)
    annualised = change * 12
    signal = "strong" if abs(change) >= threshold else "weak"
    return SeasonalCarrySignal(month, change, annualised, signal)


# ---- RV analysis (slice 169) ----

@dataclass
class CrossMarketInflationRV:
    market_a: str
    market_b: str
    breakeven_a: float
    breakeven_b: float
    spread_bps: float
    z_score: float | None
    signal: str


def cross_market_inflation_rv(
    market_a: str, breakeven_a: float,
    market_b: str, breakeven_b: float,
    history_bps: list[float] | None = None,
    threshold: float = 2.0,
) -> CrossMarketInflationRV:
    """Cross-market: e.g. US TIPS vs UK linkers breakeven spread."""
    spread = (breakeven_a - breakeven_b) * 10_000
    z, signal = _zscore(spread, history_bps or [], threshold)
    return CrossMarketInflationRV(market_a, market_b, breakeven_a, breakeven_b,
                                   spread, z, signal)


# ---- Risk decomposition (slice 169 step 2) ----

@dataclass
class InflationRiskDecomposition:
    """IE01 + real DV01 ≈ nominal DV01."""
    ie01: float
    real_dv01: float
    nominal_dv01: float
    residual: float


def inflation_risk_decomposition(
    ie01: float,
    real_dv01: float,
    nominal_dv01: float,
) -> InflationRiskDecomposition:
    """Decompose: nominal DV01 = IE01 + real DV01 + residual."""
    residual = nominal_dv01 - (ie01 + real_dv01)
    return InflationRiskDecomposition(ie01, real_dv01, nominal_dv01, residual)


# ---- Regulatory (slice 170) ----

@dataclass
class InflationCapitalReport:
    inflation_capital: float
    total_capital: float
    total_rwa: float
    n_positions: int


def inflation_frtb_capital(
    inflation_sensitivities: list[dict],
) -> InflationCapitalReport:
    """FRTB SA inflation risk charge.

    Inflation sits within GIRR. Each position contributes:
    ``weighted = sensitivity × GIRR_INFLATION_RW / 100``.

    Args:
        inflation_sensitivities: list of ``{bucket, sensitivity}`` dicts.
            ``bucket`` is the currency; ``sensitivity`` is the IE01.
    """
    girr_pos = []
    for pos in inflation_sensitivities:
        girr_pos.append({
            "bucket": pos.get("bucket", "USD"),
            "sensitivity": pos.get("sensitivity", 0),
            "risk_weight": GIRR_INFLATION_RW,
        })

    result = calculate_frtb_sa(
        delta_positions={"GIRR": girr_pos},
        vega_positions={}, curvature_positions={},
        drc_positions=[], rrao_positions=[],
    )

    girr_cap = result["sbm_by_risk_class"].get("GIRR", {}).get("total_capital", 0.0)

    return InflationCapitalReport(
        inflation_capital=girr_cap,
        total_capital=result["total_capital"],
        total_rwa=result["total_rwa"],
        n_positions=len(inflation_sensitivities),
    )
