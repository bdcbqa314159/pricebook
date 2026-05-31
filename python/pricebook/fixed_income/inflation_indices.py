"""Inflation index registry and linker bond factory.

Each inflation index has specific properties: publication lag, frequency,
interpolation method, and whether a deflation floor applies. This module
encodes those conventions and provides a factory for inflation-linked bonds.

    from pricebook.fixed_income.inflation_indices import (
        get_inflation_index, create_inflation_linker, list_inflation_indices,
    )

    idx = get_inflation_index("IPCA")
    bond = create_inflation_linker("IPCA", issue, maturity, coupon=0.06, base_cpi=6500.0)

References:
    Deacon, Derry & Mirfendereski (2004). Inflation-Indexed Securities.
    Barclays (2010). Global Inflation-Linked Products: A User's Guide.
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.core.serialisable import serialisable_convention
from datetime import date
from enum import Enum

from pricebook.core.day_count import DayCountConvention
from pricebook.core.schedule import Frequency


class IndexInterpolation(Enum):
    """How CPI values are interpolated within a month."""
    FLAT = "flat"                   # Use prior month's value (UK ILG)
    LINEAR = "linear"              # Linear between months (TIPS, most markets)
    DAILY = "daily"                # Daily accrual (UF Chile)


@serialisable_convention("inflation_index_def")
@dataclass(frozen=True)
class InflationIndexDef:
    """Definition of an inflation index for linker pricing."""
    name: str                       # e.g. "CPI_US", "IPCA"
    description: str
    currency: str                   # ISO 3-letter
    country: str
    publication_lag_months: int     # CPI lag (2-3 for most, 8 for UK)
    publication_frequency: str      # "monthly" or "daily" (UF)
    interpolation: IndexInterpolation
    deflation_floor: bool           # True = principal cannot fall below par
    base_year: int | None           # Reference base year for index (e.g. 1982 for CPI-U)
    linker_day_count: DayCountConvention
    linker_frequency: Frequency
    notes: str = ""

# ═══════════════════════════════════════════════════════════════
# Index definitions
# ═══════════════════════════════════════════════════════════════

_REGISTRY: dict[str, InflationIndexDef] = {}


def _reg(idx: InflationIndexDef) -> None:
    _REGISTRY[idx.name] = idx


# --- US ---
_reg(InflationIndexDef(
    "CPI_US", "US CPI-U (Urban Consumers, All Items, NSA)", "USD", "United States",
    publication_lag_months=3, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=True, base_year=1982,
    linker_day_count=DayCountConvention.ACT_ACT_ICMA,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="Used by TIPS. 3-month lag, linear interpolation, deflation floor on principal."))

# --- Eurozone ---
_reg(InflationIndexDef(
    "HICP_XT", "Eurostat HICPxT (ex-Tobacco)", "EUR", "Eurozone",
    publication_lag_months=3, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=False, base_year=2015,
    linker_day_count=DayCountConvention.ACT_ACT_ICMA,
    linker_frequency=Frequency.ANNUAL,
    notes="Used by OAT€i, BTP€i, Bund€i. No deflation floor (unlike TIPS)."))

# --- UK ---
_reg(InflationIndexDef(
    "RPI", "UK Retail Price Index", "GBP", "United Kingdom",
    publication_lag_months=8, publication_frequency="monthly",
    interpolation=IndexInterpolation.FLAT,
    deflation_floor=False, base_year=1987,
    linker_day_count=DayCountConvention.ACT_ACT_ICMA,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="Used by UK ILGs (index-linked gilts). 8-month lag, flat interpolation (no daily)."))

_reg(InflationIndexDef(
    "CPIH", "UK CPIH (CPI including Housing)", "GBP", "United Kingdom",
    publication_lag_months=3, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=False, base_year=2015,
    linker_day_count=DayCountConvention.ACT_ACT_ICMA,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="New UK linkers (from 2030) will use CPIH instead of RPI."))

# --- Japan ---
_reg(InflationIndexDef(
    "CPI_JP", "Japan CPI (All Items, Nationwide)", "JPY", "Japan",
    publication_lag_months=3, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=True, base_year=2020,
    linker_day_count=DayCountConvention.ACT_365_FIXED,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="Used by JGBi. Deflation floor on principal."))

# --- Canada ---
_reg(InflationIndexDef(
    "CPI_CA", "Canada CPI (All Items, NSA)", "CAD", "Canada",
    publication_lag_months=3, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=True, base_year=2002,
    linker_day_count=DayCountConvention.ACT_365_FIXED,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="Used by Canadian Real Return Bonds (RRBs)."))

# --- Australia ---
_reg(InflationIndexDef(
    "CPI_AU", "Australia CPI (All Groups, Weighted Average)", "AUD", "Australia",
    publication_lag_months=3, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=False, base_year=2011,
    linker_day_count=DayCountConvention.ACT_ACT_ICMA,
    linker_frequency=Frequency.QUARTERLY,
    notes="Used by Treasury Indexed Bonds (TIBs). Quarterly CPI published."))

# --- Brazil ---
_reg(InflationIndexDef(
    "IPCA", "IPCA (Índice Nacional de Preços ao Consumidor Amplo)", "BRL", "Brazil",
    publication_lag_months=1, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=False, base_year=1993,
    linker_day_count=DayCountConvention.BUS_252,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="Used by NTN-B. 1-month lag (shorter than most). BUS/252 day count."))

# --- Mexico ---
_reg(InflationIndexDef(
    "UDI", "UDI (Unidades de Inversión)", "MXN", "Mexico",
    publication_lag_months=0, publication_frequency="daily",
    interpolation=IndexInterpolation.DAILY,
    deflation_floor=False, base_year=2002,
    linker_day_count=DayCountConvention.ACT_360,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="UDI is published daily (Banxico). Used by UDIBONO. Zero lag, daily accrual."))

# --- Chile ---
_reg(InflationIndexDef(
    "UF", "UF (Unidad de Fomento)", "CLP", "Chile",
    publication_lag_months=0, publication_frequency="daily",
    interpolation=IndexInterpolation.DAILY,
    deflation_floor=False, base_year=1967,
    linker_day_count=DayCountConvention.ACT_365_FIXED,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="UF is published daily (BCCh). Chilean linkers denominated directly in UF."))

# --- Colombia ---
_reg(InflationIndexDef(
    "UVR", "UVR (Unidad de Valor Real)", "COP", "Colombia",
    publication_lag_months=0, publication_frequency="daily",
    interpolation=IndexInterpolation.DAILY,
    deflation_floor=False, base_year=2000,
    linker_day_count=DayCountConvention.ACT_365_FIXED,
    linker_frequency=Frequency.ANNUAL,
    notes="UVR updated daily (BanRep). Used by TES UVR."))

# --- Peru ---
_reg(InflationIndexDef(
    "IPC_PE", "Peru CPI (Lima Metropolitana)", "PEN", "Peru",
    publication_lag_months=1, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=False, base_year=2009,
    linker_day_count=DayCountConvention.ACT_365_FIXED,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="Used by VAC bonds (Valor Adquisitivo Constante)."))

# --- Argentina ---
_reg(InflationIndexDef(
    "CER", "Argentina CER (Coeficiente de Estabilización de Referencia)", "ARS", "Argentina",
    publication_lag_months=1, publication_frequency="daily",
    interpolation=IndexInterpolation.DAILY,
    deflation_floor=False, base_year=2002,
    linker_day_count=DayCountConvention.ACT_365_FIXED,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="CER: daily inflation coefficient (BCRA). Used by Lecer, Boncer."))

# --- South Africa ---
_reg(InflationIndexDef(
    "CPI_ZA", "South Africa CPI (All Urban)", "ZAR", "South Africa",
    publication_lag_months=3, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=False, base_year=2016,
    linker_day_count=DayCountConvention.ACT_365_FIXED,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="Used by SA ILBs (inflation-linked bonds). R197, I2025 series."))

# --- Israel ---
_reg(InflationIndexDef(
    "CPI_IL", "Israel CPI (Known Index)", "ILS", "Israel",
    publication_lag_months=1, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=False, base_year=1993,
    linker_day_count=DayCountConvention.ACT_365_FIXED,
    linker_frequency=Frequency.ANNUAL,
    notes="Used by Galil bonds (Israeli linkers)."))

# --- Turkey ---
_reg(InflationIndexDef(
    "CPI_TR", "Turkey CPI (TUIK)", "TRY", "Turkey",
    publication_lag_months=2, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=False, base_year=2003,
    linker_day_count=DayCountConvention.ACT_365_FIXED,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="Used by CPI-linked Turkish government bonds."))

# --- India ---
_reg(InflationIndexDef(
    "CPI_IN", "India CPI (Combined, Urban + Rural)", "INR", "India",
    publication_lag_months=3, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=True, base_year=2012,
    linker_day_count=DayCountConvention.THIRTY_360,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="Used by Indian IIBs (Inflation-Indexed Bonds). Deflation floor."))

# --- South Korea ---
_reg(InflationIndexDef(
    "CPI_KR", "Korea CPI (All Items)", "KRW", "South Korea",
    publication_lag_months=3, publication_frequency="monthly",
    interpolation=IndexInterpolation.LINEAR,
    deflation_floor=True, base_year=2020,
    linker_day_count=DayCountConvention.ACT_ACT_ICMA,
    linker_frequency=Frequency.SEMI_ANNUAL,
    notes="Used by KTBi (Korea inflation-linked bonds)."))


# Load from JSON if available
from pricebook.core.data_registry import load_registry as _load_reg
_REGISTRY = _load_reg("inflation_indices.json", InflationIndexDef, lambda i: i.name, _REGISTRY)

# ═══════════════════════════════════════════════════════════════
# Registry API
# ═══════════════════════════════════════════════════════════════


def get_inflation_index(name: str) -> InflationIndexDef:
    """Look up an inflation index by name.

    Args:
        name: e.g. "CPI_US", "IPCA", "UDI", "HICP_XT".

    Raises:
        ValueError: if not found.
    """
    key = name.upper()
    idx = _REGISTRY.get(key)
    if idx is None:
        available = sorted(_REGISTRY.keys())
        raise ValueError(f"Unknown inflation index {name!r}. Available: {available}")
    return idx


def list_inflation_indices() -> list[str]:
    """Return sorted list of available inflation index names."""
    return sorted(_REGISTRY.keys())


def indices_by_currency(currency: str) -> list[InflationIndexDef]:
    """Return all inflation indices for a given currency."""
    ccy = currency.upper()
    return [idx for idx in _REGISTRY.values() if idx.currency == ccy]


def indices_with_floor() -> list[InflationIndexDef]:
    """Return indices that have a deflation floor on principal."""
    return [idx for idx in _REGISTRY.values() if idx.deflation_floor]


def daily_indices() -> list[InflationIndexDef]:
    """Return indices with daily publication (UDI, UF, UVR)."""
    return [idx for idx in _REGISTRY.values() if idx.publication_frequency == "daily"]


# ═══════════════════════════════════════════════════════════════
# Linker factory
# ═══════════════════════════════════════════════════════════════


def create_inflation_linker(
    index_name: str,
    issue_date: date,
    maturity: date,
    coupon_rate: float,
    base_cpi: float,
    notional: float = 100.0,
) -> dict:
    """Create parameters for an InflationLinkedBond with correct conventions.

    Returns a dict of kwargs suitable for InflationLinkedBond(...).
    This avoids a direct import dependency on the bond class.

    Args:
        index_name: inflation index name (e.g. "CPI_US", "IPCA").
        issue_date: bond issue date.
        maturity: bond maturity date.
        coupon_rate: real coupon rate (e.g. 0.02 = 2%).
        base_cpi: CPI index value at issue (for indexation ratio).
        notional: face value (default 100).

    Returns:
        Dict with keys: start, end, coupon_rate, base_cpi_value, notional,
        frequency, day_count, cpi_lag_months, plus metadata.
    """
    idx = get_inflation_index(index_name)
    return {
        "start": issue_date,
        "end": maturity,
        "coupon_rate": coupon_rate,
        "base_cpi_value": base_cpi,
        "notional": notional,
        "frequency": idx.linker_frequency,
        "day_count": idx.linker_day_count,
        "cpi_lag_months": idx.publication_lag_months,
        # Metadata
        "_index_name": idx.name,
        "_deflation_floor": idx.deflation_floor,
        "_interpolation": idx.interpolation.value,
    }
