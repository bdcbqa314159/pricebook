"""Sukuk (Islamic bond) conventions (D7).

Sukuk are Sharia-compliant investment certificates. Instead of interest,
they pay profit derived from an underlying asset or activity.

    from pricebook.fixed_income.sukuk import (
        SukukType, SukukConventions, get_sukuk_conventions, list_sukuk_types,
    )

References:
    AAOIFI (2023). Sharia Standards.
    IIFM (2023). Sukuk Database.
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.core.serialisable import serialisable_convention
from enum import Enum


class SukukType(Enum):
    """Major Sukuk structures."""
    IJARA = "ijara"              # Lease-based: SPV leases asset, pays rental
    MUDARABA = "mudaraba"        # Profit-sharing: SPV manages capital
    MURABAHA = "murabaha"        # Cost-plus: SPV buys and resells at markup
    WAKALA = "wakala"            # Agency: SPV invests as agent
    MUSHARAKA = "musharaka"      # Partnership: joint venture profit sharing
    SALAM = "salam"              # Forward sale: agricultural/commodity
    ISTISNA = "istisna"          # Manufacturing: construction/project


@serialisable_convention("sukuk_conventions")
@dataclass(frozen=True)
class SukukConventions:
    """Conventions for a Sukuk type."""
    type: SukukType
    profit_mechanism: str        # how "coupon" is generated
    asset_requirement: bool      # needs underlying tangible asset?
    tradeable: bool              # can be traded on secondary market?
    typical_tenor_years: str     # typical maturity range
    major_currencies: list[str]  # USD, MYR, SAR, IDR, etc.
    pricing_approach: str        # how to price (spread, yield, etc.)
    notes: str = ""


_CONVENTIONS: dict[SukukType, SukukConventions] = {}


def _reg(c: SukukConventions) -> None:
    _CONVENTIONS[c.type] = c


_reg(SukukConventions(
    SukukType.IJARA,
    "Rental income from leased asset", True, True, "5-30Y",
    ["USD", "MYR", "SAR", "IDR"],
    "Spread over sovereign + asset quality adjustment",
    "Most common sovereign Sukuk. Saudi, Malaysia, Indonesia."))

_reg(SukukConventions(
    SukukType.MUDARABA,
    "Profit share from managed investment", False, True, "3-10Y",
    ["MYR", "BHD", "PKR"],
    "Spread over sovereign + profit rate benchmark",
    "Common in Malaysia (BNM Sukuk)."))

_reg(SukukConventions(
    SukukType.MURABAHA,
    "Cost-plus markup on commodity purchase", True, False, "1-5Y",
    ["USD", "SAR", "BHD"],
    "Discount rate (like T-Bill)",
    "Not tradeable on secondary — used for interbank."))

_reg(SukukConventions(
    SukukType.WAKALA,
    "Agency fee from managed portfolio", True, True, "3-10Y",
    ["USD", "MYR", "TRY"],
    "Spread over sovereign",
    "Increasingly popular. Turkey sovereign Sukuk."))

_reg(SukukConventions(
    SukukType.MUSHARAKA,
    "Profit/loss sharing from joint venture", True, True, "5-15Y",
    ["MYR", "PKR", "IDR"],
    "Spread over sovereign + equity-like risk premium"))

_reg(SukukConventions(
    SukukType.SALAM,
    "Forward sale of commodity at agreed price", True, False, "< 1Y",
    ["BHD", "SDR"],
    "Discount rate"))

_reg(SukukConventions(
    SukukType.ISTISNA,
    "Progress payments for construction/manufacturing", True, True, "3-20Y",
    ["USD", "SAR", "QAR"],
    "Spread over sovereign + project risk"))


def get_sukuk_conventions(sukuk_type: SukukType | str) -> SukukConventions:
    """Get conventions for a Sukuk type."""
    if isinstance(sukuk_type, str):
        sukuk_type = SukukType(sukuk_type.lower())
    conv = _CONVENTIONS.get(sukuk_type)
    if conv is None:
        raise ValueError(f"No conventions for {sukuk_type}")
    return conv


def list_sukuk_types() -> list[str]:
    """Return available Sukuk types."""
    return [t.value for t in SukukType]


def price_sukuk_as_bond(
    profit_rate: float,
    maturity_years: float,
    spread_bp: float,
    risk_free_rate: float,
    face: float = 100.0,
    freq: int = 2,
) -> float:
    """Price a Sukuk using conventional bond pricing (spread approach).

    Most traded Sukuk (Ijara, Wakala) are priced as spread over sovereign,
    identical to conventional bond pricing mechanics.

    price = Σ c/(1+y)^t + face/(1+y)^T where y = risk_free + spread
    """
    import math
    y = risk_free_rate + spread_bp / 10_000
    cpn = profit_rate / freq * face
    pv = 0.0
    n = int(maturity_years * freq)
    for i in range(1, n + 1):
        pv += cpn / (1 + y / freq) ** i
    pv += face / (1 + y / freq) ** n
    return pv
