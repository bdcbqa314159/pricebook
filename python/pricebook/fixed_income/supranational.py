"""Supranational bond conventions (D9).

EIB, IBRD, ADB, EBRD, AfDB, IFC, AIIB — AAA-rated quasi-sovereign issuers.

    from pricebook.fixed_income.supranational import (
        get_supranational, list_supranationals, SupranationalIssuer,
    )

References:
    Bloomberg (2024). SRCH <GO> — Supranational Bond Search.
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.core.serialisable import serialisable_convention


@serialisable_convention("supranational_issuer")
@dataclass(frozen=True)
class SupranationalIssuer:
    """Supranational issuer definition."""
    code: str                    # e.g. "EIB", "IBRD"
    name: str
    rating: str                  # Moody's/S&P rating
    typical_currencies: list[str]
    typical_maturities: str      # e.g. "2-30Y"
    spread_vs_sovereign_bp: float  # typical spread over domestic sovereign
    notes: str = ""


_REGISTRY: dict[str, SupranationalIssuer] = {}


def _reg(s: SupranationalIssuer) -> None:
    _REGISTRY[s.code] = s


_reg(SupranationalIssuer(
    "EIB", "European Investment Bank", "AAA",
    ["EUR", "USD", "GBP"], "2-30Y", 5.0,
    "Largest supranational issuer. EU policy bank."))

_reg(SupranationalIssuer(
    "IBRD", "International Bank for Reconstruction and Development", "AAA",
    ["USD", "EUR", "AUD", "NZD"], "2-30Y", 8.0,
    "World Bank lending arm."))

_reg(SupranationalIssuer(
    "IFC", "International Finance Corporation", "AAA",
    ["USD", "EUR", "AUD"], "3-10Y", 10.0,
    "World Bank private sector arm."))

_reg(SupranationalIssuer(
    "ADB", "Asian Development Bank", "AAA",
    ["USD", "AUD", "NZD", "INR"], "2-20Y", 7.0,
    "Regional development bank for Asia-Pacific."))

_reg(SupranationalIssuer(
    "EBRD", "European Bank for Reconstruction and Development", "AAA",
    ["USD", "EUR", "GBP", "TRY"], "2-15Y", 8.0,
    "Focus on Central/Eastern Europe and Central Asia."))

_reg(SupranationalIssuer(
    "AFDB", "African Development Bank", "AAA",
    ["USD", "EUR", "ZAR", "NGN"], "3-15Y", 12.0,
    "Regional development bank for Africa."))

_reg(SupranationalIssuer(
    "AIIB", "Asian Infrastructure Investment Bank", "AAA",
    ["USD", "EUR", "CNY"], "3-10Y", 10.0,
    "China-led multilateral. Founded 2016."))

_reg(SupranationalIssuer(
    "IADB", "Inter-American Development Bank", "AAA",
    ["USD", "EUR", "BRL", "MXN"], "2-20Y", 9.0,
    "Regional development bank for Latin America."))

_reg(SupranationalIssuer(
    "NIB", "Nordic Investment Bank", "AAA",
    ["EUR", "USD", "SEK", "NOK"], "2-10Y", 6.0,
    "Nordic/Baltic regional bank."))

_reg(SupranationalIssuer(
    "KFW", "KfW Bankengruppe", "AAA",
    ["EUR", "USD", "GBP"], "2-30Y", 3.0,
    "German state development bank. Quasi-sovereign, not strictly supranational."))


def get_supranational(code: str) -> SupranationalIssuer:
    """Look up a supranational issuer by code."""
    key = code.upper()
    s = _REGISTRY.get(key)
    if s is None:
        available = sorted(_REGISTRY.keys())
        raise ValueError(f"Unknown supranational {key!r}. Available: {available}")
    return s


def list_supranationals() -> list[str]:
    """Return sorted list of supranational issuer codes."""
    return sorted(_REGISTRY.keys())
