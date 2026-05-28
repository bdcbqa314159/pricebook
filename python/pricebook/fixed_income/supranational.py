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

from pricebook.core.data_registry import load_registry as _load_reg
_REGISTRY = _load_reg("supranational_issuers.json", SupranationalIssuer, lambda s: s.code, _REGISTRY)


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


def create_supranational_bond(
    issuer_code: str,
    currency: str,
    issue_date: "date",
    maturity: "date",
    coupon_rate: float,
    face_value: float = 100.0,
):
    """Create a FixedRateBond for a supranational issuer.

    Uses the currency's sovereign bond conventions (frequency, day_count,
    settlement) since supranationals follow the domestic market convention
    of the issuance currency.

    Args:
        issuer_code: supranational code (e.g. "EIB", "IBRD").
        currency: issuance currency (e.g. "EUR", "USD").
        issue_date: bond issue date.
        maturity: bond maturity date.
        coupon_rate: annual coupon rate.
        face_value: face value (default 100).

    Returns:
        FixedRateBond configured with the currency's sovereign conventions.
    """
    from datetime import date as _date
    issuer = get_supranational(issuer_code)
    if currency.upper() not in [c.upper() for c in issuer.typical_currencies]:
        import warnings
        warnings.warn(
            f"{issuer_code} does not typically issue in {currency}. "
            f"Typical currencies: {issuer.typical_currencies}",
            RuntimeWarning, stacklevel=2,
        )

    # Map currency to sovereign market convention
    _CURRENCY_TO_MARKET = {
        "USD": "UST", "EUR": "BUND", "GBP": "GILT", "JPY": "JGB",
        "CHF": "CONFED", "CAD": "CGB_CA", "AUD": "ACGB", "NZD": "NZGB",
        "SEK": "SGB", "NOK": "NGB",
    }
    market_code = _CURRENCY_TO_MARKET.get(currency.upper(), "UST")

    from pricebook.fixed_income.sovereign_bonds import get_conventions
    from pricebook.fixed_income.bond import FixedRateBond
    conv = get_conventions(market_code)
    return FixedRateBond.from_convention(conv, issue_date, maturity, coupon_rate, face_value)


@dataclass
class SupranationalBondResult:
    """Pricing result for a supranational bond."""
    clean_price: float
    dirty_price: float
    yield_to_maturity: float
    spread_vs_sovereign_bp: float
    issuer: str
    currency: str
    rating: str

    def to_dict(self) -> dict:
        return vars(self)


def price_supranational(
    issuer_code: str,
    currency: str,
    issue_date: "date",
    maturity: "date",
    coupon_rate: float,
    discount_curve: "DiscountCurve",
    sovereign_curve: "DiscountCurve | None" = None,
    face_value: float = 100.0,
) -> SupranationalBondResult:
    """Price a supranational bond and compute spread vs sovereign.

    Args:
        discount_curve: OIS/risk-free discount curve.
        sovereign_curve: sovereign bond curve (for spread computation).
            If None, uses discount_curve (spread will be ~0).
    """
    from pricebook.core.discount_curve import DiscountCurve

    issuer = get_supranational(issuer_code)
    bond = create_supranational_bond(issuer_code, currency, issue_date, maturity,
                                      coupon_rate, face_value)

    dirty = bond.dirty_price(discount_curve)
    clean = bond.clean_price(discount_curve)
    ytm = bond.yield_to_maturity(dirty / 100.0 * face_value)

    # Spread vs sovereign
    if sovereign_curve is not None:
        sov_dirty = bond.dirty_price(sovereign_curve)
        sov_ytm = bond.yield_to_maturity(sov_dirty / 100.0 * face_value)
        spread_bp = (ytm - sov_ytm) * 10_000
    else:
        spread_bp = issuer.spread_vs_sovereign_bp

    return SupranationalBondResult(
        clean_price=clean, dirty_price=dirty,
        yield_to_maturity=ytm, spread_vs_sovereign_bp=spread_bp,
        issuer=issuer_code, currency=currency, rating=issuer.rating,
    )
