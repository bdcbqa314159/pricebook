"""Sovereign bond factory — correct conventions for 50 markets.

Each sovereign bond market has specific settlement, day count, frequency,
and calendar conventions. This module encodes those conventions and provides
a factory to create correctly-configured FixedRateBond instances.

    from pricebook.fixed_income.sovereign_bonds import (
        create_sovereign_bond, get_conventions, list_markets,
    )

    bond = create_sovereign_bond("UST", issue, maturity, coupon=0.04)
    conv = get_conventions("BUND")

References:
    ICMA (2024). Primary Market Handbook.
    Bloomberg FLDS for sovereign bond conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.core.calendar import (
    Calendar, BusinessDayConvention, get_calendar,
)
from pricebook.core.day_count import DayCountConvention
from pricebook.core.schedule import Frequency
from pricebook.fixed_income.bond import FixedRateBond


@dataclass(frozen=True)
class SovereignConventions:
    """Bond market conventions for a sovereign issuer."""
    market_code: str        # e.g. "UST", "BUND", "NTN_F"
    country: str
    currency: str           # ISO 3-letter
    frequency: Frequency
    day_count: DayCountConvention
    settlement_days: int    # T+N
    calendar_currency: str  # currency code for get_calendar()
    ex_div_days: int = 0
    notes: str = ""


# ═══════════════════════════════════════════════════════════════
# Convention definitions — 50 markets
# ═══════════════════════════════════════════════════════════════


_CONVENTIONS: dict[str, SovereignConventions] = {}


def _reg(c: SovereignConventions) -> None:
    _CONVENTIONS[c.market_code] = c


# --- G10 core (6 existing + 4 other DM) ---

_reg(SovereignConventions(
    "UST", "United States", "USD",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 1, "USD",
    notes="US Treasury notes/bonds. T-Bills use ACT/360 discount."))

_reg(SovereignConventions(
    "BUND", "Germany", "EUR",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "EUR",
    notes="Bundesanleihen. Also covers Schatz (2Y), Bobl (5Y)."))

_reg(SovereignConventions(
    "GILT", "United Kingdom", "GBP",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 1, "GBP",
    ex_div_days=7, notes="UK Gilts. 7 business day ex-dividend period."))

_reg(SovereignConventions(
    "JGB", "Japan", "JPY",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_365_FIXED, 2, "JPY",
    notes="Japanese Government Bonds. 30/360 sometimes used for yield quotes."))

_reg(SovereignConventions(
    "OAT", "France", "EUR",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "EUR",
    notes="Obligations Assimilables du Trésor."))

_reg(SovereignConventions(
    "BTP", "Italy", "EUR",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "EUR",
    notes="Buoni del Tesoro Poliennali."))

# --- Other DM ---

_reg(SovereignConventions(
    "ACGB", "Australia", "AUD",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "AUD",
    notes="Australian Commonwealth Government Bonds."))

_reg(SovereignConventions(
    "NZGB", "New Zealand", "NZD",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "NZD",
    notes="New Zealand Government Bonds."))

_reg(SovereignConventions(
    "CGB_CA", "Canada", "CAD",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_365_FIXED, 2, "CAD",
    notes="Canadian Government Bonds. ACT/365F, not ACT/ACT."))

_reg(SovereignConventions(
    "DGB", "Denmark", "DKK",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "DKK",
    notes="Danish Government Bonds."))

_reg(SovereignConventions(
    "SGB", "Sweden", "SEK",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "SEK",
    notes="Swedish Government Bonds."))

_reg(SovereignConventions(
    "NGB", "Norway", "NOK",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "NOK",
    notes="Norwegian Government Bonds."))

_reg(SovereignConventions(
    "CONFED", "Switzerland", "CHF",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "CHF",
    notes="Swiss Confederation Bonds."))

# --- Eurozone periphery + core (9) ---

_reg(SovereignConventions(
    "BONO", "Spain", "EUR",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "EUR",
    notes="Bonos del Estado (Spain)."))

_reg(SovereignConventions(
    "BGB", "Belgium", "EUR",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "EUR",
    notes="Belgian Government Bonds (OLOs)."))

_reg(SovereignConventions(
    "DSL", "Netherlands", "EUR",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "EUR",
    notes="Dutch State Loans."))

_reg(SovereignConventions(
    "RAGB", "Austria", "EUR",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "EUR",
    notes="Republic of Austria Government Bonds."))

_reg(SovereignConventions(
    "RFGB", "Finland", "EUR",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "EUR",
    notes="Republic of Finland Government Bonds."))

_reg(SovereignConventions(
    "IRISH", "Ireland", "EUR",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "EUR",
    notes="Irish Government Bonds."))

_reg(SovereignConventions(
    "PGB", "Portugal", "EUR",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "EUR",
    notes="Portuguese Government Bonds."))

_reg(SovereignConventions(
    "GGB", "Greece", "EUR",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 3, "EUR",
    notes="Greek Government Bonds. T+3 settlement."))

# --- CEE (4) ---

_reg(SovereignConventions(
    "POLGB", "Poland", "PLN",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "PLN",
    notes="Polish Government Bonds."))

_reg(SovereignConventions(
    "CZGB", "Czech Republic", "CZK",
    Frequency.ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "CZK",
    notes="Czech Government Bonds."))

_reg(SovereignConventions(
    "HGB", "Hungary", "HUF",
    Frequency.ANNUAL, DayCountConvention.ACT_365_FIXED, 2, "HUF",
    notes="Hungarian Government Bonds. ACT/365F."))

_reg(SovereignConventions(
    "ROMGB", "Romania", "RON",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "RON",
    notes="Romanian Government Bonds. Semi-annual."))

# --- Turkey & MENA (6) ---

_reg(SovereignConventions(
    "TURKGB", "Turkey", "TRY",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_365_FIXED, 0, "TRY",
    notes="Turkish Government Bonds. T+0 settlement!"))

_reg(SovereignConventions(
    "SAGB_SA", "Saudi Arabia", "SAR",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "SAR",
    notes="Saudi Government Bonds (Sukuk + conventional)."))

_reg(SovereignConventions(
    "ADGB", "UAE", "AED",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "SAR",
    notes="Abu Dhabi Government Bonds. Uses SAR calendar (similar)."))

_reg(SovereignConventions(
    "QATGB", "Qatar", "QAR",
    Frequency.SEMI_ANNUAL, DayCountConvention.THIRTY_360, 2, "SAR",
    notes="Qatar Government Bonds. 30/360 day count."))

_reg(SovereignConventions(
    "ILGB", "Israel", "ILS",
    Frequency.ANNUAL, DayCountConvention.ACT_365_FIXED, 1, "ILS",
    notes="Israel Government Bonds (Shahar fixed, Galil linkers)."))

_reg(SovereignConventions(
    "EGGB", "Egypt", "EGP",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_365_FIXED, 2, "EGP",
    notes="Egyptian Government Bonds."))

# --- Africa (3) ---

_reg(SovereignConventions(
    "SAGB", "South Africa", "ZAR",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_365_FIXED, 3, "ZAR",
    notes="South African Government Bonds. T+3 settlement."))

_reg(SovereignConventions(
    "NGGB", "Nigeria", "NGN",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "NGN",
    notes="Nigerian Government Bonds (FGN Bonds)."))

_reg(SovereignConventions(
    "KEGB", "Kenya", "KES",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_365_FIXED, 2, "KES",
    notes="Kenyan Government Bonds."))

# --- LatAm (7) ---

_reg(SovereignConventions(
    "NTN_F", "Brazil", "BRL",
    Frequency.SEMI_ANNUAL, DayCountConvention.BUS_252, 1, "BRL",
    notes="NTN-F (fixed coupon, BUS/252). Principal + coupon at maturity."))

_reg(SovereignConventions(
    "NTN_B", "Brazil", "BRL",
    Frequency.SEMI_ANNUAL, DayCountConvention.BUS_252, 1, "BRL",
    notes="NTN-B (inflation-linked, IPCA). BUS/252 for accrual."))

_reg(SovereignConventions(
    "LTN", "Brazil", "BRL",
    Frequency.ANNUAL, DayCountConvention.BUS_252, 1, "BRL",
    notes="LTN (zero-coupon, bullet at maturity). BUS/252."))

_reg(SovereignConventions(
    "MBONO", "Mexico", "MXN",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_360, 2, "MXN",
    notes="MBONOs. ACT/360 — unusual for a government bond!"))

_reg(SovereignConventions(
    "CETES", "Mexico", "MXN",
    Frequency.ANNUAL, DayCountConvention.ACT_360, 2, "MXN",
    notes="CETES (zero-coupon T-bills). ACT/360."))

_reg(SovereignConventions(
    "BTP_CL", "Chile", "CLP",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_365_FIXED, 2, "CLP",
    notes="Chilean Government Bonds (BTPs)."))

_reg(SovereignConventions(
    "TES", "Colombia", "COP",
    Frequency.ANNUAL, DayCountConvention.ACT_365_FIXED, 2, "COP",
    notes="TES (Títulos de Tesorería). Annual coupon."))

# --- Asia (9) ---

_reg(SovereignConventions(
    "CGB", "China", "CNY",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 1, "CNY",
    notes="Chinese Government Bonds (onshore). T+1."))

_reg(SovereignConventions(
    "KTB", "South Korea", "KRW",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 1, "KRW",
    notes="Korea Treasury Bonds."))

_reg(SovereignConventions(
    "GSEC", "India", "INR",
    Frequency.SEMI_ANNUAL, DayCountConvention.THIRTY_360, 1, "INR",
    notes="Indian Government Securities. 30/360 — unusual for sovereign."))

_reg(SovereignConventions(
    "SGS", "Singapore", "SGD",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 1, "SGD",
    notes="Singapore Government Securities."))

_reg(SovereignConventions(
    "HKGB", "Hong Kong", "HKD",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "HKD",
    notes="Hong Kong Government Bonds."))

_reg(SovereignConventions(
    "INDOGB", "Indonesia", "IDR",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "IDR",
    notes="Indonesian Government Bonds (SUN/SBN)."))

_reg(SovereignConventions(
    "MGS", "Malaysia", "MYR",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 1, "MYR",
    notes="Malaysian Government Securities."))

_reg(SovereignConventions(
    "THAIGB", "Thailand", "THB",
    Frequency.SEMI_ANNUAL, DayCountConvention.ACT_ACT_ICMA, 2, "THB",
    notes="Thai Government Bonds."))

_reg(SovereignConventions(
    "RPGB", "Philippines", "PHP",
    Frequency.QUARTERLY, DayCountConvention.ACT_365_FIXED, 1, "PHP",
    notes="Republic of the Philippines Government Bonds. Quarterly coupon!"))


# ═══════════════════════════════════════════════════════════════
# Factory and helpers
# ═══════════════════════════════════════════════════════════════


def get_conventions(market_code: str) -> SovereignConventions:
    """Get conventions for a sovereign bond market.

    Args:
        market_code: e.g. "UST", "BUND", "NTN_F", "MBONO".

    Raises:
        ValueError: if market code not found.
    """
    code = market_code.upper()
    conv = _CONVENTIONS.get(code)
    if conv is None:
        available = sorted(_CONVENTIONS.keys())
        raise ValueError(f"Unknown market {code!r}. Available: {available}")
    return conv


def list_markets() -> list[str]:
    """Return sorted list of available sovereign bond market codes."""
    return sorted(_CONVENTIONS.keys())


def create_sovereign_bond(
    market_code: str,
    issue_date: date,
    maturity: date,
    coupon_rate: float,
    face_value: float = 100.0,
) -> FixedRateBond:
    """Create a FixedRateBond with correct conventions for the given sovereign market.

    Args:
        market_code: sovereign market code (e.g. "UST", "BUND", "MBONO").
        issue_date: bond issue date.
        maturity: bond maturity date.
        coupon_rate: annual coupon rate (e.g. 0.04 = 4%).
        face_value: face value (default 100).

    Returns:
        FixedRateBond configured with the correct day count, frequency,
        calendar, and settlement conventions.

    Example:
        >>> bond = create_sovereign_bond("BUND", date(2024,1,1), date(2034,1,1), 0.025)
        >>> bond.frequency == Frequency.ANNUAL  # Bunds are annual
        True
        >>> bond.day_count == DayCountConvention.ACT_ACT_ICMA
        True
    """
    conv = get_conventions(market_code)
    cal = get_calendar(conv.calendar_currency)

    return FixedRateBond(
        issue_date=issue_date,
        maturity=maturity,
        coupon_rate=coupon_rate,
        frequency=conv.frequency,
        face_value=face_value,
        day_count=conv.day_count,
        calendar=cal,
        convention=BusinessDayConvention.MODIFIED_FOLLOWING,
        settlement_days=conv.settlement_days,
        ex_div_days=conv.ex_div_days,
    )


def markets_by_region() -> dict[str, list[str]]:
    """Return market codes grouped by region."""
    regions: dict[str, list[str]] = {
        "G10_core": ["UST", "BUND", "GILT", "JGB", "OAT", "BTP"],
        "other_dm": ["ACGB", "NZGB", "CGB_CA", "DGB", "SGB", "NGB", "CONFED"],
        "eurozone": ["BONO", "BGB", "DSL", "RAGB", "RFGB", "IRISH", "PGB", "GGB"],
        "cee": ["POLGB", "CZGB", "HGB", "ROMGB"],
        "turkey_mena": ["TURKGB", "SAGB_SA", "ADGB", "QATGB", "ILGB", "EGGB"],
        "africa": ["SAGB", "NGGB", "KEGB"],
        "latam": ["NTN_F", "NTN_B", "LTN", "MBONO", "CETES", "BTP_CL", "TES"],
        "asia": ["CGB", "KTB", "GSEC", "SGS", "HKGB", "INDOGB", "MGS", "THAIGB", "RPGB"],
    }
    return regions
