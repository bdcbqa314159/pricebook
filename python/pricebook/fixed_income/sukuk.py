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


# ═══════════════════════════════════════════════════════════════
# SukukBond instrument
# ═══════════════════════════════════════════════════════════════


class SukukBond:
    """Sukuk instrument — Sharia-compliant bond equivalent.

    Priced identically to a conventional fixed-rate bond: the "profit rate"
    is economically equivalent to a coupon. The underlying asset structure
    (Ijara lease, Murabaha cost-plus, Wakala agency) determines the legal
    wrapper but not the pricing mechanics for liquid issues.

    For illiquid/project Sukuk (Istisna, Salam), spread adjustments are
    applied on top of the bond pricing framework.

    Args:
        issue_date: issue/settlement date.
        maturity: maturity date.
        profit_rate: annual profit rate (equivalent to coupon rate).
        sukuk_type: SukukType enum or string.
        face_value: face value (default 100).
        currency: issuance currency (default "USD").
        frequency: profit payment frequency (default semi-annual).
    """

    def __init__(
        self,
        issue_date,
        maturity,
        profit_rate: float,
        sukuk_type: SukukType | str = SukukType.IJARA,
        face_value: float = 100.0,
        currency: str = "USD",
        frequency: int = 2,
    ):
        from datetime import date as _date
        self.issue_date = issue_date
        self.maturity = maturity
        self.profit_rate = profit_rate
        self.sukuk_type = SukukType(sukuk_type) if isinstance(sukuk_type, str) else sukuk_type
        self.face_value = face_value
        self.currency = currency
        self.frequency = frequency

        # Build as a FixedRateBond internally for curve-based pricing
        from pricebook.core.schedule import Frequency
        freq_map = {1: Frequency.ANNUAL, 2: Frequency.SEMI_ANNUAL,
                    4: Frequency.QUARTERLY, 12: Frequency.MONTHLY}
        self._freq_enum = freq_map.get(frequency, Frequency.SEMI_ANNUAL)

    def _as_bond(self):
        """Create an equivalent FixedRateBond for pricing."""
        from pricebook.fixed_income.bond import FixedRateBond
        from pricebook.core.day_count import DayCountConvention
        return FixedRateBond(
            issue_date=self.issue_date, maturity=self.maturity,
            coupon_rate=self.profit_rate, frequency=self._freq_enum,
            face_value=self.face_value,
            day_count=DayCountConvention.ACT_365_FIXED,
        )

    def dirty_price(self, discount_curve) -> float:
        """Dirty price using discount curve (same as conventional bond)."""
        return self._as_bond().dirty_price(discount_curve)

    def clean_price(self, discount_curve) -> float:
        """Clean price."""
        return self._as_bond().clean_price(discount_curve)

    def yield_to_maturity(self, market_price: float) -> float:
        """Yield to maturity from market price."""
        return self._as_bond().yield_to_maturity(market_price)

    def price_from_spread(self, risk_free_rate: float, spread_bp: float) -> float:
        """Price using yield-based approach (risk-free + spread)."""
        return price_sukuk_as_bond(
            self.profit_rate, self._maturity_years(),
            spread_bp, risk_free_rate, self.face_value, self.frequency,
        )

    def _maturity_years(self) -> float:
        from pricebook.core.day_count import DayCountConvention, year_fraction
        return year_fraction(self.issue_date, self.maturity, DayCountConvention.ACT_365_FIXED)

    def pv(self, discount_curve) -> float:
        """PV alias for dirty_price × face / 100."""
        return self.dirty_price(discount_curve) * self.face_value / 100.0

    def pv_ctx(self, ctx) -> float:
        """PV using PricingContext."""
        curve = ctx.discount_curve
        if curve is None:
            raise ValueError("No discount curve in context")
        return self.pv(curve)

    @classmethod
    def from_convention(cls, conv, issue_date, maturity, profit_rate,
                        face_value=100.0, currency="USD"):
        """Create SukukBond from SukukConventions."""
        return cls(issue_date, maturity, profit_rate, conv.type,
                   face_value, currency)

    def to_dict(self) -> dict:
        return {
            "type": "sukuk_bond",
            "params": {
                "issue_date": self.issue_date.isoformat(),
                "maturity": self.maturity.isoformat(),
                "profit_rate": self.profit_rate,
                "sukuk_type": self.sukuk_type.value,
                "face_value": self.face_value,
                "currency": self.currency,
                "frequency": self.frequency,
            }
        }

    @classmethod
    def from_dict(cls, d):
        from datetime import date as _date
        p = d["params"]
        return cls(
            issue_date=_date.fromisoformat(p["issue_date"]),
            maturity=_date.fromisoformat(p["maturity"]),
            profit_rate=p["profit_rate"],
            sukuk_type=p["sukuk_type"],
            face_value=p.get("face_value", 100.0),
            currency=p.get("currency", "USD"),
            frequency=p.get("frequency", 2),
        )


SukukBond._SERIAL_TYPE = "sukuk_bond"
from pricebook.core.serialisable import _register as _reg_sukuk
_reg_sukuk(SukukBond)


def create_sukuk(
    sukuk_type: SukukType | str,
    issue_date,
    maturity,
    profit_rate: float,
    face_value: float = 100.0,
    currency: str = "USD",
) -> SukukBond:
    """Create a SukukBond with correct conventions for the given type.

    Args:
        sukuk_type: SukukType enum or string (e.g. "ijara", "wakala").
        issue_date: issue date.
        maturity: maturity date.
        profit_rate: annual profit rate.
    """
    conv = get_sukuk_conventions(sukuk_type)
    return SukukBond.from_convention(conv, issue_date, maturity, profit_rate,
                                     face_value, currency)
