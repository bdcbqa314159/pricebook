"""Per-currency swaption vol conventions.

Different markets quote swaption vols differently:
- USD: Black (lognormal) vol, semi-annual fixed, quarterly float
- EUR: Normal (Bachelier) vol, annual fixed, semi-annual float
- JPY: Both Black and Normal used, semi-annual
- BRL: DI-based swaption, BUS/252

    from pricebook.options.swaption_conventions import (
        get_swaption_convention, SwaptionConvention,
    )

References:
    ISDA (2024). ISDA Definitions.
    CME (2024). Interest Rate Swaption Specifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pricebook.core.day_count import DayCountConvention
from pricebook.core.schedule import Frequency


class VolQuoteType(Enum):
    BLACK = "black"           # lognormal (Black-76)
    NORMAL = "normal"         # Bachelier (normal/absolute)
    SHIFTED_BLACK = "shifted_black"  # shifted lognormal


class SmileType(Enum):
    SABR = "sabr"
    SHIFTED_SABR = "shifted_sabr"
    FLAT = "flat"             # ATM only, no smile


@dataclass(frozen=True)
class SwaptionConvention:
    """Swaption market convention per currency."""
    currency: str
    quote_type: VolQuoteType
    fixed_frequency: Frequency
    float_frequency: Frequency
    fixed_day_count: DayCountConvention
    float_day_count: DayCountConvention
    smile_type: SmileType
    standard_expiries: list[float]    # years
    standard_tenors: list[float]      # years
    settlement_days: int = 2
    premium_currency: str = ""
    notes: str = ""


_CONVENTIONS: dict[str, SwaptionConvention] = {}


def _reg(c: SwaptionConvention) -> None:
    _CONVENTIONS[c.currency] = c


# ═══════════════════════════════════════════════════════════════
# G10
# ═══════════════════════════════════════════════════════════════

_STANDARD_EXPIRIES = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
_STANDARD_TENORS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

_reg(SwaptionConvention(
    "USD", VolQuoteType.BLACK, Frequency.SEMI_ANNUAL, Frequency.QUARTERLY,
    DayCountConvention.THIRTY_360, DayCountConvention.ACT_360,
    SmileType.SHIFTED_SABR, _STANDARD_EXPIRIES, _STANDARD_TENORS, 2,
    "USD", "Shifted SABR standard. CME cleared."))

_reg(SwaptionConvention(
    "EUR", VolQuoteType.NORMAL, Frequency.ANNUAL, Frequency.SEMI_ANNUAL,
    DayCountConvention.THIRTY_360, DayCountConvention.ACT_360,
    SmileType.SABR, _STANDARD_EXPIRIES, _STANDARD_TENORS, 2,
    "EUR", "Normal (Bachelier) vol. Eurex cleared."))

_reg(SwaptionConvention(
    "GBP", VolQuoteType.BLACK, Frequency.SEMI_ANNUAL, Frequency.QUARTERLY,
    DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    SmileType.SABR, _STANDARD_EXPIRIES, _STANDARD_TENORS, 0,
    "GBP", "Black vol. LCH cleared. T+0 premium."))

_reg(SwaptionConvention(
    "JPY", VolQuoteType.BLACK, Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL,
    DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_360,
    SmileType.SABR, _STANDARD_EXPIRIES, _STANDARD_TENORS, 2,
    "JPY", "Black vol. Market also quotes Normal. JSCC cleared."))

_reg(SwaptionConvention(
    "CHF", VolQuoteType.NORMAL, Frequency.ANNUAL, Frequency.SEMI_ANNUAL,
    DayCountConvention.THIRTY_360, DayCountConvention.ACT_360,
    SmileType.SABR, _STANDARD_EXPIRIES, [1, 2, 5, 10, 20, 30], 2,
    "CHF", "Normal vol (like EUR). Can go negative."))

_reg(SwaptionConvention(
    "CAD", VolQuoteType.BLACK, Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL,
    DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    SmileType.SABR, [0.5, 1, 2, 5, 10], [1, 2, 5, 10, 30], 2,
    "CAD", "Black vol. Smaller grid than USD."))

_reg(SwaptionConvention(
    "AUD", VolQuoteType.BLACK, Frequency.SEMI_ANNUAL, Frequency.QUARTERLY,
    DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    SmileType.SABR, [0.5, 1, 2, 5, 10], [1, 2, 5, 10], 2,
    "AUD", "Black vol."))

# ═══════════════════════════════════════════════════════════════
# EM / Other
# ═══════════════════════════════════════════════════════════════

_reg(SwaptionConvention(
    "BRL", VolQuoteType.BLACK, Frequency.ANNUAL, Frequency.ANNUAL,
    DayCountConvention.BUS_252, DayCountConvention.BUS_252,
    SmileType.FLAT, [0.25, 0.5, 1, 2, 3], [1, 2, 3, 5], 1,
    "BRL", "DI-based swaption. BUS/252. Smaller grid."))

_reg(SwaptionConvention(
    "MXN", VolQuoteType.BLACK, Frequency.QUARTERLY, Frequency.QUARTERLY,
    DayCountConvention.ACT_360, DayCountConvention.ACT_360,
    SmileType.FLAT, [0.5, 1, 2, 5], [1, 2, 5, 10], 2,
    "MXN", "TIIE swaption. Limited liquidity."))

_reg(SwaptionConvention(
    "KRW", VolQuoteType.BLACK, Frequency.QUARTERLY, Frequency.QUARTERLY,
    DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    SmileType.FLAT, [0.5, 1, 2, 5], [1, 2, 5, 10], 1,
    "KRW", "KOFR swaption. Growing market."))

_reg(SwaptionConvention(
    "ZAR", VolQuoteType.BLACK, Frequency.QUARTERLY, Frequency.QUARTERLY,
    DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    SmileType.FLAT, [0.5, 1, 2, 5], [1, 2, 5, 10], 2,
    "ZAR", "JIBAR swaption."))


def get_swaption_convention(currency: str) -> SwaptionConvention:
    """Look up swaption convention by currency."""
    key = currency.upper()
    if key not in _CONVENTIONS:
        raise ValueError(f"No swaption convention for {key}. "
                         f"Available: {sorted(_CONVENTIONS.keys())}")
    return _CONVENTIONS[key]


def list_swaption_currencies() -> list[str]:
    """Return currencies with swaption conventions."""
    return sorted(_CONVENTIONS.keys())
