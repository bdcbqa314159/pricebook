"""Market-convention yield quotation per sovereign market.

Each sovereign market quotes yields with specific compounding:
- US/UK/JP: semi-annual compounding (street convention)
- Germany/France/Eurozone: annual compounding
- Brazil: BUS/252 continuous equivalent
- Mexico: ACT/360 discount basis for CETES

    from pricebook.fixed_income.yield_convention import (
        yield_to_price, price_to_yield, YieldConvention,
    )

    # US Treasury: semi-annual compounding
    price = yield_to_price(0.04, 0.05, 10.0, YieldConvention.SEMI_ANNUAL)
    ytm = price_to_yield(price, 0.05, 10.0, YieldConvention.SEMI_ANNUAL)

References:
    Tuckman & Serrat (2012). Fixed Income Securities, Ch 3.
    Fabozzi (2012). Bond Markets, Analysis and Strategies, Ch 4.
"""

from __future__ import annotations

import math
from enum import Enum

from pricebook.core.solvers import brentq


class YieldConvention(Enum):
    """How yields are compounded in market quotation."""
    SEMI_ANNUAL = "semi_annual"    # y_sa: P = Σ CF / (1 + y/2)^(2t)
    ANNUAL = "annual"              # y_a:  P = Σ CF / (1 + y)^t
    QUARTERLY = "quarterly"        # y_q:  P = Σ CF / (1 + y/4)^(4t)
    CONTINUOUS = "continuous"       # y_c:  P = Σ CF × exp(-y × t)
    SIMPLE = "simple"              # y_s:  P = Face / (1 + y × t)  [zero-coupon only]
    DISCOUNT = "discount"          # d:    P = Face × (1 - d × t)  [T-Bill bank discount]


def yield_to_price(
    ytm: float,
    coupon_rate: float,
    maturity_years: float,
    convention: YieldConvention,
    frequency: int = 2,
    face: float = 100.0,
) -> float:
    """Convert yield-to-maturity to clean price under a given convention.

    Args:
        ytm: yield-to-maturity under the specified convention.
        coupon_rate: annual coupon rate (e.g. 0.05 = 5%).
        maturity_years: time to maturity in years.
        convention: yield compounding convention.
        frequency: coupons per year (used for coupon schedule, not compounding).
        face: face value.

    Returns:
        Clean price per face.
    """
    if convention == YieldConvention.SIMPLE:
        return face / (1.0 + ytm * maturity_years)

    if convention == YieldConvention.DISCOUNT:
        return face * (1.0 - ytm * maturity_years)

    if convention == YieldConvention.CONTINUOUS:
        return _price_from_continuous(ytm, coupon_rate, maturity_years, frequency, face)

    # Compounding conventions (semi-annual, annual, quarterly)
    comp_freq = _comp_freq(convention)
    return _price_from_compounded(ytm, coupon_rate, maturity_years, frequency, comp_freq, face)


def price_to_yield(
    price: float,
    coupon_rate: float,
    maturity_years: float,
    convention: YieldConvention,
    frequency: int = 2,
    face: float = 100.0,
) -> float:
    """Convert clean price to yield-to-maturity under a given convention.

    Uses Brent's method for compounded conventions.
    """
    if price <= 0 or maturity_years <= 0:
        return 0.0

    if convention == YieldConvention.SIMPLE:
        return (face / price - 1.0) / maturity_years

    if convention == YieldConvention.DISCOUNT:
        return (1.0 - price / face) / maturity_years

    # Numerical solve for compounded/continuous
    def objective(y: float) -> float:
        return yield_to_price(y, coupon_rate, maturity_years, convention, frequency, face) - price

    try:
        return brentq(objective, -0.20, 2.0)
    except ValueError:
        return 0.0


def convert_yield(
    ytm: float,
    from_conv: YieldConvention,
    to_conv: YieldConvention,
    coupon_rate: float = 0.0,
    maturity_years: float = 1.0,
    frequency: int = 2,
    face: float = 100.0,
) -> float:
    """Convert a yield from one convention to another.

    For zero-coupon (coupon_rate=0), uses exact formulas.
    For coupon bonds, round-trips through price.
    """
    # Zero-coupon exact conversions
    if coupon_rate == 0.0:
        return _convert_zero_yield(ytm, from_conv, to_conv, maturity_years)

    # Coupon bond: price roundtrip
    price = yield_to_price(ytm, coupon_rate, maturity_years, from_conv, frequency, face)
    return price_to_yield(price, coupon_rate, maturity_years, to_conv, frequency, face)


# ═══════════════════════════════════════════════════════════════
# Sovereign market → yield convention mapping
# ═══════════════════════════════════════════════════════════════

_MARKET_YIELD_CONVENTION: dict[str, YieldConvention] = {
    # Semi-annual compounding
    "UST": YieldConvention.SEMI_ANNUAL,
    "GILT": YieldConvention.SEMI_ANNUAL,
    "JGB": YieldConvention.SEMI_ANNUAL,
    "BTP": YieldConvention.SEMI_ANNUAL,
    "ACGB": YieldConvention.SEMI_ANNUAL,
    "NZGB": YieldConvention.SEMI_ANNUAL,
    "CGB_CA": YieldConvention.SEMI_ANNUAL,
    "KTB": YieldConvention.SEMI_ANNUAL,
    "GSEC": YieldConvention.SEMI_ANNUAL,
    "SGS": YieldConvention.SEMI_ANNUAL,
    "HKGB": YieldConvention.SEMI_ANNUAL,
    "SAGB": YieldConvention.SEMI_ANNUAL,
    "ROMGB": YieldConvention.SEMI_ANNUAL,
    # Annual compounding
    "BUND": YieldConvention.ANNUAL,
    "OAT": YieldConvention.ANNUAL,
    "BONO": YieldConvention.ANNUAL,
    "BGB": YieldConvention.ANNUAL,
    "DSL": YieldConvention.ANNUAL,
    "RAGB": YieldConvention.ANNUAL,
    "RFGB": YieldConvention.ANNUAL,
    "IRISH": YieldConvention.ANNUAL,
    "PGB": YieldConvention.ANNUAL,
    "GGB": YieldConvention.ANNUAL,
    "CONFED": YieldConvention.ANNUAL,
    "DGB": YieldConvention.ANNUAL,
    "SGB": YieldConvention.ANNUAL,
    "NGB": YieldConvention.ANNUAL,
    "POLGB": YieldConvention.ANNUAL,
    "CZGB": YieldConvention.ANNUAL,
    "HGB": YieldConvention.ANNUAL,
    "TES": YieldConvention.ANNUAL,
    "ILGB": YieldConvention.ANNUAL,
    # Continuous (BUS/252 markets)
    "NTN_F": YieldConvention.CONTINUOUS,
    "NTN_B": YieldConvention.CONTINUOUS,
    "LTN": YieldConvention.CONTINUOUS,
    # Semi-annual (EM)
    "MBONO": YieldConvention.SEMI_ANNUAL,
    "TURKGB": YieldConvention.SEMI_ANNUAL,
    "SAGB_SA": YieldConvention.SEMI_ANNUAL,
    "EGGB": YieldConvention.SEMI_ANNUAL,
    "NGGB": YieldConvention.SEMI_ANNUAL,
    "KEGB": YieldConvention.SEMI_ANNUAL,
    "BTP_CL": YieldConvention.SEMI_ANNUAL,
    "CGB": YieldConvention.SEMI_ANNUAL,
    "INDOGB": YieldConvention.SEMI_ANNUAL,
    "MGS": YieldConvention.SEMI_ANNUAL,
    "THAIGB": YieldConvention.SEMI_ANNUAL,
    "ADGB": YieldConvention.SEMI_ANNUAL,
    "QATGB": YieldConvention.SEMI_ANNUAL,
    # Quarterly
    "RPGB": YieldConvention.QUARTERLY,
    # Zero-coupon / T-Bill
    "USTBILL": YieldConvention.DISCOUNT,
    "UKTBILL": YieldConvention.DISCOUNT,
    "EURTBILL": YieldConvention.DISCOUNT,
    "CETES": YieldConvention.DISCOUNT,
}


def get_yield_convention(market_code: str) -> YieldConvention:
    """Get the street yield convention for a sovereign market.

    Args:
        market_code: e.g. "UST", "BUND", "NTN_F".

    Raises:
        ValueError: if market not found.
    """
    code = market_code.upper()
    conv = _MARKET_YIELD_CONVENTION.get(code)
    if conv is None:
        raise ValueError(f"No yield convention for {code!r}")
    return conv


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════


def _comp_freq(convention: YieldConvention) -> int:
    if convention == YieldConvention.SEMI_ANNUAL:
        return 2
    elif convention == YieldConvention.ANNUAL:
        return 1
    elif convention == YieldConvention.QUARTERLY:
        return 4
    raise ValueError(f"Not a compounding convention: {convention}")


def _price_from_compounded(
    ytm: float,
    coupon_rate: float,
    maturity_years: float,
    coupon_freq: int,
    comp_freq: int,
    face: float,
) -> float:
    """Price from periodically compounded yield.

    P = Σ (c/f × Face) / (1 + y/m)^(m×t_i) + Face / (1 + y/m)^(m×T)

    where m = compounding frequency, f = coupon frequency.
    """
    n_periods = max(int(round(maturity_years * coupon_freq)), 1)
    coupon = coupon_rate / coupon_freq * face

    pv = 0.0
    for i in range(1, n_periods + 1):
        t = i / coupon_freq
        df = (1.0 + ytm / comp_freq) ** (-comp_freq * t)
        pv += coupon * df

    # Principal
    df_mat = (1.0 + ytm / comp_freq) ** (-comp_freq * maturity_years)
    pv += face * df_mat

    return pv


def _price_from_continuous(
    ytm: float,
    coupon_rate: float,
    maturity_years: float,
    coupon_freq: int,
    face: float,
) -> float:
    """Price from continuously compounded yield."""
    n_periods = max(int(round(maturity_years * coupon_freq)), 1)
    coupon = coupon_rate / coupon_freq * face

    pv = 0.0
    for i in range(1, n_periods + 1):
        t = i / coupon_freq
        pv += coupon * math.exp(-ytm * t)

    pv += face * math.exp(-ytm * maturity_years)
    return pv


def _convert_zero_yield(
    ytm: float,
    from_conv: YieldConvention,
    to_conv: YieldConvention,
    t: float,
) -> float:
    """Exact yield conversion for zero-coupon instruments."""
    # Convert to continuous first
    if from_conv == YieldConvention.CONTINUOUS:
        r_cont = ytm
    elif from_conv == YieldConvention.SIMPLE:
        r_cont = math.log(1.0 + ytm * t) / t if t > 0 else 0.0
    elif from_conv == YieldConvention.DISCOUNT:
        p = 1.0 - ytm * t
        r_cont = -math.log(max(p, 1e-10)) / t if t > 0 else 0.0
    else:
        m = _comp_freq(from_conv)
        r_cont = m * math.log(1.0 + ytm / m)

    # Convert from continuous to target
    if to_conv == YieldConvention.CONTINUOUS:
        return r_cont
    elif to_conv == YieldConvention.SIMPLE:
        return (math.exp(r_cont * t) - 1.0) / t if t > 0 else 0.0
    elif to_conv == YieldConvention.DISCOUNT:
        return (1.0 - math.exp(-r_cont * t)) / t if t > 0 else 0.0
    else:
        m = _comp_freq(to_conv)
        return m * (math.exp(r_cont / m) - 1.0)
