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
from typing import TYPE_CHECKING

from pricebook.core.serialisable import serialisable_convention

if TYPE_CHECKING:
    from datetime import date
    from pricebook.core.discount_curve import DiscountCurve


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
        return dict(vars(self))


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


# ═══════════════════════════════════════════════════════════════
# Relative value analytics
# ═══════════════════════════════════════════════════════════════


@dataclass
class SupraRVResult:
    """Relative value analysis across supranational issuers."""
    issuer: str
    currency: str
    tenor_years: float
    yield_pct: float
    spread_vs_sovereign_bp: float
    spread_vs_ois_bp: float
    z_score: float              # vs historical average spread
    signal: str                 # "RICH", "CHEAP", "FAIR"
    peer_rank: int              # rank within peer group (1 = tightest)

    def to_dict(self) -> dict:
        return dict(vars(self))


def supranational_rv(
    issuer_code: str,
    currency: str,
    tenor_years: float,
    current_spread_bp: float,
    historical_mean_bp: float,
    historical_std_bp: float,
    ois_spread_bp: float | None = None,
    peer_spreads: dict[str, float] | None = None,
) -> SupraRVResult:
    """Relative value analysis for a supranational bond.

    Computes z-score vs historical spread and ranks against peers.

    Args:
        current_spread_bp: current spread over sovereign.
        historical_mean_bp: average spread over lookback.
        historical_std_bp: spread volatility over lookback.
        ois_spread_bp: spread over OIS (if available).
        peer_spreads: {issuer_code: spread_bp} for ranking.
    """
    issuer = get_supranational(issuer_code)

    z = (current_spread_bp - historical_mean_bp) / max(historical_std_bp, 0.1)
    if z < -1.0:
        signal = "RICH"
    elif z > 1.0:
        signal = "CHEAP"
    else:
        signal = "FAIR"

    # Peer ranking
    rank = 1
    if peer_spreads:
        sorted_peers = sorted(peer_spreads.items(), key=lambda x: x[1])
        for i, (code, sp) in enumerate(sorted_peers):
            if code.upper() == issuer_code.upper():
                rank = i + 1
                break

    return SupraRVResult(
        issuer=issuer_code, currency=currency,
        tenor_years=tenor_years,
        yield_pct=0.0,  # caller fills if needed
        spread_vs_sovereign_bp=current_spread_bp,
        spread_vs_ois_bp=ois_spread_bp or current_spread_bp,
        z_score=z, signal=signal, peer_rank=rank,
    )


@dataclass
class SupraUniverseResult:
    """Result of pricing a universe of supranational bonds."""
    bonds: list[SupranationalBondResult]
    n_issuers: int
    n_currencies: int
    average_spread_bp: float
    widest: str                 # issuer with widest spread
    tightest: str               # issuer with tightest spread

    def to_dict(self) -> dict:
        return {
            "n_issuers": self.n_issuers,
            "n_currencies": self.n_currencies,
            "average_spread_bp": self.average_spread_bp,
            "widest": self.widest,
            "tightest": self.tightest,
            "bonds": [b.to_dict() for b in self.bonds],
        }


def price_supranational_universe(
    issue_date,
    maturity,
    coupon_rate: float,
    discount_curve,
    issuers: list[str] | None = None,
    currencies: list[str] | None = None,
) -> SupraUniverseResult:
    """Price bonds across multiple supranational issuers and currencies.

    Creates and prices one bond per (issuer, currency) pair.

    Args:
        issuers: list of issuer codes (default: all registered).
        currencies: list of currencies to price in (default: ["USD", "EUR"]).
    """
    if issuers is None:
        issuers = list_supranationals()
    if currencies is None:
        currencies = ["USD", "EUR"]

    results = []
    for code in issuers:
        issuer = get_supranational(code)
        for ccy in currencies:
            if ccy.upper() not in [c.upper() for c in issuer.typical_currencies]:
                continue
            try:
                r = price_supranational(code, ccy, issue_date, maturity,
                                        coupon_rate, discount_curve)
                results.append(r)
            except Exception:
                continue

    if not results:
        return SupraUniverseResult([], 0, 0, 0.0, "", "")

    avg_spread = sum(r.spread_vs_sovereign_bp for r in results) / len(results)
    widest = max(results, key=lambda r: r.spread_vs_sovereign_bp)
    tightest = min(results, key=lambda r: r.spread_vs_sovereign_bp)

    unique_issuers = len(set(r.issuer for r in results))
    unique_currencies = len(set(r.currency for r in results))

    return SupraUniverseResult(
        bonds=results, n_issuers=unique_issuers,
        n_currencies=unique_currencies,
        average_spread_bp=avg_spread,
        widest=widest.issuer, tightest=tightest.issuer,
    )


def supranational_curve_spread(
    issuer_code: str,
    currency: str,
    discount_curve,
    tenors_years: list[int] | None = None,
    coupon_rate: float = 0.03,
    face_value: float = 100.0,
) -> list[dict]:
    """Compute spread term structure for a supranational issuer.

    Returns a list of {tenor, yield, spread_bp} dicts across tenors.
    """
    from datetime import date
    from dateutil.relativedelta import relativedelta

    if tenors_years is None:
        tenors_years = [2, 3, 5, 7, 10, 15, 20, 30]

    ref = discount_curve.reference_date
    results = []
    for t in tenors_years:
        mat = ref + relativedelta(years=t)
        issue = ref - relativedelta(months=6)
        try:
            r = price_supranational(issuer_code, currency, issue, mat,
                                    coupon_rate, discount_curve)
            results.append({
                "tenor_years": t,
                "yield_pct": r.yield_to_maturity * 100,
                "spread_bp": r.spread_vs_sovereign_bp,
                "clean_price": r.clean_price,
            })
        except Exception:
            continue

    return results
