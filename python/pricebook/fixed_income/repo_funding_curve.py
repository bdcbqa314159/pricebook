"""Multi-currency dealer funding curve.

Secured (GC repo) + unsecured (CP/deposit) funding rates, tiered by
maturity, with per-currency market conventions.

    from pricebook.fixed_income.repo_funding_curve import (
        build_dealer_funding_curve, DealerFundingCurve,
        RepoMarketConventions, get_repo_conventions,
    )

References:
    Choudhry (2010). The Repo Handbook, Ch 3-4.
    Duffie & Krishnamurthy (2016). Passthrough Efficiency in the Fed's
    New Monetary Policy Setting. Fed Jackson Hole Symposium.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve


@dataclass(frozen=True)
class RepoMarketConventions:
    """Per-currency repo market conventions."""
    currency: str
    day_count: DayCountConvention
    settlement_days: int         # T+N for GC
    benchmark_index: str         # reference overnight rate
    gc_collateral: list[str]     # eligible GC collateral types
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "currency": self.currency,
            "day_count": self.day_count.value,
            "settlement_days": self.settlement_days,
            "benchmark_index": self.benchmark_index,
            "gc_collateral": self.gc_collateral,
        }


_REPO_CONVENTIONS: dict[str, RepoMarketConventions] = {
    "USD": RepoMarketConventions("USD", DayCountConvention.ACT_360, 1, "SOFR",
                                  ["UST", "Agency", "Agency_MBS"]),
    "EUR": RepoMarketConventions("EUR", DayCountConvention.ACT_360, 0, "ESTR",
                                  ["Bund", "OAT", "BTP", "BONO", "DSL"]),
    "GBP": RepoMarketConventions("GBP", DayCountConvention.ACT_365_FIXED, 2, "SONIA",
                                  ["Gilt"]),
    "JPY": RepoMarketConventions("JPY", DayCountConvention.ACT_365_FIXED, 2, "TONA",
                                  ["JGB"]),
    "CHF": RepoMarketConventions("CHF", DayCountConvention.ACT_360, 0, "SARON",
                                  ["Confed", "Cantonal"]),
    "CAD": RepoMarketConventions("CAD", DayCountConvention.ACT_365_FIXED, 1, "CORRA",
                                  ["GoC"]),
    "AUD": RepoMarketConventions("AUD", DayCountConvention.ACT_365_FIXED, 1, "AONIA",
                                  ["ACGB"]),
    "BRL": RepoMarketConventions("BRL", DayCountConvention.BUS_252, 0, "CDI",
                                  ["NTN_F", "NTN_B", "LTN"]),
    "MXN": RepoMarketConventions("MXN", DayCountConvention.ACT_360, 1, "TIIE",
                                  ["MBONO", "CETES"]),
    "ZAR": RepoMarketConventions("ZAR", DayCountConvention.ACT_365_FIXED, 0, "JIBAR",
                                  ["SAGB"]),
    "TRY": RepoMarketConventions("TRY", DayCountConvention.ACT_365_FIXED, 0, "TLREF",
                                  ["TURKGB"]),
}


def get_repo_conventions(currency: str) -> RepoMarketConventions:
    """Get repo market conventions for a currency."""
    ccy = currency.upper()
    conv = _REPO_CONVENTIONS.get(ccy)
    if conv is None:
        available = sorted(_REPO_CONVENTIONS.keys())
        raise ValueError(f"No repo conventions for {ccy!r}. Available: {available}")
    return conv


def list_repo_currencies() -> list[str]:
    """Return currencies with repo conventions."""
    return sorted(_REPO_CONVENTIONS.keys())


@dataclass
class DealerFundingCurve:
    """Dealer funding curve: secured + unsecured legs."""
    currency: str
    reference_date: date
    secured_tenors_days: list[int]
    secured_rates: list[float]       # GC repo rates
    unsecured_tenors_days: list[int]
    unsecured_rates: list[float]     # CP / deposit rates
    day_count: DayCountConvention

    def secured_rate(self, days: int) -> float:
        """Interpolated secured (repo) rate at given tenor."""
        return float(np.interp(days, self.secured_tenors_days, self.secured_rates))

    def unsecured_rate(self, days: int) -> float:
        """Interpolated unsecured rate at given tenor."""
        return float(np.interp(days, self.unsecured_tenors_days, self.unsecured_rates))

    def funding_spread(self, days: int) -> float:
        """Unsecured - secured spread (the funding basis)."""
        return self.unsecured_rate(days) - self.secured_rate(days)

    def blended_rate(self, days: int, haircut: float) -> float:
        """Blended funding rate given a haircut.

        r_blend = (1 - h) × r_secured + h × r_unsecured

        The haircut portion is funded unsecured.
        """
        return (1 - haircut) * self.secured_rate(days) + haircut * self.unsecured_rate(days)

    def to_discount_curve(self, leg: str = "secured") -> DiscountCurve:
        """Convert to a DiscountCurve for integration with pricing."""
        tenors = self.secured_tenors_days if leg == "secured" else self.unsecured_tenors_days
        rates = self.secured_rates if leg == "secured" else self.unsecured_rates
        denom = 360.0 if self.day_count == DayCountConvention.ACT_360 else 365.0

        pillar_dates = []
        pillar_dfs = []
        from datetime import timedelta
        for days, rate in zip(tenors, rates):
            d = self.reference_date + timedelta(days=days)
            t = days / denom
            pillar_dates.append(d)
            pillar_dfs.append(1.0 / (1.0 + rate * t))

        return DiscountCurve(self.reference_date, pillar_dates, pillar_dfs)

    def to_dict(self) -> dict:
        return {
            "currency": self.currency,
            "reference_date": self.reference_date.isoformat(),
            "secured_tenors": self.secured_tenors_days,
            "secured_rates": self.secured_rates,
            "unsecured_tenors": self.unsecured_tenors_days,
            "unsecured_rates": self.unsecured_rates,
            "funding_spread_1m_bp": self.funding_spread(30) * 10_000,
            "funding_spread_3m_bp": self.funding_spread(90) * 10_000,
        }


def build_dealer_funding_curve(
    currency: str,
    reference_date: date,
    secured_quotes: dict[int, float],
    unsecured_quotes: dict[int, float],
) -> DealerFundingCurve:
    """Build a dealer funding curve from market quotes.

    Args:
        currency: ISO 3-letter.
        reference_date: valuation date.
        secured_quotes: {tenor_days: rate} for GC repo rates.
        unsecured_quotes: {tenor_days: rate} for CP/deposit rates.

    Returns:
        DealerFundingCurve with both legs.
    """
    conv = get_repo_conventions(currency)

    if not secured_quotes:
        raise ValueError("At least one secured quote required")
    if not unsecured_quotes:
        raise ValueError("At least one unsecured quote required")

    sec_tenors = sorted(secured_quotes.keys())
    sec_rates = [secured_quotes[t] for t in sec_tenors]
    unsec_tenors = sorted(unsecured_quotes.keys())
    unsec_rates = [unsecured_quotes[t] for t in unsec_tenors]

    return DealerFundingCurve(
        currency=currency.upper(),
        reference_date=reference_date,
        secured_tenors_days=sec_tenors,
        secured_rates=sec_rates,
        unsecured_tenors_days=unsec_tenors,
        unsecured_rates=unsec_rates,
        day_count=conv.day_count,
    )
