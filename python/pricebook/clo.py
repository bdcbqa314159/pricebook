"""CLO structure, waterfall, compliance, and portfolio analytics.

Collateralised Loan Obligation: tranched securitisation of leveraged loans.
Senior-to-equity waterfall with OC/IC tests, diversity score, reinvestment.

    from pricebook.clo import (
        CLOTranche, CLOWaterfall,
        oc_ratio, ic_ratio, ccc_concentration,
        wal_test, warf_test,
        moody_diversity_score,
        break_even_default_rate,
        BorrowingBase,
    )

References:
    Moody's (2021). CLO Monitor: Methodology and Assumptions.
    S&P (2022). Global CLO Criteria.
    LSTA (2022). The Handbook of Loan Syndications and Trading, Ch. 21-22.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# CLO Tranche
# ---------------------------------------------------------------------------

@dataclass
class CLOTranche:
    """A single tranche in a CLO capital structure.

    Args:
        name: tranche identifier (e.g. "AAA", "AA", "A", "BBB", "BB", "Equity").
        notional: outstanding notional.
        coupon: spread over SOFR (e.g. 0.012 for 120bp).
        seniority: 1 = most senior, higher = more junior.
    """
    name: str
    notional: float
    coupon: float
    seniority: int

    @property
    def coupon_due(self) -> float:
        """Annual coupon payment due."""
        return self.notional * self.coupon

    def to_dict(self) -> dict:
        return {
            "name": self.name, "notional": self.notional,
            "coupon": self.coupon, "seniority": self.seniority,
        }


# ---------------------------------------------------------------------------
# CLO Waterfall
# ---------------------------------------------------------------------------

class CLOWaterfall:
    """CLO waterfall: distributes cashflows by seniority.

    Interest waterfall: fees → senior coupon → ... → equity residual.
    Principal waterfall: sequential (pay down most senior first) or pro-rata.

    Args:
        tranches: list of CLOTranche, will be sorted by seniority.
        mgmt_fee: annual management fee as fraction of assets.
        sub_mgmt_fee: subordinate management fee (paid after senior tranches).
    """

    def __init__(
        self,
        tranches: list[CLOTranche],
        mgmt_fee: float = 0.0015,
        sub_mgmt_fee: float = 0.0010,
    ):
        self.tranches = sorted(tranches, key=lambda t: t.seniority)
        self.mgmt_fee = mgmt_fee
        self.sub_mgmt_fee = sub_mgmt_fee

    @property
    def total_notional(self) -> float:
        return sum(t.notional for t in self.tranches)

    @property
    def debt_tranches(self) -> list[CLOTranche]:
        """All tranches except equity (highest seniority number)."""
        if not self.tranches:
            return []
        max_sen = max(t.seniority for t in self.tranches)
        return [t for t in self.tranches if t.seniority < max_sen]

    @property
    def equity_tranche(self) -> CLOTranche | None:
        if not self.tranches:
            return None
        max_sen = max(t.seniority for t in self.tranches)
        eq = [t for t in self.tranches if t.seniority == max_sen]
        return eq[0] if eq else None

    def distribute_interest(
        self,
        period_income: float,
        asset_balance: float,
    ) -> dict[str, float]:
        """Distribute interest income through the waterfall.

        Order: senior mgmt fee → tranche coupons (senior first) →
               sub mgmt fee → equity residual.

        Returns:
            dict mapping tranche name → payment received.
        """
        remaining = period_income
        payments: dict[str, float] = {}

        # Senior management fee
        senior_fee = asset_balance * self.mgmt_fee
        fee_paid = min(remaining, senior_fee)
        payments["mgmt_fee"] = fee_paid
        remaining -= fee_paid

        # Tranche coupons in seniority order
        for tranche in self.tranches:
            due = tranche.coupon_due
            paid = min(remaining, due)
            payments[tranche.name] = paid
            remaining -= paid

        # Subordinate management fee
        sub_fee = asset_balance * self.sub_mgmt_fee
        sub_paid = min(remaining, sub_fee)
        payments["sub_mgmt_fee"] = sub_paid
        remaining -= sub_paid

        # Any residual goes to equity
        eq = self.equity_tranche
        if eq and remaining > 0:
            payments[eq.name] = payments.get(eq.name, 0) + remaining

        return payments

    def distribute_principal(
        self,
        principal_proceeds: float,
        sequential: bool = True,
    ) -> dict[str, float]:
        """Distribute principal proceeds through the waterfall.

        Sequential: pay down most senior tranche first.
        Pro-rata: proportional to outstanding.

        Returns:
            dict mapping tranche name → principal received.
        """
        payments: dict[str, float] = {}
        remaining = principal_proceeds

        if sequential:
            for tranche in self.tranches:
                paydown = min(remaining, tranche.notional)
                payments[tranche.name] = paydown
                remaining -= paydown
                if remaining <= 0:
                    break
        else:
            # Pro-rata across debt tranches
            total = sum(t.notional for t in self.tranches)
            if total > 0:
                for tranche in self.tranches:
                    share = tranche.notional / total
                    paydown = min(principal_proceeds * share, tranche.notional)
                    payments[tranche.name] = paydown

        return payments

    def distribute(
        self,
        period_income: float,
        losses: float,
        recovery: float,
        asset_balance: float,
    ) -> dict[str, float]:
        """Full period distribution: absorb losses, then distribute income.

        Losses reduce equity first (reverse seniority).
        Recovery adds back to income.
        Remaining income distributed through interest waterfall.

        Returns:
            dict mapping tranche name → net payment.
        """
        payments: dict[str, float] = {t.name: 0.0 for t in self.tranches}

        # Losses absorbed bottom-up
        remaining_loss = losses - recovery
        if remaining_loss > 0:
            for tranche in reversed(self.tranches):
                absorbed = min(remaining_loss, tranche.notional)
                payments[tranche.name] -= absorbed
                remaining_loss -= absorbed
                if remaining_loss <= 0:
                    break

        # Distribute income
        interest_payments = self.distribute_interest(period_income, asset_balance)
        for name, amount in interest_payments.items():
            if name in payments:
                payments[name] += amount

        return payments

    def to_dict(self) -> dict:
        return {
            "tranches": [t.to_dict() for t in self.tranches],
            "mgmt_fee": self.mgmt_fee,
            "sub_mgmt_fee": self.sub_mgmt_fee,
        }


# ---------------------------------------------------------------------------
# Portfolio compliance tests
# ---------------------------------------------------------------------------

def oc_ratio(asset_par: float, tranche_notional: float) -> float:
    """Over-collateralisation ratio.

    OC = asset_par / tranche_notional_and_above

    OC > 1.0 means tranche is overcollateralised (passes).
    """
    if tranche_notional <= 0:
        return float("inf")
    return asset_par / tranche_notional


def ic_ratio(period_income: float, tranche_coupon: float) -> float:
    """Interest coverage ratio.

    IC = period_income / tranche_coupon_and_above

    IC > 1.0 means enough income to cover coupons (passes).
    """
    if tranche_coupon <= 0:
        return float("inf")
    return period_income / tranche_coupon


def ccc_concentration(ratings: list[str], notionals: list[float]) -> float:
    """CCC bucket concentration: % of portfolio rated CCC+ or below.

    Standard CLO limit: 7.5% of par. Excess treated as loss.
    """
    total = sum(notionals)
    if total <= 0:
        return 0.0
    ccc_bucket = {"CCC+", "CCC", "CCC-", "CC", "C", "D"}
    ccc_amount = sum(n for r, n in zip(ratings, notionals) if r in ccc_bucket)
    return ccc_amount / total


def wal_test(portfolio_wal: float, max_wal: float) -> bool:
    """WAL test: portfolio WAL must be below limit.

    Typical CLO limit: 5-7 years during reinvestment period.
    """
    return portfolio_wal <= max_wal


def warf_test(
    ratings: list[str],
    notionals: list[float],
    max_warf: float,
) -> bool:
    """Weighted Average Rating Factor test.

    WARF = Σ(notional_i × RF_i) / Σ(notional_i)

    Moody's rating factors (higher = worse credit).
    """
    rf = _rating_factor(ratings, notionals)
    return rf <= max_warf


def weighted_average_rating_factor(
    ratings: list[str],
    notionals: list[float],
) -> float:
    """Compute WARF for a portfolio."""
    return _rating_factor(ratings, notionals)


# Moody's rating factor table (simplified)
RATING_FACTORS: dict[str, int] = {
    "Aaa": 1, "Aa1": 10, "Aa2": 20, "Aa3": 40,
    "A1": 70, "A2": 120, "A3": 180,
    "Baa1": 260, "Baa2": 360, "Baa3": 610,
    "Ba1": 940, "Ba2": 1350, "Ba3": 1766,
    "B1": 2220, "B2": 2720, "B3": 3490,
    "Caa1": 4770, "Caa2": 6500, "Caa3": 8070,
    "Ca": 10000, "C": 10000,
    # S&P equivalents
    "AAA": 1, "AA+": 10, "AA": 20, "AA-": 40,
    "A+": 70, "A": 120, "A-": 180,
    "BBB+": 260, "BBB": 360, "BBB-": 610,
    "BB+": 940, "BB": 1350, "BB-": 1766,
    "B+": 2220, "B": 2720, "B-": 3490,
    "CCC+": 4770, "CCC": 6500, "CCC-": 8070,
    "CC": 10000, "C": 10000, "D": 10000,
}


def _rating_factor(ratings: list[str], notionals: list[float]) -> float:
    total = sum(notionals)
    if total <= 0:
        return 0.0
    warf = sum(
        n * RATING_FACTORS.get(r, 6500)  # default to CCC if unknown
        for r, n in zip(ratings, notionals)
    )
    return warf / total


# ---------------------------------------------------------------------------
# Moody's diversity score
# ---------------------------------------------------------------------------

# Moody's industry classification (33 industries, simplified)
MOODY_INDUSTRIES = {
    "aerospace": 1, "automotive": 2, "banking": 3, "beverages": 4,
    "broadcasting": 5, "building_materials": 6, "capital_equipment": 7,
    "chemicals": 8, "consumer_goods": 9, "containers": 10,
    "diversified": 11, "electronics": 12, "energy": 13, "environmental": 14,
    "finance": 15, "food": 16, "forest_products": 17, "gaming": 18,
    "healthcare": 19, "hotels": 20, "insurance": 21, "media": 22,
    "metals": 23, "mining": 24, "oil_gas": 25, "pharma": 26,
    "real_estate": 27, "retail": 28, "services": 29, "tech": 30,
    "telecom": 31, "transportation": 32, "utilities": 33,
}


def moody_diversity_score(
    industries: list[str],
    notionals: list[float],
) -> float:
    """Moody's diversity score: equivalent number of independent obligors.

    Groups by industry. Within each industry, maps N obligors to
    equivalent independent names using Moody's table.

    Simplified: diversity ≈ Σ_industry min(N_i, 1 + 0.5 × (N_i - 1))
    where N_i = number of equal-sized names in industry i.
    """
    if not industries or not notionals:
        return 0.0

    # Group by industry
    industry_groups: dict[str, list[float]] = {}
    for ind, notl in zip(industries, notionals):
        industry_groups.setdefault(ind, []).append(notl)

    total_score = 0.0
    for ind, group_notionals in industry_groups.items():
        n = len(group_notionals)
        if n == 0:
            continue
        # Moody's equivalent diversity within industry
        # For equal-sized: 1 name → 1.0, 2 → 1.5, 3 → 2.0, etc.
        # General: D_i = 1 + (n-1) × (1 - HHI_within) where HHI captures concentration
        total_group = sum(group_notionals)
        if total_group <= 0:
            continue
        hhi = sum((x / total_group) ** 2 for x in group_notionals)
        # Effective number of names = 1/HHI
        eff_n = 1.0 / hhi if hhi > 0 else n
        # Moody's mapping: each additional name adds 0.5 diversity
        industry_div = min(eff_n, 1.0 + 0.5 * (eff_n - 1.0))
        total_score += industry_div

    return total_score


# ---------------------------------------------------------------------------
# Reinvestment & break-even
# ---------------------------------------------------------------------------

def reinvestment_capacity(
    proceeds: float,
    avg_price: float = 99.0,
) -> float:
    """Par amount that can be purchased with reinvestment proceeds.

    capacity = proceeds / (avg_price / 100)
    """
    if avg_price <= 0:
        return 0.0
    return proceeds / (avg_price / 100.0)


def break_even_default_rate(
    equity_notional: float,
    total_assets: float,
    avg_recovery: float = 0.70,
) -> float:
    """Break-even default rate: annual default rate equity can absorb.

    BEDR = equity / (total_assets × (1 - recovery))

    Higher BEDR → more equity cushion → safer for debt tranches.
    """
    if total_assets <= 0 or avg_recovery >= 1.0:
        return 0.0
    lgd = 1.0 - avg_recovery
    return equity_notional / (total_assets * lgd)


# ---------------------------------------------------------------------------
# Borrowing base
# ---------------------------------------------------------------------------

@dataclass
class BorrowingBase:
    """Borrowing base for asset-backed revolving facilities.

    Available draw = advance_rate × eligible_receivables, capped by limit.

    Args:
        eligible_receivables: total eligible collateral.
        advance_rate: fraction advanced (e.g. 0.85 for 85%).
        facility_limit: maximum draw amount.
        concentration_limit: max % from single obligor.
    """
    eligible_receivables: float
    advance_rate: float = 0.85
    facility_limit: float = float("inf")
    concentration_limit: float = 0.10

    @property
    def gross_availability(self) -> float:
        """Advance rate × eligible receivables."""
        return self.eligible_receivables * self.advance_rate

    @property
    def available_draw(self) -> float:
        """Net available: min(gross_availability, facility_limit)."""
        return min(self.gross_availability, self.facility_limit)

    def concentration_excess(
        self,
        obligor_amounts: dict[str, float],
    ) -> dict[str, float]:
        """Find obligors exceeding concentration limit.

        Returns dict of obligor → excess amount.
        """
        total = sum(obligor_amounts.values())
        if total <= 0:
            return {}
        excess = {}
        limit = total * self.concentration_limit
        for name, amount in obligor_amounts.items():
            if amount > limit:
                excess[name] = amount - limit
        return excess

    def to_dict(self) -> dict:
        return {
            "eligible_receivables": self.eligible_receivables,
            "advance_rate": self.advance_rate,
            "facility_limit": self.facility_limit,
            "concentration_limit": self.concentration_limit,
        }
