"""Mandate compliance engine: configurable policy rules for buy-side portfolios.

    from pricebook.core.mandate import Mandate, check_mandate, investment_grade_mandate

References:
    CFA Institute (2014). Global Investment Performance Standards.
    UCITS Directive 2009/65/EC. Eligible assets and diversification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# Rating Utilities
# ═══════════════════════════════════════════════════════════════

RATING_ORDER: dict[str, int] = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4,
    "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10,
    "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16,
    "CCC+": 17, "CCC": 18, "CCC-": 19,
    "CC": 20, "C": 21, "D": 22,
    "NR": 99,
}


def rating_at_least(rating: str, floor: str) -> bool:
    """True if rating is at or above floor (e.g., 'A' >= 'BBB-')."""
    return RATING_ORDER.get(rating, 99) <= RATING_ORDER.get(floor, 99)


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class PortfolioHolding:
    """Flat representation of one holding for mandate checking."""
    trade_id: str
    asset_class: str = ""
    issuer: str = ""
    rating: str = "NR"
    sector: str = ""
    country: str = ""
    currency: str = "USD"
    weight_pct: float = 0.0
    notional: float = 0.0
    maturity_years: float = 0.0
    duration: float = 0.0
    issue_size: float = 0.0

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class MandateCheckResult:
    """Result of checking a single rule."""
    rule_type: str
    passed: bool
    description: str
    actual_value: object = None
    limit_value: object = None
    breach_details: str = ""

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class MandateReport:
    """Full mandate compliance report."""
    mandate_name: str
    reference_date: date
    n_rules: int
    n_passed: int
    n_failed: int
    results: list[MandateCheckResult]
    is_compliant: bool

    def to_dict(self) -> dict:
        return {
            "mandate_name": self.mandate_name,
            "reference_date": self.reference_date.isoformat(),
            "n_rules": self.n_rules, "n_passed": self.n_passed,
            "n_failed": self.n_failed, "is_compliant": self.is_compliant,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class Mandate:
    """Investment mandate specification."""
    name: str
    eligible_asset_classes: list[str] | None = None
    eligible_ratings: list[str] | None = None
    min_rating: str | None = None
    max_single_name_pct: float | None = None
    max_sector_pct: float | None = None
    max_country_pct: float | None = None
    max_leverage: float | None = None
    currency_restrictions: list[str] | None = None
    max_duration: float | None = None
    max_maturity_years: float | None = None
    min_issue_size: float | None = None

    def to_dict(self) -> dict:
        return vars(self)


# ═══════════════════════════════════════════════════════════════
# Predefined Templates
# ═══════════════════════════════════════════════════════════════

def investment_grade_mandate(name: str = "Investment Grade") -> Mandate:
    return Mandate(name=name, min_rating="BBB-", max_single_name_pct=0.05,
                   max_sector_pct=0.25, max_duration=10.0)

def sovereign_only_mandate(name: str = "Sovereign") -> Mandate:
    return Mandate(name=name, eligible_asset_classes=["govt_bond"],
                   min_rating="A-", max_single_name_pct=0.20)

def balanced_mandate(name: str = "Balanced") -> Mandate:
    return Mandate(name=name, min_rating="BB-", max_single_name_pct=0.10,
                   max_sector_pct=0.30, max_country_pct=0.40, max_duration=8.0)

def high_yield_mandate(name: str = "High Yield") -> Mandate:
    return Mandate(name=name, min_rating="CCC", max_single_name_pct=0.03,
                   max_sector_pct=0.20)


# ═══════════════════════════════════════════════════════════════
# Mandate Checking
# ═══════════════════════════════════════════════════════════════

def check_mandate(
    holdings: list[PortfolioHolding],
    mandate: Mandate,
    reference_date: date | None = None,
) -> MandateReport:
    """Check all mandate rules against portfolio holdings."""
    if reference_date is None:
        reference_date = date.today()

    results = []
    total_weight = sum(h.weight_pct for h in holdings)

    # 1. Asset class eligibility
    if mandate.eligible_asset_classes is not None:
        violations = [h for h in holdings if h.asset_class not in mandate.eligible_asset_classes]
        passed = len(violations) == 0
        results.append(MandateCheckResult(
            "asset_class", passed,
            f"Eligible: {mandate.eligible_asset_classes}",
            actual_value=[v.trade_id for v in violations[:5]],
            limit_value=mandate.eligible_asset_classes,
            breach_details=f"{len(violations)} ineligible holdings" if not passed else "",
        ))

    # 2. Minimum rating
    if mandate.min_rating is not None:
        violations = [h for h in holdings if not rating_at_least(h.rating, mandate.min_rating)]
        passed = len(violations) == 0
        results.append(MandateCheckResult(
            "min_rating", passed,
            f"Min rating: {mandate.min_rating}",
            actual_value=[f"{v.trade_id}({v.rating})" for v in violations[:5]],
            limit_value=mandate.min_rating,
            breach_details=f"{len(violations)} below floor" if not passed else "",
        ))

    # 3. Single name concentration
    if mandate.max_single_name_pct is not None and total_weight > 0:
        by_issuer = defaultdict(float)
        for h in holdings:
            by_issuer[h.issuer] += h.weight_pct
        worst = max(by_issuer.items(), key=lambda x: x[1]) if by_issuer else ("", 0)
        worst_pct = worst[1] / total_weight
        passed = worst_pct <= mandate.max_single_name_pct
        results.append(MandateCheckResult(
            "single_name", passed,
            f"Max single name: {mandate.max_single_name_pct:.0%}",
            actual_value=f"{worst[0]}: {worst_pct:.1%}",
            limit_value=mandate.max_single_name_pct,
            breach_details=f"{worst[0]} at {worst_pct:.1%}" if not passed else "",
        ))

    # 4. Sector concentration
    if mandate.max_sector_pct is not None and total_weight > 0:
        by_sector = defaultdict(float)
        for h in holdings:
            by_sector[h.sector] += h.weight_pct
        worst = max(by_sector.items(), key=lambda x: x[1]) if by_sector else ("", 0)
        worst_pct = worst[1] / total_weight
        passed = worst_pct <= mandate.max_sector_pct
        results.append(MandateCheckResult(
            "sector", passed,
            f"Max sector: {mandate.max_sector_pct:.0%}",
            actual_value=f"{worst[0]}: {worst_pct:.1%}",
            limit_value=mandate.max_sector_pct,
        ))

    # 5. Country concentration
    if mandate.max_country_pct is not None and total_weight > 0:
        by_country = defaultdict(float)
        for h in holdings:
            by_country[h.country] += h.weight_pct
        worst = max(by_country.items(), key=lambda x: x[1]) if by_country else ("", 0)
        worst_pct = worst[1] / total_weight
        passed = worst_pct <= mandate.max_country_pct
        results.append(MandateCheckResult(
            "country", passed,
            f"Max country: {mandate.max_country_pct:.0%}",
            actual_value=f"{worst[0]}: {worst_pct:.1%}",
            limit_value=mandate.max_country_pct,
        ))

    # 6. Currency restrictions
    if mandate.currency_restrictions is not None:
        violations = [h for h in holdings if h.currency not in mandate.currency_restrictions]
        passed = len(violations) == 0
        results.append(MandateCheckResult(
            "currency", passed,
            f"Allowed currencies: {mandate.currency_restrictions}",
            actual_value=[f"{v.trade_id}({v.currency})" for v in violations[:5]],
            limit_value=mandate.currency_restrictions,
        ))

    # 7. Duration limit
    if mandate.max_duration is not None:
        portfolio_duration = sum(h.weight_pct * h.duration for h in holdings) / total_weight if total_weight > 0 else 0
        passed = portfolio_duration <= mandate.max_duration
        results.append(MandateCheckResult(
            "duration", passed,
            f"Max portfolio duration: {mandate.max_duration}",
            actual_value=portfolio_duration,
            limit_value=mandate.max_duration,
        ))

    # 8. Maturity limit
    if mandate.max_maturity_years is not None:
        violations = [h for h in holdings if h.maturity_years > mandate.max_maturity_years]
        passed = len(violations) == 0
        results.append(MandateCheckResult(
            "maturity", passed,
            f"Max maturity: {mandate.max_maturity_years}Y",
            actual_value=len(violations),
            limit_value=mandate.max_maturity_years,
        ))

    # 9. Minimum issue size
    if mandate.min_issue_size is not None:
        violations = [h for h in holdings if h.issue_size > 0 and h.issue_size < mandate.min_issue_size]
        passed = len(violations) == 0
        results.append(MandateCheckResult(
            "issue_size", passed,
            f"Min issue size: {mandate.min_issue_size:,.0f}",
            actual_value=len(violations),
            limit_value=mandate.min_issue_size,
        ))

    n_passed = sum(1 for r in results if r.passed)
    n_failed = len(results) - n_passed

    return MandateReport(
        mandate_name=mandate.name,
        reference_date=reference_date,
        n_rules=len(results),
        n_passed=n_passed,
        n_failed=n_failed,
        results=results,
        is_compliant=n_failed == 0,
    )
