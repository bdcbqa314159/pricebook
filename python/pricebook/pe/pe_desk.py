"""Private equity trading desk: unified book for LP interests, LBO deals,
and portfolio company valuations.

9-component desk protocol for PE fund management and deal monitoring.

    from pricebook.pe.pe_desk import (
        pe_risk_metrics, PERiskMetrics,
        PEBook, PEBookEntry,
        pe_carry_decomposition, PECarryDecomposition,
        pe_daily_pnl, PEDailyPnL,
        pe_dashboard, PEDashboard,
        pe_stress_suite, PEStressResult,
        pe_capital, PECapitalResult,
        pe_hedge_recommendations, PEHedgeRecommendation,
        PELifecycle,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


# ═══════════════════════════════════════════════════════════════
# 1. Risk Metrics
# ═══════════════════════════════════════════════════════════════

@dataclass
class PERiskMetrics:
    """Risk metrics for a PE position."""
    nav: float
    irr: float
    tvpi: float
    dpi: float
    moic: float
    j_curve_trough: float
    unfunded_commitment: float
    product_type: str  # "fund", "lbo", "dcf"
    notional: float = 0.0

    def to_dict(self) -> dict:
        return vars(self)


def pe_risk_metrics(entry: PEBookEntry) -> PERiskMetrics:
    """Compute risk metrics for a PE position."""
    inst = entry.instrument
    product_type = entry.product_type

    if product_type == "fund":
        from pricebook.credit.fund_participation import FundParticipation
        m = inst.metrics()
        called = sum(frac for _, frac in inst.drawdown_schedule)
        unfunded = inst.commitment * max(1 - called, 0)
        return PERiskMetrics(
            nav=m.nav, irr=m.irr, tvpi=m.tvpi, dpi=m.dpi,
            moic=m.moic, j_curve_trough=m.j_curve_trough,
            unfunded_commitment=unfunded,
            product_type=product_type,
            notional=inst.commitment,
        )

    elif product_type == "lbo":
        from pricebook.pe.lbo import LBOModel
        result = inst.run()
        # Use middle exit analysis as representative
        mid_idx = len(result.exit_analyses) // 2
        ea = result.exit_analyses[mid_idx] if result.exit_analyses else None
        return PERiskMetrics(
            nav=ea.equity_value if ea else 0.0,
            irr=ea.equity_irr if ea else 0.0,
            tvpi=ea.moic if ea else 0.0,
            dpi=0.0,  # LBO deals are unrealised
            moic=ea.moic if ea else 0.0,
            j_curve_trough=0.0,
            unfunded_commitment=0.0,
            product_type=product_type,
            notional=inst.enterprise_value,
        )

    elif product_type == "dcf":
        from pricebook.pe.dcf import DCFModel
        result = inst.value()
        return PERiskMetrics(
            nav=result.ev_bridge.equity_value,
            irr=0.0, tvpi=0.0, dpi=0.0, moic=0.0,
            j_curve_trough=0.0, unfunded_commitment=0.0,
            product_type=product_type,
            notional=result.enterprise_value,
        )

    raise ValueError(f"Unknown product_type: {product_type}")


# ═══════════════════════════════════════════════════════════════
# 2. Book
# ═══════════════════════════════════════════════════════════════

@dataclass
class PEBookEntry:
    """Single PE position."""
    trade_id: str
    instrument: object
    product_type: str  # "fund", "lbo", "dcf"
    fund_manager: str = ""
    vintage_year: int = 0
    sector: str = ""
    geography: str = ""


class PEBook:
    """PE portfolio book."""

    def __init__(self, name: str = "pe_book"):
        self.name = name
        self._entries: list[PEBookEntry] = []

    def add(self, entry: PEBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[PEBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_commitment(self) -> float:
        total = 0.0
        for e in self._entries:
            if hasattr(e.instrument, 'commitment'):
                total += e.instrument.commitment
            elif hasattr(e.instrument, 'enterprise_value'):
                total += e.instrument.enterprise_value
        return total

    def by_vintage(self) -> dict[int, list[PEBookEntry]]:
        result: dict[int, list[PEBookEntry]] = {}
        for e in self._entries:
            vy = e.vintage_year
            if vy not in result:
                result[vy] = []
            result[vy].append(e)
        return result

    def by_manager(self) -> dict[str, list[PEBookEntry]]:
        result: dict[str, list[PEBookEntry]] = {}
        for e in self._entries:
            mgr = e.fund_manager or "Unknown"
            if mgr not in result:
                result[mgr] = []
            result[mgr].append(e)
        return result

    def by_sector(self) -> dict[str, list[PEBookEntry]]:
        result: dict[str, list[PEBookEntry]] = {}
        for e in self._entries:
            sec = e.sector or "Unknown"
            if sec not in result:
                result[sec] = []
            result[sec].append(e)
        return result


# ═══════════════════════════════════════════════════════════════
# 3. Carry Decomposition
# ═══════════════════════════════════════════════════════════════

@dataclass
class PECarryDecomposition:
    """PE carry/cost decomposition."""
    management_fee: float
    carried_interest: float
    distribution_income: float
    j_curve_drag: float
    net_carry: float

    def to_dict(self) -> dict:
        return vars(self)


def pe_carry_decomposition(entry: PEBookEntry) -> PECarryDecomposition:
    """Decompose PE position carry."""
    if entry.product_type == "fund":
        cfs = entry.instrument.project()
        mgmt_fee = sum(cf.management_fee for cf in cfs)
        carry = sum(cf.carried_interest for cf in cfs)
        distributions = sum(cf.distribution for cf in cfs)
        m = entry.instrument.metrics()
        j_drag = max(1.0 - m.j_curve_trough, 0.0) * entry.instrument.commitment
        net = distributions - mgmt_fee - carry
        return PECarryDecomposition(
            management_fee=mgmt_fee, carried_interest=carry,
            distribution_income=distributions, j_curve_drag=j_drag,
            net_carry=net,
        )

    # LBO/DCF: no periodic carry
    return PECarryDecomposition(
        management_fee=0, carried_interest=0,
        distribution_income=0, j_curve_drag=0, net_carry=0,
    )


# ═══════════════════════════════════════════════════════════════
# 4. Daily P&L
# ═══════════════════════════════════════════════════════════════

@dataclass
class PEDailyPnL:
    """PE daily P&L attribution."""
    date: date
    total: float
    nav_change: float
    fee_drag: float
    distribution: float
    unexplained: float

    def to_dict(self) -> dict:
        return vars(self)


def pe_daily_pnl(
    entry: PEBookEntry,
    nav_t0: float,
    nav_t1: float,
    pnl_date: date,
) -> PEDailyPnL:
    """Attribute daily P&L for a PE position."""
    nav_change = nav_t1 - nav_t0

    fee_drag = 0.0
    distribution = 0.0
    if entry.product_type == "fund" and hasattr(entry.instrument, 'mgmt_fee_rate'):
        # Daily fee accrual
        fee_drag = -entry.instrument.commitment * entry.instrument.mgmt_fee_rate / 365

    total = nav_change + fee_drag + distribution
    unexplained = total - (nav_change + fee_drag + distribution)

    return PEDailyPnL(
        date=pnl_date, total=total, nav_change=nav_change,
        fee_drag=fee_drag, distribution=distribution,
        unexplained=unexplained,
    )


# ═══════════════════════════════════════════════════════════════
# 5. Dashboard
# ═══════════════════════════════════════════════════════════════

@dataclass
class PEDashboard:
    """PE portfolio dashboard — morning meeting summary."""
    date: date
    n_positions: int
    total_commitment: float
    total_nav: float
    total_unfunded: float
    weighted_irr: float
    weighted_tvpi: float
    by_vintage: dict[int, int]
    by_manager: dict[str, int]
    by_sector: dict[str, int]

    def to_dict(self) -> dict:
        return vars(self)


def pe_dashboard(
    book: PEBook,
    reference_date: date,
) -> PEDashboard:
    """Generate PE portfolio dashboard."""
    metrics_list = []
    total_nav = 0.0
    total_unfunded = 0.0
    total_commitment = book.total_commitment()

    for entry in book.entries:
        m = pe_risk_metrics(entry)
        metrics_list.append(m)
        total_nav += m.nav
        total_unfunded += m.unfunded_commitment

    # NAV-weighted IRR and TVPI
    weighted_irr = 0.0
    weighted_tvpi = 0.0
    if total_nav > 0:
        for m in metrics_list:
            w = m.nav / total_nav if total_nav > 0 else 0
            weighted_irr += w * m.irr
            weighted_tvpi += w * m.tvpi

    by_vintage = {vy: len(entries) for vy, entries in book.by_vintage().items()}
    by_manager = {mgr: len(entries) for mgr, entries in book.by_manager().items()}
    by_sector = {sec: len(entries) for sec, entries in book.by_sector().items()}

    return PEDashboard(
        date=reference_date, n_positions=len(book),
        total_commitment=total_commitment, total_nav=total_nav,
        total_unfunded=total_unfunded,
        weighted_irr=weighted_irr, weighted_tvpi=weighted_tvpi,
        by_vintage=by_vintage, by_manager=by_manager, by_sector=by_sector,
    )


# ═══════════════════════════════════════════════════════════════
# 6. Stress Suite
# ═══════════════════════════════════════════════════════════════

@dataclass
class PEStressResult:
    """PE stress scenario result."""
    scenario: str
    description: str
    nav_impact: float
    pnl: float

    def to_dict(self) -> dict:
        return vars(self)


def pe_stress_suite(book: PEBook) -> list[PEStressResult]:
    """Parametric stress scenarios for PE portfolio."""
    total_nav = sum(pe_risk_metrics(e).nav for e in book.entries)

    scenarios = [
        ("NAV -10%", "Moderate markdown", -0.10),
        ("NAV -25%", "Significant markdown", -0.25),
        ("NAV -50%", "Severe downturn", -0.50),
        ("NAV +10%", "Moderate mark-up", 0.10),
        ("NAV +25%", "Strong performance", 0.25),
    ]

    results = []
    for name, desc, shock in scenarios:
        impact = total_nav * shock
        results.append(PEStressResult(
            scenario=name, description=desc,
            nav_impact=impact, pnl=impact,
        ))
    return results


# ═══════════════════════════════════════════════════════════════
# 7. Capital
# ═══════════════════════════════════════════════════════════════

@dataclass
class PECapitalResult:
    """PE regulatory capital."""
    nav_exposure: float
    unfunded_exposure: float
    total_exposure: float
    risk_weight: float
    rwa: float
    capital: float

    def to_dict(self) -> dict:
        return vars(self)


def pe_capital(entry: PEBookEntry) -> PECapitalResult:
    """Compute PE regulatory capital (Basel equity investment framework).

    PE fund investments typically receive 250% or 400% risk weight
    depending on granularity and look-through.
    """
    m = pe_risk_metrics(entry)
    nav = m.nav
    unfunded = m.unfunded_commitment

    # CCF for unfunded = 100% (Basel treatment of undrawn commitments > 1yr)
    total_exposure = nav + unfunded
    risk_weight = 2.50  # 250% for PE equity (non-look-through)
    rwa = total_exposure * risk_weight
    capital = rwa * 0.08

    return PECapitalResult(
        nav_exposure=nav, unfunded_exposure=unfunded,
        total_exposure=total_exposure, risk_weight=risk_weight,
        rwa=rwa, capital=capital,
    )


# ═══════════════════════════════════════════════════════════════
# 8. Hedge Recommendations
# ═══════════════════════════════════════════════════════════════

@dataclass
class PEHedgeRecommendation:
    """PE risk mitigation recommendation."""
    risk_type: str
    current: float
    limit: float
    breach_pct: float
    action: str

    def to_dict(self) -> dict:
        return vars(self)


def pe_hedge_recommendations(
    book: PEBook,
    concentration_limit: float = 0.25,
    unfunded_limit: float = 0.50,
) -> list[PEHedgeRecommendation]:
    """Identify PE risk limit breaches and suggest hedges."""
    recommendations = []
    total_commitment = book.total_commitment()
    if total_commitment <= 0:
        return recommendations

    total_nav = sum(pe_risk_metrics(e).nav for e in book.entries)
    total_unfunded = sum(pe_risk_metrics(e).unfunded_commitment for e in book.entries)

    # Manager concentration
    for mgr, entries in book.by_manager().items():
        mgr_commitment = sum(
            e.instrument.commitment if hasattr(e.instrument, 'commitment')
            else getattr(e.instrument, 'enterprise_value', 0)
            for e in entries
        )
        pct = mgr_commitment / total_commitment
        if pct > concentration_limit:
            recommendations.append(PEHedgeRecommendation(
                risk_type="manager_concentration",
                current=pct, limit=concentration_limit,
                breach_pct=pct / concentration_limit,
                action=f"Reduce exposure to {mgr}: {pct:.0%} > {concentration_limit:.0%} limit. "
                       f"Consider secondary sale or GP-led continuation.",
            ))

    # Unfunded ratio
    if total_commitment > 0:
        unfunded_ratio = total_unfunded / total_commitment
        if unfunded_ratio > unfunded_limit:
            recommendations.append(PEHedgeRecommendation(
                risk_type="unfunded_ratio",
                current=unfunded_ratio, limit=unfunded_limit,
                breach_pct=unfunded_ratio / unfunded_limit,
                action=f"Unfunded commitments {unfunded_ratio:.0%} of total > {unfunded_limit:.0%} limit. "
                       f"Ensure liquidity reserves or credit facility.",
            ))

    return recommendations


# ═══════════════════════════════════════════════════════════════
# 9. Lifecycle
# ═══════════════════════════════════════════════════════════════

class PELifecycle:
    """PE position lifecycle management."""

    CAPITAL_CALL = "capital_call"
    DISTRIBUTION = "distribution"
    NAV_UPDATE = "nav_update"
    SECONDARY_SALE = "secondary_sale"
    GP_LED_CONTINUATION = "gp_led_continuation"
    FUND_MATURITY = "fund_maturity"

    def __init__(self, entry: PEBookEntry, creation_date: date | None = None):
        self._entry = entry
        self._events: list[dict] = []
        if creation_date:
            self._events.append({
                "type": "creation", "date": creation_date,
                "trade_id": entry.trade_id,
            })

    @property
    def history(self) -> list[dict]:
        return list(self._events)

    def record_event(self, event_type: str, event_date: date, **kwargs) -> dict:
        """Record a lifecycle event."""
        event = {"type": event_type, "date": event_date, **kwargs}
        self._events.append(event)
        return event

    def maturity_alert(self, as_of: date, alert_days: int = 365) -> dict | None:
        """Check if fund is approaching maturity."""
        if self._entry.product_type != "fund":
            return None

        inst = self._entry.instrument
        if hasattr(inst, 'fund_life_years') and hasattr(inst, 'vintage_year'):
            maturity_year = inst.vintage_year + inst.fund_life_years
            maturity_date = date(maturity_year, 12, 31)
            days_to_maturity = (maturity_date - as_of).days
            if 0 < days_to_maturity <= alert_days:
                return {
                    "type": "maturity_alert",
                    "trade_id": self._entry.trade_id,
                    "maturity_date": maturity_date,
                    "days_remaining": days_to_maturity,
                }
        return None

    def upcoming_events(self, as_of: date, horizon_days: int = 90) -> list[dict]:
        """List upcoming lifecycle events within horizon."""
        events = []
        alert = self.maturity_alert(as_of, horizon_days)
        if alert:
            events.append(alert)
        return events
