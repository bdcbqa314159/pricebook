"""Total capital aggregation + unified regulatory portfolio.

Aggregates RWA across all risk types (credit, securitisation, CCR, CVA,
market, operational), applies the output floor, and computes capital
ratios. The RegulatoryPortfolio class provides a single container that
runs VaR, IRC, and RWA calculations from a unified interface.

    from pricebook.regulatory.total_capital import (
        calculate_total_rwa, calculate_capital_ratios,
        RegulatoryPortfolio,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pricebook.regulatory.capital_framework import calculate_output_floor
from pricebook.regulatory.counterparty import calculate_sa_ccr_ead, calculate_ba_cva
from pricebook.regulatory.credit_rwa import (
    SAExposure, calculate_sa_rwa, calculate_irb_rwa, calculate_airb_rwa,
)
from pricebook.regulatory.irc import IRCPosition, IRCConfig, calculate_irc
from pricebook.regulatory.liquidity_op import calculate_sma_capital
from pricebook.regulatory.market_risk_sa import calculate_frtb_sa
from pricebook.regulatory.securitization import (
    calculate_sec_sa_rwa, calculate_sec_irba_rwa, calculate_erba_rwa,
)
from pricebook.regulatory.var_es import quick_var


# ---- Total RWA aggregation ----

def calculate_total_rwa(
    credit_exposures_sa: list[dict] | None = None,
    credit_exposures_irb: list[dict] | None = None,
    use_airb: bool = False,
    securitization_exposures: list[dict] | None = None,
    securitization_approach: str = "SEC-SA",
    derivative_trades: list[dict] | None = None,
    derivative_collateral: float = 0,
    cva_counterparties: list[dict] | None = None,
    trading_positions: dict | None = None,
    drc_positions: list[dict] | None = None,
    rrao_positions: list[dict] | None = None,
    business_indicator: float = 0,
    average_annual_loss: float = 0,
    apply_output_floor: bool = True,
    floor_year: int = 2028,
) -> dict:
    """Aggregate RWA across all risk types.

    Returns total RWA with breakdown by risk type and optional output floor.
    """
    credit_exposures_sa = credit_exposures_sa or []
    credit_exposures_irb = credit_exposures_irb or []
    securitization_exposures = securitization_exposures or []
    derivative_trades = derivative_trades or []
    cva_counterparties = cva_counterparties or []
    trading_positions = trading_positions or {}
    drc_positions = drc_positions or []
    rrao_positions = rrao_positions or []

    results: dict = {
        "credit_risk": {"sa": 0.0, "irb": 0.0, "approach": "SA"},
        "securitization": {"rwa": 0.0, "approach": securitization_approach},
        "counterparty_risk": {"ead": 0.0, "rwa": 0.0},
        "cva_risk": {"rwa": 0.0},
        "market_risk": {"rwa": 0.0},
        "operational_risk": {"rwa": 0.0},
    }

    # Credit risk
    credit_rwa_sa = 0.0
    for exp in credit_exposures_sa:
        sa_exp = SAExposure(
            ead=exp["ead"],
            asset_class=exp.get("asset_class", "corporate"),
            rating=exp.get("rating", "unrated"),
            ltv=exp.get("ltv"),
            is_sme=exp.get("is_sme", False),
            short_term=exp.get("short_term", False),
            income_producing=exp.get("income_producing", False),
        )
        credit_rwa_sa += calculate_sa_rwa(sa_exp)["rwa"]

    credit_rwa_irb = 0.0
    for exp in credit_exposures_irb:
        if use_airb:
            r = calculate_airb_rwa(
                exp["ead"], exp["pd"], exp["lgd"],
                exp.get("maturity", 2.5), exp.get("asset_class", "corporate"),
            )
        else:
            r = calculate_irb_rwa(
                exp["ead"], exp["pd"], exp.get("lgd", 0.45),
                exp.get("maturity", 2.5), exp.get("asset_class", "corporate"),
            )
        credit_rwa_irb += r["rwa"]

    results["credit_risk"]["sa"] = credit_rwa_sa
    results["credit_risk"]["irb"] = credit_rwa_irb
    if credit_exposures_irb:
        results["credit_risk"]["rwa"] = credit_rwa_irb
        results["credit_risk"]["approach"] = "A-IRB" if use_airb else "F-IRB"
    else:
        results["credit_risk"]["rwa"] = credit_rwa_sa
        results["credit_risk"]["approach"] = "SA-CR"

    # Securitisation
    sec_rwa = 0.0
    for exp in securitization_exposures:
        if securitization_approach == "SEC-IRBA":
            r = calculate_sec_irba_rwa(
                exp["ead"], exp["attachment"], exp["detachment"],
                kirb=exp.get("kirb", 0.06), n=exp.get("n", 25),
            )
        elif securitization_approach == "ERBA":
            r = calculate_erba_rwa(
                exp["ead"], exp.get("cqs", 5), exp.get("seniority", "senior"),
                maturity=exp.get("maturity", 5), thickness=exp.get("thickness"),
            )
        else:
            r = calculate_sec_sa_rwa(
                exp["ead"], exp["attachment"], exp["detachment"],
                ksa=exp.get("ksa", 0.08), n=exp.get("n", 25),
            )
        sec_rwa += r["rwa"]
    results["securitization"]["rwa"] = sec_rwa

    # CCR
    if derivative_trades:
        ccr = calculate_sa_ccr_ead(derivative_trades, collateral_held=derivative_collateral)
        results["counterparty_risk"]["ead"] = ccr["ead"]
        # Apply 100% RW (simplified — would normally use counterparty PD)
        results["counterparty_risk"]["rwa"] = ccr["ead"] * 1.0
        results["counterparty_risk"]["detail"] = ccr

    # CVA
    if cva_counterparties:
        cva = calculate_ba_cva(cva_counterparties)
        results["cva_risk"]["rwa"] = cva["rwa"]
        results["cva_risk"]["detail"] = cva

    # Market risk (FRTB-SA)
    if trading_positions or drc_positions or rrao_positions:
        frtb = calculate_frtb_sa(
            delta_positions=trading_positions,
            drc_positions=drc_positions,
            rrao_positions=rrao_positions,
        )
        results["market_risk"]["rwa"] = frtb["total_rwa"]
        results["market_risk"]["detail"] = frtb

    # Operational risk (SMA)
    if business_indicator > 0:
        sma = calculate_sma_capital(business_indicator, average_annual_loss)
        results["operational_risk"]["rwa"] = sma["rwa"]
        results["operational_risk"]["detail"] = sma

    # Total RWA (IRB and SA basis)
    other_rwa = (
        results["securitization"]["rwa"]
        + results["counterparty_risk"]["rwa"]
        + results["cva_risk"]["rwa"]
        + results["market_risk"]["rwa"]
        + results["operational_risk"]["rwa"]
    )
    total_rwa_irb = results["credit_risk"]["irb"] + other_rwa
    total_rwa_sa = results["credit_risk"]["sa"] + other_rwa

    results["total_rwa_irb_based"] = total_rwa_irb
    results["total_rwa_sa_based"] = total_rwa_sa

    # Output floor
    if apply_output_floor and credit_exposures_irb and credit_exposures_sa:
        floor = calculate_output_floor(total_rwa_irb, total_rwa_sa, floor_year)
        results["output_floor"] = floor
        results["total_rwa"] = floor["floored_rwa"]
    else:
        results["total_rwa"] = total_rwa_irb if credit_exposures_irb else total_rwa_sa

    return results


# ---- Capital ratios ----

def calculate_capital_ratios(
    total_rwa: float,
    cet1_capital: float,
    at1_capital: float = 0,
    tier2_capital: float = 0,
    countercyclical_buffer: float = 0,
    gsib_buffer: float = 0,
) -> dict:
    """Compute CET1, Tier1, total capital ratios with combined buffer requirements."""
    tier1 = cet1_capital + at1_capital
    total_capital = tier1 + tier2_capital

    cet1_ratio = cet1_capital / total_rwa if total_rwa > 0 else 0
    tier1_ratio = tier1 / total_rwa if total_rwa > 0 else 0
    total_ratio = total_capital / total_rwa if total_rwa > 0 else 0

    # Pillar 1 minimums
    min_cet1 = 0.045
    min_tier1 = 0.06
    min_total = 0.08

    # Capital conservation buffer
    conservation_buffer = 0.025

    total_buffer = conservation_buffer + countercyclical_buffer + gsib_buffer
    combined_cet1 = min_cet1 + total_buffer
    combined_tier1 = min_tier1 + total_buffer
    combined_total = min_total + total_buffer

    return {
        "cet1_capital": cet1_capital,
        "tier1_capital": tier1,
        "total_capital": total_capital,
        "total_rwa": total_rwa,
        "cet1_ratio_pct": cet1_ratio * 100,
        "tier1_ratio_pct": tier1_ratio * 100,
        "total_ratio_pct": total_ratio * 100,
        "requirements": {
            "min_cet1_pct": min_cet1 * 100,
            "min_tier1_pct": min_tier1 * 100,
            "min_total_pct": min_total * 100,
            "conservation_buffer_pct": conservation_buffer * 100,
            "countercyclical_buffer_pct": countercyclical_buffer * 100,
            "gsib_buffer_pct": gsib_buffer * 100,
            "combined_cet1_pct": combined_cet1 * 100,
            "combined_tier1_pct": combined_tier1 * 100,
            "combined_total_pct": combined_total * 100,
        },
        "compliance": {
            "cet1": cet1_ratio >= combined_cet1,
            "tier1": tier1_ratio >= combined_tier1,
            "total": total_ratio >= combined_total,
        },
        "surplus_pct": {
            "cet1": (cet1_ratio - combined_cet1) * 100,
            "tier1": (tier1_ratio - combined_tier1) * 100,
            "total": (total_ratio - combined_total) * 100,
        },
    }


# ---- Unified Regulatory Portfolio ----

@dataclass
class RegulatoryPosition:
    """A position in the regulatory portfolio."""
    position_id: str
    issuer: str
    notional: float
    rating: str = "BBB"
    tenor_years: float = 5.0
    seniority: str = "senior_unsecured"
    sector: str = "corporate"
    is_long: bool = True
    asset_class: str = "credit"
    coupon_rate: float = 0.05
    ccy: str = "USD"


class RegulatoryPortfolio:
    """Unified portfolio for VaR + IRC + RWA calculations.

    Provides a single container with methods for all the major
    regulatory metrics.

        port = RegulatoryPortfolio(name="Trading Book")
        port.add("Apple", notional=10_000_000, rating="AA", tenor_years=5)
        port.add("Microsoft", notional=15_000_000, rating="AAA", tenor_years=7)
        summary = port.risk_summary()
    """

    def __init__(self, name: str = "Portfolio", reference_ccy: str = "USD"):
        self.name = name
        self.reference_ccy = reference_ccy
        self.positions: list[RegulatoryPosition] = []
        self._counter = 0

    def add(
        self,
        issuer: str,
        notional: float,
        rating: str = "BBB",
        tenor_years: float = 5.0,
        seniority: str = "senior_unsecured",
        sector: str = "corporate",
        is_long: bool = True,
        asset_class: str = "credit",
        coupon_rate: float = 0.05,
        ccy: str | None = None,
        position_id: str | None = None,
    ) -> "RegulatoryPortfolio":
        """Add a position. Returns self for chaining."""
        self._counter += 1
        pid = position_id or f"pos_{self._counter}"
        self.positions.append(RegulatoryPosition(
            position_id=pid, issuer=issuer, notional=notional,
            rating=rating, tenor_years=tenor_years,
            seniority=seniority, sector=sector,
            is_long=is_long, asset_class=asset_class,
            coupon_rate=coupon_rate, ccy=ccy or self.reference_ccy,
        ))
        return self

    @property
    def total_notional(self) -> float:
        return sum(p.notional for p in self.positions)

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    @property
    def n_issuers(self) -> int:
        return len({p.issuer for p in self.positions})

    def to_irc_positions(self) -> list[IRCPosition]:
        """Convert to IRCPosition list for IRC calculation."""
        return [
            IRCPosition(
                position_id=p.position_id, issuer=p.issuer,
                notional=p.notional, market_value=p.notional,
                rating=p.rating, tenor_years=p.tenor_years,
                seniority=p.seniority, sector=p.sector,
                is_long=p.is_long, coupon_rate=p.coupon_rate,
            )
            for p in self.positions
        ]

    def to_credit_exposures(self) -> list[dict]:
        """Convert credit positions to SA exposures."""
        return [
            {
                "ead": p.notional,
                "asset_class": "corporate" if p.sector == "corporate" else p.sector,
                "rating": p.rating,
            }
            for p in self.positions if p.asset_class == "credit"
        ]

    def irc(self, num_simulations: int = 50_000, matrix: str = "global") -> dict:
        """Compute IRC for the portfolio."""
        config = IRCConfig(num_simulations=num_simulations, transition_matrix=matrix)
        return calculate_irc(self.to_irc_positions(), config)

    def var(self, returns: list | None = None, confidence: float = 0.99) -> dict:
        """Compute parametric VaR. If returns not given, uses synthetic."""
        if returns is None:
            import numpy as np
            rng = np.random.default_rng(42)
            returns = rng.normal(0.0003, 0.013, 252)
        return quick_var(returns, confidence=confidence,
                         position_value=self.total_notional)

    def credit_rwa(self, approach: str = "sa") -> dict:
        """Compute credit RWA across all positions."""
        if approach == "sa":
            exposures_sa = self.to_credit_exposures()
            return calculate_total_rwa(credit_exposures_sa=exposures_sa)
        else:
            exposures_irb = [
                {
                    "ead": p.notional,
                    "pd": _rating_to_pd(p.rating),
                    "lgd": 0.45,
                    "maturity": p.tenor_years,
                    "asset_class": "corporate",
                }
                for p in self.positions if p.asset_class == "credit"
            ]
            return calculate_total_rwa(
                credit_exposures_irb=exposures_irb,
                apply_output_floor=False,
            )

    def risk_summary(self) -> dict:
        """One-call risk summary: total notional, RWA, IRC."""
        rwa = self.credit_rwa("sa")
        return {
            "name": self.name,
            "n_positions": self.n_positions,
            "n_issuers": self.n_issuers,
            "total_notional": self.total_notional,
            "credit_rwa": rwa["credit_risk"]["rwa"],
            "total_rwa": rwa["total_rwa"],
            "irc": self.irc(num_simulations=10_000)["irc"],
        }


def _rating_to_pd(rating: str) -> float:
    """Quick rating-to-PD lookup (delegates to ratings module)."""
    from pricebook.regulatory.ratings import resolve_pd
    return resolve_pd(rating=rating)
