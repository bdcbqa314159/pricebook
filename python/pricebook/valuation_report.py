"""Unified cross-department valuation report.

Single entry point that prices an instrument for ALL departments:
- Front Office: mid price, Greeks, carry
- Prudent Valuation: mid - AVA → prudent value
- XVA: CVA, FVA, MVA, KVA
- Regulatory: SA-CCR EAD, SIMM IM, FRTB capital
- Market Risk: VaR contribution, stress P&L

    from pricebook.valuation_report import valuation_report, ValuationReport

    report = valuation_report(instrument, curve, ...)
    report.mid_price       # Trading
    report.prudent_value   # Prudent valuation
    report.xva             # XVA desk
    report.capital         # Regulatory
    report.risk_summary    # Market risk

References:
    EBA (2014). RTS on prudent valuation.
    Gregory (2020). The xVA Challenge. Wiley, 4th ed.
    Basel Committee (2019). Minimum capital requirements for market risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.discount_curve import DiscountCurve


@dataclass
class TradingView:
    """Front office view: mid price + Greeks."""
    mid_price: float
    dv01: float
    cs01: float
    delta: float
    gamma: float
    vega: float
    theta: float
    carry_1d: float

    def to_dict(self) -> dict:
        return {
            "mid": self.mid_price, "dv01": self.dv01, "cs01": self.cs01,
            "delta": self.delta, "gamma": self.gamma, "vega": self.vega,
            "theta": self.theta, "carry_1d": self.carry_1d,
        }


@dataclass
class PrudentView:
    """Prudent valuation view: mid - AVA."""
    mid_price: float
    total_ava: float
    prudent_value: float
    ava_breakdown: dict

    def to_dict(self) -> dict:
        return {"mid": self.mid_price, "ava": self.total_ava,
                "prudent": self.prudent_value, "breakdown": self.ava_breakdown}


@dataclass
class XVAView:
    """XVA desk view: all valuation adjustments."""
    cva: float
    dva: float
    fva: float
    mva: float
    kva: float
    total_xva: float

    def to_dict(self) -> dict:
        return {"cva": self.cva, "dva": self.dva, "fva": self.fva,
                "mva": self.mva, "kva": self.kva, "total": self.total_xva}


@dataclass
class RegulatoryView:
    """Regulatory capital view."""
    ead: float            # SA-CCR exposure at default
    rwa: float            # risk-weighted assets
    capital: float        # 8% × RWA
    simm_im: float        # ISDA SIMM initial margin
    frtb_charge: float    # FRTB market risk capital

    def to_dict(self) -> dict:
        return {"ead": self.ead, "rwa": self.rwa, "capital": self.capital,
                "simm_im": self.simm_im, "frtb": self.frtb_charge}


@dataclass
class RiskView:
    """Market risk view: VaR contribution + stress."""
    var_contribution: float     # marginal VaR
    stress_worst: float         # worst stress scenario P&L
    stress_scenarios: dict[str, float]

    def to_dict(self) -> dict:
        return {"var": self.var_contribution, "stress_worst": self.stress_worst,
                "scenarios": self.stress_scenarios}


@dataclass
class ValuationReport:
    """Unified valuation report across all departments."""
    instrument_type: str
    reference_date: date
    notional: float
    trading: TradingView
    prudent: PrudentView
    xva: XVAView
    regulatory: RegulatoryView
    risk: RiskView

    def to_dict(self) -> dict:
        return {
            "type": self.instrument_type,
            "date": self.reference_date.isoformat(),
            "notional": self.notional,
            "trading": self.trading.to_dict(),
            "prudent": self.prudent.to_dict(),
            "xva": self.xva.to_dict(),
            "regulatory": self.regulatory.to_dict(),
            "risk": self.risk.to_dict(),
        }

    @property
    def mid_price(self) -> float:
        return self.trading.mid_price

    @property
    def prudent_value(self) -> float:
        return self.prudent.prudent_value

    @property
    def total_xva(self) -> float:
        return self.xva.total_xva

    @property
    def capital(self) -> float:
        return self.regulatory.capital


def valuation_report(
    instrument_type: str,
    mid_price: float,
    notional: float,
    reference_date: date,
    *,
    dv01: float = 0.0,
    cs01: float = 0.0,
    delta: float = 0.0,
    gamma: float = 0.0,
    vega: float = 0.0,
    theta: float = 0.0,
    carry_1d: float = 0.0,
    # Prudent valuation inputs
    bid_price: float | None = None,
    ask_price: float | None = None,
    model_prices: list[float] | None = None,
    asset_class: str = "bond_ig",
    illiquidity_bp: float = 0.0,
    maturity_years: float = 5.0,
    # XVA inputs
    cva: float = 0.0,
    dva: float = 0.0,
    fva: float = 0.0,
    mva: float = 0.0,
    kva: float = 0.0,
    # Regulatory inputs
    ead: float = 0.0,
    rwa: float = 0.0,
    capital_req: float = 0.0,
    simm_im: float = 0.0,
    frtb_charge: float = 0.0,
    # Risk inputs
    var_contribution: float = 0.0,
    stress_scenarios: dict[str, float] | None = None,
) -> ValuationReport:
    """Build a unified valuation report for an instrument.

    This is a composition layer — it takes pre-computed values from each
    department's engine and packages them into a single report.

    In a production system, each department computes its own values:
    - Trading: desk risk_metrics + carry
    - Prudent: prudent_valuation.compute_prudent_value()
    - XVA: xva.py or new_desk_xva.py
    - Regulatory: SA-CCR, SIMM, FRTB
    - Risk: incremental_var, stress

    This function assembles the output.
    """
    from pricebook.prudent_valuation import (
        market_price_uncertainty_ava, close_out_cost_ava,
        model_risk_ava, investing_funding_ava, compute_prudent_value,
    )

    # Trading view
    trading = TradingView(
        mid_price=mid_price, dv01=dv01, cs01=cs01,
        delta=delta, gamma=gamma, vega=vega, theta=theta,
        carry_1d=carry_1d,
    )

    # Prudent valuation
    mpu = None
    if bid_price is not None and ask_price is not None:
        mpu = market_price_uncertainty_ava(mid_price, bid_price, ask_price)
    coc = close_out_cost_ava(notional, asset_class)
    mr = model_risk_ava(model_prices) if model_prices and len(model_prices) >= 2 else None
    ifc = investing_funding_ava(notional, illiquidity_bp, maturity_years) if illiquidity_bp > 0 else None

    pv_report = compute_prudent_value(mid_price, mpu=mpu, coc=coc, mr=mr, ifc=ifc)
    prudent = PrudentView(
        mid_price=mid_price,
        total_ava=pv_report.total_ava_diversified,
        prudent_value=pv_report.prudent_value,
        ava_breakdown=pv_report.to_dict(),
    )

    # XVA view
    xva_view = XVAView(cva=cva, dva=dva, fva=fva, mva=mva, kva=kva,
                        total_xva=cva - dva + fva + mva + kva)

    # Regulatory view
    reg = RegulatoryView(ead=ead, rwa=rwa, capital=capital_req,
                          simm_im=simm_im, frtb_charge=frtb_charge)

    # Risk view
    scenarios = stress_scenarios or {}
    stress_worst = min(scenarios.values()) if scenarios else 0.0
    risk_view = RiskView(var_contribution=var_contribution,
                          stress_worst=stress_worst,
                          stress_scenarios=scenarios)

    return ValuationReport(
        instrument_type=instrument_type,
        reference_date=reference_date,
        notional=notional,
        trading=trading,
        prudent=prudent,
        xva=xva_view,
        regulatory=reg,
        risk=risk_view,
    )
