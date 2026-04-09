"""Stress testing + IRRBB (Interest Rate Risk in Banking Book).

Macroeconomic stress testing framework with PD/LGD multipliers, market
loss estimation, integrated stress test runner. Plus IRRBB EVE/NII
analysis under standardised shock scenarios (SRP31).

    from pricebook.regulatory.stress_irrbb import (
        ScenarioType, MacroVariable, StressScenario, PortfolioData,
        STANDARD_SCENARIOS, calculate_pd_stress_multiplier,
        stress_credit_portfolio, stress_market_portfolio,
        run_integrated_stress_test,
        calculate_duration_gap, calculate_eve_impact, calculate_eve_all_scenarios,
        calculate_irrbb_capital, IRRBB_SHOCK_SCENARIOS,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Stress testing
# =============================================================================

class ScenarioType(Enum):
    BASELINE = "baseline"
    ADVERSE = "adverse"
    SEVERELY_ADVERSE = "severely_adverse"
    CUSTOM = "custom"


class MacroVariable(Enum):
    GDP_GROWTH = "gdp_growth"
    UNEMPLOYMENT = "unemployment"
    INFLATION = "inflation"
    INTEREST_RATE = "interest_rate"
    HOUSE_PRICES = "house_prices"
    EQUITY_PRICES = "equity_prices"
    CREDIT_SPREADS = "credit_spreads"
    FX_RATE = "fx_rate"


# Standard scenario parameters (EBA / Fed CCAR style)
STANDARD_SCENARIOS: dict[ScenarioType, dict[MacroVariable, float]] = {
    ScenarioType.BASELINE: {
        MacroVariable.GDP_GROWTH: 0.02,
        MacroVariable.UNEMPLOYMENT: 0.05,
        MacroVariable.INFLATION: 0.02,
        MacroVariable.INTEREST_RATE: 0.03,
        MacroVariable.HOUSE_PRICES: 0.03,
        MacroVariable.EQUITY_PRICES: 0.05,
        MacroVariable.CREDIT_SPREADS: 0.01,
        MacroVariable.FX_RATE: 0.0,
    },
    ScenarioType.ADVERSE: {
        MacroVariable.GDP_GROWTH: -0.02,
        MacroVariable.UNEMPLOYMENT: 0.08,
        MacroVariable.INFLATION: 0.00,
        MacroVariable.INTEREST_RATE: 0.01,
        MacroVariable.HOUSE_PRICES: -0.15,
        MacroVariable.EQUITY_PRICES: -0.25,
        MacroVariable.CREDIT_SPREADS: 0.03,
        MacroVariable.FX_RATE: -0.10,
    },
    ScenarioType.SEVERELY_ADVERSE: {
        MacroVariable.GDP_GROWTH: -0.05,
        MacroVariable.UNEMPLOYMENT: 0.12,
        MacroVariable.INFLATION: -0.01,
        MacroVariable.INTEREST_RATE: 0.00,
        MacroVariable.HOUSE_PRICES: -0.30,
        MacroVariable.EQUITY_PRICES: -0.50,
        MacroVariable.CREDIT_SPREADS: 0.06,
        MacroVariable.FX_RATE: -0.20,
    },
}


@dataclass
class StressScenario:
    """A multi-year stress test scenario."""
    name: str
    scenario_type: ScenarioType
    horizon_years: int
    macro_paths: dict[MacroVariable, list[float]]
    description: str = ""


@dataclass
class PortfolioData:
    """Portfolio inputs for stress testing."""
    credit_exposure: float
    credit_rwa: float
    average_pd: float
    average_lgd: float
    market_var: float = 0
    market_exposure: float = 0
    operational_bir: float = 0
    liquidity_hqla: float = 0
    liquidity_outflows: float = 0


# ---- Credit stress ----

def calculate_pd_stress_multiplier(
    gdp_growth: float,
    unemployment: float,
    house_prices: float,
) -> float:
    """PD stress multiplier from macro shocks.

    Each impact term is positive when conditions worsen:
    - GDP drop: (base_gdp - gdp_growth) × 5
    - Unemployment up: (unemployment - base_unemp) × 2
    - House prices drop: (base_hp - house_prices) × 1.5
    Multiplier = 1 + Σ positive impacts (floored at 1, capped at 10x).
    """
    base_gdp, base_unemp, base_hp = 0.02, 0.05, 0.03
    gdp_impact = (base_gdp - gdp_growth) * 5.0
    unemp_impact = (unemployment - base_unemp) * 2.0
    hp_impact = (base_hp - house_prices) * 1.5
    multiplier = 1.0 + max(gdp_impact + unemp_impact + hp_impact, 0)
    return max(1.0, min(multiplier, 10.0))


def calculate_lgd_stress_multiplier(
    house_prices: float,
    credit_spreads: float,
) -> float:
    """LGD stress multiplier from collateral and recovery shocks."""
    base_hp, base_spreads = 0.03, 0.01
    hp_impact = (base_hp - house_prices) * 0.8
    spread_impact = (credit_spreads - base_spreads) * 3.0
    multiplier = 1.0 + max(hp_impact + spread_impact, 0)
    return max(1.0, min(multiplier, 2.0))


def stress_credit_portfolio(
    portfolio: PortfolioData,
    scenario: StressScenario,
    year: int,
) -> dict:
    """Apply credit stress for one year of the scenario."""
    gdp = scenario.macro_paths[MacroVariable.GDP_GROWTH][year]
    unemp = scenario.macro_paths[MacroVariable.UNEMPLOYMENT][year]
    hp = scenario.macro_paths[MacroVariable.HOUSE_PRICES][year]
    spreads = scenario.macro_paths[MacroVariable.CREDIT_SPREADS][year]

    pd_mult = calculate_pd_stress_multiplier(gdp, unemp, hp)
    lgd_mult = calculate_lgd_stress_multiplier(hp, spreads)

    stressed_pd = min(portfolio.average_pd * pd_mult, 1.0)
    stressed_lgd = min(portfolio.average_lgd * lgd_mult, 1.0)

    baseline_el = portfolio.average_pd * portfolio.average_lgd * portfolio.credit_exposure
    stressed_el = stressed_pd * stressed_lgd * portfolio.credit_exposure

    rwa_mult = (stressed_pd / portfolio.average_pd) * math.sqrt(stressed_lgd / portfolio.average_lgd)
    stressed_rwa = portfolio.credit_rwa * rwa_mult

    return {
        "pd_multiplier": pd_mult, "lgd_multiplier": lgd_mult,
        "baseline_pd": portfolio.average_pd, "stressed_pd": stressed_pd,
        "baseline_lgd": portfolio.average_lgd, "stressed_lgd": stressed_lgd,
        "baseline_el": baseline_el, "stressed_el": stressed_el,
        "incremental_losses": stressed_el - baseline_el,
        "baseline_rwa": portfolio.credit_rwa, "stressed_rwa": stressed_rwa,
        "rwa_increase": stressed_rwa - portfolio.credit_rwa,
    }


def stress_market_portfolio(
    portfolio: PortfolioData,
    scenario: StressScenario,
    year: int,
) -> dict:
    """Apply market stress for one year."""
    equity = scenario.macro_paths[MacroVariable.EQUITY_PRICES][year]
    rates = scenario.macro_paths[MacroVariable.INTEREST_RATE][year]
    fx = scenario.macro_paths[MacroVariable.FX_RATE][year]

    eq_pnl = portfolio.market_exposure * 0.40 * equity
    rate_pnl = portfolio.market_exposure * 0.30 * (rates - 0.03) * -2  # duration ≈ 2
    fx_pnl = portfolio.market_exposure * 0.15 * fx

    total_loss = -(eq_pnl + rate_pnl + fx_pnl)
    vol_mult = 1.0 + abs(equity) + abs(fx)
    stressed_var = portfolio.market_var * vol_mult

    return {
        "equity_shock": equity, "rate_shock": rates, "fx_shock": fx,
        "equity_pnl": eq_pnl, "rate_pnl": rate_pnl, "fx_pnl": fx_pnl,
        "total_market_loss": max(0, total_loss),
        "baseline_var": portfolio.market_var, "stressed_var": stressed_var,
    }


def create_scenario_paths(
    scenario_type: ScenarioType,
    horizon_years: int = 3,
) -> StressScenario:
    """Create a stress scenario with year-by-year paths.

    For simplicity, applies the standard shock to all years (constant path).
    """
    base = STANDARD_SCENARIOS[scenario_type]
    macro_paths = {var: [val] * horizon_years for var, val in base.items()}
    return StressScenario(
        name=scenario_type.value,
        scenario_type=scenario_type,
        horizon_years=horizon_years,
        macro_paths=macro_paths,
    )


def run_integrated_stress_test(
    portfolio: PortfolioData,
    scenario: StressScenario,
) -> dict:
    """Run integrated multi-year stress test on a portfolio."""
    yearly_results = []
    cumulative_credit_loss = 0.0
    cumulative_market_loss = 0.0

    for year in range(scenario.horizon_years):
        credit = stress_credit_portfolio(portfolio, scenario, year)
        market = stress_market_portfolio(portfolio, scenario, year)
        cumulative_credit_loss += credit["incremental_losses"]
        cumulative_market_loss += market["total_market_loss"]
        yearly_results.append({
            "year": year + 1,
            "credit": credit, "market": market,
            "total_loss": credit["incremental_losses"] + market["total_market_loss"],
        })

    return {
        "scenario": scenario.name,
        "horizon_years": scenario.horizon_years,
        "yearly": yearly_results,
        "cumulative_credit_loss": cumulative_credit_loss,
        "cumulative_market_loss": cumulative_market_loss,
        "cumulative_total_loss": cumulative_credit_loss + cumulative_market_loss,
    }


# =============================================================================
# IRRBB (SRP31)
# =============================================================================

# Standardised IR shock scenarios (basis points)
IRRBB_SHOCK_SCENARIOS: dict[str, dict[str, int]] = {
    "parallel_up": {
        "USD": 200, "EUR": 200, "GBP": 250, "JPY": 100, "CHF": 100, "other": 200,
    },
    "parallel_down": {
        "USD": -200, "EUR": -200, "GBP": -250, "JPY": -100, "CHF": -100, "other": -200,
    },
    "steepener": {"short_shock": -100, "long_shock": 100},
    "flattener": {"short_shock": 100, "long_shock": -100},
    "short_up": {"shock": 300},
    "short_down": {"shock": -300},
}


def calculate_pv01(notional: float, duration: float) -> float:
    """PV01 ≈ notional × duration × 0.0001."""
    return notional * duration * 0.0001


def calculate_duration_gap(assets: list[dict], liabilities: list[dict]) -> dict:
    """Duration gap analysis for the banking book.

    assets/liabilities: list of {notional, duration}.
    """
    total_a = sum(a["notional"] for a in assets)
    total_l = sum(l["notional"] for l in liabilities)

    wad_a = sum(a["notional"] * a["duration"] for a in assets) / total_a if total_a > 0 else 0
    wad_l = sum(l["notional"] * l["duration"] for l in liabilities) / total_l if total_l > 0 else 0

    duration_gap = wad_a - (total_l / total_a) * wad_l if total_a > 0 else 0

    pv01_a = sum(calculate_pv01(a["notional"], a["duration"]) for a in assets)
    pv01_l = sum(calculate_pv01(l["notional"], l["duration"]) for l in liabilities)
    net_pv01 = pv01_a - pv01_l

    return {
        "total_assets": total_a, "total_liabilities": total_l,
        "equity": total_a - total_l,
        "wa_duration_assets": wad_a, "wa_duration_liabilities": wad_l,
        "duration_gap": duration_gap,
        "pv01_assets": pv01_a, "pv01_liabilities": pv01_l,
        "net_pv01": net_pv01,
    }


def calculate_eve_impact(gap_analysis: dict, rate_shock_bps: int = 200) -> dict:
    """ΔEVE from interest rate shock.

    ΔEVE ≈ -DurationGap × Equity × Δr
    """
    equity = gap_analysis["equity"]
    duration_gap = gap_analysis["duration_gap"]
    net_pv01 = gap_analysis["net_pv01"]

    rate_shock = rate_shock_bps / 10000
    eve_change = -duration_gap * equity * rate_shock
    eve_change_pv01 = -net_pv01 * rate_shock_bps  # negative because PV decreases when rates rise
    eve_change_pct = (eve_change / equity * 100) if equity > 0 else 0

    return {
        "equity": equity, "duration_gap": duration_gap,
        "rate_shock_bps": rate_shock_bps,
        "eve_change": eve_change,
        "eve_change_pv01": eve_change_pv01,
        "eve_change_pct": eve_change_pct,
        "new_equity": equity + eve_change,
    }


def calculate_eve_all_scenarios(
    assets: list[dict],
    liabilities: list[dict],
    currency: str = "USD",
) -> dict:
    """EVE impact under all standardised IRRBB shocks."""
    gap = calculate_duration_gap(assets, liabilities)
    results: dict = {"gap_analysis": gap, "scenarios": {}}

    for sc in ["parallel_up", "parallel_down"]:
        shock = IRRBB_SHOCK_SCENARIOS[sc].get(currency, IRRBB_SHOCK_SCENARIOS[sc]["other"])
        results["scenarios"][sc] = calculate_eve_impact(gap, shock)

    # Find worst-case (most negative)
    worst = min(results["scenarios"].items(), key=lambda x: x[1]["eve_change"])
    results["worst_scenario"] = worst[0]
    results["worst_eve_change"] = worst[1]["eve_change"]
    results["worst_eve_change_pct"] = worst[1]["eve_change_pct"]
    return results


def calculate_nii_sensitivity(
    assets_by_bucket: dict[str, float],
    liabilities_by_bucket: dict[str, float],
    rate_shock_bps: int = 200,
    horizon_years: int = 1,
) -> dict:
    """NII sensitivity from rate shock over horizon.

    ΔNII ≈ Net repricing gap × Δr × time_in_horizon
    """
    rate_shock = rate_shock_bps / 10000
    bucket_results = []
    total_nii_change = 0.0

    for bucket, asset_amt in assets_by_bucket.items():
        liab_amt = liabilities_by_bucket.get(bucket, 0)
        gap = asset_amt - liab_amt
        # Approximate fraction of horizon affected (simplified)
        nii_change = gap * rate_shock * horizon_years
        total_nii_change += nii_change
        bucket_results.append({
            "bucket": bucket, "assets": asset_amt, "liabilities": liab_amt,
            "gap": gap, "nii_change": nii_change,
        })

    return {
        "rate_shock_bps": rate_shock_bps,
        "horizon_years": horizon_years,
        "total_nii_change": total_nii_change,
        "buckets": bucket_results,
    }


def calculate_irrbb_capital(
    eve_result: dict,
    tier1_capital: float,
    threshold_pct: float = 0.15,
) -> dict:
    """IRRBB capital charge.

    SOT (Supervisory Outlier Test): bank is outlier if max EVE loss > 15% of Tier1.
    """
    worst_loss = abs(eve_result.get("worst_eve_change", 0))
    threshold = threshold_pct * tier1_capital
    is_outlier = worst_loss > threshold
    capital_charge = max(worst_loss - threshold, 0)

    return {
        "worst_eve_loss": worst_loss,
        "tier1_capital": tier1_capital,
        "threshold_pct": threshold_pct * 100,
        "threshold": threshold,
        "is_outlier": is_outlier,
        "capital_charge": capital_charge,
    }
