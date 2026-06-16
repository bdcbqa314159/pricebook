"""CCAR/DFAST: 9-quarter capital projection under stress scenarios.

    from pricebook.regulatory.ccar import (
        project_capital_trajectory, run_ccar_suite, CCARConfig,
    )

References:
    Federal Reserve (2023). Comprehensive Capital Analysis and Review.
    Federal Reserve (2022). Dodd-Frank Act Stress Tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pricebook.regulatory.stress_irrbb import (
    ScenarioType, MacroVariable, StressScenario, PortfolioData,
    create_scenario_paths, stress_credit_portfolio, stress_market_portfolio,
)


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class CCARConfig:
    """CCAR projection configuration."""
    n_quarters: int = 9
    starting_cet1: float = 0.0
    starting_at1: float = 0.0
    starting_tier2: float = 0.0
    starting_rwa: float = 0.0
    # Revenue
    quarterly_ppnr: float = 0.0
    ppnr_stress_factors: dict[str, float] = field(default_factory=lambda: {
        "baseline": 1.0, "adverse": 0.70, "severely_adverse": 0.40,
    })
    # Capital actions
    quarterly_dividend: float = 0.0
    quarterly_buyback: float = 0.0
    suspend_buyback_under_stress: bool = True
    # Thresholds (CET1 / Tier1 / Total / Leverage)
    cet1_minimum: float = 0.045
    tier1_minimum: float = 0.06
    total_minimum: float = 0.08
    # Stress loss parameters
    quarterly_op_loss_pct: float = 0.001  # operational loss as % of RWA

    def to_dict(self) -> dict:
        return {
            "n_quarters": self.n_quarters,
            "starting_cet1": self.starting_cet1,
            "starting_rwa": self.starting_rwa,
            "quarterly_ppnr": self.quarterly_ppnr,
            "quarterly_dividend": self.quarterly_dividend,
        }


@dataclass
class QuarterResult:
    """Single quarter in CCAR projection."""
    quarter: int
    ppnr: float
    credit_losses: float
    market_losses: float
    operational_losses: float
    total_losses: float
    net_income: float
    dividends: float
    buybacks: float
    capital_change: float
    cet1_capital: float
    rwa: float
    cet1_ratio: float
    is_breach: bool

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class CCARResult:
    """Full CCAR projection result."""
    scenario_name: str
    quarterly_results: list[QuarterResult]
    minimum_cet1_ratio: float
    minimum_quarter: int
    starting_cet1_ratio: float
    ending_cet1_ratio: float
    cumulative_losses: float
    cumulative_ppnr: float
    total_capital_actions: float
    passes_minimum: bool

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "minimum_cet1_ratio": self.minimum_cet1_ratio,
            "minimum_quarter": self.minimum_quarter,
            "starting_cet1_ratio": self.starting_cet1_ratio,
            "ending_cet1_ratio": self.ending_cet1_ratio,
            "cumulative_losses": self.cumulative_losses,
            "cumulative_ppnr": self.cumulative_ppnr,
            "passes_minimum": self.passes_minimum,
            "quarterly_results": [q.to_dict() for q in self.quarterly_results],
        }


# ═══════════════════════════════════════════════════════════════
# Core Functions
# ═══════════════════════════════════════════════════════════════

def project_capital_trajectory(
    config: CCARConfig,
    scenario: StressScenario,
    portfolio: PortfolioData,
) -> CCARResult:
    """Project CET1 capital over 9 quarters under a stress scenario.

    Each quarter:
    1. PPNR (scaled by scenario severity)
    2. Credit losses from stress_credit_portfolio()
    3. Market losses from stress_market_portfolio()
    4. Operational losses (flat % of RWA)
    5. Net income = PPNR - total losses
    6. Capital actions (dividends; buybacks suspended under stress)
    7. CET1 += net income - capital actions
    8. RWA adjustment from stressed PD/LGD
    9. Compute CET1 ratio, check minimum
    """
    cet1 = config.starting_cet1
    rwa = config.starting_rwa
    scenario_name = scenario.name

    # PPNR stress factor
    ppnr_factor = config.ppnr_stress_factors.get(scenario_name, 0.50)
    is_stress = scenario.scenario_type in (ScenarioType.ADVERSE, ScenarioType.SEVERELY_ADVERSE)

    starting_ratio = cet1 / rwa if rwa > 0 else 0.0
    quarterly_results = []
    cumulative_losses = 0.0
    cumulative_ppnr = 0.0
    total_actions = 0.0
    min_ratio = starting_ratio
    min_quarter = 0

    for q in range(1, config.n_quarters + 1):
        # Map quarter to year index for stress_credit/market
        year_idx = min((q - 1) // 4, scenario.horizon_years - 1)

        # 1. PPNR
        ppnr = config.quarterly_ppnr * ppnr_factor

        # 2. Credit losses (annualised → quarterly)
        credit_result = stress_credit_portfolio(portfolio, scenario, year_idx)
        credit_losses = credit_result["incremental_losses"] / 4.0

        # 3. Market losses (annualised → quarterly)
        market_result = stress_market_portfolio(portfolio, scenario, year_idx)
        market_losses = market_result["total_market_loss"] / 4.0

        # 4. Operational losses
        op_losses = rwa * config.quarterly_op_loss_pct

        total_losses = credit_losses + market_losses + op_losses
        cumulative_losses += total_losses

        # 5. Net income
        net_income = ppnr - total_losses
        cumulative_ppnr += ppnr

        # 6. Capital actions
        dividends = config.quarterly_dividend
        buybacks = config.quarterly_buyback
        if is_stress and config.suspend_buyback_under_stress:
            buybacks = 0.0
        total_actions += dividends + buybacks

        # 7. Capital change
        capital_change = net_income - dividends - buybacks
        cet1 += capital_change

        # 8. RWA adjustment
        rwa_mult = credit_result["stressed_rwa"] / credit_result["baseline_rwa"] \
            if credit_result["baseline_rwa"] > 0 else 1.0
        rwa = config.starting_rwa * rwa_mult

        # 9. Ratio
        cet1_ratio = cet1 / rwa if rwa > 0 else 0.0
        is_breach = cet1_ratio < config.cet1_minimum

        if cet1_ratio < min_ratio:
            min_ratio = cet1_ratio
            min_quarter = q

        quarterly_results.append(QuarterResult(
            quarter=q, ppnr=ppnr,
            credit_losses=credit_losses, market_losses=market_losses,
            operational_losses=op_losses, total_losses=total_losses,
            net_income=net_income, dividends=dividends, buybacks=buybacks,
            capital_change=capital_change, cet1_capital=cet1,
            rwa=rwa, cet1_ratio=cet1_ratio, is_breach=is_breach,
        ))

    ending_ratio = cet1 / rwa if rwa > 0 else 0.0

    return CCARResult(
        scenario_name=scenario_name,
        quarterly_results=quarterly_results,
        minimum_cet1_ratio=min_ratio,
        minimum_quarter=min_quarter,
        starting_cet1_ratio=starting_ratio,
        ending_cet1_ratio=ending_ratio,
        cumulative_losses=cumulative_losses,
        cumulative_ppnr=cumulative_ppnr,
        total_capital_actions=total_actions,
        passes_minimum=min_ratio >= config.cet1_minimum,
    )


def run_ccar_suite(
    config: CCARConfig,
    portfolio: PortfolioData,
) -> dict[str, CCARResult]:
    """Run all 3 standard scenarios: baseline, adverse, severely_adverse.

    Returns:
        dict mapping scenario name to CCARResult.
    """
    results = {}
    for st in (ScenarioType.BASELINE, ScenarioType.ADVERSE, ScenarioType.SEVERELY_ADVERSE):
        # Create 3-year paths, CCAR uses 9 quarters ≈ 2.25 years
        scenario = create_scenario_paths(st, horizon_years=3)
        result = project_capital_trajectory(config, scenario, portfolio)
        results[st.value] = result
    return results


def ccar_summary(results: dict[str, CCARResult]) -> dict:
    """Executive summary: worst scenario, trough ratios, pass/fail."""
    worst = min(results.values(), key=lambda r: r.minimum_cet1_ratio)
    return {
        "worst_scenario": worst.scenario_name,
        "worst_trough_ratio": worst.minimum_cet1_ratio,
        "worst_trough_quarter": worst.minimum_quarter,
        "all_pass": all(r.passes_minimum for r in results.values()),
        "scenarios": {name: {
            "min_ratio": r.minimum_cet1_ratio,
            "passes": r.passes_minimum,
            "cumulative_losses": r.cumulative_losses,
        } for name, r in results.items()},
    }
