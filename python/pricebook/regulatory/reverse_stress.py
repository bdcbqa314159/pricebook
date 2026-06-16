"""Reverse stress testing: find minimum-severity scenario that breaches a threshold.

    from pricebook.regulatory.reverse_stress import (
        reverse_stress_portfolio, reverse_stress_ccar, ReverseStressTarget,
    )

References:
    EBA (2018). Guidelines on stress testing (EBA/GL/2018/04).
    Basel Committee (2009). Principles for sound stress testing practices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from pricebook.regulatory.stress_irrbb import (
    ScenarioType, MacroVariable, StressScenario, PortfolioData,
    run_integrated_stress_test,
)
from pricebook.regulatory.ccar import (
    CCARConfig, project_capital_trajectory,
)


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class ReverseStressTarget:
    """What outcome to search for."""
    metric: str           # "cumulative_total_loss", "cet1_ratio", "lcr"
    threshold: float      # e.g. 0.045 for CET1 < 4.5%
    direction: str = "below"  # "below" = breach when metric < threshold

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class ReverseStressResult:
    """Result of reverse stress test."""
    target: ReverseStressTarget
    found: bool
    scenario_params: dict[str, float]
    scenario_severity: float  # L2 norm of shock vector
    metric_value: float
    breach_margin: float
    n_iterations: int

    def to_dict(self) -> dict:
        return {
            "found": self.found,
            "metric": self.target.metric,
            "threshold": self.target.threshold,
            "scenario_params": self.scenario_params,
            "severity": self.scenario_severity,
            "metric_value": self.metric_value,
            "breach_margin": self.breach_margin,
            "n_iterations": self.n_iterations,
        }


# ═══════════════════════════════════════════════════════════════
# Default Bounds
# ═══════════════════════════════════════════════════════════════

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    MacroVariable.GDP_GROWTH.value: (-0.10, 0.05),
    MacroVariable.UNEMPLOYMENT.value: (0.03, 0.20),
    MacroVariable.INFLATION.value: (-0.03, 0.10),
    MacroVariable.INTEREST_RATE.value: (-0.02, 0.10),
    MacroVariable.HOUSE_PRICES.value: (-0.50, 0.10),
    MacroVariable.EQUITY_PRICES.value: (-0.80, 0.20),
    MacroVariable.CREDIT_SPREADS.value: (0.0, 0.15),
    MacroVariable.FX_RATE.value: (-0.40, 0.10),
}

MACRO_VARS = [
    MacroVariable.GDP_GROWTH,
    MacroVariable.UNEMPLOYMENT,
    MacroVariable.INTEREST_RATE,
    MacroVariable.HOUSE_PRICES,
    MacroVariable.EQUITY_PRICES,
    MacroVariable.CREDIT_SPREADS,
    MacroVariable.FX_RATE,
]


# ═══════════════════════════════════════════════════════════════
# Core Functions
# ═══════════════════════════════════════════════════════════════

def reverse_stress_portfolio(
    portfolio: PortfolioData,
    target: ReverseStressTarget,
    bounds: dict[str, tuple[float, float]] | None = None,
    horizon_years: int = 3,
    max_iter: int = 200,
) -> ReverseStressResult:
    """Find minimum-severity scenario that breaches target.

    Objective: minimize ||shock_vector||₂
    Constraint: metric(shock_vector) breaches threshold

    Args:
        portfolio: portfolio data for stress testing.
        target: what to breach (metric, threshold, direction).
        bounds: per-variable bounds. Defaults to DEFAULT_BOUNDS.
        horizon_years: stress horizon.
        max_iter: maximum optimisation iterations.
    """
    bnds = bounds or DEFAULT_BOUNDS
    n_vars = len(MACRO_VARS)

    # Bounds for scipy
    scipy_bounds = [bnds.get(v.value, (-0.10, 0.10)) for v in MACRO_VARS]

    n_evals = [0]

    def _metric_from_params(x: np.ndarray) -> float:
        """Run stress test and extract metric."""
        macro_paths = {}
        for i, var in enumerate(MACRO_VARS):
            macro_paths[var] = [float(x[i])] * horizon_years

        # Add missing variables with baseline values
        if MacroVariable.INFLATION not in macro_paths:
            macro_paths[MacroVariable.INFLATION] = [0.02] * horizon_years

        scenario = StressScenario(
            name="reverse_stress",
            scenario_type=ScenarioType.CUSTOM,
            horizon_years=horizon_years,
            macro_paths=macro_paths,
        )
        result = run_integrated_stress_test(portfolio, scenario)
        n_evals[0] += 1

        if target.metric == "cumulative_total_loss":
            return result["cumulative_total_loss"]
        elif target.metric == "cumulative_credit_loss":
            return result["cumulative_credit_loss"]
        elif target.metric == "final_stressed_rwa":
            return result["yearly_results"][-1]["credit"]["stressed_rwa"]
        return result.get(target.metric, 0.0)

    def objective(x: np.ndarray) -> float:
        """Minimize severity (L2 norm) while breaching threshold."""
        return float(np.linalg.norm(x))

    def constraint(x: np.ndarray) -> float:
        """Constraint: metric must breach threshold.

        For "below": metric - threshold ≤ 0 (i.e., we want metric < threshold)
        Returns negative when breached.
        """
        metric_val = _metric_from_params(x)
        if target.direction == "below":
            return metric_val - target.threshold  # negative when breached
        return target.threshold - metric_val  # negative when metric > threshold

    # Starting point: midpoint of bounds
    x0 = np.array([(b[0] + b[1]) / 2 for b in scipy_bounds])

    try:
        result = minimize(
            objective, x0,
            method="SLSQP",
            bounds=scipy_bounds,
            constraints={"type": "ineq", "fun": lambda x: -constraint(x)},
            options={"maxiter": max_iter, "ftol": 1e-6},
        )

        # Extract final metric value
        final_metric = _metric_from_params(result.x)
        if target.direction == "below":
            found = final_metric < target.threshold
            breach_margin = target.threshold - final_metric
        else:
            found = final_metric > target.threshold
            breach_margin = final_metric - target.threshold

        params = {MACRO_VARS[i].value: float(result.x[i]) for i in range(n_vars)}

        return ReverseStressResult(
            target=target,
            found=found,
            scenario_params=params,
            scenario_severity=float(np.linalg.norm(result.x)),
            metric_value=final_metric,
            breach_margin=breach_margin,
            n_iterations=n_evals[0],
        )
    except Exception:
        return ReverseStressResult(
            target=target, found=False,
            scenario_params={}, scenario_severity=0.0,
            metric_value=0.0, breach_margin=0.0,
            n_iterations=n_evals[0],
        )


def reverse_stress_ccar(
    config: CCARConfig,
    portfolio: PortfolioData,
    target: ReverseStressTarget,
    bounds: dict[str, tuple[float, float]] | None = None,
    max_iter: int = 100,
) -> ReverseStressResult:
    """Reverse stress against CCAR capital trajectory.

    Finds mildest scenario where minimum CET1 ratio breaches threshold.
    """
    bnds = bounds or DEFAULT_BOUNDS
    n_vars = len(MACRO_VARS)
    scipy_bounds = [bnds.get(v.value, (-0.10, 0.10)) for v in MACRO_VARS]
    n_evals = [0]

    def _cet1_from_params(x: np.ndarray) -> float:
        macro_paths = {MACRO_VARS[i]: [float(x[i])] * 3 for i in range(n_vars)}
        if MacroVariable.INFLATION not in macro_paths:
            macro_paths[MacroVariable.INFLATION] = [0.02] * 3

        scenario = StressScenario(
            name="reverse_ccar",
            scenario_type=ScenarioType.CUSTOM,
            horizon_years=3,
            macro_paths=macro_paths,
        )
        result = project_capital_trajectory(config, scenario, portfolio)
        n_evals[0] += 1
        return result.minimum_cet1_ratio

    x0 = np.array([(b[0] + b[1]) / 2 for b in scipy_bounds])

    try:
        result = minimize(
            lambda x: float(np.linalg.norm(x)),
            x0,
            method="SLSQP",
            bounds=scipy_bounds,
            constraints={"type": "ineq", "fun": lambda x: target.threshold - _cet1_from_params(x)},
            options={"maxiter": max_iter, "ftol": 1e-6},
        )

        final_cet1 = _cet1_from_params(result.x)
        found = final_cet1 < target.threshold
        params = {MACRO_VARS[i].value: float(result.x[i]) for i in range(n_vars)}

        return ReverseStressResult(
            target=target, found=found,
            scenario_params=params,
            scenario_severity=float(np.linalg.norm(result.x)),
            metric_value=final_cet1,
            breach_margin=target.threshold - final_cet1,
            n_iterations=n_evals[0],
        )
    except Exception:
        return ReverseStressResult(
            target=target, found=False,
            scenario_params={}, scenario_severity=0.0,
            metric_value=0.0, breach_margin=0.0,
            n_iterations=n_evals[0],
        )


def scenario_surface(
    portfolio: PortfolioData,
    var1: MacroVariable,
    var2: MacroVariable,
    var1_range: tuple[float, float] = (-0.05, 0.05),
    var2_range: tuple[float, float] = (-0.30, 0.10),
    metric: str = "cumulative_total_loss",
    n_grid: int = 10,
    horizon_years: int = 3,
) -> dict:
    """2D grid of metric values across two macro variables.

    Returns:
        dict with "var1_values", "var2_values", "metric_grid" (2D list).
    """
    v1s = np.linspace(var1_range[0], var1_range[1], n_grid)
    v2s = np.linspace(var2_range[0], var2_range[1], n_grid)
    grid = []

    # Baseline values for other variables
    from pricebook.regulatory.stress_irrbb import STANDARD_SCENARIOS
    baseline = STANDARD_SCENARIOS[ScenarioType.BASELINE]

    for v1 in v1s:
        row = []
        for v2 in v2s:
            macro_paths = {var: [val] * horizon_years for var, val in baseline.items()}
            macro_paths[var1] = [float(v1)] * horizon_years
            macro_paths[var2] = [float(v2)] * horizon_years

            scenario = StressScenario(
                name="surface", scenario_type=ScenarioType.CUSTOM,
                horizon_years=horizon_years, macro_paths=macro_paths,
            )
            result = run_integrated_stress_test(portfolio, scenario)
            row.append(result.get(metric, 0.0))
        grid.append(row)

    return {
        "var1": var1.value, "var2": var2.value,
        "var1_values": v1s.tolist(), "var2_values": v2s.tolist(),
        "metric": metric, "metric_grid": grid,
    }
