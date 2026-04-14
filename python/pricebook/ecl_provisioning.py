"""Expected Credit Loss provisioning: IFRS 9 / CECL framework.

Implements the three-stage impairment model for financial instruments:

* Stage 1: 12-month ECL (performing, no significant increase in risk).
* Stage 2: Lifetime ECL (significant increase in credit risk).
* Stage 3: Lifetime ECL (credit-impaired, interest on net carrying amount).

ECL = PD × LGD × EAD × df, summed over the relevant horizon.

* :func:`ecl_12_month` — stage 1 ECL.
* :func:`ecl_lifetime` — stage 2/3 lifetime ECL.
* :func:`stage_classification` — determine the IFRS 9 stage.
* :func:`ecl_portfolio` — portfolio-level ECL with macro scenarios.
* :class:`ECLResult` — structured ECL output.

References:
    IFRS 9, *Financial Instruments*, IASB, 2014, §5.5.
    CECL (ASC 326), FASB, 2016.
    EBA, *Guidelines on Credit Risk Management — ECL Estimation*, 2017.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Stage classification ----

@dataclass
class StageResult:
    """IFRS 9 stage classification."""
    stage: int          # 1, 2, or 3
    reason: str
    pd_origination: float
    pd_current: float
    pd_change: float


def stage_classification(
    pd_origination: float,
    pd_current: float,
    is_impaired: bool = False,
    relative_threshold: float = 2.0,
    absolute_threshold: float = 0.005,
    days_past_due: int = 0,
) -> StageResult:
    """Classify a loan into IFRS 9 stages.

    Stage 1: no significant increase in credit risk (SICR).
    Stage 2: SICR occurred but not credit-impaired.
    Stage 3: credit-impaired (specific evidence of loss).

    SICR triggers:
    - PD has increased by more than relative_threshold × PD_origination, OR
    - PD has increased by more than absolute_threshold in absolute terms, OR
    - Days past due > 30 (rebuttable presumption).

    Args:
        pd_origination: PD at initial recognition.
        pd_current: current PD.
        is_impaired: explicit impairment flag (stage 3).
        relative_threshold: relative PD increase for SICR (default 2×).
        absolute_threshold: absolute PD increase for SICR (default 50bp).
        days_past_due: number of days past due.
    """
    if is_impaired or days_past_due > 90:
        return StageResult(3, "credit-impaired", pd_origination, pd_current,
                           pd_current - pd_origination)

    pd_change = pd_current - pd_origination
    relative_increase = pd_current / pd_origination if pd_origination > 0 else float("inf")

    if (relative_increase > relative_threshold
            or pd_change > absolute_threshold
            or days_past_due > 30):
        return StageResult(2, "significant increase in credit risk",
                           pd_origination, pd_current, pd_change)

    return StageResult(1, "performing", pd_origination, pd_current, pd_change)


# ---- ECL computation ----

@dataclass
class ECLResult:
    """Expected credit loss result."""
    ecl: float
    stage: int
    pd_used: float
    lgd: float
    ead: float
    horizon: str  # "12-month" or "lifetime"
    n_periods: int


def ecl_12_month(
    pd_12m: float,
    lgd: float,
    ead: float,
    discount_rate: float = 0.0,
) -> ECLResult:
    """Stage 1: 12-month expected credit loss.

    ECL = PD_12m × LGD × EAD × df(1Y)

    Args:
        pd_12m: 12-month PD.
        lgd: loss given default (fraction).
        ead: exposure at default.
        discount_rate: effective interest rate for discounting.
    """
    df = math.exp(-discount_rate * 1.0) if discount_rate > 0 else 1.0
    ecl = pd_12m * lgd * ead * df

    return ECLResult(ecl, 1, pd_12m, lgd, ead, "12-month", 1)


def ecl_lifetime(
    marginal_pds: list[float],
    lgd: float,
    ead: float,
    discount_rate: float = 0.0,
    stage: int = 2,
) -> ECLResult:
    """Stage 2/3: lifetime expected credit loss.

    ECL = Σ_t marginal_PD(t) × LGD × EAD(t) × df(t)

    where marginal_PD(t) = Q(t-1) − Q(t) is the default probability
    in year t conditional on surviving to t-1.

    Args:
        marginal_pds: list of marginal (period) default probabilities.
        lgd: loss given default (can be time-varying in principle).
        ead: exposure at default (assumed constant; could amortise).
        discount_rate: effective interest rate.
        stage: 2 or 3.
    """
    ecl = 0.0
    n = len(marginal_pds)

    for t, pd_t in enumerate(marginal_pds, 1):
        df = math.exp(-discount_rate * t) if discount_rate > 0 else 1.0
        ecl += pd_t * lgd * ead * df

    return ECLResult(ecl, stage, sum(marginal_pds), lgd, ead, "lifetime", n)


def marginal_pds_from_cumulative(
    cumulative_pds: list[float],
) -> list[float]:
    """Convert cumulative PDs to marginal (period) PDs.

    marginal_PD(t) = cumulative_PD(t) − cumulative_PD(t−1).
    """
    marginals = [cumulative_pds[0]]
    for i in range(1, len(cumulative_pds)):
        marginals.append(max(cumulative_pds[i] - cumulative_pds[i - 1], 0.0))
    return marginals


def cumulative_pds_from_hazard(
    hazard_rate: float,
    n_years: int,
) -> list[float]:
    """Generate cumulative PDs from a constant hazard rate.

    cumulative_PD(t) = 1 − exp(−λt).
    """
    return [1 - math.exp(-hazard_rate * t) for t in range(1, n_years + 1)]


# ---- Portfolio ECL with macro scenarios ----

@dataclass
class ScenarioECL:
    """ECL under a single macro scenario."""
    scenario_name: str
    probability: float
    ecl: float
    weighted_ecl: float


@dataclass
class PortfolioECLResult:
    """Portfolio-level probability-weighted ECL."""
    total_ecl: float
    by_scenario: list[ScenarioECL]
    n_exposures: int


def ecl_portfolio(
    exposures: list[dict],
    scenarios: list[tuple[str, float, float]],
    lgd: float = 0.45,
    discount_rate: float = 0.0,
) -> PortfolioECLResult:
    """Compute probability-weighted portfolio ECL under macro scenarios.

    IFRS 9 requires forward-looking ECL using multiple economic
    scenarios, each with a probability weight.

    Args:
        exposures: list of dicts with keys:
            'ead': exposure at default,
            'pd_base': base-case PD,
            'stage': 1 or 2 or 3,
            'maturity_years': remaining maturity (for lifetime).
        scenarios: list of (name, probability, pd_multiplier) tuples.
            pd_multiplier scales the base PD (1.0 = base, 1.5 = stress).
        lgd: portfolio-level LGD.
        discount_rate: effective interest rate.

    Returns:
        :class:`PortfolioECLResult`.
    """
    scenario_results = []

    for name, prob, pd_mult in scenarios:
        total_ecl = 0.0

        for exp in exposures:
            pd_adj = min(exp["pd_base"] * pd_mult, 1.0)
            ead = exp["ead"]
            stage = exp.get("stage", 1)
            maturity = exp.get("maturity_years", 1)

            if stage == 1:
                result = ecl_12_month(pd_adj, lgd, ead, discount_rate)
            else:
                cum_pds = cumulative_pds_from_hazard(
                    -math.log(1 - pd_adj) if pd_adj < 1 else 10.0,
                    maturity,
                )
                marg_pds = marginal_pds_from_cumulative(cum_pds)
                result = ecl_lifetime(marg_pds, lgd, ead, discount_rate, stage)

            total_ecl += result.ecl

        scenario_results.append(ScenarioECL(name, prob, total_ecl, prob * total_ecl))

    portfolio_ecl = sum(s.weighted_ecl for s in scenario_results)

    return PortfolioECLResult(portfolio_ecl, scenario_results, len(exposures))
