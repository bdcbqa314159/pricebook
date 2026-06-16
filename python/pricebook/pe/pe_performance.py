"""PE performance benchmarking: PME, direct alpha, vintage cohorts, GP economics.

    from pricebook.pe.pe_performance import (
        kaplan_schoar_pme, direct_alpha, long_nickels_pme,
        vintage_cohort, commitment_pacing, gp_economics,
    )

References:
    Kaplan & Schoar (2005). Private Equity Performance: Returns, Persistence, and Capital Flows.
    Long & Nickels (1996). A Private Investment Benchmark.
    Phalippou (2014). Performance of Buyout Funds Revisited.
    Giles (2008). ILPA Commitment Pacing Model.
    Metrick & Yasuda (2010). The Economics of Private Equity Funds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pricebook.credit.fund_participation import FundParticipation

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class PMEResult:
    """Public Market Equivalent result.

    PME > 1: fund outperformed the public market.
    PME < 1: fund underperformed.
    """
    pme_ratio: float        # Kaplan-Schoar PME
    direct_alpha: float     # IRR(fund) - IRR(index)
    long_nickels_pme: float # since-inception wealth ratio

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class VintageCohort:
    """Performance metrics aggregated by vintage year."""
    vintage_year: int
    n_funds: int
    median_irr: float
    mean_irr: float
    median_tvpi: float
    upper_quartile_irr: float
    lower_quartile_irr: float

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class PacingResult:
    """Commitment pacing model output for a single year."""
    year: int
    new_commitments: float
    expected_calls: float
    expected_distributions: float
    expected_nav: float
    net_cashflow: float  # distributions - calls

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class GPEconomics:
    """GP-level economics over fund life."""
    mgmt_fee_total: float
    mgmt_fee_npv: float
    carry_total: float
    carry_npv: float
    gp_commitment: float
    gp_commitment_return: float
    total_gp_revenue: float
    clawback_exposure: float

    def to_dict(self) -> dict:
        return dict(vars(self))


# ═══════════════════════════════════════════════════════════════
# PME (Kaplan-Schoar)
# ═══════════════════════════════════════════════════════════════

def kaplan_schoar_pme(
    contributions: list[float],
    distributions: list[float],
    index_returns: list[float],
    fund_nav: float = 0.0,
) -> PMEResult:
    """Kaplan-Schoar Public Market Equivalent.

    PME = PV(distributions + NAV, invested in index) / PV(contributions, invested in index)

    Each cashflow is "invested" in the public index from its date to the end.

    Args:
        contributions: capital calls per period (positive = LP pays in).
        distributions: distributions per period (positive = LP receives).
        index_returns: public market index returns per period.
        fund_nav: residual NAV at end of fund life.

    Returns:
        PMEResult with pme_ratio, direct_alpha, long_nickels_pme.
    """
    n = len(contributions)
    if n == 0:
        return PMEResult(pme_ratio=1.0, direct_alpha=0.0, long_nickels_pme=1.0)

    # Compute index growth factors from each period to end.
    # index_growth[i] = product of (1 + r_j) for j = i..n-1
    # So index_growth[0] = full compound, index_growth[n] = 1.0
    index_growth = np.ones(n + 1)
    for i in range(n - 1, -1, -1):
        r = index_returns[i] if i < len(index_returns) else 0.0
        index_growth[i] = index_growth[i + 1] * (1 + r)

    # FV of contributions at end: contribution at period i grows for (n-i) periods.
    # Cashflow at period i occurs at start of period i → grows by index_growth[i].
    fv_contributions = 0.0
    for i in range(n):
        fv_contributions += contributions[i] * index_growth[i]

    # FV of distributions at end: distribution at period i grows by index_growth[i].
    # Final NAV is already at end → no further growth.
    fv_distributions = fund_nav
    for i in range(n):
        fv_distributions += distributions[i] * index_growth[i]

    # Kaplan-Schoar PME
    pme = fv_distributions / fv_contributions if fv_contributions > 0 else 1.0

    # Direct alpha: fund IRR vs index IRR
    fund_cfs = [-contributions[i] + distributions[i] for i in range(n)]
    fund_cfs[-1] += fund_nav
    fund_irr = _pe_irr(fund_cfs)
    index_irr = _compound_return(index_returns)
    d_alpha = fund_irr - index_irr

    # Long-Nickels PME
    ln_pme = _long_nickels(contributions, distributions, index_returns, fund_nav)

    return PMEResult(pme_ratio=pme, direct_alpha=d_alpha, long_nickels_pme=ln_pme)


def direct_alpha(fund_irr: float, index_irr: float) -> float:
    """Direct alpha: excess return of fund over public market.

    direct_alpha = fund_irr - index_irr
    """
    return fund_irr - index_irr


def long_nickels_pme(
    contributions: list[float],
    distributions: list[float],
    index_returns: list[float],
    fund_nav: float = 0.0,
) -> float:
    """Long-Nickels PME: since-inception wealth ratio.

    Invest each contribution in the index, compare to actual distributions + NAV.
    """
    return _long_nickels(contributions, distributions, index_returns, fund_nav)


# ═══════════════════════════════════════════════════════════════
# Vintage Cohort
# ═══════════════════════════════════════════════════════════════

def vintage_cohort(
    funds: list[FundParticipation],
) -> list[VintageCohort]:
    """Aggregate fund metrics by vintage year.

    Args:
        funds: list of FundParticipation objects.

    Returns:
        List of VintageCohort, one per vintage year.
    """
    by_vintage: dict[int, list] = {}
    for f in funds:
        m = f.metrics()
        vy = f.vintage_year
        if vy not in by_vintage:
            by_vintage[vy] = []
        by_vintage[vy].append(m)

    cohorts = []
    for vy in sorted(by_vintage.keys()):
        metrics_list = by_vintage[vy]
        irrs = sorted([m.irr for m in metrics_list])
        tvpis = [m.tvpi for m in metrics_list]
        n = len(irrs)

        median_irr = _median(irrs)
        mean_irr = sum(irrs) / n
        uq = float(np.percentile(irrs, 75)) if n >= 4 else irrs[-1]
        lq = float(np.percentile(irrs, 25)) if n >= 4 else irrs[0]

        cohorts.append(VintageCohort(
            vintage_year=vy, n_funds=n,
            median_irr=median_irr, mean_irr=mean_irr,
            median_tvpi=_median(sorted(tvpis)),
            upper_quartile_irr=uq, lower_quartile_irr=lq,
        ))
    return cohorts


# ═══════════════════════════════════════════════════════════════
# Commitment Pacing
# ═══════════════════════════════════════════════════════════════

def commitment_pacing(
    target_allocation: float,
    portfolio_value: float,
    existing_nav: float = 0.0,
    existing_unfunded: float = 0.0,
    horizon: int = 10,
    call_rate: float = 0.25,
    distribution_rate: float = 0.15,
    growth_rate: float = 0.10,
    portfolio_growth: float = 0.05,
) -> list[PacingResult]:
    """Model commitment pacing to reach/maintain target PE allocation.

    Simple deterministic model: NAV grows at `growth_rate`, new commitments
    are called at `call_rate` per year, distributions at `distribution_rate` of NAV.

    Args:
        target_allocation: target PE allocation as fraction of portfolio (e.g. 0.15).
        portfolio_value: total portfolio value.
        existing_nav: current PE NAV.
        existing_unfunded: current unfunded commitments.
        horizon: projection horizon in years.
        call_rate: fraction of unfunded called per year.
        distribution_rate: fraction of NAV distributed per year.
        growth_rate: annual NAV growth rate (net of distributions).
        portfolio_growth: annual total portfolio growth.

    Returns:
        List of PacingResult per year.
    """
    nav = existing_nav
    unfunded = existing_unfunded
    total_portfolio = portfolio_value
    results = []

    for yr in range(1, horizon + 1):
        # Target NAV
        target_nav = total_portfolio * target_allocation

        # New commitments needed.
        # Estimate how much of unfunded will convert to NAV:
        # after ~2 call cycles, (1-call_rate)^2 remains unfunded.
        expected_from_unfunded = unfunded * (1 - (1 - call_rate) ** 2)
        gap = max(target_nav - nav - expected_from_unfunded, 0.0)
        new_commitment = gap * 0.5  # conservative: commit half the gap

        # Calls from existing + new unfunded
        unfunded += new_commitment
        calls = unfunded * call_rate
        unfunded -= calls

        # Distributions
        distributions = nav * distribution_rate

        # NAV evolution: growth + calls - distributions
        nav = nav * (1 + growth_rate) + calls - distributions

        # Portfolio grows
        total_portfolio *= (1 + portfolio_growth)

        results.append(PacingResult(
            year=yr,
            new_commitments=new_commitment,
            expected_calls=calls,
            expected_distributions=distributions,
            expected_nav=nav,
            net_cashflow=distributions - calls,
        ))

    return results


# ═══════════════════════════════════════════════════════════════
# GP Economics
# ═══════════════════════════════════════════════════════════════

def gp_economics(
    fund_size: float,
    fund_life: int = 10,
    mgmt_fee_rate: float = 0.015,
    carry_rate: float = 0.20,
    hurdle_rate: float = 0.08,
    gp_commitment_pct: float = 0.02,
    gross_return: float = 0.15,
    discount_rate: float = 0.10,
    fee_basis: str = "committed",
    investment_period: int = 5,
) -> GPEconomics:
    """Model GP-level economics over fund life.

    Args:
        fund_size: total fund AUM.
        fund_life: fund life in years.
        mgmt_fee_rate: annual management fee rate.
        carry_rate: carried interest rate (typically 20%).
        hurdle_rate: preferred return before carry.
        gp_commitment_pct: GP's co-investment as % of fund.
        gross_return: expected gross annual portfolio return.
        discount_rate: rate to discount GP cashflows.
        fee_basis: "committed" (fee on full fund) or "invested" (fee on deployed).
        investment_period: years over which capital is deployed.
    """
    gp_commitment = fund_size * gp_commitment_pct

    # Management fees
    mgmt_fee_total = 0.0
    mgmt_fee_npv = 0.0
    invested = 0.0

    for yr in range(1, fund_life + 1):
        # Capital deployment (linear over investment period)
        if yr <= investment_period:
            invested += fund_size / investment_period

        if fee_basis == "committed":
            fee = fund_size * mgmt_fee_rate
        else:
            fee = invested * mgmt_fee_rate

        # Post-investment period: common to reduce fee basis
        if yr > investment_period and fee_basis == "committed":
            fee = invested * mgmt_fee_rate  # step-down

        mgmt_fee_total += fee
        mgmt_fee_npv += fee / (1 + discount_rate) ** yr

    # Carried interest
    # Simple model: fund grows at gross_return, carry on profits above hurdle
    terminal_nav = fund_size * (1 + gross_return) ** fund_life
    total_profit = terminal_nav - fund_size
    hurdle_return = fund_size * ((1 + hurdle_rate) ** fund_life - 1)

    carry_base = max(total_profit - hurdle_return, 0.0)
    carry_total = carry_base * carry_rate
    carry_npv = carry_total / (1 + discount_rate) ** fund_life

    # GP commitment return
    gp_terminal = gp_commitment * (1 + gross_return) ** fund_life
    gp_commitment_return = gp_terminal - gp_commitment

    # Clawback exposure: if deal-by-deal carry were distributed early and
    # remaining portfolio dropped to zero, how much would GP owe back?
    # In a whole-fund model this is the carry itself (worst-case exposure).
    # In practice: max(carry_distributed - entitled_on_realized, 0).
    clawback = carry_total  # worst-case: all carry could be clawed back

    total_revenue = mgmt_fee_total + carry_total + gp_commitment_return

    return GPEconomics(
        mgmt_fee_total=mgmt_fee_total,
        mgmt_fee_npv=mgmt_fee_npv,
        carry_total=carry_total,
        carry_npv=carry_npv,
        gp_commitment=gp_commitment,
        gp_commitment_return=gp_commitment_return,
        total_gp_revenue=total_revenue,
        clawback_exposure=clawback,
    )


def clawback_exposure(
    total_carry_distributed: float,
    entitled_carry: float,
    gp_commitment_return: float = 0.0,
) -> float:
    """Compute GP clawback exposure.

    Clawback = max(carry distributed - carry entitled, 0).
    In practice, clawback is limited to carry net of taxes paid.

    Args:
        total_carry_distributed: total carry already distributed to GP.
        entitled_carry: carry the GP is actually entitled to (whole-fund basis).
        gp_commitment_return: GP's own investment return (not subject to clawback).
    """
    return max(total_carry_distributed - entitled_carry, 0.0)


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════

def _pe_irr(cashflows: list[float]) -> float:
    """IRR via Newton-Raphson on periodic cashflows."""
    if not cashflows or all(abs(cf) < 1e-15 for cf in cashflows):
        return 0.0

    def npv(r: float) -> float:
        return sum(cf / (1 + r) ** i for i, cf in enumerate(cashflows))

    def npv_d(r: float) -> float:
        return sum(-i * cf / (1 + r) ** (i + 1) for i, cf in enumerate(cashflows))

    r = 0.10
    for _ in range(200):
        f = npv(r)
        fp = npv_d(r)
        if abs(fp) < 1e-15:
            break
        r -= f / fp
        r = max(min(r, 5.0), -0.99)
        if abs(f) < 1e-10:
            break
    return r


def _compound_return(returns: list[float]) -> float:
    """Annualised compound return from periodic returns."""
    if not returns:
        return 0.0
    product = 1.0
    for r in returns:
        product *= (1 + r)
    n = len(returns)
    return product ** (1.0 / n) - 1.0 if n > 0 else 0.0


def _long_nickels(
    contributions: list[float],
    distributions: list[float],
    index_returns: list[float],
    fund_nav: float,
) -> float:
    """Long-Nickels PME implementation.

    Simulates investing contributions in the index and withdrawing distributions.
    PME = (fund_value) / (index_portfolio_value) at end.
    """
    n = len(contributions)
    index_portfolio = 0.0

    for i in range(n):
        r = index_returns[i] if i < len(index_returns) else 0.0
        # Grow existing portfolio
        index_portfolio *= (1 + r)
        # Add contribution, subtract distribution
        index_portfolio += contributions[i] - distributions[i]
        index_portfolio = max(index_portfolio, 0.0)

    # Final index growth (if we had held)
    if index_portfolio <= 0:
        return float('inf') if fund_nav > 0 else 1.0

    return (fund_nav) / index_portfolio if index_portfolio > 0 else 1.0


def _median(sorted_values: list[float]) -> float:
    """Median of a sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return sorted_values[n // 2]
    return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
