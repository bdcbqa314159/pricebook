"""CLO equity Monte Carlo: IRR distribution, loss analysis, warehouse risk.

    from pricebook.credit.clo_equity import CLOEquityMC, warehouse_risk

    mc = CLOEquityMC(waterfall=wf, n_loans=200, default_prob=0.02)
    result = mc.simulate(n_paths=1000)
    print(result.equity_irr_mean)

References:
    Cordell, Roberts & Schwert (2023). CLO Performance. JFE.
    Lucas, Goodman & Fabozzi (2006). Collateralized Debt Obligations.
    Moody's (2021). CLO Monitor: Methodology and Assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricebook.credit.clo import CLOWaterfall


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class CLOEquityCashflow:
    """Single-period CLO equity cashflow."""
    period: int
    interest_income: float
    default_losses: float
    recovery_proceeds: float
    tranche_interest_paid: float
    equity_distribution: float
    par_remaining: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CLOEquityResult:
    """Monte Carlo result for CLO equity analysis."""
    equity_irr_mean: float
    equity_irr_std: float
    equity_irr_percentiles: dict[int, float]
    equity_nav_mean: float
    equity_nav_std: float
    loss_rate_mean: float
    loss_rate_std: float
    n_paths: int
    mean_cashflows: list[CLOEquityCashflow]

    def to_dict(self) -> dict:
        return {
            "equity_irr_mean": self.equity_irr_mean,
            "equity_irr_std": self.equity_irr_std,
            "equity_irr_percentiles": self.equity_irr_percentiles,
            "equity_nav_mean": self.equity_nav_mean,
            "equity_nav_std": self.equity_nav_std,
            "loss_rate_mean": self.loss_rate_mean,
            "loss_rate_std": self.loss_rate_std,
            "n_paths": self.n_paths,
        }


@dataclass
class WarehouseRiskResult:
    """Warehouse (pre-CLO) mark-to-market risk."""
    expected_spread_income: float
    funding_cost: float
    net_carry: float
    mtm_var_95: float
    ramp_shortfall_prob: float

    def to_dict(self) -> dict:
        return vars(self)


# ═══════════════════════════════════════════════════════════════
# CLO Equity MC Engine
# ═══════════════════════════════════════════════════════════════

class CLOEquityMC:
    """Monte Carlo engine for CLO equity IRR and loss distribution.

    Simulates correlated defaults (Gaussian one-factor copula),
    recoveries, and prepayments through a CLOWaterfall.

    During reinvestment period: defaulted par is replaced at par.
    Post-reinvestment: portfolio amortises from defaults/prepayments.

    Known limitation: debt tranche notionals are held constant — prepayment
    proceeds are not applied via distribute_principal() to reduce debt
    outstanding. This makes terminal equity NAV conservative (understated)
    when prepayment rates are high. Enhancement: call
    waterfall.distribute_principal(prepay) each period post-reinvestment.

    Args:
        waterfall: CLOWaterfall instance defining the tranche structure.
        n_loans: number of loans in the reference portfolio.
        avg_spread: average portfolio spread (annual).
        avg_life: average loan life in years.
        default_prob: annual probability of default per loan.
        recovery_mean: mean recovery rate on defaults.
        recovery_vol: recovery rate volatility.
        correlation: asset correlation for one-factor Gaussian copula.
        reinvestment_years: reinvestment period length.
        deal_life: total deal life in years.
        periods_per_year: periods per year (4 = quarterly).
        prepay_cpr: annual conditional prepayment rate.
    """

    def __init__(
        self,
        waterfall: CLOWaterfall,
        n_loans: int = 200,
        avg_spread: float = 0.035,
        avg_life: float = 5.0,
        default_prob: float = 0.02,
        recovery_mean: float = 0.70,
        recovery_vol: float = 0.20,
        correlation: float = 0.20,
        reinvestment_years: float = 4.0,
        deal_life: float = 8.0,
        periods_per_year: int = 4,
        prepay_cpr: float = 0.20,
    ):
        self.waterfall = waterfall
        self.n_loans = n_loans
        self.avg_spread = avg_spread
        self.avg_life = avg_life
        self.default_prob = default_prob
        self.recovery_mean = recovery_mean
        self.recovery_vol = recovery_vol
        self.correlation = correlation
        self.reinvestment_years = reinvestment_years
        self.deal_life = deal_life
        self.periods_per_year = periods_per_year
        self.prepay_cpr = prepay_cpr
        self.n_periods = int(deal_life * periods_per_year)
        self.dt = 1.0 / periods_per_year

        # Per-period default probability
        self._period_pd = 1.0 - (1.0 - default_prob) ** self.dt
        # Per-period prepay rate (SMM from CPR)
        self._period_smm = 1.0 - (1.0 - prepay_cpr) ** self.dt

    def simulate(self, n_paths: int = 1000, seed: int = 42) -> CLOEquityResult:
        """Run MC simulation.

        Returns:
            CLOEquityResult with IRR distribution, loss stats, mean cashflows.
        """
        rng = np.random.default_rng(seed)
        eq_tranche = self.waterfall.equity_tranche
        if eq_tranche is None:
            raise ValueError("Waterfall has no equity tranche")

        equity_investment = eq_tranche.notional
        total_par = self.waterfall.total_notional

        irrs = []
        final_navs = []
        loss_rates = []
        all_cashflows = []

        for _ in range(n_paths):
            cfs, irr, final_nav, loss_rate = self._simulate_single_path(
                rng, equity_investment, total_par,
            )
            irrs.append(irr)
            final_navs.append(final_nav)
            loss_rates.append(loss_rate)
            all_cashflows.append(cfs)

        irrs_arr = np.array(irrs)
        navs_arr = np.array(final_navs)
        loss_arr = np.array(loss_rates)

        # Mean cashflows across paths
        mean_cfs = []
        for p in range(self.n_periods):
            mean_cfs.append(CLOEquityCashflow(
                period=p + 1,
                interest_income=float(np.mean([cfs[p]["income"] for cfs in all_cashflows])),
                default_losses=float(np.mean([cfs[p]["defaults"] for cfs in all_cashflows])),
                recovery_proceeds=float(np.mean([cfs[p]["recovery"] for cfs in all_cashflows])),
                tranche_interest_paid=float(np.mean([cfs[p]["tranche_int"] for cfs in all_cashflows])),
                equity_distribution=float(np.mean([cfs[p]["equity_dist"] for cfs in all_cashflows])),
                par_remaining=float(np.mean([cfs[p]["par"] for cfs in all_cashflows])),
            ))

        return CLOEquityResult(
            equity_irr_mean=float(np.mean(irrs_arr)),
            equity_irr_std=float(np.std(irrs_arr)),
            equity_irr_percentiles={
                5: float(np.percentile(irrs_arr, 5)),
                25: float(np.percentile(irrs_arr, 25)),
                50: float(np.percentile(irrs_arr, 50)),
                75: float(np.percentile(irrs_arr, 75)),
                95: float(np.percentile(irrs_arr, 95)),
            },
            equity_nav_mean=float(np.mean(navs_arr)),
            equity_nav_std=float(np.std(navs_arr)),
            loss_rate_mean=float(np.mean(loss_arr)),
            loss_rate_std=float(np.std(loss_arr)),
            n_paths=n_paths,
            mean_cashflows=mean_cfs,
        )

    def _simulate_single_path(
        self, rng, equity_investment: float, initial_par: float,
    ) -> tuple[list[dict], float, float, float]:
        """Single MC path: returns (period_data, equity_irr, final_nav, loss_rate)."""
        par = initial_par
        reinvest_end = int(self.reinvestment_years * self.periods_per_year)
        total_defaults = 0.0
        cumulative_net_losses = 0.0  # net losses absorbed by equity
        period_data = []
        equity_cfs = [-equity_investment]  # t=0: invest

        # Systematic factor for correlated defaults
        rho = self.correlation
        sqrt_rho = np.sqrt(rho) if rho > 0 else 0.0
        sqrt_1_rho = np.sqrt(1.0 - rho)
        market_factor = rng.standard_normal()

        for p in range(self.n_periods):
            # Default simulation: one-factor Gaussian copula
            # Each loan: Z_i = sqrt(rho)*M + sqrt(1-rho)*eps_i
            # Default if Phi(Z_i) < PD
            idio = rng.standard_normal(self.n_loans)
            z = sqrt_rho * market_factor + sqrt_1_rho * idio
            default_indicators = norm.cdf(z) < self._period_pd
            n_defaults = int(np.sum(default_indicators))

            # Per-loan notional (equal weight)
            loan_notional = par / max(self.n_loans, 1)
            default_par = n_defaults * loan_notional

            # Recovery on defaults (per-loan recovery sampling)
            if default_par > 0 and n_defaults > 0:
                recovery_rates = np.clip(
                    rng.normal(self.recovery_mean, self.recovery_vol, n_defaults),
                    0.0, 1.0,
                )
                recovery = loan_notional * float(np.sum(recovery_rates))
            else:
                recovery = 0.0

            loss = default_par - recovery  # net loss after recovery
            total_defaults += default_par

            # Prepayments
            prepay = par * self._period_smm

            # Portfolio income
            income = par * self.avg_spread * self.dt

            # Reinvestment: replace defaulted + prepaid par during reinvestment period
            if p < reinvest_end:
                # Manager reinvests — par stays constant
                # But net losses still erode equity cushion
                cumulative_net_losses += loss
            else:
                # Post-reinvestment: par declines from defaults and prepayments
                par = par - default_par - prepay
                par = max(par, 0.0)

            # Waterfall: income net of losses
            net_income = income - loss

            if net_income > 0:
                payments = self.waterfall.distribute_interest(net_income, par)
                eq_name = self.waterfall.equity_tranche.name
                equity_dist = payments.get(eq_name, 0.0)
                tranche_int = sum(v for k, v in payments.items()
                                  if k != eq_name and k != "mgmt_fee" and k != "sub_mgmt_fee")
            else:
                equity_dist = 0.0
                tranche_int = 0.0
                payments = {}

            equity_cfs.append(max(equity_dist, 0.0))

            period_data.append({
                "income": income, "defaults": default_par,
                "recovery": recovery, "tranche_int": tranche_int,
                "equity_dist": equity_dist, "par": par,
            })

            # Resample market factor periodically (annual)
            if (p + 1) % self.periods_per_year == 0:
                market_factor = rng.standard_normal()

        # Terminal: equity gets residual par minus debt minus cumulative losses
        debt_outstanding = sum(t.notional for t in self.waterfall.debt_tranches)
        final_nav = max(par - debt_outstanding - cumulative_net_losses, 0.0)
        equity_cfs[-1] += final_nav

        irr = _mc_irr(equity_cfs, self.periods_per_year)
        loss_rate = total_defaults / initial_par if initial_par > 0 else 0.0

        return period_data, irr, final_nav, loss_rate


# ═══════════════════════════════════════════════════════════════
# Warehouse Risk
# ═══════════════════════════════════════════════════════════════

def warehouse_risk(
    pipeline_notional: float,
    avg_spread: float,
    funding_cost: float,
    ramp_months: float = 6.0,
    spread_vol: float = 0.01,
    target_par: float | None = None,
    n_sims: int = 1000,
    seed: int = 42,
) -> WarehouseRiskResult:
    """Warehouse risk during CLO ramp: spread MTM + ramp shortfall.

    Args:
        pipeline_notional: current warehouse notional.
        avg_spread: average spread on warehouse loans.
        funding_cost: warehouse funding rate.
        ramp_months: expected ramp-up period.
        spread_vol: annual spread volatility.
        target_par: target CLO collateral par (None = pipeline_notional).
        n_sims: number of spread simulations.
        seed: random seed.
    """
    if target_par is None:
        target_par = pipeline_notional

    rng = np.random.default_rng(seed)
    ramp_years = ramp_months / 12.0

    spread_income = pipeline_notional * avg_spread * ramp_years
    funding = pipeline_notional * funding_cost * ramp_years

    # Simulate spread changes → MTM impact
    spread_shocks = rng.normal(0, spread_vol * np.sqrt(ramp_years), n_sims)
    # Duration ≈ 4 years for leveraged loans
    duration = 4.0
    mtm_changes = -pipeline_notional * duration * spread_shocks
    var_95 = float(-np.percentile(mtm_changes, 5))  # 95% loss

    # Ramp shortfall: probability of not filling to target
    fill_rate = pipeline_notional / target_par if target_par > 0 else 1.0
    shortfall_prob = max(1.0 - fill_rate, 0.0)

    return WarehouseRiskResult(
        expected_spread_income=spread_income,
        funding_cost=funding,
        net_carry=spread_income - funding,
        mtm_var_95=var_95,
        ramp_shortfall_prob=shortfall_prob,
    )


# ═══════════════════════════════════════════════════════════════
# Internal
# ═══════════════════════════════════════════════════════════════

def _mc_irr(cashflows: list[float], periods_per_year: int) -> float:
    """IRR from periodic cashflows, annualised."""
    if not cashflows or all(abs(cf) < 1e-15 for cf in cashflows):
        return 0.0

    # Newton-Raphson on periodic NPV
    def npv(r: float) -> float:
        return sum(cf / (1 + r) ** i for i, cf in enumerate(cashflows))

    def npv_d(r: float) -> float:
        return sum(-i * cf / (1 + r) ** (i + 1) for i, cf in enumerate(cashflows))

    r = 0.02  # periodic guess
    for _ in range(200):
        f = npv(r)
        fp = npv_d(r)
        if abs(fp) < 1e-15:
            break
        r -= f / fp
        r = max(min(r, 0.50), -0.50)  # clamp periodic rate
        if abs(f) < 1e-10:
            break

    # Annualise: (1 + r_period)^periods_per_year - 1
    annual_irr = (1 + r) ** periods_per_year - 1
    # Clamp annualised IRR to reasonable range
    return max(min(annual_irr, 2.0), -1.0)
