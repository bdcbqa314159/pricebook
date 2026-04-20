"""Commodity real options: mine valuation, power plant dispatch, pipeline FTR.

* :class:`MineValuation` — Brennan-Schwartz natural resource investment.
* :func:`power_plant_dispatch_value` — spark-spread plant valuation.
* :func:`unit_commitment_value` — start-up/shut-down optimisation.
* :func:`pipeline_ftr_value` — Financial Transmission Rights valuation.

References:
    Brennan & Schwartz, *Evaluating Natural Resource Investments*,
    J. Business, 1985.
    Dixit & Pindyck, *Investment Under Uncertainty*, Princeton UP, 1994.
    Thompson, Davison & Rasmussen, *Valuation and Optimal Operation of
    Electric Power Plants in Competitive Markets*, Ops. Res., 2004.
    Hogan, *Contract Networks for Electric Power Transmission*,
    J. Reg. Econ., 1992 (FTR).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Mine valuation ----

@dataclass
class MineValuationResult:
    """Mine valuation result."""
    value: float
    reserves: float
    annual_production: float
    production_cost: float
    shutdown_value: float       # value if operated to exhaustion
    restart_option: float       # extra value from flexibility


class MineValuation:
    """Brennan-Schwartz (1985) mine valuation with output flexibility.

    Mine characteristics:
    - Initial reserves Q (units).
    - Annual production q (can be 0 if shut down).
    - Variable cost c per unit.
    - Fixed cost f per year when operating.
    - Shutdown / restart options.

    Commodity price follows a stochastic model (provided as MC paths).
    Optimal policy: operate when spot > marginal cost, shut down otherwise.

    Args:
        initial_reserves: total extractable resource.
        annual_production: production rate when operating.
        variable_cost: extraction cost per unit.
        fixed_cost: annual fixed cost.
        shutdown_cost: cost to shut down (one-off).
        restart_cost: cost to restart.
    """

    def __init__(
        self,
        initial_reserves: float,
        annual_production: float,
        variable_cost: float,
        fixed_cost: float = 0.0,
        shutdown_cost: float = 0.0,
        restart_cost: float = 0.0,
    ):
        self.initial_reserves = initial_reserves
        self.annual_production = annual_production
        self.variable_cost = variable_cost
        self.fixed_cost = fixed_cost
        self.shutdown_cost = shutdown_cost
        self.restart_cost = restart_cost

    def value(
        self,
        price_paths: np.ndarray,        # (n_paths, n_years+1) commodity price
        discount_factors: np.ndarray,   # (n_years+1,) DF per year
    ) -> MineValuationResult:
        """Value the mine with optimal operation (naive: operate when S > c)."""
        n_paths, n_times = price_paths.shape
        n_years = n_times - 1

        total_value = np.zeros(n_paths)
        total_shutdown_baseline = np.zeros(n_paths)
        reserves_remaining = np.full(n_paths, float(self.initial_reserves))

        for yr in range(n_years):
            price = price_paths[:, yr]
            df = discount_factors[min(yr + 1, len(discount_factors) - 1)]

            # Annual production capped at remaining reserves
            produce = np.minimum(self.annual_production, reserves_remaining)

            # Profit per unit if operating
            unit_margin = price - self.variable_cost
            # Operate if unit_margin × produce > fixed_cost
            operate = (unit_margin * produce > self.fixed_cost)
            # With option to shut down
            cash_flow_operate = unit_margin * produce - self.fixed_cost
            cash_flow = np.where(operate, cash_flow_operate, 0.0)

            total_value += cash_flow * df
            reserves_remaining -= produce * operate.astype(float)

            # Always-operate baseline for comparison
            total_shutdown_baseline += cash_flow_operate * df

        total = float(total_value.mean())
        baseline = float(total_shutdown_baseline.mean())
        restart_option_value = total - baseline

        return MineValuationResult(
            value=total,
            reserves=self.initial_reserves,
            annual_production=self.annual_production,
            production_cost=self.variable_cost,
            shutdown_value=baseline,
            restart_option=restart_option_value,
        )


# ---- Power plant dispatch ----

@dataclass
class PowerPlantResult:
    """Power plant dispatch valuation."""
    value: float
    mean_generation_hours: float
    heat_rate: float
    spark_spread_mean: float
    dispatch_ratio: float           # fraction of hours operating


def power_plant_dispatch_value(
    power_paths: np.ndarray,        # (n_paths, n_hours)
    gas_paths: np.ndarray,          # (n_paths, n_hours)
    heat_rate: float = 7.5,         # MMBtu/MWh
    variable_om: float = 5.0,       # $/MWh
    capacity_mw: float = 100.0,
    discount_factors: np.ndarray | None = None,
) -> PowerPlantResult:
    """Simplified power plant dispatch value (spark spread).

    Plant generates if spark_spread = P_power − heat_rate × P_gas − VOM > 0.

    Value = Σ hours of max(spark_spread, 0) × capacity.

    Args:
        power_paths: (n_paths, n_hours) power prices.
        gas_paths: (n_paths, n_hours) gas prices.
        heat_rate: gas burn per MWh.
        variable_om: variable operating cost.
        capacity_mw: plant capacity.
        discount_factors: per-hour DF (default: all 1).
    """
    n_paths, n_hours = power_paths.shape
    if discount_factors is None:
        discount_factors = np.ones(n_hours)

    spark = power_paths - heat_rate * gas_paths - variable_om
    operating = spark > 0

    hourly_value = np.where(operating, spark, 0.0) * capacity_mw
    discounted = hourly_value * discount_factors[np.newaxis, :]
    total = discounted.sum(axis=1)

    return PowerPlantResult(
        value=float(total.mean()),
        mean_generation_hours=float(operating.sum(axis=1).mean()),
        heat_rate=heat_rate,
        spark_spread_mean=float(spark.mean()),
        dispatch_ratio=float(operating.mean()),
    )


@dataclass
class UnitCommitmentResult:
    """Unit commitment with startup/shutdown costs."""
    value: float
    n_startups_mean: float
    n_hours_operating_mean: float
    startup_cost: float


def unit_commitment_value(
    power_paths: np.ndarray,
    gas_paths: np.ndarray,
    heat_rate: float = 7.5,
    variable_om: float = 5.0,
    capacity_mw: float = 100.0,
    startup_cost: float = 10_000.0,
    shutdown_cost: float = 2_000.0,
    min_up_hours: int = 4,
    min_down_hours: int = 4,
    discount_factors: np.ndarray | None = None,
) -> UnitCommitmentResult:
    """Simplified unit commitment with startup/shutdown costs & min up/down times.

    Greedy heuristic: start when spark > startup_cost/capacity over
    min_up_hours window; shutdown similarly. Production-grade would use MILP.

    Args:
        startup_cost: $ per start event.
        shutdown_cost: $ per shutdown event.
        min_up_hours: minimum operating hours once started.
        min_down_hours: minimum downtime once stopped.
    """
    n_paths, n_hours = power_paths.shape
    if discount_factors is None:
        discount_factors = np.ones(n_hours)

    spark = power_paths - heat_rate * gas_paths - variable_om

    total_values = np.zeros(n_paths)
    n_startups = np.zeros(n_paths)
    n_ops = np.zeros(n_paths)

    for p in range(n_paths):
        operating = False
        hours_in_state = 0
        for h in range(n_hours):
            if operating:
                # Can only shutdown after min_up_hours
                if hours_in_state >= min_up_hours and spark[p, h] < 0:
                    # Shutdown
                    operating = False
                    hours_in_state = 1
                    total_values[p] -= shutdown_cost * discount_factors[h]
                else:
                    total_values[p] += spark[p, h] * capacity_mw * discount_factors[h]
                    n_ops[p] += 1
                    hours_in_state += 1
            else:
                # Can only startup after min_down_hours
                if hours_in_state >= min_down_hours and spark[p, h] > 0:
                    operating = True
                    hours_in_state = 1
                    total_values[p] -= startup_cost * discount_factors[h]
                    total_values[p] += spark[p, h] * capacity_mw * discount_factors[h]
                    n_ops[p] += 1
                    n_startups[p] += 1
                else:
                    hours_in_state += 1

    return UnitCommitmentResult(
        value=float(total_values.mean()),
        n_startups_mean=float(n_startups.mean()),
        n_hours_operating_mean=float(n_ops.mean()),
        startup_cost=startup_cost,
    )


# ---- Pipeline / FTR ----

@dataclass
class FTRResult:
    """Financial Transmission Rights valuation."""
    value: float
    capacity_mw: float
    mean_congestion_price: float
    positive_congestion_hours: float


def pipeline_ftr_value(
    lmp_source: np.ndarray,         # (n_paths, n_hours) source node prices
    lmp_sink: np.ndarray,           # (n_paths, n_hours) sink node prices
    capacity_mw: float = 100.0,
    discount_factors: np.ndarray | None = None,
    is_obligation: bool = False,
) -> FTRResult:
    """Financial Transmission Rights (FTR) valuation.

    An FTR pays (LMP_sink − LMP_source) × capacity each hour.

    For an "option" FTR: pays max((sink - source), 0) × capacity (no downside).
    For an "obligation" FTR: pays (sink - source) × capacity (can be negative).

    Args:
        lmp_source, lmp_sink: hourly LMPs at source and sink.
        capacity_mw: FTR capacity.
        is_obligation: True for obligation FTR (two-sided), False for option.
    """
    n_paths, n_hours = lmp_source.shape
    if discount_factors is None:
        discount_factors = np.ones(n_hours)

    spread = lmp_sink - lmp_source

    if is_obligation:
        payoff = spread
    else:
        payoff = np.maximum(spread, 0.0)

    hourly = payoff * capacity_mw * discount_factors[np.newaxis, :]
    total = hourly.sum(axis=1)

    pos_cong_hours = float((spread > 0).sum(axis=1).mean())
    mean_cong = float(spread.mean())

    return FTRResult(
        value=float(total.mean()),
        capacity_mw=capacity_mw,
        mean_congestion_price=mean_cong,
        positive_congestion_hours=pos_cong_hours,
    )
