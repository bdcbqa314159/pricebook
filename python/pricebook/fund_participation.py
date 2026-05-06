"""Fund participation: LP economics, capital calls, NAV pricing.

Models a limited partner's interest in a credit fund with:
- Capital commitments and drawdowns (J-curve)
- Management fees and carried interest
- NAV-based secondary pricing
- Performance metrics: MOIC, DPI, TVPI, IRR

    from pricebook.fund_participation import FundParticipation

References:
    Metrick & Yasuda (2010). The Economics of Private Equity Funds.
    Phalippou (2014). Performance of Buyout Funds Revisited.
    ILPA (2011). Private Equity Principles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date


@dataclass
class FundCashflow:
    """A single fund cashflow event."""
    period: int
    date: date | None = None
    capital_call: float = 0.0      # LP pays in (positive)
    distribution: float = 0.0      # LP receives (positive)
    nav: float = 0.0               # net asset value after event
    management_fee: float = 0.0
    carried_interest: float = 0.0


@dataclass
class FundMetrics:
    """Fund performance metrics."""
    moic: float          # multiple on invested capital (total value / invested)
    dpi: float           # distributions to paid-in (distributions / invested)
    tvpi: float          # total value to paid-in ((distributions + nav) / invested)
    irr: float           # internal rate of return
    invested: float      # total capital called
    distributed: float   # total distributions
    nav: float           # current NAV
    j_curve_trough: float  # minimum TVPI during fund life

    def to_dict(self) -> dict:
        return {
            "moic": self.moic, "dpi": self.dpi, "tvpi": self.tvpi,
            "irr": self.irr, "invested": self.invested,
            "distributed": self.distributed, "nav": self.nav,
            "j_curve_trough": self.j_curve_trough,
        }


@dataclass
class SecondaryPricing:
    """NAV-based secondary market pricing for fund interests."""
    nav: float
    discount_pct: float       # discount to NAV (e.g. 0.10 = 10%)
    secondary_price: float    # NAV × (1 - discount)
    unfunded_commitment: float
    implied_tvpi: float       # (secondary_price + future_distributions) / invested

    def to_dict(self) -> dict:
        return {
            "nav": self.nav, "discount_pct": self.discount_pct,
            "secondary_price": self.secondary_price,
            "unfunded": self.unfunded_commitment,
            "implied_tvpi": self.implied_tvpi,
        }


class FundParticipation:
    """LP interest in a credit fund.

    Models the full lifecycle: commitment → drawdown → investment →
    harvest → distribution, with management fees and carry.

    Args:
        commitment: total LP commitment.
        vintage_year: fund inception year.
        fund_life_years: expected fund life (typically 7-10).
        mgmt_fee_rate: annual management fee (on committed or invested).
        carry_rate: carried interest rate (typically 20%).
        hurdle_rate: preferred return before carry (typically 8%).
        drawdown_schedule: list of (period, fraction_of_commitment) for calls.
        gross_return: expected gross annual return of the portfolio.
        fee_basis: "committed" or "invested" for management fee.
    """

    def __init__(
        self,
        commitment: float,
        vintage_year: int = 2024,
        fund_life_years: int = 8,
        mgmt_fee_rate: float = 0.015,
        carry_rate: float = 0.20,
        hurdle_rate: float = 0.08,
        drawdown_schedule: list[tuple[int, float]] | None = None,
        gross_return: float = 0.10,
        fee_basis: str = "committed",
    ):
        if commitment <= 0:
            raise ValueError(f"commitment must be positive, got {commitment}")

        self.commitment = commitment
        self.vintage_year = vintage_year
        self.fund_life_years = fund_life_years
        self.mgmt_fee_rate = mgmt_fee_rate
        self.carry_rate = carry_rate
        self.hurdle_rate = hurdle_rate
        self.gross_return = gross_return
        self.fee_basis = fee_basis

        # Default drawdown: 25% per year in years 1-4
        self.drawdown_schedule = drawdown_schedule or [
            (1, 0.25), (2, 0.25), (3, 0.25), (4, 0.25),
        ]

    def project(self) -> list[FundCashflow]:
        """Project fund cashflows over the full life."""
        cashflows = []
        invested = 0.0
        nav = 0.0
        total_distributed = 0.0
        drawdown_map = dict(self.drawdown_schedule)

        for period in range(1, self.fund_life_years + 1):
            # Capital call
            call_frac = drawdown_map.get(period, 0.0)
            call = self.commitment * call_frac
            invested += call

            # Management fee
            if self.fee_basis == "committed":
                fee = self.commitment * self.mgmt_fee_rate
            else:
                fee = invested * self.mgmt_fee_rate

            # Portfolio return (net of fee)
            gross_income = nav * self.gross_return
            nav = nav + call + gross_income - fee

            # Distributions: start harvesting after investment period
            distribution = 0.0
            carry = 0.0
            harvest_start = max(p for p, _ in self.drawdown_schedule) + 1

            if period >= harvest_start and nav > 0:
                # Distribute proportionally over remaining life
                remaining = self.fund_life_years - period + 1
                dist_target = nav / max(remaining, 1)

                # Check hurdle
                total_value = total_distributed + nav
                if total_value > invested * (1 + self.hurdle_rate) ** period:
                    # Above hurdle: carry applies
                    excess = total_value - invested * (1 + self.hurdle_rate) ** period
                    carry = min(excess * self.carry_rate, dist_target * self.carry_rate)

                distribution = max(dist_target - carry, 0.0)
                nav -= distribution + carry
                total_distributed += distribution

            cashflows.append(FundCashflow(
                period=period, capital_call=call,
                distribution=distribution, nav=max(nav, 0.0),
                management_fee=fee, carried_interest=carry,
            ))

        return cashflows

    def metrics(self) -> FundMetrics:
        """Compute fund performance metrics."""
        cashflows = self.project()
        invested = sum(cf.capital_call for cf in cashflows)
        distributed = sum(cf.distribution for cf in cashflows)
        final_nav = cashflows[-1].nav if cashflows else 0.0

        moic = (distributed + final_nav) / max(invested, 1e-10)
        dpi = distributed / max(invested, 1e-10)
        tvpi = (distributed + final_nav) / max(invested, 1e-10)

        # J-curve trough
        cumulative_invested = 0.0
        cumulative_distributed = 0.0
        min_tvpi = float('inf')
        for cf in cashflows:
            cumulative_invested += cf.capital_call
            cumulative_distributed += cf.distribution
            if cumulative_invested > 0:
                current_tvpi = (cumulative_distributed + cf.nav) / cumulative_invested
                min_tvpi = min(min_tvpi, current_tvpi)

        # IRR
        irr_cfs = []
        for cf in cashflows:
            irr_cfs.append(-cf.capital_call + cf.distribution)
        irr_cfs[-1] += final_nav  # terminal NAV as final cashflow
        irr = _fund_irr(irr_cfs)

        return FundMetrics(
            moic=moic, dpi=dpi, tvpi=tvpi, irr=irr,
            invested=invested, distributed=distributed,
            nav=final_nav,
            j_curve_trough=min_tvpi if min_tvpi != float('inf') else 0.0,
        )

    def secondary_pricing(
        self,
        current_nav: float,
        discount_pct: float = 0.10,
        unfunded_remaining: float | None = None,
    ) -> SecondaryPricing:
        """Price an LP interest on the secondary market.

        Args:
            current_nav: current net asset value of the LP interest.
            discount_pct: discount to NAV (e.g. 0.10 = buyer pays 90% of NAV).
            unfunded_remaining: unfunded commitment the buyer assumes.
        """
        if unfunded_remaining is None:
            called = sum(frac for _, frac in self.drawdown_schedule)
            unfunded_remaining = self.commitment * max(1 - called, 0)

        secondary_price = current_nav * (1 - discount_pct)
        invested = self.commitment * sum(frac for _, frac in self.drawdown_schedule)
        implied_tvpi = (secondary_price + current_nav * 0.5) / max(invested, 1e-10)

        return SecondaryPricing(
            nav=current_nav,
            discount_pct=discount_pct,
            secondary_price=secondary_price,
            unfunded_commitment=unfunded_remaining,
            implied_tvpi=implied_tvpi,
        )


def _fund_irr(cashflows: list[float]) -> float:
    """IRR via Newton-Raphson on annual cashflows."""
    if not cashflows or all(cf == 0 for cf in cashflows):
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
        r = max(min(r, 5.0), -0.99)  # clamp
        if abs(f) < 1e-10:
            break
    return r
