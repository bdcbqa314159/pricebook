"""Fund participation: LP economics, capital calls, NAV pricing.

Models a limited partner's interest in a credit fund with:
- Capital commitments and drawdowns (J-curve)
- Management fees and carried interest
- NAV-based secondary pricing
- Performance metrics: MOIC, DPI, TVPI, IRR

    from pricebook.credit.fund_participation import FundParticipation

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



    def to_dict(self) -> dict:
        return dict(vars(self))
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
        self.notional = commitment  # alias for desk compatibility
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

        harvest_start = max(p for p, _ in self.drawdown_schedule) + 1

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


# ═══════════════════════════════════════════════════════════════
# PE Fund Waterfall Extensions
# ═══════════════════════════════════════════════════════════════

@dataclass
class WaterfallConfig:
    """PE fund distribution waterfall configuration.

    European (whole-fund): carry only after all capital returned + preferred
    to ALL deals. GP bears cross-collateralisation risk.

    American (deal-by-deal): carry paid on each realised deal independently.
    GP receives carry earlier but faces clawback risk.
    """
    style: str = "european"         # "european" or "american"
    carry_rate: float = 0.20
    hurdle_rate: float = 0.08
    catchup_rate: float = 1.0       # fraction to GP during catch-up (1.0 = 100%)
    gp_commitment_pct: float = 0.02
    clawback: bool = True
    recycling: bool = False
    recycling_limit: float = 0.0    # max fraction of fund recyclable

    def to_dict(self) -> dict:
        return {
            "style": self.style, "carry_rate": self.carry_rate,
            "hurdle_rate": self.hurdle_rate, "catchup_rate": self.catchup_rate,
            "gp_commitment_pct": self.gp_commitment_pct,
            "clawback": self.clawback, "recycling": self.recycling,
            "recycling_limit": self.recycling_limit,
        }


@dataclass
class WaterfallResult:
    """Single-period distribution waterfall output."""
    period: int
    available: float
    return_of_capital: float
    preferred_return: float
    gp_catchup: float
    carried_interest: float
    lp_residual: float
    total_distribution: float

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class ClawbackResult:
    """GP clawback analysis."""
    total_carry_distributed: float
    entitled_carry: float
    clawback_amount: float
    triggered: bool

    def to_dict(self) -> dict:
        return dict(vars(self))


class PEFundParticipation(FundParticipation):
    """Extended LP participation with PE waterfall mechanics.

    Adds European/American waterfall, GP catch-up, clawback analysis,
    and GP commitment economics to the base FundParticipation.

    Args:
        commitment: total LP commitment.
        waterfall: waterfall configuration (defaults to European 80/20).
        **kwargs: passed to FundParticipation.
    """

    def __init__(
        self,
        commitment: float,
        waterfall: WaterfallConfig | None = None,
        **kwargs,
    ):
        super().__init__(commitment, **kwargs)
        self.waterfall = waterfall or WaterfallConfig()

    def _distribute_european(
        self,
        available: float,
        total_invested: float,
        total_returned: float,
        preferred_accrued: float,
        period: int,
    ) -> WaterfallResult:
        """European (whole-fund) waterfall distribution.

        1. Return of capital (until all invested capital returned)
        2. Preferred return (hurdle rate on invested capital)
        3. GP catch-up (100% to GP until carry share reached)
        4. Residual split (80/20 LP/GP)
        """
        wf = self.waterfall
        remaining = available

        # 1. Return of capital
        capital_shortfall = max(total_invested - total_returned, 0.0)
        roc = min(remaining, capital_shortfall)
        remaining -= roc

        # 2. Preferred return
        pref = min(remaining, max(preferred_accrued, 0.0))
        remaining -= pref

        # 3. GP catch-up
        # GP receives catchup_rate of distributions until GP has received
        # carry_rate / (1 - carry_rate) of what LP got in preferred
        total_pref_to_lp = pref
        gp_catchup_target = total_pref_to_lp * wf.carry_rate / (1 - wf.carry_rate)
        catchup = min(remaining * wf.catchup_rate, gp_catchup_target)
        catchup = min(catchup, remaining)
        remaining -= catchup

        # 4. Residual: 80/20 split
        carry = remaining * wf.carry_rate
        lp_residual = remaining - carry

        return WaterfallResult(
            period=period, available=available,
            return_of_capital=roc, preferred_return=pref,
            gp_catchup=catchup, carried_interest=carry,
            lp_residual=lp_residual,
            total_distribution=roc + pref + catchup + carry + lp_residual,
        )

    def project_waterfall(self) -> list[WaterfallResult]:
        """Project waterfall distributions over fund life."""
        cashflows = self.project()
        wf = self.waterfall

        results = []
        total_invested = 0.0
        total_returned = 0.0
        cumulative_preferred = 0.0

        for cf in cashflows:
            total_invested += cf.capital_call

            # NAV growth is the "available" for distribution
            available = cf.distribution + cf.carried_interest

            # Preferred return accrual (compound: accrue on unreturned + unpaid pref)
            unreturned_capital = max(total_invested - total_returned, 0.0)
            preferred_base = unreturned_capital + cumulative_preferred
            preferred_this_period = preferred_base * wf.hurdle_rate
            cumulative_preferred += preferred_this_period

            if available > 0:
                wr = self._distribute_european(
                    available, total_invested, total_returned,
                    cumulative_preferred, cf.period,
                )
                total_returned += wr.return_of_capital
                cumulative_preferred -= wr.preferred_return
                results.append(wr)
            else:
                results.append(WaterfallResult(
                    period=cf.period, available=0, return_of_capital=0,
                    preferred_return=0, gp_catchup=0, carried_interest=0,
                    lp_residual=0, total_distribution=0,
                ))

        return results

    def clawback_analysis(self) -> ClawbackResult:
        """Analyse GP clawback exposure.

        Compares total carry distributed vs entitled carry on whole-fund basis.
        """
        waterfall_results = self.project_waterfall()
        total_carry = sum(wr.carried_interest + wr.gp_catchup for wr in waterfall_results)

        # Entitled carry: whole-fund basis
        m = self.metrics()
        total_profit = max(m.distributed + m.nav - m.invested, 0.0)
        hurdle_profit = m.invested * ((1 + self.waterfall.hurdle_rate) ** self.fund_life_years - 1)
        excess = max(total_profit - hurdle_profit, 0.0)
        entitled = excess * self.waterfall.carry_rate

        clawback = max(total_carry - entitled, 0.0)

        return ClawbackResult(
            total_carry_distributed=total_carry,
            entitled_carry=entitled,
            clawback_amount=clawback,
            triggered=clawback > 0,
        )

    def gp_commitment_cashflows(self) -> list[FundCashflow]:
        """GP's own commitment cashflows (pro-rata with LP)."""
        lp_cashflows = self.project()
        gp_pct = self.waterfall.gp_commitment_pct

        gp_flows = []
        for cf in lp_cashflows:
            gp_flows.append(FundCashflow(
                period=cf.period,
                date=cf.date,
                capital_call=cf.capital_call * gp_pct,
                distribution=cf.distribution * gp_pct,
                nav=cf.nav * gp_pct,
                management_fee=0.0,  # GP doesn't pay mgmt fee on own commitment
                carried_interest=0.0,
            ))
        return gp_flows


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
