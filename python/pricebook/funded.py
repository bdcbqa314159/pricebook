"""
Funded structures: repo, total return swap, funded participation.

Repo: sell bond + agree to buy back at repo rate.
TRS: receive total return (interest + price change) vs pay floating + spread.
Participation: partial funded credit risk transfer.

    from pricebook.funded import Repo, TotalReturnSwap, FundedParticipation

    repo = Repo(bond_pv=101, repo_rate=0.04, T=0.25)
    trs = TotalReturnSwap(reference_pv=100, funding_rate=0.05, spread=0.01, T=1.0)
"""

from __future__ import annotations

import math
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


class Repo:
    """Classic repurchase agreement.

    Sell bond at start, buy back at maturity at agreed price.
    Repo PV at inception ≈ 0 (fair repo rate).

    Args:
        bond_dirty_price: current dirty price of the collateral bond (per 100).
        repo_rate: agreed annualised repo rate.
        T: repo term in years.
        haircut: overcollateralisation (e.g. 0.02 = 2% haircut).
        notional: face value of the bond.
    """

    def __init__(
        self,
        bond_dirty_price: float,
        repo_rate: float,
        T: float,
        haircut: float = 0.0,
        notional: float = 1_000_000.0,
    ):
        self.bond_dirty_price = bond_dirty_price
        self.repo_rate = repo_rate
        self.T = T
        self.haircut = haircut
        self.notional = notional

    @property
    def cash_lent(self) -> float:
        """Cash amount lent (bond value minus haircut)."""
        bond_value = self.notional * self.bond_dirty_price / 100.0
        return bond_value * (1 - self.haircut)

    @property
    def repurchase_price(self) -> float:
        """Price to buy back the bond at maturity."""
        return self.cash_lent * (1 + self.repo_rate * self.T)

    @property
    def effective_funding_rate(self) -> float:
        """Effective funding rate accounting for haircut."""
        if self.haircut >= 1:
            return float("inf")
        return self.repo_rate / (1 - self.haircut)

    def pv(self, discount_curve: DiscountCurve, valuation_date: date, maturity_date: date) -> float:
        """PV of the repo (from the cash lender's perspective).

        PV = df(T) * repurchase_price - cash_lent (at inception ≈ 0).
        """
        df = discount_curve.df(maturity_date)
        return df * self.repurchase_price - self.cash_lent

    @staticmethod
    def implied_repo_rate(
        bond_spot_price: float,
        bond_forward_price: float,
        T: float,
        coupon_income: float = 0.0,
    ) -> float:
        """Implied repo rate from bond spot and forward prices.

        repo_rate = (forward - spot + coupon_income) / (spot * T)
        """
        if T <= 0 or bond_spot_price <= 0:
            return 0.0
        return (bond_forward_price - bond_spot_price + coupon_income) / (bond_spot_price * T)


class TotalReturnSwap:
    """Total Return Swap: unfunded replication of a reference asset.

    Total return receiver gets: interest + price appreciation.
    Total return payer gets: floating rate + spread.

    PV = PV(total_return_leg) - PV(funding_leg).

    Args:
        reference_pv_start: reference asset PV at trade inception.
        reference_pv_current: current reference asset PV.
        funding_rate: floating rate for the funding leg.
        spread: spread over funding rate.
        T: remaining maturity.
        notional: TRS notional.
    """

    def __init__(
        self,
        reference_pv_start: float,
        reference_pv_current: float,
        funding_rate: float,
        spread: float = 0.0,
        T: float = 1.0,
        notional: float = 1_000_000.0,
    ):
        self.ref_start = reference_pv_start
        self.ref_current = reference_pv_current
        self.funding_rate = funding_rate
        self.spread = spread
        self.T = T
        self.notional = notional

    @property
    def total_return(self) -> float:
        """Total return: price change + income (simplified as PV change)."""
        return (self.ref_current - self.ref_start) / self.ref_start

    @property
    def funding_cost(self) -> float:
        """Funding cost over the period."""
        return (self.funding_rate + self.spread) * self.T

    def pv(self, discount_curve: DiscountCurve | None = None) -> float:
        """PV of the TRS (receiver perspective).

        PV = notional * (total_return - funding_cost).
        Simplified: no term structure, single-period.
        """
        return self.notional * (self.total_return - self.funding_cost)

    @staticmethod
    def fair_spread(
        reference_yield: float,
        funding_rate: float,
    ) -> float:
        """Fair TRS spread: spread such that PV = 0 at inception.

        At inception ref_current = ref_start, so total_return = 0.
        For PV = 0: spread = reference_yield - funding_rate (approximately).
        """
        return reference_yield - funding_rate


class FundedParticipation:
    """Funded participation: partial risk transfer.

    The participant funds a portion of a loan/bond and receives
    pro-rata interest. Credit risk on the funded amount.

    Args:
        total_notional: total loan/bond notional.
        participation_rate: fraction funded by participant (0 to 1).
        asset_yield: yield of the reference asset.
        funding_cost: participant's cost of funding.
        T: term in years.
        expected_loss: expected credit loss rate (annualised).
    """

    def __init__(
        self,
        total_notional: float,
        participation_rate: float,
        asset_yield: float,
        funding_cost: float,
        T: float = 1.0,
        expected_loss: float = 0.0,
    ):
        if not 0 <= participation_rate <= 1:
            raise ValueError(f"participation_rate must be in [0,1], got {participation_rate}")
        self.total_notional = total_notional
        self.participation_rate = participation_rate
        self.asset_yield = asset_yield
        self.funding_cost = funding_cost
        self.T = T
        self.expected_loss = expected_loss

    @property
    def funded_amount(self) -> float:
        return self.total_notional * self.participation_rate

    @property
    def net_carry(self) -> float:
        """Net carry: asset yield - funding cost - expected loss."""
        return self.asset_yield - self.funding_cost - self.expected_loss

    def pv(self) -> float:
        """PV of the participation (simplified: carry * funded * T)."""
        return self.funded_amount * self.net_carry * self.T

    @staticmethod
    def cash_cds_basis(
        funded_spread: float,
        cds_spread: float,
    ) -> float:
        """Cash-CDS basis: funded spread - CDS spread.

        Positive basis: funded is cheaper (CDS protection costs more).
        Negative basis: funded is more expensive.
        """
        return funded_spread - cds_spread
