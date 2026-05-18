"""Special Purpose Vehicle: pooled assets, tranched liabilities, cashflow projection.

An SPV pools assets (loans, bonds) and issues tranched liabilities.
Cashflows from the pool are distributed via a waterfall engine.

Supports:
- Static pools (no reinvestment, amortising)
- Managed pools (reinvestment period + amortisation)
- Cashflow projection with default and prepayment assumptions
- Tranche IRR, credit enhancement, break-even default rate

    from pricebook.credit.spv import SPV, SPVTranche, SPVProjection

References:
    Fabozzi (2007). Fixed Income Analysis, Ch. 11 (ABS/MBS).
    LSTA (2022). Handbook of Loan Syndications and Trading, Ch. 21.
    Moody's (2021). CLO Monitor Methodology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve


# ---------------------------------------------------------------------------
# SPV Tranche
# ---------------------------------------------------------------------------

@dataclass
class SPVTranche:
    """A single tranche in the SPV capital structure."""
    name: str
    notional: float
    coupon: float           # spread over base rate (decimal)
    seniority: int          # 1 = most senior
    balance: float = 0.0    # outstanding (initialised to notional)
    interest_received: float = 0.0
    principal_received: float = 0.0
    losses_absorbed: float = 0.0

    def __post_init__(self):
        if self.balance == 0.0:
            self.balance = self.notional

    @property
    def outstanding(self) -> float:
        return max(self.balance - self.principal_received, 0.0)

    @property
    def is_paid_off(self) -> bool:
        return self.outstanding <= 0.01

    def reset(self):
        self.balance = self.notional
        self.interest_received = 0.0
        self.principal_received = 0.0
        self.losses_absorbed = 0.0



    def to_dict(self) -> dict:
        return vars(self)
# ---------------------------------------------------------------------------
# SPV Cashflow Projection
# ---------------------------------------------------------------------------

@dataclass
class SPVPeriodResult:
    """Cashflow result for one projection period."""
    period: int
    pool_balance: float
    pool_interest: float
    pool_principal: float
    pool_defaults: float
    pool_recoveries: float
    tranche_interest: dict[str, float]
    tranche_principal: dict[str, float]
    tranche_losses: dict[str, float]
    oc_ratio: float
    equity_cashflow: float



    def to_dict(self) -> dict:
        return vars(self)
@dataclass
class SPVProjection:
    """Full cashflow projection for an SPV."""
    periods: list[SPVPeriodResult]
    tranche_irr: dict[str, float]
    total_losses: float
    credit_enhancement: dict[str, float]   # subordination per tranche
    break_even_cdr: float                  # annual CDR equity can absorb

    def to_dict(self) -> dict:
        return {
            "n_periods": len(self.periods),
            "tranche_irr": self.tranche_irr,
            "total_losses": self.total_losses,
            "credit_enhancement": self.credit_enhancement,
            "break_even_cdr": self.break_even_cdr,
        }


# ---------------------------------------------------------------------------
# SPV
# ---------------------------------------------------------------------------

class SPV:
    """Special Purpose Vehicle with pooled assets and tranched liabilities.

    Args:
        pool_balance: initial pool par balance.
        pool_coupon: weighted average coupon of the pool.
        tranches: list of SPVTranche (capital structure).
        n_periods: number of projection periods (typically quarterly).
        cdr: constant default rate (annual, decimal).
        recovery: recovery rate on defaults.
        cpr: constant prepayment rate (annual, decimal).
        reinvestment_periods: number of periods with reinvestment (0 = static).
        reinvestment_spread: spread on reinvested assets.
        mgmt_fee: annual management fee (as fraction of pool).
        base_rate: floating base rate for tranche coupons.
    """

    def __init__(
        self,
        pool_balance: float,
        pool_coupon: float,
        tranches: list[SPVTranche],
        n_periods: int = 20,
        cdr: float = 0.02,
        recovery: float = 0.40,
        cpr: float = 0.10,
        reinvestment_periods: int = 0,
        reinvestment_spread: float = 0.0,
        mgmt_fee: float = 0.005,
        base_rate: float = 0.04,
    ):
        if pool_balance <= 0:
            raise ValueError(f"pool_balance must be positive, got {pool_balance}")

        self.pool_balance = pool_balance
        self.pool_coupon = pool_coupon
        self.tranches = sorted(tranches, key=lambda t: t.seniority)
        self.n_periods = n_periods
        self.cdr = cdr
        self.recovery = recovery
        self.cpr = cpr
        self.reinvestment_periods = reinvestment_periods
        self.reinvestment_spread = reinvestment_spread or pool_coupon
        self.mgmt_fee = mgmt_fee
        self.base_rate = base_rate

    def _period_fraction(self) -> float:
        """Fraction of year per period (assume quarterly)."""
        return 0.25

    def project(self) -> SPVProjection:
        """Run full cashflow projection through the waterfall."""
        dt = self._period_fraction()
        balance = self.pool_balance
        periods = []
        total_losses = 0.0

        # Reset tranches
        for t in self.tranches:
            t.reset()

        # Track cashflows per tranche for IRR
        tranche_cfs: dict[str, list[float]] = {t.name: [-t.notional] for t in self.tranches}

        for period in range(1, self.n_periods + 1):
            # Pool cashflows
            # Defaults: convert annual CDR to periodic
            smm_default = 1 - (1 - self.cdr) ** dt
            defaults = balance * smm_default
            recoveries = defaults * self.recovery
            net_loss = defaults - recoveries

            # Prepayments
            smm_prepay = 1 - (1 - self.cpr) ** dt
            prepayments = (balance - defaults) * smm_prepay

            # Interest
            interest = balance * self.pool_coupon * dt

            # Scheduled principal (amortising assumption: equal across periods)
            sched_principal = self.pool_balance / self.n_periods

            # Total principal
            total_principal = prepayments + sched_principal + recoveries

            # Update pool balance
            balance -= defaults + prepayments + sched_principal

            # Reinvestment
            if period <= self.reinvestment_periods:
                reinvested = prepayments + sched_principal
                balance += reinvested
                total_principal -= reinvested

            balance = max(balance, 0.0)
            total_losses += net_loss

            # Management fee
            fee = balance * self.mgmt_fee * dt
            available_interest = max(interest - fee, 0.0)

            # Waterfall: distribute interest (senior first)
            remaining_interest = available_interest
            period_tranche_interest = {}
            period_tranche_principal = {}
            period_tranche_losses = {}

            for tr in self.tranches:
                coupon_due = tr.outstanding * (tr.coupon + self.base_rate) * dt
                paid = min(coupon_due, remaining_interest)
                tr.interest_received += paid
                remaining_interest -= paid
                period_tranche_interest[tr.name] = paid
                period_tranche_principal[tr.name] = 0.0
                period_tranche_losses[tr.name] = 0.0

            # Equity gets residual interest
            equity_cf = remaining_interest

            # Absorb losses (bottom-up)
            loss_remaining = net_loss
            for tr in reversed(self.tranches):
                absorbed = min(loss_remaining, tr.outstanding)
                tr.losses_absorbed += absorbed
                tr.balance -= absorbed
                loss_remaining -= absorbed
                period_tranche_losses[tr.name] = absorbed
                if loss_remaining <= 0:
                    break

            # Distribute principal (senior first, sequential)
            remaining_principal = total_principal
            for tr in self.tranches:
                if tr.is_paid_off:
                    continue
                principal_paid = min(tr.outstanding, remaining_principal)
                tr.principal_received += principal_paid
                remaining_principal -= principal_paid
                period_tranche_principal[tr.name] = principal_paid

            # Equity gets residual principal
            equity_cf += remaining_principal

            # OC ratio
            total_senior = sum(tr.outstanding for tr in self.tranches if tr.seniority < 100)
            oc = balance / max(total_senior, 1e-10)

            # Track cashflows for IRR
            for tr in self.tranches:
                tranche_cfs[tr.name].append(
                    period_tranche_interest[tr.name] + period_tranche_principal[tr.name]
                )

            periods.append(SPVPeriodResult(
                period=period, pool_balance=balance,
                pool_interest=interest, pool_principal=total_principal,
                pool_defaults=defaults, pool_recoveries=recoveries,
                tranche_interest=period_tranche_interest,
                tranche_principal=period_tranche_principal,
                tranche_losses=period_tranche_losses,
                oc_ratio=oc, equity_cashflow=equity_cf,
            ))

        # Compute tranche IRRs
        tranche_irr = {}
        for tr in self.tranches:
            cfs = tranche_cfs[tr.name]
            # Add final outstanding as terminal cashflow
            cfs[-1] += tr.outstanding
            tranche_irr[tr.name] = _compute_irr(cfs, dt)

        # Credit enhancement per tranche
        total_liabilities = sum(t.notional for t in self.tranches)
        credit_enhancement = {}
        for i, tr in enumerate(self.tranches):
            subordination = sum(t.notional for t in self.tranches[i+1:])
            credit_enhancement[tr.name] = subordination / total_liabilities if total_liabilities > 0 else 0.0

        # Break-even CDR
        equity_notional = sum(t.notional for t in self.tranches if t.seniority == max(t2.seniority for t2 in self.tranches))
        be_cdr = equity_notional / self.pool_balance / (1 - self.recovery) if self.pool_balance > 0 else 0.0

        return SPVProjection(
            periods=periods,
            tranche_irr=tranche_irr,
            total_losses=total_losses,
            credit_enhancement=credit_enhancement,
            break_even_cdr=be_cdr,
        )

    def tranche_pv(
        self,
        tranche_name: str,
        discount_curve: DiscountCurve,
    ) -> float:
        """PV of a specific tranche's cashflows."""
        projection = self.project()
        dt = self._period_fraction()
        ref = discount_curve.reference_date

        pv = 0.0
        for p in projection.periods:
            cf = p.tranche_interest.get(tranche_name, 0.0) + p.tranche_principal.get(tranche_name, 0.0)
            t = p.period * dt
            # Approximate DF from time
            df = math.exp(-discount_curve.zero_rate(ref) * t) if t > 0 else 1.0
            pv += cf * df

        return pv


def _compute_irr(cashflows: list[float], dt: float) -> float:
    """Compute IRR from periodic cashflows via Newton's method."""
    if not cashflows or all(cf == 0 for cf in cashflows):
        return 0.0

    def npv(r: float) -> float:
        return sum(cf / (1 + r * dt) ** i for i, cf in enumerate(cashflows))

    def npv_deriv(r: float) -> float:
        return sum(
            -i * dt * cf / (1 + r * dt) ** (i + 1)
            for i, cf in enumerate(cashflows)
        )

    # Newton-Raphson
    r = 0.05  # initial guess
    for _ in range(100):
        f = npv(r)
        fp = npv_deriv(r)
        if abs(fp) < 1e-15:
            break
        r -= f / fp
        if abs(f) < 1e-10:
            break

    return r
