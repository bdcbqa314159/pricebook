"""
P&L Explain: attribution of portfolio value changes.

Decomposes total P&L into: carry, roll-down, rate moves, vol moves,
credit moves, FX moves, and unexplained.

    from pricebook.pnl_explain import pnl_decompose, PnLResult

    result = pnl_decompose(
        base_pv=100_000, current_pv=101_500,
        carry=500, rolldown=200,
        greeks={"rate": (-50, 0.02), "vol": (300, 0.01)},
    )
    print(result.total, result.unexplained)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PnLResult:
    """Result of P&L decomposition."""

    base_pv: float
    current_pv: float
    total: float
    carry: float = 0.0
    rolldown: float = 0.0
    rate_pnl: float = 0.0
    vol_pnl: float = 0.0
    credit_pnl: float = 0.0
    fx_pnl: float = 0.0
    theta_pnl: float = 0.0
    other: dict[str, float] = field(default_factory=dict)

    @property
    def explained(self) -> float:
        return (self.carry + self.rolldown + self.rate_pnl + self.vol_pnl +
                self.credit_pnl + self.fx_pnl + self.theta_pnl +
                sum(self.other.values()))

    @property
    def unexplained(self) -> float:
        return self.total - self.explained


def compute_carry(
    coupon_income: float,
    funding_cost: float,
    dt: float = 1.0 / 252,
) -> float:
    """Daily carry = (coupon accrual - funding cost) × dt.

    Args:
        coupon_income: annualised coupon income (rate × notional).
        funding_cost: annualised funding cost (rate × notional).
        dt: day fraction (default: 1/252 business day).

    Returns daily carry by scaling annualised amounts by dt.
    """
    return (coupon_income - funding_cost) * dt


def compute_rolldown(
    pricer,
    base_curve,
    days: int = 1,
) -> float:
    """Roll-down: PV change from time passing with unchanged yield curve.

    Rolls the curve forward by `days` (shifts reference_date, keeps shape),
    then reprices. The difference is the P&L from the passage of time
    with no market moves.

    Roll-down = pricer(rolled_curve) - pricer(base_curve)
    """
    if not hasattr(base_curve, 'roll_down'):
        return 0.0
    rolled = base_curve.roll_down(days)
    return pricer(rolled) - pricer(base_curve)


def greek_pnl(
    sensitivity: float,
    risk_factor_change: float,
    gamma: float = 0.0,
) -> float:
    """P&L from a single risk factor via Taylor expansion.

    pnl ≈ delta * dx + 0.5 * gamma * dx^2
    """
    return sensitivity * risk_factor_change + 0.5 * gamma * risk_factor_change**2


def pnl_decompose(
    base_pv: float,
    current_pv: float,
    carry: float = 0.0,
    rolldown: float = 0.0,
    rate_delta: float = 0.0,
    rate_change: float = 0.0,
    rate_gamma: float = 0.0,
    vol_vega: float = 0.0,
    vol_change: float = 0.0,
    credit_cs01: float = 0.0,
    credit_change: float = 0.0,
    fx_delta: float = 0.0,
    fx_change: float = 0.0,
    theta: float = 0.0,
    dt: float = 1.0 / 252,
) -> PnLResult:
    """Full P&L decomposition.

    Total P&L = current_pv - base_pv
    Explained = carry + rolldown + rate + vol + credit + FX + theta
    Unexplained = total - explained
    """
    total = current_pv - base_pv

    rate_pnl = greek_pnl(rate_delta, rate_change, rate_gamma)
    vol_pnl = greek_pnl(vol_vega, vol_change)
    credit_pnl = greek_pnl(credit_cs01, credit_change)
    fx_pnl = greek_pnl(fx_delta, fx_change)
    theta_pnl = theta * dt

    return PnLResult(
        base_pv=base_pv,
        current_pv=current_pv,
        total=total,
        carry=carry,
        rolldown=rolldown,
        rate_pnl=rate_pnl,
        vol_pnl=vol_pnl,
        credit_pnl=credit_pnl,
        fx_pnl=fx_pnl,
        theta_pnl=theta_pnl,
    )
