"""Bond forward: forward pricing, forward DV01, forward spread.

A bond forward is an agreement to buy/sell a bond at a future date.
Forward price is determined by cash-and-carry arbitrage.

    from pricebook.bond_forward import BondForward

    fwd = BondForward(bond, settlement, delivery, repo_rate)
    result = fwd.price(curve)

Standalone functions for raw forward price under repo (Pucci 2019):

    from pricebook.bond_forward import forward_price_repo, forward_price_haircut

References:
    Tuckman & Serrat, *Fixed Income Securities*, Wiley, 2012, Ch. 15.
    Pucci, M. (2019). Hedging the Treasury Lock. SSRN 3386521, Section 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.bond import FixedRateBond
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


# ---- Repo financing utilities ----

def repo_financing_factor(rs_minus_r: float, T: float, t: float = 0.0) -> float:
    """Repo-vs-OIS financing factor exp((rs-r)(T-t)).

    Used by TRS (Lou 2018 Eq 8), bond forward, and T-Lock.
    Returns 1.0 when rs = r (no financing cost).
    """
    return math.exp(rs_minus_r * (T - t))


def blended_repo_rate(
    repo_rate: float,
    funding_rate: float,
    haircut: float,
) -> float:
    """All-in bond financing rate with haircut (Lou 2018 Eq 19).

    r̄s = (1-h) rs + h rN

    The haircut portion is funded at the unsecured rate rN.
    """
    return (1 - haircut) * repo_rate + haircut * funding_rate


# ---- Standalone forward functions (Pucci Eq 21, 24) ----

def forward_price_repo(
    market_price: float,
    repo_rate: float,
    time_to_expiry: float,
    coupon_rate: float,
    coupon_accruals: list[float],
    coupon_times_to_expiry: list[float],
    repo_rates_coupon: list[float] | None = None,
) -> float:
    """Forward price under repo, zero haircut (Pucci Eq 21).

    ForwardPrice = P^mkt * (1 + r_repo * tau)
                   - c * sum alpha_i * (1 + r_repo_i * tau_i)

    Args:
        market_price: current bond dirty price.
        repo_rate: repo rate from t to t_e (simply-compounded).
        time_to_expiry: t_e - t in years.
        coupon_accruals: alpha_i for coupons paid in (t, t_e].
        coupon_times_to_expiry: t_e - t_i for each coupon in (t, t_e].
        repo_rates_coupon: repo rate from t_i to t_e for each coupon.
    """
    fwd = market_price * (1 + repo_rate * time_to_expiry)
    if repo_rates_coupon is None:
        repo_rates_coupon = [repo_rate] * len(coupon_accruals)
    for alpha, tau, r in zip(coupon_accruals, coupon_times_to_expiry,
                              repo_rates_coupon):
        fwd -= coupon_rate * alpha * (1 + r * tau)
    return fwd


def forward_price_haircut(
    market_price: float,
    repo_rate: float,
    funding_rate: float,
    haircut: float,
    time_to_expiry: float,
    coupon_amounts: list[float],
    coupon_times_to_expiry: list[float],
) -> float:
    """Forward price with haircut and funding blend (Pucci Eq 24).

    ForwardPrice = X * [(1-h)*exp(r_repo*T) + h*exp(r_fun*T)]
                   - sum c_i * exp((h*r_fun + (1-h)*r_repo) * tau_i)

    Args:
        haircut: h^cut in [0, 1]. 0 = fully repo, 1 = fully unsecured.
        funding_rate: unsecured funding rate (continuously compounded).
    """
    if not 0 <= haircut <= 1:
        raise ValueError(f"haircut must be in [0, 1], got {haircut}")
    T = time_to_expiry
    blend_rate = haircut * funding_rate + (1 - haircut) * repo_rate
    fwd = market_price * (
        (1 - haircut) * math.exp(repo_rate * T)
        + haircut * math.exp(funding_rate * T)
    )
    for ci, tau_i in zip(coupon_amounts, coupon_times_to_expiry):
        fwd -= ci * math.exp(blend_rate * tau_i)
    return fwd


@dataclass
class BondForwardResult:
    """Bond forward pricing result."""
    forward_dirty: float
    forward_clean: float
    spot_dirty: float
    carry: float
    repo_cost: float
    coupon_income: float
    forward_dv01: float


class BondForward:
    """Forward contract on a fixed-rate bond.

    Args:
        bond: the underlying bond.
        settlement: spot settlement date (today or T+1/T+2).
        delivery: forward delivery date.
        repo_rate: financing rate for the carry period.
    """

    def __init__(
        self,
        bond: FixedRateBond,
        settlement: date,
        delivery: date,
        repo_rate: float,
        repo_day_count: DayCountConvention = DayCountConvention.ACT_360,
    ):
        if delivery <= settlement:
            raise ValueError(f"delivery ({delivery}) must be after settlement ({settlement})")
        self.bond = bond
        self.settlement = settlement
        self.delivery = delivery
        self.repo_rate = repo_rate
        self.repo_day_count = repo_day_count

    def _coupon_income(self) -> float:
        """Sum of coupons received between settlement and delivery."""
        total = 0.0
        for cf in self.bond.coupon_leg.cashflows:
            if self.settlement < cf.payment_date <= self.delivery:
                total += cf.amount
        return total / self.bond.face_value * 100.0

    def price(self, curve: DiscountCurve) -> BondForwardResult:
        """Compute forward price from spot price + carry.

        Forward dirty = spot dirty × (1 + repo × T) - coupon_income.
        """
        spot_dirty = self.bond.dirty_price(curve)
        dt = year_fraction(self.settlement, self.delivery, self.repo_day_count)
        repo_cost = spot_dirty * self.repo_rate * dt
        coupon_income = self._coupon_income()
        fwd_dirty = spot_dirty + repo_cost - coupon_income
        accrued_at_delivery = self.bond.accrued_interest(self.delivery)
        fwd_clean = fwd_dirty - accrued_at_delivery
        carry = coupon_income - repo_cost

        # Forward DV01: bump yield 1bp and recompute forward
        ytm = self.bond.yield_to_maturity(spot_dirty, self.settlement)
        price_up = self.bond._price_from_ytm(ytm + 0.0001, self.settlement)
        fwd_up = price_up + price_up * self.repo_rate * dt - coupon_income
        fwd_dv01 = fwd_dirty - fwd_up

        return BondForwardResult(
            forward_dirty=fwd_dirty,
            forward_clean=fwd_clean,
            spot_dirty=spot_dirty,
            carry=carry,
            repo_cost=repo_cost,
            coupon_income=coupon_income,
            forward_dv01=fwd_dv01,
        )

    # ---- Serialisation ----

    _SERIAL_TYPE = "bond_forward"

    def to_dict(self) -> dict:
        from pricebook.serialisable import _serialise_atom
        return {"type": self._SERIAL_TYPE, "params": {
            "bond": self.bond.to_dict(),
            "settlement": self.settlement.isoformat(),
            "delivery": self.delivery.isoformat(),
            "repo_rate": self.repo_rate,
            "repo_day_count": _serialise_atom(self.repo_day_count),
        }}

    @classmethod
    def from_dict(cls, d: dict) -> "BondForward":
        from pricebook.serialisable import from_dict as _fd
        p = d["params"]
        return cls(
            bond=_fd(p["bond"]),
            settlement=date.fromisoformat(p["settlement"]),
            delivery=date.fromisoformat(p["delivery"]),
            repo_rate=p["repo_rate"],
            repo_day_count=DayCountConvention(p.get("repo_day_count", "ACT/360")),
        )


from pricebook.serialisable import _register as _reg_bf
BondForward._SERIAL_TYPE = "bond_forward"
_reg_bf(BondForward)
