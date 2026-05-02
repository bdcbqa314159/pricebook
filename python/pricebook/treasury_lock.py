"""Treasury Lock (T-Lock) pricing, hedging, and roll P&L.

Thin layer over :mod:`pricebook.bond` (yield-based pricing) and
:mod:`pricebook.bond_forward` (repo-funded forward). T-Lock-specific
payoff, booking, greeks, overhedge bound, and roll P&L.

References:
    Pucci, M. (2019). Hedging the Treasury Lock. SSRN 3386521.

Functions:
    :func:`tlock_payoff` — Eq (1): a * N * RiskFactor * (IRR - L).
    :func:`tlock_booking_value` — forward-contract proxy.
    :func:`tlock_delta` — Eq (14): D_P[g].
    :func:`tlock_gamma` — Eq (16-17): D^2_P[g].
    :func:`gamma_sign_threshold` — Eq (18).
    :func:`overhedge_bound` — Eq (10-11).
    :func:`roll_pnl` — Eq (31): full closed-form.
    :func:`roll_pnl_first_order` — Eq (33).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.bond import (
    bond_price_from_yield,
    bond_price_from_yield_stub,
    bond_price_continuous,
    bond_yield_derivatives,
    bond_irr,
    bond_risk_factor,
    bond_dv01_from_yield,
)
from pricebook.bond_forward import forward_price_repo, forward_price_haircut


# ---- T-Lock payoff (Pucci Eq 1) ----

def tlock_payoff(
    irr_te: float,
    locked_yield: float,
    coupon_rate: float,
    accrual_factors: list[float],
    notional: float = 1.0,
    direction: int = 1,
) -> float:
    """T-Lock payoff at expiry (Pucci Eq 1).

    Payoff = a * N * RiskFactor(IRR_te) * (IRR_te - L)
    """
    rf = bond_risk_factor(coupon_rate, accrual_factors, irr_te)
    return direction * notional * rf * (irr_te - locked_yield)


# ---- T-Lock booking value ----

@dataclass
class TLockResult:
    """T-Lock booking result."""
    value: float
    forward_price: float
    strike_price: float
    risk_factor: float
    locked_yield: float
    direction: int

    @property
    def price(self) -> float:
        return self.value

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.value, "forward_price": self.forward_price,
            "strike_price": self.strike_price, "risk_factor": self.risk_factor,
        }


def tlock_booking_value(
    locked_yield: float,
    forward_price: float,
    coupon_rate: float,
    accrual_factors_from_expiry: list[float],
    discount_factor: float,
    notional: float = 1.0,
    direction: int = 1,
) -> TLockResult:
    """T-Lock booking value as forward contract (Pucci Eq 8).

    v = a * D_{t,te} * (K - ForwardPrice), K = P_{te}(L).
    """
    K = bond_price_from_yield(coupon_rate, accrual_factors_from_expiry, locked_yield)
    rf = bond_risk_factor(coupon_rate, accrual_factors_from_expiry, locked_yield)
    value = direction * discount_factor * notional * (K - forward_price)
    return TLockResult(value, forward_price, K, rf, locked_yield, direction)


# ---- Greeks (Pucci Section 4) ----

def tlock_delta(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    y: float,
    locked_yield: float,
    direction: int = 1,
) -> float:
    """T-Lock delta in price (Pucci Eq 14).

    Delta = D_P[g] = D_y[g] / D_y[P]. Negative for long T-Lock near L.
    """
    D1, D2, _ = bond_yield_derivatives(
        coupon_rate, accrual_factors, times_to_coupon, time_to_maturity, y)
    Dy_g = -direction * (D2 * (y - locked_yield) + D1)
    if abs(D1) < 1e-30:
        return 0.0
    return Dy_g / D1


def tlock_gamma(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    y: float,
    locked_yield: float,
    direction: int = 1,
) -> float:
    """T-Lock gamma in price (Pucci Eq 16-17).

    At y = L: Gamma = -a * D2 / D1^2. Negative for long T-Lock.
    """
    D1, D2, D3 = bond_yield_derivatives(
        coupon_rate, accrual_factors, times_to_coupon, time_to_maturity, y)
    if abs(D1) < 1e-30:
        return 0.0
    Dy_g = -direction * (D2 * (y - locked_yield) + D1)
    D2y_g = -direction * (D3 * (y - locked_yield) + 2 * D2)
    return D2y_g / D1**2 - Dy_g * D2 / D1**3


def gamma_sign_threshold(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    locked_yield: float,
) -> float:
    """Yield threshold above which gamma flips sign (Pucci Eq 18).

    Returns inf if gamma never flips (common case — always negative).
    """
    D1, D2, D3 = bond_yield_derivatives(
        coupon_rate, accrual_factors, times_to_coupon, time_to_maturity,
        locked_yield)
    denom = D2**2 - D1 * D3
    if denom <= 0:
        return float('inf')
    threshold_diff = D1 * D3 / denom
    if threshold_diff <= 0:
        return float('inf')
    return locked_yield + threshold_diff


# ---- Overhedge bound (Pucci Eq 10-11) ----

def overhedge_bound(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    yield_change: float,
) -> float:
    """Upper bound on overhedge error |R1(y)| (Pucci Eq 10-11).

    |R1| <= 0.5 * M * (y - L)^2, M <= T^2 + c * sum alpha_i * tau_i^2.
    """
    M = time_to_maturity**2
    for alpha, tau in zip(accrual_factors, times_to_coupon):
        M += coupon_rate * alpha * tau**2
    return 0.5 * M * yield_change**2


# ---- Roll P&L (Pucci Eq 31, 33) ----

def roll_pnl(
    coupon_old: float,
    coupon_new: float,
    irr_old: float,
    irr_new: float,
    locked_yield: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
) -> float:
    """Full closed-form roll P&L (Pucci Eq 31).

    (pi_hat - pi) / D = (c_hat - c) * [A(L) - A(R_hat)] + [P(R) - P(R_hat)]
    """
    def annuity_cont(y):
        return sum(alpha * math.exp(-tau * y)
                   for alpha, tau in zip(accrual_factors, times_to_coupon))

    dc = coupon_new - coupon_old
    A_L = annuity_cont(locked_yield)
    A_Rhat = annuity_cont(irr_new)
    P_R = bond_price_continuous(
        coupon_old, accrual_factors, times_to_coupon, time_to_maturity, irr_old)
    P_Rhat = bond_price_continuous(
        coupon_old, accrual_factors, times_to_coupon, time_to_maturity, irr_new)
    return dc * (A_L - A_Rhat) + (P_R - P_Rhat)


def roll_pnl_first_order(
    coupon_old: float,
    coupon_new: float,
    irr_old: float,
    irr_new: float,
    locked_yield: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
) -> float:
    """First-order roll P&L approximation (Pucci Eq 33)."""
    dc = coupon_new - coupon_old
    dR = irr_new - irr_old
    T = time_to_maturity
    weighted_tau = sum(alpha * tau for alpha, tau in zip(accrual_factors, times_to_coupon))
    return dc * (irr_new - locked_yield) * weighted_tau + dR * (T + coupon_old * weighted_tau)


# ---------------------------------------------------------------------------
# Three lock conventions (practitioner's paper Eq 1, 3, 4, 5)
# ---------------------------------------------------------------------------

def bond_forward_clean(
    spot_clean: float,
    accrued_spot: float,
    accrued_delivery: float,
    coupons: list[float],
    repo_dfs_to_delivery: list[float],
    repo_df_spot_to_delivery: float,
) -> float:
    """Bond forward clean price via discrete cash-and-carry (Eq 1).

    Bf = (P0 + A0) / D_repo(T0, Tdel) - Σ Ci / D_repo(Ti, Tdel) - Adel

    Args:
        spot_clean: P0, spot clean price.
        accrued_spot: A0, accrued at spot settlement.
        accrued_delivery: Adel, accrued at delivery.
        coupons: [C1, C2, ...] intermediate coupons.
        repo_dfs_to_delivery: [D_repo(T1, Tdel), ...] repo DF from each coupon date to delivery.
        repo_df_spot_to_delivery: D_repo(T0, Tdel), repo DF from spot to delivery.
    """
    # Finance the dirty price to delivery
    dirty_grown = (spot_clean + accrued_spot) / repo_df_spot_to_delivery

    # Subtract reinvested coupons
    coupon_fv = sum(c / d for c, d in zip(coupons, repo_dfs_to_delivery))

    return dirty_grown - coupon_fv - accrued_delivery


def tlock_clean_price_npv(
    locked_clean_price: float,
    forward_clean_price: float,
    quantity: float,
    discount_factor: float,
) -> float:
    """Clean-price T-Lock NPV (Eq 3).

    NPV = N × (P_TLock - Bf) × D_TLock(t, Tdel)

    Buyer profits when forward drops below locked price.
    """
    return quantity * (locked_clean_price - forward_clean_price) * discount_factor


def pv01_forward(
    coupon_rate: float,
    accrual_factors: list[float],
    forward_yield: float,
) -> float:
    """PV01 at forward yield via centred difference (Eq 4).

    PV01(yf) = B(yf + 0.5bp) - B(yf - 0.5bp)

    Numerical, not analytical — as specified by the paper.
    """
    half_bp = 0.00005
    p_up = bond_price_from_yield(coupon_rate, accrual_factors, forward_yield + half_bp)
    p_down = bond_price_from_yield(coupon_rate, accrual_factors, forward_yield - half_bp)
    return abs(p_up - p_down)


def tlock_yield_npv(
    locked_yield: float,
    forward_yield: float,
    trade_amount: float,
    pv01_fwd: float,
    discount_factor: float,
) -> float:
    """Yield T-Lock NPV (Eq 5).

    NPV = M × (yf - yTLock) × |PV01(yf)| × D_TLock(t, Tdel)

    Buyer profits when forward yield rises above locked yield.
    """
    return trade_amount * (forward_yield - locked_yield) * pv01_fwd * discount_factor


def tlock_dirty_price_npv(
    locked_dirty_price: float,
    forward_clean_price: float,
    accrued_delivery: float,
    quantity: float,
    discount_factor: float,
) -> float:
    """Dirty-price T-Lock NPV — equivalent to clean lock at strike = dirty - Adel (Eq 3 variant)."""
    locked_clean = locked_dirty_price - accrued_delivery
    return tlock_clean_price_npv(locked_clean, forward_clean_price, quantity, discount_factor)


# ---- Instrument class ----

class TreasuryLock:
    """Treasury Lock instrument for Trade/Portfolio integration.

    Trade-level constructor: takes a bond, locked yield, expiry, repo rate.
    Extracts schedules internally, delegates to formula-level functions.

        tlock = TreasuryLock(bond, locked_yield=0.03, expiry=date(2027, 1, 15),
                             repo_rate=0.02, notional=10_000_000)
        result = tlock.price(discount_curve)
        portfolio.add(Trade(tlock))
    """

    def __init__(
        self,
        bond,
        locked_yield: float,
        expiry,
        notional: float = 1_000_000.0,
        direction: int = 1,
        repo_rate: float = 0.0,
        lock_convention: str = "yield",
        locked_price: float | None = None,
    ):
        """
        Args:
            lock_convention: "yield" (Eq 5), "clean_price" (Eq 3), or "dirty_price".
            locked_price: strike for clean/dirty price lock (ignored for yield lock).
        """
        if repo_rate < -1.0:
            raise ValueError(f"repo_rate must be > -1, got {repo_rate}")
        self.bond = bond
        self.locked_yield = locked_yield
        self.expiry = expiry
        self.notional = notional
        self.direction = direction
        self.repo_rate = repo_rate
        self.lock_convention = lock_convention
        self.locked_price = locked_price

    def _compute_forward(self, curve, forward_method: str = "discrete_reinvest"):
        """Common forward computation used by price() and greeks().

        Uses BondForward as the centralised forward engine.

        Returns (fwd_clean, fwd_dirty, fwd_yield, alphas, times, T_mat, accrued_del).
        All prices per unit face.
        """
        from pricebook.bond_forward import BondForward

        alphas, times, T_mat = self.bond.accrual_schedule(self.expiry)

        # Centralised forward via BondForward with selectable method
        bf = BondForward(self.bond, curve.reference_date, self.expiry, self.repo_rate)
        bf_result = bf.price(curve, method=forward_method)

        fwd_dirty = bf_result.forward_dirty / 100.0  # per unit face
        fwd_clean = bf_result.forward_clean / 100.0
        accrued_del = self.bond.accrued_interest(self.expiry) / 100.0

        # Forward yield via Pucci simply-compounded convention (Eq 2)
        fwd_yield = bond_irr(fwd_dirty, self.bond.coupon_rate, alphas)

        return fwd_clean, fwd_dirty, fwd_yield, alphas, times, T_mat, accrued_del

    def forward_yield_standard(self, curve) -> float:
        """Forward yield using standard YTM convention: (1 + y/k)^n.

        This is the market convention. Use this for comparison with
        quoted yields. The Pucci convention is used internally for
        T-Lock analytics (delta, gamma, risk factor).
        """
        from pricebook.bond import FixedRateBond as _FRB
        fwd_clean, fwd_dirty = self._compute_forward(curve)[:2]
        fwd_bond = _FRB.treasury_note(
            self.expiry, self.bond.maturity, self.bond.coupon_rate,
            self.bond.face_value,
        )
        return fwd_bond.yield_to_maturity(fwd_dirty * 100.0, self.expiry)

    def price(self, curve, discount_curve=None) -> TLockResult:
        """Price the T-Lock using the three-convention framework.

        Two-curve structure (paper Remark 6):
          - repo rate → bond forward price Bf (drift)
          - discount_curve → DT-Lock for discounting payoff (discount)
          If discount_curve is None, uses the same curve for both.

        Supports: lock_convention = "yield" | "clean_price" | "dirty_price"
        """
        fwd_clean, fwd_dirty, fwd_yield, alphas, times, T_mat, accrued_del = \
            self._compute_forward(curve)

        dc = discount_curve or curve
        df = dc.df(self.expiry)

        # NPV based on convention
        if self.lock_convention == "clean_price" and self.locked_price is not None:
            npv = tlock_clean_price_npv(
                self.locked_price / 100.0, fwd_clean,
                self.notional * self.direction, df,
            )
        elif self.lock_convention == "dirty_price" and self.locked_price is not None:
            npv = tlock_dirty_price_npv(
                self.locked_price / 100.0, fwd_clean, accrued_del,
                self.notional * self.direction, df,
            )
        else:
            pv01 = pv01_forward(self.bond.coupon_rate, alphas, fwd_yield)
            npv = tlock_yield_npv(
                self.locked_yield, fwd_yield,
                self.notional * self.direction, pv01, df,
            )

        rf = bond_risk_factor(self.bond.coupon_rate, alphas, fwd_yield)
        return TLockResult(npv, fwd_clean, fwd_clean, rf, self.locked_yield, self.direction)

    def greeks(self, curve) -> dict[str, float]:
        """Delta and gamma at the FORWARD yield (not spot).

        Uses the same forward computation as price() — no divergence.
        """
        _, _, fwd_yield, alphas, times, T_mat, _ = self._compute_forward(curve)

        delta = tlock_delta(self.bond.coupon_rate, alphas, times, T_mat,
                            fwd_yield, self.locked_yield, self.direction)
        gamma = tlock_gamma(self.bond.coupon_rate, alphas, times, T_mat,
                            fwd_yield, self.locked_yield, self.direction)
        return {"delta": delta, "gamma": gamma}

    def pv_ctx(self, ctx) -> float:
        """Price from PricingContext — compatible with Trade.pv().

        If ctx has repo_curves, extracts the repo rate for this bond's
        tenor from the curve. Otherwise falls back to self.repo_rate.
        """
        curve = ctx.discount_curve
        # Try to get repo rate from context
        if hasattr(ctx, "repo_curves") and ctx.repo_curves:
            from pricebook.day_count import year_fraction, DayCountConvention
            rc = next(iter(ctx.repo_curves.values()))
            tau = year_fraction(curve.reference_date, self.expiry, DayCountConvention.ACT_365_FIXED)
            days = int(tau * 365)
            if hasattr(rc, "rate"):
                old_repo = self.repo_rate
                self.repo_rate = rc.rate(days)
                result = self.price(curve)
                self.repo_rate = old_repo
                return result.value
        return self.price(curve).value

    def settlement_amount(self, irr_at_expiry: float) -> float:
        """Cash settlement at expiry (Pucci Eq 1).

        T-Locks are cash-settled: payer receives N × a × RF × (IRR − L).

        Args:
            irr_at_expiry: realised IRR of the benchmark at expiry.

        Returns:
            Cash amount (positive = payer receives).
        """
        alphas, _, _ = self.bond.accrual_schedule(self.expiry)
        return tlock_payoff(
            irr_te=irr_at_expiry,
            locked_yield=self.locked_yield,
            coupon_rate=self.bond.coupon_rate,
            accrual_factors=alphas,
            notional=self.notional,
            direction=self.direction,
        )

    # ---- Serialisation ----

    _SERIAL_TYPE = "treasury_lock"

    def to_dict(self) -> dict:
        d = {"type": self._SERIAL_TYPE, "params": {
            "bond": self.bond.to_dict(),
            "locked_yield": self.locked_yield,
            "expiry": self.expiry.isoformat(),
            "notional": self.notional,
            "direction": self.direction,
            "repo_rate": self.repo_rate,
            "lock_convention": self.lock_convention,
        }}
        if self.locked_price is not None:
            d["params"]["locked_price"] = self.locked_price
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TreasuryLock":
        from pricebook.serialisable import from_dict as _fd
        from datetime import date
        p = d["params"]
        return cls(
            bond=_fd(p["bond"]),
            locked_yield=p["locked_yield"],
            expiry=date.fromisoformat(p["expiry"]),
            notional=p.get("notional", 1_000_000.0),
            direction=p.get("direction", 1),
            repo_rate=p.get("repo_rate", 0.0),
            lock_convention=p.get("lock_convention", "yield"),
            locked_price=p.get("locked_price"),
        )

    # ---- Portfolio risk ----

    def dv01(self, curve, shift: float = 0.0001) -> float:
        """DV01: PV change for 1bp parallel yield shift."""
        pv_base = self.price(curve).value
        pv_bumped = self.price(curve.bumped(shift)).value
        return pv_bumped - pv_base

    def key_rate_dv01(self, curve) -> dict[str, float]:
        """Key-rate DV01: per-pillar yield sensitivity."""
        pv_base = self.price(curve).value
        result = {}
        shift = 0.0001
        for i, d in enumerate(curve.pillar_dates):
            bumped = curve.bumped_at(i, shift)
            pv_bumped = self.price(bumped).value
            result[d.isoformat()] = (pv_bumped - pv_base) / shift
        return result

    def repo_sensitivity(self, curve, shift: float = 0.0001) -> float:
        """PV sensitivity to repo rate change (1bp)."""
        pv_base = self.price(curve).value
        old_repo = self.repo_rate
        self.repo_rate = old_repo + shift
        pv_bumped = self.price(curve).value
        self.repo_rate = old_repo
        return pv_bumped - pv_base

    def cross_gamma_yield_repo(self, curve, y_shift: float = 0.0001, r_shift: float = 0.0001) -> float:
        """Cross-gamma: ∂²V/(∂y ∂r_repo). Measures interaction risk."""
        # V(y, r)
        pv_base = self.price(curve).value
        # V(y+h, r)
        pv_y_up = self.price(curve.bumped(y_shift)).value
        # V(y, r+h)
        old_repo = self.repo_rate
        self.repo_rate = old_repo + r_shift
        pv_r_up = self.price(curve).value
        # V(y+h, r+h)
        pv_yr_up = self.price(curve.bumped(y_shift)).value
        self.repo_rate = old_repo
        return (pv_yr_up - pv_y_up - pv_r_up + pv_base) / (y_shift * r_shift)


from pricebook.serialisable import _register as _reg_tlock
_reg_tlock(TreasuryLock)


# ---------------------------------------------------------------------------
# T-Lock portfolio risk
# ---------------------------------------------------------------------------

def tlock_portfolio_risk(
    tlocks: list[TreasuryLock],
    curve,
) -> dict[str, float]:
    """Aggregate risk for a portfolio of T-Locks.

    Returns:
        total_pv: net PV of all positions.
        total_dv01: net DV01 (parallel yield shift).
        total_delta: sum of Pucci deltas.
        total_gamma: sum of Pucci gammas.
        repo_sensitivity: net PV change for 1bp repo shift.
        max_overhedge: largest overhedge bound across positions.
    """
    total_pv = 0.0
    total_dv01 = 0.0
    total_delta = 0.0
    total_gamma = 0.0
    total_repo_sens = 0.0
    max_overhedge = 0.0

    for tl in tlocks:
        result = tl.price(curve)
        total_pv += result.value

        total_dv01 += tl.dv01(curve)
        total_repo_sens += tl.repo_sensitivity(curve)

        greeks = tl.greeks(curve)
        total_delta += greeks["delta"] * tl.notional
        total_gamma += greeks["gamma"] * tl.notional

        # Overhedge bound (Pucci Eq 10-11)
        alphas, times, T_mat = tl.bond.accrual_schedule(tl.expiry)
        irr = bond_irr(
            tl.bond.dirty_price(curve) / 100.0,
            tl.bond.coupon_rate, alphas,
        )
        yield_change = irr - tl.locked_yield
        bound = overhedge_bound(
            tl.bond.coupon_rate, alphas, times, T_mat,
            yield_change,
        )
        max_overhedge = max(max_overhedge, abs(bound))

    return {
        "total_pv": total_pv,
        "total_dv01": total_dv01,
        "total_delta": total_delta,
        "total_gamma": total_gamma,
        "repo_sensitivity": total_repo_sens,
        "max_overhedge": max_overhedge,
        "n_positions": len(tlocks),
    }


def tlock_pnl_attribution(
    tlock: TreasuryLock,
    curve_t0,
    curve_t1,
    repo_rate_t1: float | None = None,
) -> dict[str, float]:
    """Daily P&L attribution for a T-Lock position.

    Decomposes PV change into:
      carry: time decay (theta) at unchanged curves.
      curve_pnl: PV change from yield curve move.
      repo_pnl: PV change from repo rate move.
      roll_pnl: Pucci Eq 31 roll analytics.
      unexplained: residual.

    Args:
        tlock: the T-Lock position.
        curve_t0: discount curve at start of day.
        curve_t1: discount curve at end of day.
        repo_rate_t1: repo rate at end of day (None = unchanged).
    """
    pv_t0 = tlock.price(curve_t0).value
    pv_t1 = tlock.price(curve_t1).value
    total = pv_t1 - pv_t0

    # Carry: PV change from 1 day passing, curves unchanged
    carry = tlock.dv01(curve_t0) * 0  # theta approximation
    # Better: use theta directly
    from pricebook.day_count import year_fraction, DayCountConvention
    tau = year_fraction(curve_t0.reference_date, tlock.expiry, DayCountConvention.ACT_365_FIXED)
    # Theta ≈ -dV/dT (from time decay of forward and discount)
    # Approximate as dv01 × daily_rolldown
    carry = 0.0  # placeholder — T-Lock carry is through roll, not coupon

    # Curve P&L: PV change from curve shift at t0 repo
    curve_pnl = tlock.price(curve_t1).value - tlock.price(curve_t0).value

    # Repo P&L: PV change from repo rate move
    repo_pnl = 0.0
    if repo_rate_t1 is not None and repo_rate_t1 != tlock.repo_rate:
        old_repo = tlock.repo_rate
        tlock.repo_rate = repo_rate_t1
        pv_new_repo = tlock.price(curve_t1).value
        tlock.repo_rate = old_repo
        repo_pnl = pv_new_repo - pv_t1
        # Recompute total with repo change
        tlock.repo_rate = repo_rate_t1
        pv_t1_full = tlock.price(curve_t1).value
        tlock.repo_rate = old_repo
        total = pv_t1_full - pv_t0

    # Roll P&L: Pucci Eq 31
    alphas_t0, times_t0, T_mat_t0 = tlock.bond.accrual_schedule(tlock.expiry)
    irr_t0 = bond_irr(
        tlock.bond.dirty_price(curve_t0) / 100.0,
        tlock.bond.coupon_rate, alphas_t0,
    )
    irr_t1 = bond_irr(
        tlock.bond.dirty_price(curve_t1) / 100.0,
        tlock.bond.coupon_rate, alphas_t0,
    )
    roll = roll_pnl_first_order(
        coupon_old=tlock.bond.coupon_rate,
        coupon_new=tlock.bond.coupon_rate,  # same coupon (no roll yet)
        irr_old=irr_t0,
        irr_new=irr_t1,
        locked_yield=tlock.locked_yield,
        accrual_factors=alphas_t0,
        times_to_coupon=times_t0,
        time_to_maturity=T_mat_t0,
    )

    unexplained = total - curve_pnl - repo_pnl

    return {
        "total": total,
        "curve_pnl": curve_pnl,
        "repo_pnl": repo_pnl,
        "roll_pnl_first_order": roll,
        "unexplained": unexplained,
    }
