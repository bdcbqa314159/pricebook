"""Repo desk: position management, GC/special tracking, financing optimisation.

Builds on :mod:`pricebook.bond_desk` (``RepoPosition``, ``repo_carry``,
``securities_lending_fee``) and :mod:`pricebook.funded` (``Repo``) with
desk-level tooling for repo operations.

* :class:`RepoBook` — positions by counterparty, collateral type, term.
* :func:`repo_rate_monitor` — z-score the current repo rate vs history.
* :func:`cheapest_to_deliver_repo` — select the bond that minimises
  financing cost.
* :func:`term_vs_overnight` — compare locking in term repo vs rolling
  overnight.
* :class:`FailsTracker` — track and cost settlement fails.

    book = RepoBook("GovtRepo")
    book.add(entry)
    pnl = book.net_carry()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import math

from enum import Enum

from pricebook.day_count import year_fraction as _year_fraction
from pricebook.zscore import zscore as _zscore, ZScoreSignal


class RepoDirection(str, Enum):
    """Repo direction: repo (lend bond) or reverse (lend cash)."""
    REPO = "repo"
    REVERSE = "reverse"


# ---------------------------------------------------------------------------
# Unified RepoTrade — fixes gaps 1, 2, 3, 5, 6, 7, 8
# ---------------------------------------------------------------------------

class RepoTrade:
    """A repo trade that can price itself, track lifecycle, and sit in a Portfolio.

    Unifies the old RepoTradeEntry (desk metadata) with Repo (pricing).
    Supports: pv(curve), pv_ctx(ctx), Trade(repo_trade), roll, margin.

    Args:
        counterparty: repo counterparty name.
        collateral_issuer: bond/issuer identifier.
        collateral_type: "GC" or "special".
        face_amount: face value of collateral.
        bond_price: dirty price at trade inception (per 100).
        repo_rate: annualised simple rate (ACT/360).
        term_days: repo term in calendar days (0 = open repo).
        coupon_rate: annual coupon of the collateral bond.
        direction: "repo" (lend bond, borrow cash) or "reverse".
        start_date: trade date.
        haircut: overcollateralisation (e.g. 0.02 = 2%).
        status: "live", "matured", "terminated", "rolled".
        rate_type: "fixed" (simple), "sofr_compound" (compounded ON).
    """

    def __init__(
        self,
        counterparty: str,
        collateral_issuer: str,
        collateral_type: str = "GC",
        face_amount: float = 0.0,
        bond_price: float = 100.0,
        repo_rate: float = 0.0,
        term_days: int = 1,
        coupon_rate: float = 0.0,
        direction: str | RepoDirection = "repo",
        start_date: date | None = None,
        haircut: float = 0.0,
        status: str = "live",
        rate_type: str = "fixed",
        trade_id: str = "",
        bond=None,
        settlement_days: int = 1,
        bond_currency: str = "USD",
        cash_currency: str = "USD",
        fx_rate: float = 1.0,
        fx_haircut: float = 0.0,
    ):
        if face_amount < 0:
            raise ValueError(f"face_amount must be >= 0, got {face_amount}")
        if haircut < 0 or haircut >= 1:
            raise ValueError(f"haircut must be in [0, 1), got {haircut}")
        if fx_rate <= 0:
            raise ValueError(f"fx_rate must be positive, got {fx_rate}")
        if fx_haircut < 0:
            raise ValueError(f"fx_haircut must be >= 0, got {fx_haircut}")
        if haircut + fx_haircut >= 1:
            raise ValueError(f"total haircut (haircut + fx_haircut) must be < 1, got {haircut + fx_haircut}")
        self.counterparty = counterparty
        self.collateral_issuer = collateral_issuer
        self.collateral_type = collateral_type
        self.face_amount = face_amount
        self.bond_price = bond_price
        self.repo_rate = repo_rate
        self.term_days = term_days
        self.coupon_rate = coupon_rate
        # Normalize direction to string for backward compat (== "repo" still works)
        self.direction = RepoDirection(direction).value
        self.start_date = start_date
        self.haircut = haircut
        self.status = status
        self.rate_type = rate_type
        self.trade_id = trade_id
        self.bond = bond  # Issue 6: reference to FixedRateBond (optional)
        self.settlement_days = settlement_days  # Issue 5: T+1 for UST
        self._margin_posted: float = 0.0
        # Cross-currency fields
        self.bond_currency = bond_currency
        self.cash_currency = cash_currency
        self.fx_rate = fx_rate            # cash_ccy per 1 bond_ccy
        self.fx_haircut = fx_haircut      # extra haircut for FX mismatch

    # ---- Core properties ----

    @property
    def settlement_date(self) -> date | None:
        """Settlement date = trade date + settlement_days (Issue 5)."""
        if self.start_date is None:
            return None
        return self.start_date + timedelta(days=self.settlement_days)

    @property
    def market_value(self) -> float:
        """Market value of the collateral."""
        return self.face_amount * self.bond_price / 100.0

    @property
    def is_cross_currency(self) -> bool:
        return self.bond_currency != self.cash_currency

    @property
    def cash_amount(self) -> float:
        """Cash lent/borrowed (after haircut + FX haircut).

        For cross-currency repos: bond value converted at fx_rate,
        then total haircut = haircut + fx_haircut applied.
        """
        mv = self.market_value
        if self.is_cross_currency:
            mv = mv * self.fx_rate  # convert to cash currency
        total_haircut = self.haircut + self.fx_haircut
        return mv * (1 - total_haircut)

    @property
    def maturity_date(self) -> date | None:
        """Maturity date = settlement_date + term_days (None for open repos).

        Term runs from settlement, not trade date. For T+1 UST repos,
        maturity is 1 day later than trade + term.
        """
        sd = self.settlement_date
        if sd is None or self.term_days == 0:
            return None
        return sd + timedelta(days=self.term_days)

    @property
    def is_open(self) -> bool:
        return self.term_days == 0

    @property
    def repurchase_amount(self) -> float:
        """Amount to repay at maturity (ACT/360)."""
        dt = self.term_days / 360.0
        return self.cash_amount * (1 + self.repo_rate * dt)

    @property
    def interest(self) -> float:
        return self.repurchase_amount - self.cash_amount

    @property
    def financing_cost(self) -> float:
        """Financing cost over the term (ACT/360)."""
        return self.cash_amount * self.repo_rate * self.term_days / 360.0

    @property
    def carry(self) -> float:
        """Net carry = coupon income − financing cost.

        Issue 1 fix: coupon uses bond's day count if bond is attached,
        otherwise ACT/365. Financing always ACT/360.
        """
        # Coupon accrual
        if self.bond is not None:
            sd = self.settlement_date or self.start_date
            mat = self.maturity_date
            if sd and mat:
                yf = _year_fraction(sd, mat, self.bond.day_count)
            else:
                yf = self.term_days / 365.0
            coupon = self.face_amount * self.coupon_rate * yf
        else:
            coupon = self.face_amount * self.coupon_rate * self.term_days / 365.0

        # Financing cost (always ACT/360)
        financing = self.cash_amount * self.repo_rate * self.term_days / 360.0

        sign = 1.0 if self.direction == "repo" else -1.0
        return sign * (coupon - financing)

    @property
    def effective_rate(self) -> float:
        """Effective funding rate accounting for haircut."""
        if self.haircut >= 1:
            return float("inf")
        return self.repo_rate / (1 - self.haircut)

    # ---- Coupon pass-through (Issue 2) ----

    def coupons_during_term(self) -> list[tuple[date, float]]:
        """Coupons paid on the bond during the repo term.

        In a repo, the bond buyer receives coupons and must pass them
        through to the seller (economic owner). This is called
        "manufactured payment" or "coupon pass-through".

        Returns list of (coupon_date, amount).
        """
        if self.bond is None:
            return []
        sd = self.settlement_date or self.start_date
        mat = self.maturity_date
        if sd is None or mat is None:
            return []
        result = []
        for cf in self.bond.coupon_leg.cashflows:
            if sd < cf.payment_date < mat:
                # Coupon at maturity belongs to the bond holder post-repo
                result.append((cf.payment_date, cf.amount))
        return result

    @property
    def coupon_pass_through(self) -> float:
        """Total coupon amount passed through during repo term."""
        return sum(amt for _, amt in self.coupons_during_term())

    # ---- Repo accrued interest (Issue 3) ----

    def accrued_interest(self, as_of: date) -> float:
        """Repo interest accrued from settlement to as_of date.

        For intraday/mid-term marking. ACT/360 convention.
        """
        sd = self.settlement_date or self.start_date
        if sd is None:
            return 0.0
        days_elapsed = max(0, (as_of - sd).days)
        days_elapsed = min(days_elapsed, self.term_days) if self.term_days > 0 else days_elapsed
        return self.cash_amount * self.repo_rate * days_elapsed / 360.0

    # ---- Mark-to-market (Issue 4) ----

    def mark_to_market(self, market_rate: float, as_of: date) -> float:
        """Replacement-value mark-to-market.

        The value of the position if you could close and re-enter at
        the current market repo rate for the remaining term.

        MTM = (contract_rate - market_rate) × cash × remaining_days / 360

        Positive = you locked in below market (ahead).
        Negative = you're above market (behind).
        """
        remaining = self.remaining_days(as_of)
        if remaining <= 0:
            return 0.0
        mtm = (self.repo_rate - market_rate) * self.cash_amount * remaining / 360.0
        # For repo direction: locked in low rate = good (negative mtm = you pay less)
        # For reverse direction: locked in high rate = good
        if self.direction == "repo":
            return -mtm  # positive when contract rate < market
        else:
            return mtm   # positive when contract rate > market

    # ---- Intraday snapshot ----

    def snapshot(
        self,
        as_of: date,
        current_bond_price: float | None = None,
        current_repo_rate: float | None = None,
        current_fx_rate: float | None = None,
    ) -> dict[str, float]:
        """Intraday position snapshot — everything you need at any point in time.

        Args:
            as_of: snapshot timestamp.
            current_bond_price: live bond dirty price (None = use inception).
            current_repo_rate: current market repo rate (None = use contract).
            current_fx_rate: current FX rate for xccy repos (None = use inception).
        """
        price = current_bond_price or self.bond_price
        rate = current_repo_rate or self.repo_rate
        fx = current_fx_rate or self.fx_rate

        remaining = self.remaining_days(as_of)
        accrued = self.accrued_interest(as_of)
        mtm = self.mark_to_market(rate, as_of) if rate != self.repo_rate else 0.0
        vm = self.variation_margin(price) if price != self.bond_price else 0.0

        # FX P&L for xccy
        fx_pnl = 0.0
        if self.is_cross_currency and fx != self.fx_rate:
            fx_pnl = self.market_value * (fx - self.fx_rate)

        return {
            "trade_id": self.trade_id,
            "as_of": as_of.isoformat(),
            "status": self.status,
            "remaining_days": remaining,
            "accrued_interest": accrued,
            "mark_to_market": mtm,
            "variation_margin": vm,
            "fx_pnl": fx_pnl,
            "current_price": price,
            "current_rate": rate,
            "current_fx": fx,
            "total_unrealised": mtm + vm + fx_pnl,
        }

    # ---- Cross-currency margin ----

    def xccy_margin_call(
        self,
        current_bond_price: float,
        current_fx_rate: float,
    ) -> float:
        """Margin call for cross-currency repos (all in cash currency).

        Computes entirely in cash currency to avoid mixing dimensions.
        margin_call = (current_collateral_value_cash - initial_collateral_value_cash)
                      × (haircut + fx_haircut)
        """
        initial_value_cash = self.market_value * self.fx_rate
        current_value_cash = self.face_amount * current_bond_price / 100.0 * current_fx_rate
        total_haircut = self.haircut + self.fx_haircut
        return (current_value_cash - initial_value_cash) * total_haircut

    # ---- Pricing (Gap 8) ----

    def pv(self, discount_curve, reference_date: date | None = None,
           projection_curve=None) -> float:
        """Present value against a discount curve.

        For fixed repos: PV = df(mat) × repurchase − cash.
        For floating repos: uses projection_curve for forward SOFR rates.
        """
        if reference_date is None and self.start_date is None:
            raise ValueError("reference_date required when start_date is None")
        ref = reference_date or self.start_date
        mat = self.maturity_date
        if mat is None:
            return 0.0  # open repo: PV = 0 at inception

        df = discount_curve.df(mat)

        if self.rate_type == "sofr_compound" and projection_curve is not None:
            repurchase = self.cash_amount + self.floating_interest(
                projection_curve=projection_curve)
        else:
            repurchase = self.repurchase_amount

        if self.direction == "repo":
            return df * repurchase - self.cash_amount
        else:
            return self.cash_amount - df * repurchase

    def pv_ctx(self, ctx) -> float:
        """Price from PricingContext — Trade/Portfolio integration.

        Uses projection_curves from context for SOFR repos.
        """
        proj = None
        if self.rate_type == "sofr_compound" and hasattr(ctx, "projection_curves"):
            if ctx.projection_curves:
                proj = next(iter(ctx.projection_curves.values()), None)
        return self.pv(ctx.discount_curve, projection_curve=proj)

    # ---- Variation Margin (Gap 3) ----

    def margin_required(self, current_price: float) -> float:
        """Margin required at current bond price.

        margin = face × current_price / 100 × haircut
        """
        return self.face_amount * current_price / 100.0 * self.haircut

    def margin_call(self, current_price: float) -> float:
        """Margin call amount: positive = must post more, negative = receive back.

        If bond drops → more margin needed (for cash lender protection).
        """
        new_margin = self.margin_required(current_price)
        initial_margin = self.market_value * self.haircut
        return new_margin - initial_margin

    def variation_margin(self, current_price: float) -> float:
        """Variation margin from price move.

        VM = face × (current - inception) / 100 (sign depends on direction).
        """
        price_move = (current_price - self.bond_price) / 100.0 * self.face_amount
        return price_move if self.direction == "reverse" else -price_move

    # ---- Lifecycle (Gap 7) ----

    def remaining_days(self, as_of: date | None = None) -> int:
        """Days remaining from as_of to maturity."""
        mat = self.maturity_date
        if mat is None:
            return 0
        if as_of is None:
            raise ValueError("as_of date required for remaining_days()")
        return max(0, (mat - as_of).days)

    def mature(self) -> None:
        """Mark the trade as matured."""
        self.status = "matured"

    def terminate_early(self, termination_date: date | None = None) -> None:
        """Early termination."""
        self.status = "terminated"

    def roll(self, new_rate: float, new_term_days: int, new_date: date | None = None) -> "RepoTrade":
        """Roll into a new repo at a new rate and term (Gap 2).

        Marks the current trade as rolled and returns the new trade.
        """
        self.status = "rolled"
        if new_date is None and self.maturity_date is None:
            raise ValueError("new_date required when maturity_date is None (open repo)")
        roll_date = new_date or self.maturity_date
        return RepoTrade(
            counterparty=self.counterparty,
            collateral_issuer=self.collateral_issuer,
            collateral_type=self.collateral_type,
            face_amount=self.face_amount,
            bond_price=self.bond_price,  # same price for now
            repo_rate=new_rate,
            term_days=new_term_days,
            coupon_rate=self.coupon_rate,
            direction=self.direction,
            start_date=roll_date,
            haircut=self.haircut,
            status="live",
            rate_type=self.rate_type,
            trade_id=self.trade_id + "_roll",
        )

    def roll_cost(self, new_rate: float, new_term_days: int) -> float:
        """Cost of rolling: difference in financing at new vs old rate."""
        old_cost = self.cash_amount * self.repo_rate * self.term_days / 360.0
        new_cost = self.cash_amount * new_rate * new_term_days / 360.0
        return new_cost - old_cost

    # ---- Floating-rate repo ----

    def floating_interest(
        self,
        daily_rates: list[float] | None = None,
        projection_curve=None,
    ) -> float:
        """Interest for a floating-rate (SOFR-linked) repo.

        Three modes:
        - rate_type="fixed": simple interest at repo_rate (ignores this method).
        - rate_type="sofr_compound" + daily_rates: compound from fixings.
        - rate_type="sofr_compound" + projection_curve: compound from curve forwards.

        For SOFR repos, the rate is:
          interest = cash × [∏(1 + SOFR_i / 360) - 1 + spread × days/360]

        where spread = repo_rate (the agreed spread over SOFR).
        """
        if self.rate_type == "fixed":
            return self.interest

        if daily_rates is not None:
            # From actual fixings
            from pricebook.rfr import compound_rfr
            day_fracs = [1.0 / 360.0] * len(daily_rates)
            compounded = compound_rfr(daily_rates, day_fracs)
            total_yf = len(daily_rates) / 360.0
            # Spread on top
            all_in = compounded + self.repo_rate * total_yf / max(total_yf, 1e-10)
            return self.cash_amount * all_in * total_yf

        if projection_curve is not None:
            # From curve forwards (for projection / pricing)
            from pricebook.rfr import compound_rfr_from_curve
            mat = self.maturity_date
            if mat is None or self.start_date is None:
                return self.interest
            fwd_rate = compound_rfr_from_curve(projection_curve, self.start_date, mat)
            dt = self.term_days / 360.0
            # All-in = compounded SOFR + spread
            return self.cash_amount * (fwd_rate + self.repo_rate) * dt

        # Fallback: use repo_rate as all-in
        return self.interest

    def sofr_interest(self, daily_sofr_rates: list[float]) -> float:
        """Backward-compat alias for floating_interest with fixings."""
        return self.floating_interest(daily_rates=daily_sofr_rates)

    # ---- Serialisation ----

    def to_dict(self) -> dict:
        return {"type": "repo_trade", "params": {
            "counterparty": self.counterparty,
            "collateral_issuer": self.collateral_issuer,
            "collateral_type": self.collateral_type,
            "face_amount": self.face_amount,
            "bond_price": self.bond_price,
            "repo_rate": self.repo_rate,
            "term_days": self.term_days,
            "coupon_rate": self.coupon_rate,
            "direction": self.direction,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "haircut": self.haircut,
            "status": self.status,
            "rate_type": self.rate_type,
            "trade_id": self.trade_id,
            "bond_currency": self.bond_currency,
            "cash_currency": self.cash_currency,
            "fx_rate": self.fx_rate,
            "fx_haircut": self.fx_haircut,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> "RepoTrade":
        p = d["params"]
        sd = date.fromisoformat(p["start_date"]) if p.get("start_date") else None
        return cls(
            counterparty=p["counterparty"],
            collateral_issuer=p["collateral_issuer"],
            collateral_type=p.get("collateral_type", "GC"),
            face_amount=p["face_amount"],
            bond_price=p["bond_price"],
            repo_rate=p["repo_rate"],
            term_days=p["term_days"],
            coupon_rate=p.get("coupon_rate", 0.0),
            direction=p.get("direction", "repo"),
            start_date=sd,
            haircut=p.get("haircut", 0.0),
            status=p.get("status", "live"),
            rate_type=p.get("rate_type", "fixed"),
            trade_id=p.get("trade_id", ""),
            bond_currency=p.get("bond_currency", "USD"),
            cash_currency=p.get("cash_currency", "USD"),
            fx_rate=p.get("fx_rate", 1.0),
            fx_haircut=p.get("fx_haircut", 0.0),
        )

    # Backward compat: convert from old RepoTradeEntry
    @classmethod
    def from_entry(cls, entry: "RepoTradeEntry") -> "RepoTrade":
        return cls(
            counterparty=entry.counterparty,
            collateral_issuer=entry.collateral_issuer,
            collateral_type=entry.collateral_type,
            face_amount=entry.face_amount,
            bond_price=entry.bond_price,
            repo_rate=entry.repo_rate,
            term_days=entry.term_days,
            coupon_rate=entry.coupon_rate,
            direction=entry.direction,
            start_date=entry.start_date,
        )


    # ---- Product type constructors (Phase 3) ----

    @classmethod
    def buy_sell_back(
        cls,
        counterparty: str,
        collateral_issuer: str,
        face_amount: float,
        spot_dirty_price: float,
        forward_dirty_price: float,
        term_days: int,
        start_date: date,
        **kwargs,
    ) -> "RepoTrade":
        """Buy/sell-back: two separate cash trades (not a repo legally).

        Sell at spot dirty, agree to buy back at forward dirty.
        The implied repo rate = (forward / spot - 1) × 360 / term.
        Forward price includes accrued (no separate interest).
        """
        if spot_dirty_price <= 0:
            raise ValueError("spot_dirty_price must be positive")
        implied_rate = (forward_dirty_price / spot_dirty_price - 1) * 360.0 / term_days
        return cls(
            counterparty=counterparty, collateral_issuer=collateral_issuer,
            face_amount=face_amount, bond_price=spot_dirty_price,
            repo_rate=implied_rate, term_days=term_days, start_date=start_date,
            direction="repo", **kwargs,
        )

    @classmethod
    def repo_to_maturity(
        cls,
        counterparty: str,
        bond,
        repo_rate: float,
        start_date: date,
        **kwargs,
    ) -> "RepoTrade":
        """Repo-to-maturity: term = bond maturity - settlement.

        Finances the bond from now until it matures.
        """
        from pricebook.day_count import year_fraction, DayCountConvention
        settlement = start_date + timedelta(days=kwargs.get("settlement_days", 1))
        term = (bond.maturity - settlement).days
        if term <= 0:
            raise ValueError(f"Bond already matured: {bond.maturity} <= {settlement}")
        dirty = bond.dirty_price(kwargs.pop("discount_curve")) if "discount_curve" in kwargs else kwargs.pop("bond_price", 100.0)
        return cls(
            counterparty=counterparty, collateral_issuer=str(bond.maturity),
            face_amount=kwargs.pop("face_amount", bond.face_value),
            bond_price=dirty, repo_rate=repo_rate,
            term_days=term, start_date=start_date,
            direction="repo", bond=bond,
            coupon_rate=bond.coupon_rate, **kwargs,
        )

    @classmethod
    def equity_repo(
        cls,
        counterparty: str,
        stock_id: str,
        shares: int,
        stock_price: float,
        repo_rate: float,
        term_days: int,
        start_date: date,
        dividend_yield: float = 0.0,
        **kwargs,
    ) -> "RepoTrade":
        """Equity repo: stock as collateral, higher haircuts.

        Uses regulatory haircut for equities (15-25%).
        Dividend pass-through instead of coupon.
        """
        from pricebook.repo_analytics import regulatory_haircut as _reg_hc
        haircut = kwargs.pop("haircut", _reg_hc("equity_main_index", 0) / 100.0)
        face = shares * stock_price  # "face" = market value for equities
        return cls(
            counterparty=counterparty, collateral_issuer=stock_id,
            collateral_type="equity",
            face_amount=face, bond_price=100.0,  # price=100 so MV = face
            repo_rate=repo_rate, term_days=term_days,
            coupon_rate=dividend_yield,  # dividend yield instead of coupon
            start_date=start_date, direction="repo",
            haircut=haircut, **kwargs,
        )


RepoTrade._SERIAL_TYPE = "repo_trade"
from pricebook.serialisable import _register as _reg_rt
_reg_rt(RepoTrade)


# ---------------------------------------------------------------------------
# Collateral Pool (Gap 4)
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Daily P&L (Gap 6)
# ---------------------------------------------------------------------------

@dataclass
class RepoDailyPnL:
    """Daily P&L for the repo book."""
    date: date
    total_pnl: float
    carry_pnl: float
    rate_pnl: float
    unexplained: float

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "total": self.total_pnl,
            "carry": self.carry_pnl, "rate": self.rate_pnl,
            "unexplained": self.unexplained,
        }


def repo_daily_pnl(
    book: RepoBook,
    curve_t0,
    curve_t1,
    ref_t0: date,
    ref_t1: date,
) -> RepoDailyPnL:
    """Daily P&L: PV change + carry accrual, attributed (Gap 6).

    total = pv(t1) - pv(t0)
    carry = 1-day carry accrual
    rate = total - carry (attributed to rate moves)
    """
    from pricebook.repo_desk import repo_book_pv

    pv_t0 = repo_book_pv(book, curve_t0, ref_t0)["total_pv"]
    pv_t1 = repo_book_pv(book, curve_t1, ref_t1)["total_pv"]
    total = pv_t1 - pv_t0

    # 1-day carry accrual
    daily_carry = book.net_carry() / max(
        sum(e.term_days for e in book.entries) / len(book.entries) if len(book) > 0 else 1, 1
    )

    rate_pnl = total - daily_carry
    unexplained = 0.0  # placeholder

    return RepoDailyPnL(ref_t1, total, daily_carry, rate_pnl, unexplained)


# ---- RepoTradeEntry: factory alias for RepoTrade (backward compat) ----

def RepoTradeEntry(
    counterparty: str,
    collateral_issuer: str,
    collateral_type: str = "GC",
    face_amount: float = 0.0,
    bond_price: float = 100.0,
    repo_rate: float = 0.0,
    term_days: int = 1,
    coupon_rate: float = 0.0,
    direction: str = "repo",
    start_date: date | None = None,
) -> RepoTrade:
    """Create a RepoTrade with haircut=0 (legacy compatibility).

    DEPRECATED: Use RepoTrade directly.
    """
    return RepoTrade(
        counterparty=counterparty,
        collateral_issuer=collateral_issuer,
        collateral_type=collateral_type,
        face_amount=face_amount,
        bond_price=bond_price,
        repo_rate=repo_rate,
        term_days=term_days,
        coupon_rate=coupon_rate,
        direction=direction,
        start_date=start_date,
        haircut=0.0,
    )


# ---- Repo book ----

@dataclass
class RepoCounterpartyExposure:
    """Aggregate repo exposure per counterparty."""
    counterparty: str
    total_cash: float
    n_trades: int
    avg_rate: float


@dataclass
class RepoCollateralSummary:
    """Aggregate by collateral type (GC vs special)."""
    collateral_type: str
    total_cash: float
    avg_rate: float
    n_trades: int


class RepoBook:
    """A collection of repo positions with aggregation.

    Stores RepoTrade objects. Accepts both RepoTrade and legacy
    RepoTradeEntry (auto-converted to RepoTrade).

    Args:
        name: book name (e.g. "GovtRepo", "IG_Repo").
    """

    def __init__(self, name: str):
        self.name = name
        self._entries: list[RepoTrade] = []

    def add(self, entry: RepoTrade) -> None:
        """Add a RepoTrade to the book."""
        self._entries.append(entry)

    @property
    def entries(self) -> list[RepoTrade]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def net_carry(self) -> float:
        """Total net carry across all repo positions."""
        return sum(e.carry for e in self._entries)

    def total_cash_out(self) -> float:
        """Total cash borrowed (repo direction)."""
        return sum(
            e.cash_amount for e in self._entries if e.direction == "repo"
        )

    def total_cash_in(self) -> float:
        """Total cash lent (reverse repo direction)."""
        return sum(
            e.cash_amount for e in self._entries if e.direction == "reverse"
        )

    def by_counterparty(self) -> list[RepoCounterpartyExposure]:
        """Aggregate exposure per counterparty (cash-weighted average rate)."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            cp = e.counterparty
            if cp not in agg:
                agg[cp] = {"cash": 0.0, "weighted_rate": 0.0, "count": 0}
            agg[cp]["cash"] += e.cash_amount
            agg[cp]["weighted_rate"] += e.cash_amount * e.repo_rate
            agg[cp]["count"] += 1

        return [
            RepoCounterpartyExposure(
                counterparty=cp,
                total_cash=d["cash"],
                n_trades=d["count"],
                avg_rate=d["weighted_rate"] / d["cash"] if d["cash"] > 0 else 0.0,
            )
            for cp, d in sorted(agg.items())
        ]

    def by_collateral_type(self) -> list[RepoCollateralSummary]:
        """Aggregate by GC vs special (cash-weighted average rate)."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            ct = e.collateral_type
            if ct not in agg:
                agg[ct] = {"cash": 0.0, "weighted_rate": 0.0, "count": 0}
            agg[ct]["cash"] += e.cash_amount
            agg[ct]["weighted_rate"] += e.cash_amount * e.repo_rate
            agg[ct]["count"] += 1

        return [
            RepoCollateralSummary(
                collateral_type=ct,
                total_cash=d["cash"],
                avg_rate=d["weighted_rate"] / d["cash"] if d["cash"] > 0 else 0.0,
                n_trades=d["count"],
            )
            for ct, d in sorted(agg.items())
        ]

    def gc_rate(self) -> float | None:
        """Weighted-average GC repo rate, or None if no GC trades."""
        gc = [e for e in self._entries if e.collateral_type == "GC"]
        if not gc:
            return None
        total_cash = sum(e.cash_amount for e in gc)
        if total_cash <= 0:
            return 0.0
        return sum(e.cash_amount * e.repo_rate for e in gc) / total_cash

    def special_rate(self, issuer: str) -> float | None:
        """Weighted-average special repo rate for a specific issuer."""
        sp = [
            e for e in self._entries
            if e.collateral_type == "special" and e.collateral_issuer == issuer
        ]
        if not sp:
            return None
        total_cash = sum(e.cash_amount for e in sp)
        if total_cash <= 0:
            return 0.0
        return sum(e.cash_amount * e.repo_rate for e in sp) / total_cash

    def positions(self) -> list[RepoTrade]:
        """Return all entries (desk protocol compat)."""
        return list(self._entries)

    def aggregate_risk(self, curve=None) -> dict:
        """Aggregate risk for cross-asset desk integration.

        DV01 computed per-position: sum of cash_i × (term_i / 360) per 1bp.
        """
        total_cash = 0.0
        total_carry = 0.0
        total_notional = 0.0
        total_dv01 = 0.0

        for e in self._entries:
            total_cash += e.cash_amount
            total_carry += e.carry
            total_notional += e.face_amount
            # Per-position DV01: interest change for 1bp rate move
            total_dv01 += e.cash_amount * e.term_days / 360.0 * 0.0001

        return {
            "total_pv": total_carry,
            "total_dv01": total_dv01,
            "total_notional": total_notional,
            "total_cash": total_cash,
            "total_carry": total_carry,
            "n_positions": len(self._entries),
        }


# ---- Repo rate monitor ----



# ---- Cheapest-to-deliver repo ----





# ---- Term vs overnight ----





# ---- Fails tracking ----





# ---------------------------------------------------------------------------
# Maturity / Cash Ladder
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Repo Rate DV01
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Carry P&L Decomposition
# ---------------------------------------------------------------------------

@dataclass
class CarryDecomposition:
    """Carry P&L split into components."""
    total_carry: float
    coupon_income: float
    repo_financing_cost: float
    specialness_benefit: float  # GC_cost - actual_cost (positive when on special)
    net_cash_position: float

    def to_dict(self) -> dict:
        return {
            "total_carry": self.total_carry,
            "coupon_income": self.coupon_income,
            "repo_financing_cost": self.repo_financing_cost,
            "specialness_benefit": self.specialness_benefit,
            "net_cash_position": self.net_cash_position,
        }


def carry_pnl_decomposition(
    book: RepoBook,
    gc_rate: float,
) -> CarryDecomposition:
    """Decompose book carry into coupon, repo cost, and specialness.

    coupon_income: total coupon earned on bonds held.
    repo_financing_cost: total interest paid on borrowed cash.
    specialness_benefit: savings from financing below GC (positive = good).
    """
    coupon_income = 0.0
    financing_cost = 0.0
    specialness = 0.0

    for e in book.entries:
        dt_coupon = e.term_days / 365.0   # ACT/365 for coupon
        dt_fin = e.term_days / 360.0      # ACT/360 for financing
        sign = 1.0 if e.direction == "repo" else -1.0

        coupon = e.face_amount * e.coupon_rate * dt_coupon * sign
        financing = e.cash_amount * e.repo_rate * dt_fin * sign
        # Specialness: what would financing cost at GC?
        gc_financing = e.cash_amount * gc_rate * dt_fin * sign
        spec_benefit = gc_financing - financing  # positive when repo < GC

        coupon_income += coupon
        financing_cost += financing
        specialness += spec_benefit

    total = coupon_income - financing_cost
    net_cash = book.total_cash_out() - book.total_cash_in()

    return CarryDecomposition(
        total_carry=total,
        coupon_income=coupon_income,
        repo_financing_cost=financing_cost,
        specialness_benefit=specialness,
        net_cash_position=net_cash,
    )


# ---------------------------------------------------------------------------
# Rollover Risk
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Counterparty Exposure Monitor
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Tier 2: Dynamic Haircut Adjustment
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Tier 2: Margin Call Simulation
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Tier 2: Specialness Forecast
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Tier 2: Repo Curve Stress Scenarios
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Tier 3: Settlement Fail Workflow
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Tier 3: Collateral Substitution
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Tier 3: Balance Sheet Efficiency
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Tier 3: Stress Testing Suite
# ---------------------------------------------------------------------------

@dataclass
class StressTestResult:
    """One stress scenario result."""
    scenario_name: str
    description: str
    carry_impact: float
    margin_call: float
    fails_impact: float
    total_impact: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name, "description": self.description,
            "carry": self.carry_impact, "margin_call": self.margin_call,
            "fails": self.fails_impact, "total": self.total_impact,
        }


def stress_test_suite(
    book: RepoBook,
    haircut_pct: float = 2.0,
) -> list[StressTestResult]:
    """Pre-built stress scenarios: 2008, COVID, inversion, sector default, CB tightening."""
    total_cash = sum(e.cash_amount for e in book.entries)
    on_cash = sum(e.cash_amount for e in book.entries if e.term_days <= 7)
    special_face = sum(e.face_amount for e in book.entries if e.collateral_type == "special")

    scenarios = []

    # 2008
    cs_2008 = repo_curve_stress(book, [("2008", 500, 200)])
    mc_2008 = margin_call_simulation(book, haircut_pct, [("2008", 200)])
    fails_2008 = total_cash * 0.10 * 0.003
    scenarios.append(StressTestResult(
        "2008_crisis", "ON +500bp, term +200bp, 10% fails",
        cs_2008[0].carry_impact, mc_2008[0].total_margin_call, fails_2008,
        cs_2008[0].carry_impact + mc_2008[0].total_margin_call + fails_2008,
    ))

    # COVID
    covid_cost = on_cash * 0.03 * 5 / 365.0
    mc_covid = margin_call_simulation(book, haircut_pct, [("covid", 100)])
    scenarios.append(StressTestResult(
        "covid_mar2020", "ON +300bp for 5d, margin spike",
        -covid_cost, mc_covid[0].total_margin_call, 0.0,
        -covid_cost + mc_covid[0].total_margin_call,
    ))

    # Inversion
    inv = repo_curve_stress(book, [("inv", 100, -50)])
    scenarios.append(StressTestResult(
        "sustained_inversion", "ON > term by 100bp, 30d",
        inv[0].carry_impact * 30, 0.0, 0.0, inv[0].carry_impact * 30,
    ))

    # Sector default
    fails_sector = special_face * 0.20 * 0.003
    gc_cost = total_cash * 0.005 * 30 / 365.0
    scenarios.append(StressTestResult(
        "sector_default", "20% specials fail, GC +50bp",
        -gc_cost, 0.0, fails_sector, -gc_cost + fails_sector,
    ))

    # CB tightening
    cb = repo_curve_stress(book, [("cb", 100, 100)])
    extra_capital = total_cash * 0.01
    scenarios.append(StressTestResult(
        "cb_tightening", "Parallel +100bp, haircuts +1%",
        cb[0].carry_impact, extra_capital, 0.0,
        cb[0].carry_impact + extra_capital,
    ))

    return scenarios


# ---------------------------------------------------------------------------
# Tier 4: Daily Risk Dashboard
# ---------------------------------------------------------------------------

@dataclass
class RepoDashboard:
    """Morning-meeting summary for the repo desk."""
    date: date
    net_cash: float
    gc_rate: float | None
    n_positions: int
    total_carry: float
    repo_dv01: float
    n_fails: int
    total_fail_face: float
    top_cp_exposures: list[dict]
    top_specials: list[dict]
    rollover_exposure: float  # cash in O/N + 1W bucket

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "net_cash": self.net_cash,
            "gc_rate": self.gc_rate,
            "n_positions": self.n_positions,
            "total_carry": self.total_carry,
            "repo_dv01": self.repo_dv01,
            "n_fails": self.n_fails,
            "total_fail_face": self.total_fail_face,
            "top_cp_exposures": self.top_cp_exposures,
            "top_specials": self.top_specials,
            "rollover_exposure": self.rollover_exposure,
        }


def daily_dashboard(
    book: RepoBook,
    reference_date: date,
    tracker: FailsTracker | None = None,
    gc_rate: float | None = None,
    cp_limits: dict[str, float] | None = None,
) -> RepoDashboard:
    """Build the morning-meeting dashboard in one call.

    Aggregates: cash position, carry, DV01, fails, CP exposure,
    specials, rollover risk.
    """
    net_cash = book.total_cash_out() - book.total_cash_in()
    gc = gc_rate if gc_rate is not None else book.gc_rate()
    carry = book.net_carry()
    dv01 = repo_rate_dv01(book)["total_dv01"]

    # Fails
    n_fails = len(tracker) if tracker else 0
    fail_face = tracker.total_face_outstanding() if tracker else 0.0

    # Top CP exposures
    cp_monitor = counterparty_exposure_monitor(book, cp_limits)
    top_cp = [r.to_dict() for r in cp_monitor[:5]]

    # Specials
    specials = [
        {"issuer": e.collateral_issuer, "rate": e.repo_rate, "face": e.face_amount}
        for e in book.entries if e.collateral_type == "special"
    ]
    specials.sort(key=lambda s: s["rate"])

    # Rollover exposure (O/N + 1W)
    on_1w = sum(
        e.cash_amount for e in book.entries
        if e.term_days <= 7 and e.direction == "repo"
    )

    return RepoDashboard(
        date=reference_date, net_cash=net_cash, gc_rate=gc,
        n_positions=len(book), total_carry=carry,
        repo_dv01=dv01, n_fails=n_fails, total_fail_face=fail_face,
        top_cp_exposures=top_cp, top_specials=specials[:5],
        rollover_exposure=on_1w,
    )


# ---------------------------------------------------------------------------
# Tier 4: Hedge Recommendations
# ---------------------------------------------------------------------------

@dataclass
class HedgeRecommendation:
    """A suggested hedge action."""
    action: str          # "reduce_on", "extend_term", "short_bond", "buy_futures"
    reason: str
    notional: float
    urgency: str         # "immediate", "eod", "monitor"

    def to_dict(self) -> dict:
        return {
            "action": self.action, "reason": self.reason,
            "notional": self.notional, "urgency": self.urgency,
        }


def hedge_recommendations(
    book: RepoBook,
    dv01_limit: float = 50_000.0,
    rollover_limit: float = 500_000_000.0,
    concentration_limit_pct: float = 30.0,
) -> list[HedgeRecommendation]:
    """Generate hedge recommendations based on risk limits.

    Rules:
    - DV01 > limit → reduce exposure or extend term.
    - Rollover > limit → lock in term repo.
    - Single CP > concentration% of total → diversify.
    """
    recs = []
    dv01 = abs(repo_rate_dv01(book)["total_dv01"])

    # DV01 check
    if dv01 > dv01_limit:
        recs.append(HedgeRecommendation(
            action="reduce_dv01",
            reason=f"Repo DV01 ${dv01:,.0f} exceeds limit ${dv01_limit:,.0f}",
            notional=dv01 - dv01_limit,
            urgency="eod",
        ))

    # Rollover check
    on_cash = sum(e.cash_amount for e in book.entries
                  if e.term_days <= 7 and e.direction == "repo")
    if on_cash > rollover_limit:
        excess = on_cash - rollover_limit
        recs.append(HedgeRecommendation(
            action="extend_term",
            reason=f"O/N+1W exposure ${on_cash:,.0f} exceeds ${rollover_limit:,.0f}",
            notional=excess,
            urgency="immediate",
        ))

    # Concentration check
    total = sum(e.cash_amount for e in book.entries)
    if total > 0:
        by_cp = book.by_counterparty()
        for cp in by_cp:
            pct = abs(cp.total_cash) / total * 100
            if pct > concentration_limit_pct:
                recs.append(HedgeRecommendation(
                    action="diversify_cp",
                    reason=f"{cp.counterparty} at {pct:.0f}% (limit {concentration_limit_pct:.0f}%)",
                    notional=abs(cp.total_cash) - total * concentration_limit_pct / 100,
                    urgency="eod",
                ))

    if not recs:
        recs.append(HedgeRecommendation(
            action="none", reason="All limits within bounds",
            notional=0.0, urgency="monitor",
        ))

    return recs


# ---------------------------------------------------------------------------
# Tier 4: Matched Book Analytics
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Tier 4: Cross-Desk Funding Attribution
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Tier 5: RepoBook serialisation and PV
# ---------------------------------------------------------------------------







# Import serialisation helpers from operations before using them
from pricebook.repo_operations import (  # noqa: E402
    _RepoBookMixin, _repo_book_to_dict, _repo_book_from_dict,
)

RepoBook._SERIAL_TYPE = "repo_book"
RepoBook.to_dict = _repo_book_to_dict
RepoBook.from_dict = _repo_book_from_dict

from pricebook.serialisable import _register as _reg_rb
_reg_rb(RepoBook)




# ---------------------------------------------------------------------------
# SOFR lookback + lockout (3→4)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Netting (3→4)
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# FAIL_STATES moved to repo_operations.py






# ---------------------------------------------------------------------------
# Repo key-rate DV01 (4→5)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Regulatory haircut floors (Basel III) (3→4)
# ---------------------------------------------------------------------------





# ===========================================================================
# DESK PROTOCOL COMPONENTS (added for 9-component compliance)
# ===========================================================================

# ---------------------------------------------------------------------------
# Component 1: Risk Metrics
# ---------------------------------------------------------------------------

@dataclass
class RepoRiskMetrics:
    """Unified risk metrics for a repo position."""
    pv: float                # net carry PV
    cash_amount: float       # cash lent/borrowed
    interest: float          # repo interest
    carry: float             # net carry (coupon - financing)
    dv01: float              # rate sensitivity (repo rate bump)
    notional: float          # face amount

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "cash": self.cash_amount, "interest": self.interest,
            "carry": self.carry, "dv01": self.dv01, "notional": self.notional,
        }


def repo_risk_metrics(
    trade: RepoTrade,
    rate_bump: float = 0.0001,
) -> RepoRiskMetrics:
    """Compute risk metrics for a repo trade.

    Repo is a short-dated financing instrument. Key risks:
    - Rate sensitivity (DV01): change in interest for 1bp rate move
    - Carry: coupon income vs financing cost

    Args:
        trade: RepoTrade instance.
        rate_bump: rate shift for DV01 (default 1bp).
    """
    cash = trade.cash_amount
    interest = trade.interest
    carry = trade.carry
    notional = trade.face_amount

    # DV01: change in interest for 1bp rate move
    # ΔInterest = cash × (term/360) × Δrate, where Δrate = 0.0001 (1bp)
    dt = trade.term_days / 360.0
    dv01 = cash * dt * 0.0001  # always per 1bp, independent of rate_bump param

    # PV: simplified as carry (for short-dated, PV ≈ carry)
    pv = carry

    return RepoRiskMetrics(
        pv=pv, cash_amount=cash, interest=interest,
        carry=carry, dv01=dv01, notional=notional,
    )


# ---------------------------------------------------------------------------
# Component 7: Capital
# ---------------------------------------------------------------------------

@dataclass
class RepoCapitalResult:
    """Regulatory capital for a repo position."""
    ead: float
    rwa: float
    capital: float
    simm_im: float

    def to_dict(self) -> dict:
        return {"ead": self.ead, "rwa": self.rwa, "capital": self.capital,
                "simm_im": self.simm_im}


def repo_capital(
    trade: RepoTrade,
    counterparty_rw: float = 0.20,
) -> RepoCapitalResult:
    """SA-CCR capital for a repo / SFT.

    For securities financing transactions:
    EAD = max(0, exposure - collateral_value) + add-on

    Repos are collateralised → EAD driven by haircut gap.
    SIMM: repo rate sensitivity into GIRR bucket.

    Args:
        trade: RepoTrade instance.
        counterparty_rw: counterparty risk weight.
    """
    # SFT EAD: exposure = cash_amount, collateral = market_value × (1-haircut)
    exposure = trade.cash_amount
    collateral = trade.market_value
    haircut_gap = max(exposure - collateral, 0)

    # Add-on: 5% of cash amount for counterparty risk
    add_on = exposure * 0.05
    ead = max(haircut_gap + add_on, 0)

    rwa = ead * counterparty_rw
    capital = rwa * 0.08

    # SIMM: rate sensitivity
    dt = trade.term_days / 360.0
    simm_sensitivity = trade.cash_amount * dt  # DV01-like
    girr_rw = 0.002  # GIRR risk weight
    simm_im = abs(simm_sensitivity) * girr_rw * math.sqrt(10.0 / 252.0)

    return RepoCapitalResult(ead=ead, rwa=rwa, capital=capital, simm_im=simm_im)


# ---------------------------------------------------------------------------
# Component 9: Lifecycle
# ---------------------------------------------------------------------------

class RepoEventType:
    MATURITY = "maturity"
    ROLL = "roll"
    MARGIN_CALL = "margin_call"
    COUPON_PASS_THROUGH = "coupon_pass_through"
    SETTLEMENT_FAIL = "settlement_fail"
    SUBSTITUTION = "substitution"


class RepoLifecycle:
    """Lifecycle management for repo positions."""

    def __init__(self, trade: RepoTrade):
        self._trade = trade
        self._events: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return sorted(self._events, key=lambda x: x.get("date", ""))

    def maturity_alert(self, as_of: date, alert_days: int = 3) -> dict | None:
        """Alert for upcoming repo maturity (short horizon for repos)."""
        mat = self._trade.maturity_date
        if mat is None:
            return None  # open repo has no maturity
        days = (mat - as_of).days
        if 0 < days <= alert_days:
            return {
                "type": RepoEventType.MATURITY,
                "date": mat.isoformat(),
                "days_remaining": days,
                "action": "roll or unwind",
            }
        return None

    def roll_alert(self, as_of: date) -> dict | None:
        """Alert when open repo should be reviewed for rate reset."""
        if not self._trade.is_open:
            return None
        return {
            "type": RepoEventType.ROLL,
            "date": as_of.isoformat(),
            "current_rate": self._trade.repo_rate,
            "note": "Open repo — review rate and renew",
        }

    def coupon_alert(self, as_of: date) -> list[dict]:
        """Alert for upcoming coupon pass-throughs during repo term."""
        coupons = self._trade.coupons_during_term()
        alerts = []
        for coupon_date, amount in coupons:
            days = (coupon_date - as_of).days
            if 0 < days <= 5:
                alerts.append({
                    "type": RepoEventType.COUPON_PASS_THROUGH,
                    "date": coupon_date.isoformat(),
                    "amount": amount,
                    "days_until": days,
                })
        return alerts

    def record_event(self, event_type: str, event_date: date, **kwargs) -> dict:
        event = {"type": event_type, "date": event_date.isoformat(), **kwargs}
        self._events.append(event)
        return event

    def record_roll(self, roll_date: date, new_rate: float, new_term_days: int) -> dict:
        return self.record_event(
            RepoEventType.ROLL, roll_date,
            old_rate=self._trade.repo_rate, new_rate=new_rate,
            new_term_days=new_term_days,
        )

    def record_margin_call(self, call_date: date, amount: float, reason: str = "") -> dict:
        return self.record_event(
            RepoEventType.MARGIN_CALL, call_date,
            amount=amount, reason=reason,
        )


# ---------------------------------------------------------------------------
# aggregate_risk for cross-asset compatibility
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Standardized function name aliases (desk protocol consistency)
# ---------------------------------------------------------------------------

repo_dashboard = daily_dashboard
repo_stress_suite = stress_test_suite
repo_hedge_recommendations = hedge_recommendations


# ===========================================================================
# Backward-compatible re-exports from split modules
# ===========================================================================

from pricebook.repo_operations import (  # noqa: F401, E402
    CollateralPosition, CollateralPool,
    SettlementFail, FailsTracker,
    CashLadderBucket, cash_ladder,
    HaircutAdjustment, dynamic_haircut,
    MarginCallScenario, margin_call_simulation,
    FailResolution, fail_workflow,
    SubstitutionCandidate, find_substitutes,
    NettingResult, netting_by_counterparty,
    FailState, auto_escalate_fails,
    FAIL_STATES,
)

from pricebook.repo_analytics import (  # noqa: F401, E402
    repo_rate_monitor,
    CTDRepoCandidate, cheapest_to_deliver_repo,
    TermVsOvernightResult, term_vs_overnight,
    repo_rate_dv01,
    RolloverScenario, rollover_risk,
    CounterpartyLimit, counterparty_exposure_monitor,
    SpecialnessForecast, forecast_specialness,
    RepoCurveStress, repo_curve_stress,
    BalanceSheetMetrics, balance_sheet_efficiency,
    MatchedBookEntry, matched_book_analysis,
    FundingAttribution, funding_attribution,
    repo_book_pv, sofr_compounded_with_lookback,
    repo_key_rate_dv01, regulatory_haircut,
    BASEL_HAIRCUT_FLOORS,
)
