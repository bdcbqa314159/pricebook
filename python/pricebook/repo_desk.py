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

from pricebook.zscore import zscore as _zscore, ZScoreSignal


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
        direction: str = "repo",
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
        self.counterparty = counterparty
        self.collateral_issuer = collateral_issuer
        self.collateral_type = collateral_type
        self.face_amount = face_amount
        self.bond_price = bond_price
        self.repo_rate = repo_rate
        self.term_days = term_days
        self.coupon_rate = coupon_rate
        self.direction = direction
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
    def carry(self) -> float:
        """Net carry = coupon income − financing cost.

        Issue 1 fix: coupon uses bond's day count if bond is attached,
        otherwise ACT/365. Financing always ACT/360.
        """
        # Coupon accrual
        if self.bond is not None:
            from pricebook.day_count import year_fraction
            sd = self.settlement_date or self.start_date
            mat = self.maturity_date
            if sd and mat:
                yf = year_fraction(sd, mat, self.bond.day_count)
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
            if sd < cf.payment_date <= mat:
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
        """Margin call for cross-currency repos.

        Two sources of margin change:
        1. Bond price move (same as single-ccy)
        2. FX rate move (changes collateral value in cash currency)

        Total call = price_margin + fx_margin.
        """
        # Price-driven margin
        price_margin = self.margin_call(current_bond_price)

        # FX-driven margin
        # Initial collateral in cash ccy: market_value × fx_rate_inception
        # Current collateral in cash ccy: face × current_price / 100 × current_fx
        initial_value_cash = self.market_value * self.fx_rate
        current_value_cash = self.face_amount * current_bond_price / 100.0 * current_fx_rate
        fx_margin = (current_value_cash - initial_value_cash) * self.fx_haircut

        return price_margin + fx_margin

    # ---- Pricing (Gap 8) ----

    def pv(self, discount_curve, reference_date: date | None = None,
           projection_curve=None) -> float:
        """Present value against a discount curve.

        For fixed repos: PV = df(mat) × repurchase − cash.
        For floating repos: uses projection_curve for forward SOFR rates.
        """
        ref = reference_date or self.start_date or date.today()
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
        ref = as_of or date.today()
        return max(0, (mat - ref).days)

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
        roll_date = new_date or self.maturity_date or date.today()
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


RepoTrade._SERIAL_TYPE = "repo_trade"
from pricebook.serialisable import _register as _reg_rt
_reg_rt(RepoTrade)


# ---------------------------------------------------------------------------
# Collateral Pool (Gap 4)
# ---------------------------------------------------------------------------

@dataclass
class CollateralPosition:
    """A bond in the collateral pool."""
    issuer: str
    face_amount: float
    pledged_to: dict[str, float] = field(default_factory=dict)  # {counterparty: face_pledged}

    @property
    def total_pledged(self) -> float:
        return sum(self.pledged_to.values())

    @property
    def available(self) -> float:
        return max(0, self.face_amount - self.total_pledged)

    def pledge(self, counterparty: str, amount: float) -> None:
        if amount > self.available:
            raise ValueError(
                f"Cannot pledge {amount}: only {self.available} available "
                f"({self.face_amount} total, {self.total_pledged} pledged)"
            )
        self.pledged_to[counterparty] = self.pledged_to.get(counterparty, 0) + amount

    def release(self, counterparty: str, amount: float) -> None:
        current = self.pledged_to.get(counterparty, 0)
        self.pledged_to[counterparty] = max(0, current - amount)

    def to_dict(self) -> dict:
        return {
            "issuer": self.issuer, "face": self.face_amount,
            "pledged": self.total_pledged, "available": self.available,
            "by_cp": dict(self.pledged_to),
        }


class CollateralPool:
    """Tracks bond inventory: what's pledged, what's free (Gap 4)."""

    def __init__(self):
        self._positions: dict[str, CollateralPosition] = {}

    def add_inventory(self, issuer: str, face_amount: float) -> None:
        if issuer in self._positions:
            self._positions[issuer].face_amount += face_amount
        else:
            self._positions[issuer] = CollateralPosition(issuer, face_amount)

    def pledge(self, issuer: str, counterparty: str, amount: float) -> None:
        if issuer not in self._positions:
            raise ValueError(f"No inventory for {issuer}")
        self._positions[issuer].pledge(counterparty, amount)

    def release(self, issuer: str, counterparty: str, amount: float) -> None:
        if issuer in self._positions:
            self._positions[issuer].release(counterparty, amount)

    def available(self, issuer: str) -> float:
        pos = self._positions.get(issuer)
        return pos.available if pos else 0.0

    def total_available(self) -> float:
        return sum(p.available for p in self._positions.values())

    def summary(self) -> list[dict]:
        return [p.to_dict() for p in sorted(self._positions.values(), key=lambda p: p.issuer)]

    def can_pledge(self, issuer: str, amount: float) -> bool:
        return self.available(issuer) >= amount


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


# ---- Legacy: RepoTradeEntry (backward compat) ----

@dataclass
class RepoTradeEntry:
    """A single repo position with desk metadata.

    Attributes:
        counterparty: repo counterparty.
        collateral_issuer: issuer of the collateral bond.
        collateral_type: ``"GC"`` (general collateral) or ``"special"``.
        face_amount: face value of the collateral bond.
        bond_price: dirty price of the collateral (per 100 face).
        repo_rate: annualised repo rate.
        term_days: repo term in calendar days (0 = overnight).
        coupon_rate: annual coupon of the collateral bond.
        direction: ``"repo"`` (lend bond / borrow cash) or
            ``"reverse"`` (borrow bond / lend cash).
        start_date: repo start date.
    """
    counterparty: str
    collateral_issuer: str
    collateral_type: str = "GC"
    face_amount: float = 0.0
    bond_price: float = 100.0
    repo_rate: float = 0.0
    term_days: int = 1
    coupon_rate: float = 0.0
    direction: str = "repo"
    start_date: date | None = None

    @property
    def cash_amount(self) -> float:
        """Cash lent / borrowed = face × dirty_price / 100."""
        return self.face_amount * self.bond_price / 100.0

    @property
    def carry(self) -> float:
        """Net carry = coupon income − financing cost over the term."""
        dt = self.term_days / 365.0
        coupon = self.face_amount * self.coupon_rate * dt
        financing = self.cash_amount * self.repo_rate * dt
        sign = 1.0 if self.direction == "repo" else -1.0
        return sign * (coupon - financing)

    @property
    def financing_cost(self) -> float:
        dt = self.term_days / 365.0
        return self.cash_amount * self.repo_rate * dt


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

    Args:
        name: book name (e.g. "GovtRepo", "IG_Repo").
    """

    def __init__(self, name: str):
        self.name = name
        self._entries: list[RepoTradeEntry] = []

    def add(self, entry) -> None:
        """Add a RepoTradeEntry or RepoTrade to the book."""
        if isinstance(entry, RepoTrade):
            # Convert RepoTrade to RepoTradeEntry for backward compat
            entry = RepoTradeEntry(
                counterparty=entry.counterparty,
                collateral_issuer=entry.collateral_issuer,
                collateral_type=entry.collateral_type,
                face_amount=entry.face_amount,
                bond_price=entry.bond_price,
                repo_rate=entry.repo_rate,
                term_days=max(entry.term_days, 1),
                coupon_rate=entry.coupon_rate,
                direction=entry.direction,
                start_date=entry.start_date,
            )
        self._entries.append(entry)

    @property
    def entries(self) -> list[RepoTradeEntry]:
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
        """Aggregate exposure per counterparty."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            cp = e.counterparty
            if cp not in agg:
                agg[cp] = {"cash": 0.0, "rate_sum": 0.0, "count": 0}
            agg[cp]["cash"] += e.cash_amount
            agg[cp]["rate_sum"] += e.repo_rate
            agg[cp]["count"] += 1

        return [
            RepoCounterpartyExposure(
                counterparty=cp,
                total_cash=d["cash"],
                n_trades=d["count"],
                avg_rate=d["rate_sum"] / d["count"] if d["count"] > 0 else 0.0,
            )
            for cp, d in sorted(agg.items())
        ]

    def by_collateral_type(self) -> list[RepoCollateralSummary]:
        """Aggregate by GC vs special."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            ct = e.collateral_type
            if ct not in agg:
                agg[ct] = {"cash": 0.0, "rate_sum": 0.0, "count": 0}
            agg[ct]["cash"] += e.cash_amount
            agg[ct]["rate_sum"] += e.repo_rate
            agg[ct]["count"] += 1

        return [
            RepoCollateralSummary(
                collateral_type=ct,
                total_cash=d["cash"],
                avg_rate=d["rate_sum"] / d["count"] if d["count"] > 0 else 0.0,
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


# ---- Repo rate monitor ----

def repo_rate_monitor(
    current_rate: float,
    history: list[float],
    threshold: float = 2.0,
) -> ZScoreSignal:
    """Z-score the current repo rate vs history."""
    return _zscore(current_rate, history, threshold)


# ---- Cheapest-to-deliver repo ----

@dataclass
class CTDRepoCandidate:
    """A bond candidate for repo financing."""
    issuer: str
    bond_price: float
    repo_rate: float
    coupon_rate: float
    term_days: int

    @property
    def financing_cost(self) -> float:
        return self.bond_price * self.repo_rate * self.term_days / 365.0

    @property
    def carry(self) -> float:
        dt = self.term_days / 365.0
        return self.coupon_rate * 100.0 * dt - self.financing_cost


def cheapest_to_deliver_repo(
    candidates: list[CTDRepoCandidate],
) -> CTDRepoCandidate | None:
    """Select the bond that minimises financing cost.

    Among all candidates, picks the one with the lowest financing
    cost per 100 face for the given term.
    """
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.financing_cost)


# ---- Term vs overnight ----

@dataclass
class TermVsOvernightResult:
    """Comparison of term repo vs rolling overnight."""
    term_rate: float
    overnight_rate: float
    term_days: int
    term_cost: float
    overnight_cost: float
    savings: float
    recommendation: str


def term_vs_overnight(
    face_amount: float,
    bond_price: float,
    term_rate: float,
    overnight_rate: float,
    term_days: int,
) -> TermVsOvernightResult:
    """Compare locking in a term repo vs rolling overnight.

    Assumes the overnight rate is constant over the term for simplicity.

    Args:
        face_amount: face value of collateral.
        bond_price: dirty price per 100 face.
        term_rate: annualised term repo rate.
        overnight_rate: annualised overnight repo rate.
        term_days: number of days for the term repo.

    Returns:
        :class:`TermVsOvernightResult` with costs and recommendation.
    """
    cash = face_amount * bond_price / 100.0
    dt = term_days / 365.0
    term_cost = cash * term_rate * dt
    overnight_cost = cash * overnight_rate * dt

    savings = overnight_cost - term_cost
    if term_cost < overnight_cost:
        recommendation = "term"
    elif term_cost > overnight_cost:
        recommendation = "overnight"
    else:
        recommendation = "indifferent"

    return TermVsOvernightResult(
        term_rate=term_rate,
        overnight_rate=overnight_rate,
        term_days=term_days,
        term_cost=term_cost,
        overnight_cost=overnight_cost,
        savings=savings,
        recommendation=recommendation,
    )


# ---- Fails tracking ----

@dataclass
class SettlementFail:
    """A single settlement fail."""
    counterparty: str
    issuer: str
    face_amount: float
    fail_date: date
    days_outstanding: int = 0
    penalty_rate_bps: float = 0.0

    @property
    def penalty_cost(self) -> float:
        """Penalty = face × penalty_rate × days / 365."""
        return (
            self.face_amount
            * (self.penalty_rate_bps / 10_000.0)
            * self.days_outstanding / 365.0
        )


class FailsTracker:
    """Track and cost settlement fails."""

    def __init__(self):
        self._fails: list[SettlementFail] = []

    def add(self, fail: SettlementFail) -> None:
        self._fails.append(fail)

    @property
    def fails(self) -> list[SettlementFail]:
        return list(self._fails)

    def __len__(self) -> int:
        return len(self._fails)

    def total_penalty(self) -> float:
        return sum(f.penalty_cost for f in self._fails)

    def total_face_outstanding(self) -> float:
        return sum(f.face_amount for f in self._fails)

    def by_counterparty(self) -> dict[str, float]:
        """Total fail face per counterparty."""
        result: dict[str, float] = {}
        for f in self._fails:
            result[f.counterparty] = result.get(f.counterparty, 0.0) + f.face_amount
        return result


# ---------------------------------------------------------------------------
# Maturity / Cash Ladder
# ---------------------------------------------------------------------------

@dataclass
class CashLadderBucket:
    """One bucket in the maturity ladder."""
    bucket: str          # "O/N", "1W", "1M", "3M", "6M", "1Y+"
    maturing_cash: float  # cash flowing in/out at this tenor
    n_trades: int
    avg_rate: float
    refinancing_cost: float  # cost to roll at current ON rate

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket, "maturing_cash": self.maturing_cash,
            "n_trades": self.n_trades, "avg_rate": self.avg_rate,
            "refinancing_cost": self.refinancing_cost,
        }


def cash_ladder(
    book: RepoBook,
    reference_date: date,
    overnight_rate: float = 0.0,
) -> list[CashLadderBucket]:
    """Build a maturity/cash ladder from the repo book.

    Groups positions by remaining tenor and computes the cash
    maturing in each bucket + cost to refinance at the overnight rate.

    Buckets: O/N (0-1d), 1W (2-7d), 1M (8-30d), 3M (31-90d),
             6M (91-180d), 1Y+ (181+d).
    """
    buckets_def = [
        ("O/N", 0, 1),
        ("1W", 2, 7),
        ("1M", 8, 30),
        ("3M", 31, 90),
        ("6M", 91, 180),
        ("1Y+", 181, 99999),
    ]

    result = []
    for label, lo, hi in buckets_def:
        matching = []
        for e in book.entries:
            remaining = e.term_days
            if e.start_date:
                elapsed = (reference_date - e.start_date).days
                remaining = max(0, e.term_days - elapsed)
            if lo <= remaining <= hi:
                matching.append(e)

        total_cash = sum(
            e.cash_amount * (1 if e.direction == "repo" else -1)
            for e in matching
        )
        avg_rate = (
            sum(e.repo_rate * e.cash_amount for e in matching)
            / sum(e.cash_amount for e in matching)
            if matching and sum(e.cash_amount for e in matching) > 0
            else 0.0
        )
        # Refinancing cost: if this bucket matures, roll at ON for same term
        mid_days = (lo + min(hi, 365)) / 2
        refi_cost = abs(total_cash) * overnight_rate * mid_days / 365.0

        result.append(CashLadderBucket(
            bucket=label, maturing_cash=total_cash,
            n_trades=len(matching), avg_rate=avg_rate,
            refinancing_cost=refi_cost,
        ))

    return result


# ---------------------------------------------------------------------------
# Repo Rate DV01
# ---------------------------------------------------------------------------

def repo_rate_dv01(
    book: RepoBook,
    shift_bps: float = 1.0,
) -> dict[str, float]:
    """Carry sensitivity to a parallel 1bp repo rate shift.

    Returns:
        total_dv01: change in total carry for +1bp repo shift.
        per_trade: list of per-trade carry changes.
    """
    shift = shift_bps / 10_000.0
    base_carry = book.net_carry()

    # Bump all repo rates and recompute
    bumped_carry = 0.0
    per_trade = []
    for e in book.entries:
        base_c = e.carry
        dt = e.term_days / 365.0
        # Carry = sign × (coupon_income - cash × (repo_rate + shift) × dt)
        sign = 1.0 if e.direction == "repo" else -1.0
        coupon = e.face_amount * e.coupon_rate * dt
        financing_bumped = e.cash_amount * (e.repo_rate + shift) * dt
        bumped_c = sign * (coupon - financing_bumped)
        bumped_carry += bumped_c
        per_trade.append(bumped_c - base_c)

    return {
        "total_dv01": bumped_carry - base_carry,
        "base_carry": base_carry,
        "bumped_carry": bumped_carry,
        "per_trade_dv01": per_trade,
    }


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
        dt = e.term_days / 365.0
        sign = 1.0 if e.direction == "repo" else -1.0

        coupon = e.face_amount * e.coupon_rate * dt * sign
        financing = e.cash_amount * e.repo_rate * dt * sign
        # Specialness: what would financing cost at GC?
        gc_financing = e.cash_amount * gc_rate * dt * sign
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

@dataclass
class RolloverScenario:
    """Cost of rolling O/N repo under a rate spike."""
    scenario_name: str
    on_rate_spike_bps: float
    spike_duration_days: int
    additional_cost: float
    annualised_impact_bps: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "spike_bps": self.on_rate_spike_bps,
            "days": self.spike_duration_days,
            "cost": self.additional_cost,
            "impact_bps": self.annualised_impact_bps,
        }


def rollover_risk(
    book: RepoBook,
    scenarios: list[tuple[str, float, int]] | None = None,
) -> list[RolloverScenario]:
    """Quantify cost of O/N repo rate spikes when rolling forward.

    Computes: for each scenario, how much extra financing cost
    on the O/N portion of the book during the spike.

    Default scenarios: mild (+25bp, 3d), moderate (+100bp, 5d),
    severe (+300bp, 10d), crisis (+500bp, 30d).

    Args:
        scenarios: list of (name, spike_bps, duration_days).
    """
    if scenarios is None:
        scenarios = [
            ("mild", 25, 3),
            ("moderate", 100, 5),
            ("severe", 300, 10),
            ("crisis", 500, 30),
        ]

    # O/N and short-term positions vulnerable to rollover
    on_trades = [e for e in book.entries if e.term_days <= 7]
    on_cash = sum(e.cash_amount for e in on_trades)

    results = []
    for name, spike_bps, days in scenarios:
        spike = spike_bps / 10_000.0
        extra_cost = on_cash * spike * days / 365.0
        annualised = spike_bps * (days / 365.0) if on_cash > 0 else 0.0
        results.append(RolloverScenario(
            scenario_name=name,
            on_rate_spike_bps=spike_bps,
            spike_duration_days=days,
            additional_cost=extra_cost,
            annualised_impact_bps=annualised,
        ))

    return results


# ---------------------------------------------------------------------------
# Counterparty Exposure Monitor
# ---------------------------------------------------------------------------

@dataclass
class CounterpartyLimit:
    """Counterparty limit and utilisation."""
    counterparty: str
    limit: float
    current_exposure: float
    utilisation_pct: float
    breached: bool
    headroom: float

    def to_dict(self) -> dict:
        return {
            "counterparty": self.counterparty,
            "limit": self.limit,
            "exposure": self.current_exposure,
            "utilisation_pct": self.utilisation_pct,
            "breached": self.breached,
            "headroom": self.headroom,
        }


def counterparty_exposure_monitor(
    book: RepoBook,
    limits: dict[str, float] | None = None,
    default_limit: float = 500_000_000.0,
) -> list[CounterpartyLimit]:
    """Monitor counterparty exposure against limits.

    Args:
        limits: {counterparty: max_exposure}. Missing CPs get default_limit.
        default_limit: default exposure limit per CP.

    Returns:
        List of CounterpartyLimit, sorted by utilisation (highest first).
    """
    exposures = book.by_counterparty()
    if limits is None:
        limits = {}

    results = []
    for cp_exp in exposures:
        limit = limits.get(cp_exp.counterparty, default_limit)
        exposure = abs(cp_exp.total_cash)
        util = (exposure / limit * 100.0) if limit > 0 else 0.0
        results.append(CounterpartyLimit(
            counterparty=cp_exp.counterparty,
            limit=limit,
            current_exposure=exposure,
            utilisation_pct=util,
            breached=exposure > limit,
            headroom=max(0, limit - exposure),
        ))

    return sorted(results, key=lambda r: -r.utilisation_pct)


# ---------------------------------------------------------------------------
# Tier 2: Dynamic Haircut Adjustment
# ---------------------------------------------------------------------------

@dataclass
class HaircutAdjustment:
    """Haircut adjusted for market stress."""
    base_haircut_pct: float
    vol_multiplier: float
    stress_add_on_pct: float
    adjusted_haircut_pct: float
    regime: str  # "normal", "elevated", "stressed"

    def to_dict(self) -> dict:
        return {
            "base": self.base_haircut_pct, "vol_mult": self.vol_multiplier,
            "stress_add": self.stress_add_on_pct,
            "adjusted": self.adjusted_haircut_pct, "regime": self.regime,
        }


def dynamic_haircut(
    base_haircut_pct: float,
    current_vol: float,
    normal_vol: float = 0.05,
    stress_threshold: float = 2.0,
) -> HaircutAdjustment:
    """Adjust haircut based on market volatility.

    haircut_adj = base × max(1, vol / normal_vol)
    Plus stress add-on when vol > stress_threshold × normal_vol.

    Args:
        base_haircut_pct: normal market haircut (e.g. 2.0 for treasuries).
        current_vol: current realised or implied vol of the collateral.
        normal_vol: long-run average vol.
        stress_threshold: multiplier at which stress add-on kicks in.
    """
    vol_ratio = max(1.0, current_vol / normal_vol) if normal_vol > 0 else 1.0
    adjusted = base_haircut_pct * vol_ratio

    if current_vol > stress_threshold * normal_vol:
        stress_add = base_haircut_pct * 0.5  # +50% of base in stress
        adjusted += stress_add
        regime = "stressed"
    elif current_vol > 1.5 * normal_vol:
        stress_add = base_haircut_pct * 0.2  # +20% in elevated
        adjusted += stress_add
        regime = "elevated"
    else:
        stress_add = 0.0
        regime = "normal"

    return HaircutAdjustment(
        base_haircut_pct=base_haircut_pct,
        vol_multiplier=vol_ratio,
        stress_add_on_pct=stress_add,
        adjusted_haircut_pct=adjusted,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# Tier 2: Margin Call Simulation
# ---------------------------------------------------------------------------

@dataclass
class MarginCallScenario:
    """Result of a margin call simulation under rate shock."""
    scenario_name: str
    rate_shock_bps: float
    total_margin_call: float
    n_positions_affected: int
    largest_single_call: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name, "shock_bps": self.rate_shock_bps,
            "total_call": self.total_margin_call,
            "n_affected": self.n_positions_affected,
            "largest_call": self.largest_single_call,
        }


def margin_call_simulation(
    book: RepoBook,
    haircut_pct: float = 2.0,
    scenarios: list[tuple[str, float]] | None = None,
) -> list[MarginCallScenario]:
    """Simulate margin calls under repo rate shocks.

    When rates move, bond prices move, and the margin (haircut × notional)
    changes. The desk needs to post/receive the difference.

    Approximate: ΔMargin ≈ cash_amount × duration × Δrate × haircut adjustment.

    Args:
        haircut_pct: base haircut percentage.
        scenarios: list of (name, rate_shock_bps).
    """
    if scenarios is None:
        scenarios = [
            ("mild", 25), ("moderate", 50), ("severe", 100), ("crisis", 200),
        ]

    results = []
    for name, shock_bps in scenarios:
        shock = shock_bps / 10_000.0
        total_call = 0.0
        largest = 0.0
        n_affected = 0

        for e in book.entries:
            # Rough price impact: ΔP ≈ -duration × Δy × price
            # Use term_days as rough duration proxy (scaled)
            duration_proxy = min(e.term_days / 365.0, 10.0) * 5.0  # rough
            price_move = e.bond_price * duration_proxy * shock / 100.0
            margin_change = abs(e.face_amount * price_move / 100.0 * haircut_pct / 100.0)

            if margin_change > 0:
                total_call += margin_change
                largest = max(largest, margin_change)
                n_affected += 1

        results.append(MarginCallScenario(
            scenario_name=name, rate_shock_bps=shock_bps,
            total_margin_call=total_call, n_positions_affected=n_affected,
            largest_single_call=largest,
        ))

    return results


# ---------------------------------------------------------------------------
# Tier 2: Specialness Forecast
# ---------------------------------------------------------------------------

@dataclass
class SpecialnessForecast:
    """Forecast of specialness for a bond."""
    bond_id: str
    current_specialness_bps: float
    days_to_auction: int | None
    forecast_specialness_bps: float
    trend: str  # "widening", "stable", "collapsing"
    confidence: str  # "high", "medium", "low"

    def to_dict(self) -> dict:
        return {
            "bond": self.bond_id,
            "current_bps": self.current_specialness_bps,
            "forecast_bps": self.forecast_specialness_bps,
            "days_to_auction": self.days_to_auction,
            "trend": self.trend, "confidence": self.confidence,
        }


def forecast_specialness(
    bond_id: str,
    current_specialness_bps: float,
    days_to_auction: int | None = None,
    borrowing_demand_pct: float = 0.5,
    supply_pct: float = 0.5,
) -> SpecialnessForecast:
    """Forecast specialness using supply-demand rules.

    Rules:
    - Close to auction (< 14 days): specialness widens (supply about to increase).
    - High borrowing demand (> 70%): specialness widens.
    - Post-auction (just happened): specialness collapses.
    - Low demand (< 30%): specialness narrows.

    Args:
        borrowing_demand_pct: fraction of outstanding on loan (0-1).
        supply_pct: available supply relative to demand (0-1).
    """
    forecast = current_specialness_bps
    confidence = "medium"

    # Auction proximity effect
    if days_to_auction is not None:
        if days_to_auction <= 3:
            # Just before auction — specialness at peak, about to collapse
            forecast *= 1.2
            trend = "collapsing"
            confidence = "high"
        elif days_to_auction <= 14:
            # Approaching auction — widening
            forecast *= 1.1
            trend = "widening"
        elif days_to_auction <= 30:
            trend = "stable"
        else:
            # Far from auction — demand builds slowly
            forecast *= 0.9
            trend = "stable"
    else:
        trend = "stable"

    # Demand/supply
    if borrowing_demand_pct > 0.7:
        forecast *= 1.3
        if trend == "stable":
            trend = "widening"
        confidence = "high"
    elif borrowing_demand_pct < 0.3:
        forecast *= 0.7
        if trend == "stable":
            trend = "collapsing"

    if supply_pct < 0.3:
        forecast *= 1.2  # scarce supply widens
    elif supply_pct > 0.8:
        forecast *= 0.8  # ample supply narrows

    return SpecialnessForecast(
        bond_id=bond_id,
        current_specialness_bps=current_specialness_bps,
        days_to_auction=days_to_auction,
        forecast_specialness_bps=max(0, forecast),
        trend=trend,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Tier 2: Repo Curve Stress Scenarios
# ---------------------------------------------------------------------------

@dataclass
class RepoCurveStress:
    """Repo book P&L under a curve stress scenario."""
    scenario_name: str
    on_shift_bps: float
    term_shift_bps: float
    carry_impact: float
    financing_impact: float
    total_impact: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "on_shift": self.on_shift_bps, "term_shift": self.term_shift_bps,
            "carry_impact": self.carry_impact,
            "financing_impact": self.financing_impact,
            "total": self.total_impact,
        }


def repo_curve_stress(
    book: RepoBook,
    scenarios: list[tuple[str, float, float]] | None = None,
) -> list[RepoCurveStress]:
    """Stress the repo book under curve scenarios.

    Each scenario specifies O/N and term rate shifts independently,
    capturing curve flattening/steepening as well as parallel moves.

    Default scenarios:
    - parallel_up: +50bp across all tenors
    - parallel_down: -50bp
    - steepener: O/N -25bp, term +50bp
    - flattener: O/N +50bp, term -25bp
    - inversion: O/N +100bp, term -50bp
    """
    if scenarios is None:
        scenarios = [
            ("parallel_up", 50, 50),
            ("parallel_down", -50, -50),
            ("steepener", -25, 50),
            ("flattener", 50, -25),
            ("inversion", 100, -50),
        ]

    base_carry = book.net_carry()

    results = []
    for name, on_shift, term_shift in scenarios:
        stressed_carry = 0.0
        for e in book.entries:
            # O/N positions get on_shift, longer-term get term_shift
            if e.term_days <= 7:
                shift = on_shift / 10_000.0
            else:
                shift = term_shift / 10_000.0

            dt = e.term_days / 365.0
            sign = 1.0 if e.direction == "repo" else -1.0
            coupon = e.face_amount * e.coupon_rate * dt
            financing = e.cash_amount * (e.repo_rate + shift) * dt
            stressed_carry += sign * (coupon - financing)

        carry_impact = stressed_carry - base_carry
        # Financing impact: extra cost from the shift
        financing_impact = sum(
            e.cash_amount * (on_shift if e.term_days <= 7 else term_shift) / 10_000.0
            * e.term_days / 365.0
            for e in book.entries
        )

        results.append(RepoCurveStress(
            scenario_name=name,
            on_shift_bps=on_shift,
            term_shift_bps=term_shift,
            carry_impact=carry_impact,
            financing_impact=financing_impact,
            total_impact=carry_impact,
        ))

    return results


# ---------------------------------------------------------------------------
# Tier 3: Settlement Fail Workflow
# ---------------------------------------------------------------------------

@dataclass
class FailResolution:
    """Resolution path for a settlement fail."""
    counterparty: str
    issuer: str
    face_amount: float
    days_outstanding: int
    penalty_cost: float
    category: str          # "collateral", "system", "counterparty"
    buy_in_cost: float
    escalated: bool

    def to_dict(self) -> dict:
        return {
            "counterparty": self.counterparty, "issuer": self.issuer,
            "face": self.face_amount, "days_out": self.days_outstanding,
            "penalty": self.penalty_cost, "category": self.category,
            "buy_in_cost": self.buy_in_cost, "escalated": self.escalated,
        }


def fail_workflow(
    tracker: FailsTracker,
    current_prices: dict[str, float] | None = None,
    contract_prices: dict[str, float] | None = None,
    escalation_days: int = 5,
) -> list[FailResolution]:
    """Process settlement fails: categorise, price buy-in, escalate.

    Buy-in cost = max(0, (current - contract) / 100 × face).
    Categories: system (≤1d), collateral (2-3d), counterparty (4d+).
    Escalate if days > escalation_days.
    """
    if current_prices is None:
        current_prices = {}
    if contract_prices is None:
        contract_prices = {}

    results = []
    for f in tracker.fails:
        curr = current_prices.get(f.issuer, 100.0)
        contract = contract_prices.get(f.issuer, 100.0)
        buy_in = max(0, (curr - contract) / 100.0 * f.face_amount)

        if f.days_outstanding <= 1:
            category = "system"
        elif f.days_outstanding <= 3:
            category = "collateral"
        else:
            category = "counterparty"

        results.append(FailResolution(
            counterparty=f.counterparty, issuer=f.issuer,
            face_amount=f.face_amount, days_outstanding=f.days_outstanding,
            penalty_cost=f.penalty_cost, category=category,
            buy_in_cost=buy_in, escalated=f.days_outstanding > escalation_days,
        ))

    return results


# ---------------------------------------------------------------------------
# Tier 3: Collateral Substitution
# ---------------------------------------------------------------------------

@dataclass
class SubstitutionCandidate:
    """A substitute bond ranked by cost."""
    bond_id: str
    repo_rate: float
    haircut_pct: float
    cost_vs_original_bps: float
    available: bool

    def to_dict(self) -> dict:
        return {
            "bond": self.bond_id, "repo_rate": self.repo_rate,
            "haircut": self.haircut_pct, "cost_bp": self.cost_vs_original_bps,
            "available": self.available,
        }


def find_substitutes(
    failed_repo_rate: float,
    alternatives: dict[str, tuple[float, float, bool]],
) -> list[SubstitutionCandidate]:
    """Find substitute collateral, sorted by cost.

    Args:
        failed_repo_rate: repo rate on the failed trade.
        alternatives: {bond_id: (repo_rate, haircut_pct, available)}.
    """
    candidates = []
    for bond_id, (rate, haircut, avail) in alternatives.items():
        cost_bp = (rate - failed_repo_rate) * 10_000
        candidates.append(SubstitutionCandidate(
            bond_id=bond_id, repo_rate=rate,
            haircut_pct=haircut, cost_vs_original_bps=cost_bp,
            available=avail,
        ))
    return sorted(candidates, key=lambda c: c.cost_vs_original_bps)


# ---------------------------------------------------------------------------
# Tier 3: Balance Sheet Efficiency
# ---------------------------------------------------------------------------

@dataclass
class BalanceSheetMetrics:
    """Balance sheet efficiency for the repo desk."""
    total_assets: float
    total_capital_used: float
    annual_carry: float
    return_on_capital_pct: float
    leverage_ratio: float

    def to_dict(self) -> dict:
        return {
            "total_assets": self.total_assets, "capital_used": self.total_capital_used,
            "annual_carry": self.annual_carry, "roc_pct": self.return_on_capital_pct,
            "leverage": self.leverage_ratio,
        }


def balance_sheet_efficiency(
    book: RepoBook,
    haircut_pct: float = 2.0,
) -> BalanceSheetMetrics:
    """ROC = annualised_carry / capital. Leverage = assets / capital."""
    total_assets = sum(e.cash_amount for e in book.entries)
    capital = total_assets * haircut_pct / 100.0

    annual_carry = sum(
        e.carry * (365.0 / max(e.term_days, 1)) for e in book.entries
    )

    roc = (annual_carry / capital * 100.0) if capital > 0 else 0.0
    leverage = total_assets / capital if capital > 0 else 0.0

    return BalanceSheetMetrics(total_assets, capital, annual_carry, roc, leverage)


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

@dataclass
class MatchedBookEntry:
    """A matched pair: repo + reverse on same collateral."""
    issuer: str
    repo_cash: float
    reverse_cash: float
    repo_rate: float
    reverse_rate: float
    spread_earned_bps: float
    net_carry: float

    def to_dict(self) -> dict:
        return {
            "issuer": self.issuer,
            "repo_cash": self.repo_cash, "reverse_cash": self.reverse_cash,
            "repo_rate": self.repo_rate, "reverse_rate": self.reverse_rate,
            "spread_bps": self.spread_earned_bps, "net_carry": self.net_carry,
        }


def matched_book_analysis(book: RepoBook) -> list[MatchedBookEntry]:
    """Find matched repo/reverse pairs on same collateral and compute spread.

    The desk earns the spread between the rate it borrows at (repo)
    and the rate it lends at (reverse).
    """
    # Group by issuer
    by_issuer: dict[str, dict] = {}
    for e in book.entries:
        iss = e.collateral_issuer
        if iss not in by_issuer:
            by_issuer[iss] = {"repo": [], "reverse": []}
        by_issuer[iss][e.direction].append(e)

    matches = []
    for iss, sides in by_issuer.items():
        if not sides["repo"] or not sides["reverse"]:
            continue

        repo_cash = sum(e.cash_amount for e in sides["repo"])
        repo_rate = (sum(e.cash_amount * e.repo_rate for e in sides["repo"])
                     / repo_cash if repo_cash > 0 else 0.0)
        rev_cash = sum(e.cash_amount for e in sides["reverse"])
        rev_rate = (sum(e.cash_amount * e.repo_rate for e in sides["reverse"])
                    / rev_cash if rev_cash > 0 else 0.0)

        spread = (rev_rate - repo_rate) * 10_000  # bps earned
        matched_amt = min(repo_cash, rev_cash)
        avg_term = sum(e.term_days for e in sides["repo"] + sides["reverse"]) / \
                   len(sides["repo"] + sides["reverse"])
        net_carry = matched_amt * (rev_rate - repo_rate) * avg_term / 365.0

        matches.append(MatchedBookEntry(
            issuer=iss, repo_cash=repo_cash, reverse_cash=rev_cash,
            repo_rate=repo_rate, reverse_rate=rev_rate,
            spread_earned_bps=spread, net_carry=net_carry,
        ))

    return sorted(matches, key=lambda m: -abs(m.net_carry))


# ---------------------------------------------------------------------------
# Tier 4: Cross-Desk Funding Attribution
# ---------------------------------------------------------------------------

@dataclass
class FundingAttribution:
    """P&L attribution by strategy."""
    strategy: str
    total_cash: float
    total_carry: float
    pct_of_book: float

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "cash": self.total_cash,
            "carry": self.total_carry, "pct": self.pct_of_book,
        }


def funding_attribution(book: RepoBook) -> list[FundingAttribution]:
    """Attribute carry by strategy axis: GC vs special, ON vs term, repo vs reverse."""
    total_carry = book.net_carry()
    total_cash = sum(e.cash_amount for e in book.entries) or 1.0

    axes = {
        "GC_ON": lambda e: e.collateral_type == "GC" and e.term_days <= 1,
        "GC_term": lambda e: e.collateral_type == "GC" and e.term_days > 1,
        "special_ON": lambda e: e.collateral_type == "special" and e.term_days <= 1,
        "special_term": lambda e: e.collateral_type == "special" and e.term_days > 1,
        "reverse": lambda e: e.direction == "reverse",
    }

    result = []
    for strat, predicate in axes.items():
        entries = [e for e in book.entries if predicate(e)]
        cash = sum(e.cash_amount for e in entries)
        carry = sum(e.carry for e in entries)
        pct = cash / total_cash * 100.0
        result.append(FundingAttribution(strat, cash, carry, pct))

    return sorted(result, key=lambda f: -abs(f.total_carry))


# ---------------------------------------------------------------------------
# Tier 5: RepoBook serialisation and PV
# ---------------------------------------------------------------------------

class _RepoBookMixin:
    """Added to RepoBook via monkey-patch below."""
    pass


def _repo_book_to_dict(self) -> dict:
    """Serialise the RepoBook."""
    entries = []
    for e in self._entries:
        entries.append({
            "counterparty": e.counterparty,
            "collateral_issuer": e.collateral_issuer,
            "collateral_type": e.collateral_type,
            "face_amount": e.face_amount,
            "bond_price": e.bond_price,
            "repo_rate": e.repo_rate,
            "term_days": e.term_days,
            "coupon_rate": e.coupon_rate,
            "direction": e.direction,
            "start_date": e.start_date.isoformat() if e.start_date else None,
        })
    return {"type": "repo_book", "params": {
        "name": self.name, "entries": entries,
    }}


@classmethod
def _repo_book_from_dict(cls, d: dict) -> "RepoBook":
    """Deserialise a RepoBook."""
    p = d["params"]
    book = cls(name=p.get("name", "repo_book"))
    for e in p.get("entries", []):
        sd = date.fromisoformat(e["start_date"]) if e.get("start_date") else None
        book.add(RepoTradeEntry(
            counterparty=e["counterparty"],
            collateral_issuer=e["collateral_issuer"],
            collateral_type=e.get("collateral_type", "GC"),
            face_amount=e["face_amount"],
            bond_price=e["bond_price"],
            repo_rate=e["repo_rate"],
            term_days=e["term_days"],
            coupon_rate=e.get("coupon_rate", 0.0),
            direction=e.get("direction", "repo"),
            start_date=sd,
        ))
    return book


RepoBook._SERIAL_TYPE = "repo_book"
RepoBook.to_dict = _repo_book_to_dict
RepoBook.from_dict = _repo_book_from_dict

from pricebook.serialisable import _register as _reg_rb
_reg_rb(RepoBook)


def repo_book_pv(
    book: RepoBook,
    discount_curve,
    reference_date: date,
) -> dict[str, float]:
    """Total PV of all repo positions against a discount curve.

    Each position: PV = df(maturity) × repurchase_amount − cash_lent.

    Returns total PV, per-direction breakdown.
    """
    from pricebook.day_count import year_fraction, DayCountConvention
    from datetime import timedelta

    total_pv = 0.0
    repo_pv = 0.0
    reverse_pv = 0.0

    for e in book.entries:
        start = e.start_date or reference_date
        mat = start + timedelta(days=e.term_days)
        df = discount_curve.df(mat)
        dt = e.term_days / 365.0

        repurchase = e.cash_amount * (1 + e.repo_rate * dt)

        if e.direction == "repo":
            # Lent bond, borrowed cash. PV = df × repurchase − cash_lent
            pv = df * repurchase - e.cash_amount
            repo_pv += pv
        else:
            # Reverse: lent cash, borrowed bond. PV = cash_lent − df × repurchase
            pv = e.cash_amount - df * repurchase
            reverse_pv += pv

        total_pv += pv

    return {
        "total_pv": total_pv,
        "repo_pv": repo_pv,
        "reverse_pv": reverse_pv,
        "n_positions": len(book),
    }


# ---------------------------------------------------------------------------
# SOFR lookback + lockout (3→4)
# ---------------------------------------------------------------------------

def sofr_compounded_with_lookback(
    daily_rates: list[float],
    lookback_days: int = 2,
    lockout_days: int = 0,
) -> float:
    """Compounded SOFR with lookback and lockout conventions.

    Lookback: use rate from N days ago (SOFR published with lag).
    Lockout: last N days use the rate from the lockout start.

    Returns annualised compounded rate.
    """
    n = len(daily_rates)
    if n == 0:
        return 0.0

    # Apply lookback: shift rates back by lookback_days
    shifted = [0.0] * n
    for i in range(n):
        src = max(0, i - lookback_days)
        shifted[i] = daily_rates[src]

    # Apply lockout: freeze last N days
    if lockout_days > 0 and n > lockout_days:
        lock_rate = shifted[n - lockout_days - 1]
        for i in range(n - lockout_days, n):
            shifted[i] = lock_rate

    # Compound
    compound = 1.0
    for r in shifted:
        compound *= (1 + r / 360.0)

    total_yf = n / 360.0
    if total_yf <= 0:
        return 0.0
    return (compound - 1.0) / total_yf


# ---------------------------------------------------------------------------
# Netting (3→4)
# ---------------------------------------------------------------------------

@dataclass
class NettingResult:
    """Net exposure after netting repos with same counterparty."""
    counterparty: str
    gross_repo: float
    gross_reverse: float
    net_exposure: float
    n_trades: int

    def to_dict(self) -> dict:
        return {
            "counterparty": self.counterparty,
            "gross_repo": self.gross_repo,
            "gross_reverse": self.gross_reverse,
            "net": self.net_exposure,
            "n_trades": self.n_trades,
        }


def netting_by_counterparty(book: RepoBook) -> list[NettingResult]:
    """Compute net exposure per counterparty after netting.

    Under ISDA/GMRA netting agreements, repo and reverse repo
    with the same counterparty offset.
    """
    by_cp: dict[str, dict] = {}
    for e in book.entries:
        cp = e.counterparty
        if cp not in by_cp:
            by_cp[cp] = {"repo": 0.0, "reverse": 0.0, "n": 0}
        if e.direction == "repo":
            by_cp[cp]["repo"] += e.cash_amount
        else:
            by_cp[cp]["reverse"] += e.cash_amount
        by_cp[cp]["n"] += 1

    return [
        NettingResult(
            counterparty=cp,
            gross_repo=d["repo"],
            gross_reverse=d["reverse"],
            net_exposure=abs(d["repo"] - d["reverse"]),
            n_trades=d["n"],
        )
        for cp, d in sorted(by_cp.items())
    ]


# ---------------------------------------------------------------------------
# Fail state machine (3→4)
# ---------------------------------------------------------------------------

FAIL_STATES = ["open", "investigating", "resolving", "resolved", "bought_in"]


@dataclass
class FailState:
    """Settlement fail with lifecycle state."""
    counterparty: str
    issuer: str
    face_amount: float
    fail_date: date
    days_outstanding: int
    state: str = "open"       # FAIL_STATES
    buy_in_triggered: bool = False
    buy_in_cost: float = 0.0

    def advance(self, new_state: str) -> None:
        if new_state not in FAIL_STATES:
            raise ValueError(f"Invalid state '{new_state}'. Must be one of {FAIL_STATES}")
        self.state = new_state

    def trigger_buy_in(self, current_price: float, contract_price: float) -> None:
        self.buy_in_triggered = True
        self.buy_in_cost = max(0, (current_price - contract_price) / 100.0 * self.face_amount)
        self.state = "bought_in"

    def to_dict(self) -> dict:
        return {
            "counterparty": self.counterparty, "issuer": self.issuer,
            "face": self.face_amount, "days": self.days_outstanding,
            "state": self.state, "buy_in": self.buy_in_triggered,
            "buy_in_cost": self.buy_in_cost,
        }


def auto_escalate_fails(
    fails: list[FailState],
    investigate_after: int = 2,
    resolve_after: int = 5,
    buy_in_after: int = 10,
    current_prices: dict[str, float] | None = None,
    contract_prices: dict[str, float] | None = None,
) -> list[FailState]:
    """Auto-advance fail states based on days outstanding."""
    if current_prices is None:
        current_prices = {}
    if contract_prices is None:
        contract_prices = {}

    for f in fails:
        if f.state == "resolved" or f.state == "bought_in":
            continue
        if f.days_outstanding >= buy_in_after:
            curr = current_prices.get(f.issuer, 100.0)
            contract = contract_prices.get(f.issuer, 100.0)
            f.trigger_buy_in(curr, contract)
        elif f.days_outstanding >= resolve_after:
            f.advance("resolving")
        elif f.days_outstanding >= investigate_after:
            f.advance("investigating")

    return fails


# ---------------------------------------------------------------------------
# Repo key-rate DV01 (4→5)
# ---------------------------------------------------------------------------

def repo_key_rate_dv01(
    book: RepoBook,
    repo_curve,
    shift_bps: float = 1.0,
) -> dict[int, float]:
    """Key-rate DV01 on the repo curve — carry sensitivity per tenor bucket.

    Bumps each tenor on the repo curve independently and measures
    the carry change.

    Returns: {tenor_days: carry_change_per_bp}.
    """
    from pricebook.repo_term import RepoCurve, RepoRate

    base_carry = book.net_carry()
    shift = shift_bps / 10_000.0
    result = {}

    for i, tenor_days in enumerate(repo_curve._days):
        # Bump this tenor only
        new_rates = list(repo_curve._rates)
        new_rates[i] += shift
        bumped = RepoCurve(
            repo_curve.reference_date,
            [RepoRate(d, r) for d, r in zip(repo_curve._days, new_rates)],
        )

        # Reprice all trades at bumped repo rates
        bumped_carry = 0.0
        for e in book.entries:
            remaining = e.term_days
            bumped_rate = bumped.rate(remaining)
            dt = e.term_days / 365.0
            sign = 1.0 if e.direction == "repo" else -1.0
            coupon = e.face_amount * e.coupon_rate * dt
            financing = e.cash_amount * bumped_rate * e.term_days / 360.0
            bumped_carry += sign * (coupon - financing)

        result[tenor_days] = (bumped_carry - base_carry) / shift_bps

    return result


# ---------------------------------------------------------------------------
# Regulatory haircut floors (Basel III) (3→4)
# ---------------------------------------------------------------------------

BASEL_HAIRCUT_FLOORS = {
    # Asset class → minimum haircut % (Basel III Table 1)
    "sovereign_0_1Y": 0.5,
    "sovereign_1_5Y": 2.0,
    "sovereign_5Y+": 4.0,
    "agency_0_1Y": 1.0,
    "agency_1_5Y": 3.0,
    "agency_5Y+": 6.0,
    "ig_corp_0_1Y": 2.0,
    "ig_corp_1_5Y": 4.0,
    "ig_corp_5Y+": 8.0,
    "hy_corp": 15.0,
    "equity_main_index": 15.0,
    "equity_other": 25.0,
    "cash_same_ccy": 0.0,
    "fx_mismatch_add_on": 8.0,  # additional for xccy
}


def regulatory_haircut(
    asset_class: str,
    maturity_years: float,
    is_cross_currency: bool = False,
) -> float:
    """Minimum regulatory haircut (Basel III).

    Args:
        asset_class: "sovereign", "agency", "ig_corp", "hy_corp", "equity"
        maturity_years: remaining maturity of the collateral.
        is_cross_currency: adds 8% FX add-on.
    """
    if asset_class in ("hy_corp", "equity_main_index", "equity_other"):
        key = asset_class
    else:
        if maturity_years <= 1:
            bucket = "0_1Y"
        elif maturity_years <= 5:
            bucket = "1_5Y"
        else:
            bucket = "5Y+"
        key = f"{asset_class}_{bucket}"

    haircut = BASEL_HAIRCUT_FLOORS.get(key, 10.0)

    if is_cross_currency:
        haircut += BASEL_HAIRCUT_FLOORS.get("fx_mismatch_add_on", 8.0)

    return haircut
