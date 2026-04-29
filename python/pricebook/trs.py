"""Unified Total Return Swap — equity, bond, loan underlyings.

Single class that auto-detects the underlying type and delegates to
the appropriate pricing engine. Supports repo financing (Lou 2018),
discrete dividends, multi-period with MTM notional reset, and SOFR
funding conventions.

    from pricebook.trs import TotalReturnSwap, FundingLegSpec

    # Equity TRS
    trs = TotalReturnSwap(underlying=100.0, notional=10_000_000,
                           start=date(2024,1,15), end=date(2025,1,15),
                           repo_spread=0.002)
    result = trs.price(curve)

    # Bond TRS
    trs = TotalReturnSwap(underlying=my_bond, notional=50_000_000,
                           start=date(2024,1,15), end=date(2025,1,15),
                           repo_spread=0.001, haircut=0.05)

    # Loan TRS
    trs = TotalReturnSwap(underlying=my_loan, notional=my_loan.notional,
                           start=date(2024,1,15), end=date(2025,1,15))

    # All work with Trade/Portfolio
    portfolio.add(Trade(trs))

References:
    Lou, W. (2018). Pricing Total Return Swap. SSRN 3217420.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Union

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


# ---- Result and spec dataclasses ----

@dataclass
class FundingLegSpec:
    """Funding leg conventions for the TRS."""
    spread: float = 0.0
    rate_index: str = "SOFR"
    compounding: str = "in_arrears"
    payment_delay: int = 2
    lookback: int = 0
    day_count: DayCountConvention = DayCountConvention.ACT_360


@dataclass
class LSTATerms:
    """LSTA settlement conventions for loan TRS."""
    settlement_days: int = 7         # T+7 standard
    trade_type: str = "assignment"   # "assignment" or "participation"
    minimum_transfer: float = 250_000.0


@dataclass
class TRSResult:
    """Unified TRS pricing result."""
    value: float
    total_return_leg: float
    funding_leg: float
    price_return: float
    income_return: float
    fva: float
    repo_factor: float
    dpv: float = 0.0
    xva: dict[str, float] | None = None
    period_details: list[dict] | None = None

    @property
    def price(self) -> float:
        return self.value

    @property
    def mtm(self) -> float:
        """Backward compat alias."""
        return self.value

    def to_dict(self) -> dict[str, float]:
        d = {
            "price": self.value, "total_return_leg": self.total_return_leg,
            "funding_leg": self.funding_leg, "price_return": self.price_return,
            "income_return": self.income_return, "fva": self.fva,
            "repo_factor": self.repo_factor, "dpv": self.dpv,
        }
        if self.xva:
            d.update(self.xva)
        return d


# ---- Unified TRS class ----

class TotalReturnSwap:
    """Unified Total Return Swap — equity, bond, or loan underlying."""

    def __init__(
        self,
        underlying,
        notional: float,
        start: date,
        end: date,
        funding: FundingLegSpec | None = None,
        dividends: list | None = None,
        initial_price: float | None = None,
        repo_spread: float = 0.0,
        haircut: float = 0.0,
        credit_curve_name: str | None = None,
        recovery: float = 0.4,
        reset_dates: list[date] | None = None,
        mtm_reset: bool = False,
        sigma: float = 0.20,
        # Loan-specific
        prepay_model: float | str | None = None,  # None, float (flat CPR), "PSA"
        settlement_terms: LSTATerms | None = None,
        # CLN-specific
        survival_curve=None,
    ):
        self.underlying = underlying
        self.notional = notional
        self.start = start
        self.end = end
        self.funding = funding or FundingLegSpec()
        self.dividends = dividends or []
        self.initial_price = initial_price
        self.repo_spread = repo_spread
        self.haircut = haircut
        self.credit_curve_name = credit_curve_name
        self.recovery = recovery
        self.reset_dates = reset_dates
        self.mtm_reset = mtm_reset
        self.sigma = sigma
        self.prepay_model = prepay_model
        self.settlement_terms = settlement_terms
        self.survival_curve = survival_curve

        # Detect underlying type
        self._underlying_type = self._detect_type()

    def _detect_type(self) -> str:
        if isinstance(self.underlying, (int, float)):
            return "equity"
        cls_name = type(self.underlying).__name__
        if cls_name == "FixedRateBond":
            return "bond"
        if cls_name in ("TermLoan", "RevolvingFacility"):
            return "loan"
        if cls_name in ("CreditLinkedNote",):
            return "cln"
        return "unknown"

    # ---- Main pricing ----

    def price(self, curve: DiscountCurve, projection_curve=None) -> TRSResult:
        """Price the TRS. Auto-dispatches based on underlying type."""
        if self.reset_dates:
            return self._price_multi_period(curve, projection_curve)
        if self._underlying_type == "equity":
            return self._price_equity(curve, projection_curve)
        if self._underlying_type == "bond":
            return self._price_bond(curve, projection_curve)
        if self._underlying_type == "loan":
            return self._price_loan(curve, projection_curve)
        if self._underlying_type == "cln":
            return self._price_cln(curve, projection_curve)
        raise TypeError(f"Unsupported underlying type: {type(self.underlying).__name__}")

    # ---- Equity TRS ----

    def _price_equity(self, curve, projection_curve) -> TRSResult:
        """Equity TRS: Lou (2018) Eq (7).

        V = (M0 rf T + S0) D - St exp((rs-r)(T-t))

        All values in currency units (notional-scaled).
        """
        from pricebook.bond_forward import repo_financing_factor
        from pricebook.dividend_model import Dividend

        spot = float(self.underlying)
        S_0 = self.initial_price if self.initial_price is not None else spot
        if S_0 <= 0:
            raise ValueError(f"Initial price must be positive, got {S_0}")

        T = year_fraction(self.start, self.end, DayCountConvention.ACT_365_FIXED)
        if T < 1e-5:
            raise ValueError(f"Maturity too short: {T:.6f} years")

        tau = year_fraction(curve.reference_date, self.end, DayCountConvention.ACT_365_FIXED)
        D = curve.df(self.end)

        # Funding rate: forward OIS + spread
        fwd_rate = -math.log(curve.df(self.end) / curve.df(self.start)) / T
        r_f = fwd_rate + self.funding.spread

        # Dividends: income (received) and forward adjustment (future)
        income = 0.0
        future_div_pv = 0.0
        for div in self.dividends:
            if isinstance(div, Dividend):
                if self.start < div.ex_date <= curve.reference_date:
                    income += div.amount
                elif div.ex_date > curve.reference_date:
                    future_div_pv += div.amount * curve.df(div.ex_date)

        # Scale factor: number of shares = notional / S_0
        shares = self.notional / S_0

        # Lou Eq (7): V = (M0 rf T + S0) D - St exp((rs-r) tau)
        # M0 = notional (currency), S0 and St are per-share
        repo_factor = repo_financing_factor(self.repo_spread, tau)
        funding_leg = (self.notional * r_f * T + self.notional) * D
        asset_leg = (spot - future_div_pv) * shares * repo_factor

        value = funding_leg - asset_leg
        fva = (repo_factor - 1) * spot * shares

        # Decomposition
        price_return = (spot - S_0) * shares
        income_scaled = income * shares

        return TRSResult(
            value=value, total_return_leg=price_return + income_scaled,
            funding_leg=funding_leg,
            price_return=price_return, income_return=income_scaled,
            fva=fva, repo_factor=repo_factor,
        )

    # ---- Bond TRS ----

    def _price_bond(self, curve, projection_curve) -> TRSResult:
        """Bond TRS: Lou (2018) Eq (25) with haircut blending (Eq 19)."""
        from pricebook.bond_forward import repo_financing_factor, blended_repo_rate

        bond = self.underlying
        T = year_fraction(self.start, self.end, DayCountConvention.ACT_365_FIXED)
        if T < 1e-5:
            raise ValueError(f"Maturity too short: {T:.6f} years")
        if curve.reference_date >= bond.maturity:
            raise ValueError("Bond has matured")

        D = curve.df(self.end)

        # Current and initial price
        current_dirty = bond.dirty_price(curve)
        initial_dirty = self.initial_price if self.initial_price is not None else current_dirty

        # Coupons received
        income = 0.0
        for cf in bond.coupon_leg.cashflows:
            if self.start < cf.payment_date <= curve.reference_date:
                income += cf.amount
        income_scaled = income / bond.face_value * self.notional

        # Price return (per 100 face → scale to notional)
        price_return = (current_dirty - initial_dirty) / 100.0 * self.notional

        # Funding leg
        fwd_rate = -math.log(D / curve.df(self.start)) / T
        r_f = fwd_rate + self.funding.spread
        yf = year_fraction(self.start, curve.reference_date, self.funding.day_count)
        funding_leg = self.notional * r_f * yf

        # Blended repo rate: r̄s = (1-h)rs + h rN (Lou Eq 19)
        # Use fwd_rate as proxy for rN (bank's unsecured ~ OIS + spread)
        r_s = fwd_rate + self.repo_spread
        r_N = fwd_rate + 0.02  # unsecured funding proxy
        rs_bar = blended_repo_rate(r_s, r_N, self.haircut)
        rs_bar_spread = rs_bar - fwd_rate

        # FVA using blended rate (Lou Eq 25-26)
        repo_factor = repo_financing_factor(rs_bar_spread, T)
        fva = (repo_factor - 1) * current_dirty / 100.0 * self.notional

        total_return = price_return + income_scaled
        value = total_return - funding_leg - fva

        return TRSResult(
            value=value, total_return_leg=total_return,
            funding_leg=funding_leg, price_return=price_return,
            income_return=income_scaled, fva=fva, repo_factor=repo_factor,
        )

    # ---- Loan TRS ----

    def _price_loan(self, curve, projection_curve) -> TRSResult:
        """Loan TRS: adapts bond TRS framework to TermLoan cashflows.

        Supports prepayment-adjusted cashflows (CPR/PSA) and LSTA settlement.
        """
        from pricebook.bond_forward import repo_financing_factor

        loan = self.underlying
        proj = projection_curve or curve
        T = year_fraction(self.start, self.end, DayCountConvention.ACT_365_FIXED)
        if T < 1e-5:
            raise ValueError(f"Maturity too short: {T:.6f} years")

        # Current and initial price
        current_price = loan.dirty_price(curve, proj)
        initial_price = self.initial_price if self.initial_price is not None else 100.0

        # Get cashflows — prepayment-adjusted if model specified
        if self.prepay_model is not None:
            from pricebook.exotic_loan import prepay_adjusted_loan, psa_cpr
            if isinstance(self.prepay_model, (int, float)):
                cpr = float(self.prepay_model)
            elif self.prepay_model == "PSA":
                # Use 12-month PSA as representative
                cpr = psa_cpr(12, 1.0)
            else:
                cpr = 0.0
            flows = prepay_adjusted_loan(loan, cpr, proj)
        else:
            flows = loan.cashflows(proj)

        # Interest income received (between start and valuation)
        income = 0.0
        for d, interest, _ in flows:
            if self.start < d <= curve.reference_date:
                income += interest
        income_scaled = income / loan.notional * self.notional

        # Price return
        price_return = (current_price - initial_price) / 100.0 * self.notional

        # LSTA settlement adjustment
        settlement_shift = 0
        if self.settlement_terms is not None:
            settlement_shift = self.settlement_terms.settlement_days

        # Funding leg (adjusted for settlement delay)
        yf = year_fraction(self.start, curve.reference_date, self.funding.day_count)
        fwd_rate = proj.forward_rate(self.start, self.end) if T > 0 else 0.0
        r_f = fwd_rate + self.funding.spread
        funding_leg = self.notional * r_f * yf

        # Settlement delay cost: carry over T+7
        if settlement_shift > 0:
            settle_cost = self.notional * r_f * settlement_shift / 365.0
            funding_leg += settle_cost

        # FVA
        repo_factor = repo_financing_factor(self.repo_spread, T)
        fva = (repo_factor - 1) * current_price / 100.0 * self.notional

        total_return = price_return + income_scaled
        value = total_return - funding_leg - fva

        return TRSResult(
            value=value, total_return_leg=total_return,
            funding_leg=funding_leg, price_return=price_return,
            income_return=income_scaled, fva=fva, repo_factor=repo_factor,
        )

    # ---- CLN TRS ----

    def _price_cln(self, curve, projection_curve) -> TRSResult:
        """CLN TRS: total return on a credit-linked note.

        The TR receiver gets CLN price changes + coupon income.
        The TR payer finances at floating + spread.
        """
        from pricebook.bond_forward import repo_financing_factor

        cln = self.underlying
        T = year_fraction(self.start, self.end, DayCountConvention.ACT_365_FIXED)
        if T < 1e-5:
            raise ValueError(f"Maturity too short: {T:.6f} years")

        # CLN pricing requires a survival curve — use TRS-level, then CLN-level, then error
        survival_curve = self.survival_curve
        if survival_curve is None and hasattr(cln, '_survival_curve'):
            survival_curve = cln._survival_curve
        if survival_curve is None:
            from pricebook.survival_curve import SurvivalCurve
            import warnings
            warnings.warn(
                "No survival curve provided for CLN TRS — using flat hazard=0.02 fallback. "
                "Pass survival_curve to TotalReturnSwap for accurate pricing.",
                stacklevel=2,
            )
            survival_curve = SurvivalCurve.flat(curve.reference_date, 0.02)

        current_price = cln.price_per_100(curve, survival_curve)
        initial_price = self.initial_price if self.initial_price is not None else 100.0

        # Coupon income between start and valuation date
        income = 0.0
        for i in range(1, len(cln.schedule)):
            pay_date = cln.schedule[i]
            if self.start < pay_date <= curve.reference_date:
                yf = year_fraction(cln.schedule[i-1], pay_date, cln.day_count)
                income += cln.notional * cln.coupon_rate * yf
        income_scaled = income / cln.notional * self.notional

        # Price return
        price_return = (current_price - initial_price) / 100.0 * self.notional

        # Funding leg
        D = curve.df(self.end)
        fwd_rate = -math.log(D / curve.df(self.start)) / T
        r_f = fwd_rate + self.funding.spread
        yf = year_fraction(self.start, curve.reference_date, self.funding.day_count)
        funding_leg = self.notional * r_f * yf

        # FVA
        repo_factor = repo_financing_factor(self.repo_spread, T)
        fva = (repo_factor - 1) * current_price / 100.0 * self.notional

        total_return = price_return + income_scaled
        value = total_return - funding_leg - fva

        return TRSResult(
            value=value, total_return_leg=total_return,
            funding_leg=funding_leg, price_return=price_return,
            income_return=income_scaled, fva=fva, repo_factor=repo_factor,
        )

    # ---- Multi-period ----

    def _price_multi_period(self, curve, projection_curve) -> TRSResult:
        from pricebook.trs_lou import trs_multi_period

        resets = sorted([self.start] + self.reset_dates + [self.end])
        K = len(resets) - 1
        periods = [year_fraction(resets[i], resets[i + 1], DayCountConvention.ACT_365_FIXED)
                    for i in range(K)]
        dfs = [curve.df(resets[i + 1]) for i in range(K)]

        # Forwards at each reset: F_j = spot × df(start) / df(reset_j)
        if self._underlying_type == "equity":
            spot = float(self.underlying)
            forwards = [spot * curve.df(self.start) / curve.df(resets[i])
                        for i in range(K + 1)]
        elif self._underlying_type == "bond":
            price = self.underlying.dirty_price(curve)
            forwards = [price * curve.df(self.start) / curve.df(resets[i])
                        for i in range(K + 1)]
        else:
            forwards = [100.0 * curve.df(self.start) / curve.df(resets[i])
                        for i in range(K + 1)]

        if forwards[0] <= 0:
            raise ValueError(f"Forward price must be positive, got {forwards[0]}")

        # Funding rates: period-matched forward rate + spread
        funding_rates = []
        for i in range(K):
            if periods[i] < 1e-10:
                raise ValueError(f"Period {i} too short: {periods[i]}")
            fwd_r = -math.log(dfs[i] / curve.df(resets[i])) / periods[i]
            funding_rates.append(fwd_r + self.funding.spread)

        # Notionals: MTM = F_j (currency), fixed = notional (currency)
        if self.mtm_reset:
            funding_notionals = [f * self.notional / forwards[0] for f in forwards[:K]]
        else:
            funding_notionals = [self.notional] * K

        value = trs_multi_period(
            forwards=forwards, funding_rates=funding_rates,
            funding_notionals=funding_notionals, periods=periods,
            discount_factors=dfs, recovery=self.recovery,
        )

        from pricebook.bond_forward import repo_financing_factor
        T = year_fraction(self.start, self.end, DayCountConvention.ACT_365_FIXED)
        repo_factor = repo_financing_factor(self.repo_spread, T)

        return TRSResult(
            value=value, total_return_leg=0.0,
            funding_leg=0.0, price_return=0.0,
            income_return=0.0, fva=0.0, repo_factor=repo_factor,
        )

    # ---- Tree pricing (equity only) ----

    def price_tree(self, curve, n_steps=100, margin_style="full_csa",
                   mu=1.0, r_b=None, r_c=None) -> TRSResult:
        """Price via trinomial tree (Lou Section 4). Equity only."""
        if self._underlying_type != "equity":
            raise NotImplementedError("Tree pricing only for equity TRS")

        from pricebook.trs_tree import trs_trinomial_tree

        spot = float(self.underlying)
        T = year_fraction(self.start, self.end, DayCountConvention.ACT_365_FIXED)
        r = -math.log(curve.df(self.end)) / max(T, 1e-10)
        D_period = math.exp(-r * T)
        libor = (1 / D_period - 1) / T
        r_f = libor + self.funding.spread

        result = trs_trinomial_tree(
            spot, r_f, T, r, self.repo_spread, self.sigma,
            self.sigma, n_steps=n_steps, mu=mu,
            r_b=r_b, r_c=r_c, M_0=self.notional,
            margin_style=margin_style)

        from pricebook.bond_forward import repo_financing_factor
        repo_factor = repo_financing_factor(self.repo_spread, T)

        return TRSResult(
            value=result.value, total_return_leg=0.0,
            funding_leg=0.0, price_return=0.0,
            income_return=0.0, fva=0.0, repo_factor=repo_factor,
        )

    # ---- XVA pricing (equity only) ----

    def price_xva(self, curve, r_b, r_c, s_b, s_c,
                  mu_b=0.0, mu_c=0.0, n_steps=100,
                  mu=0.0, margin_style="full_csa") -> TRSResult:
        """Price with XVA decomposition (Lou Section 5). Equity only."""
        if self._underlying_type != "equity":
            raise NotImplementedError("XVA pricing only for equity TRS")

        from pricebook.trs_tree import trs_tree_xva

        spot = float(self.underlying)
        T = year_fraction(self.start, self.end, DayCountConvention.ACT_365_FIXED)
        r = -math.log(curve.df(self.end)) / max(T, 1e-10)
        D_period = math.exp(-r * T)
        libor = (1 / D_period - 1) / T
        r_f = libor + self.funding.spread

        result = trs_tree_xva(
            spot, r_f, T, r, self.repo_spread, self.sigma,
            r_b, r_c, s_b, s_c, mu_b, mu_c,
            n_steps=n_steps, mu=mu, M_0=self.notional,
            margin_style=margin_style)

        return TRSResult(
            value=result.value, total_return_leg=0.0,
            funding_leg=0.0, price_return=0.0,
            income_return=0.0, fva=0.0, repo_factor=1.0,
            xva=result.to_dict(),
        )

    # ---- Greeks ----

    def greeks(self, curve, projection_curve=None) -> dict[str, float]:
        """Bump-and-reprice sensitivities."""
        base = self.price(curve, projection_curve)

        # Delta (underlying price sensitivity)
        if self._underlying_type == "equity":
            old = self.underlying
            bump = float(old) * 0.01
            self.underlying = float(old) + bump
            up = self.price(curve, projection_curve)
            self.underlying = float(old) - bump
            dn = self.price(curve, projection_curve)
            self.underlying = old
            delta = (up.value - dn.value) / (2 * bump)
        else:
            delta = 0.0  # bond/loan delta via DV01

        # Repo sensitivity
        old_repo = self.repo_spread
        self.repo_spread = old_repo + 0.0001
        up_repo = self.price(curve, projection_curve)
        self.repo_spread = old_repo
        repo_sens = (up_repo.value - base.value) / 0.0001

        return {"delta": delta, "repo_sensitivity": repo_sens, "fva": base.fva}

    # ---- Breakeven spread ----

    def breakeven_spread(self, curve, projection_curve=None) -> float:
        """Fair TRS spread that makes NPV = 0."""
        from pricebook.solvers import brentq

        def objective(sf):
            self.funding = FundingLegSpec(spread=sf, **{
                k: v for k, v in self.funding.__dict__.items() if k != "spread"
            })
            return self.price(curve, projection_curve).value

        old_funding = self.funding
        try:
            sf = brentq(objective, -0.10, 0.10)
        finally:
            self.funding = old_funding
        return sf

    # ---- Trade/Portfolio integration ----

    def pv_ctx(self, ctx) -> float:
        """Price from PricingContext — compatible with Trade.pv()."""
        curve = ctx.discount_curve
        proj = None
        if ctx.projection_curves:
            proj = next(iter(ctx.projection_curves.values()), None)
        return self.price(curve, proj).value

from pricebook.serialisable import _register, _serialise_atom
from pricebook.serialisable import from_dict as _s_from_dict

TotalReturnSwap._SERIAL_TYPE = "trs"

def _trs_to_dict(self):
    if isinstance(self.underlying, (int, float)):
        underlying_d = {"type": "equity_spot", "value": float(self.underlying)}
    elif hasattr(self.underlying, "to_dict"):
        underlying_d = self.underlying.to_dict()
    else:
        underlying_d = {"type": "equity_spot", "value": 100.0}
    return {"type": "trs", "params": {
        "underlying": underlying_d,
        "notional": self.notional, "start": self.start.isoformat(),
        "end": self.end.isoformat(), "repo_spread": self.repo_spread,
        "haircut": self.haircut, "sigma": self.sigma,
    }}

@classmethod
def _trs_from_dict(cls, d):
    from datetime import date as _d
    p = d["params"]
    u = p["underlying"]
    if isinstance(u, (int, float)):
        underlying = float(u)
    elif isinstance(u, dict):
        if u.get("type") == "equity_spot":
            underlying = float(u["value"])
        else:
            underlying = _s_from_dict(u)
    else:
        underlying = float(u)
    return cls(underlying=underlying, notional=p["notional"],
               start=_d.fromisoformat(p["start"]), end=_d.fromisoformat(p["end"]),
               repo_spread=p.get("repo_spread", 0.0),
               haircut=p.get("haircut", 0.0), sigma=p.get("sigma", 0.20))

TotalReturnSwap.to_dict = _trs_to_dict
TotalReturnSwap.from_dict = _trs_from_dict
_register(TotalReturnSwap)
