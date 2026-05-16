"""Unified Total Return Swap — equity, bond, loan, commodity, FX underlyings.

Single class that auto-detects the underlying type and delegates to
the appropriate pricing engine. Supports repo financing (Lou 2018),
discrete dividends, multi-period with MTM notional reset, and SOFR
funding conventions.

    from pricebook.equity.trs import TotalReturnSwap, FundingLegSpec

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

    # Commodity TRS
    trs = TotalReturnSwap(
        underlying=CommodityUnderlying("WTI", spot=75.0, storage_cost=0.02,
                                        convenience_yield=0.01),
        notional=100_000, start=..., end=...)

    # FX TRS
    trs = TotalReturnSwap(
        underlying=FXUnderlying("EUR", "USD", spot=1.08),
        notional=10_000_000, start=..., end=...)

    # Cross-currency Bond TRS
    trs = TotalReturnSwap(underlying=bund, notional=10_000_000,
        xccy=XccySpec(fx_rate=1.08, asset_currency="EUR",
                      funding_currency="USD"))

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

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve


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
class CommodityUnderlying:
    """Commodity underlying for TRS.

    Forward = spot x exp((r - convenience_yield + storage_cost) x T).
    Supports seasonal factors and StorageCostModel from commodity_seasonal.py.
    """
    name: str
    spot: float
    storage_cost: float = 0.0      # annualised % of spot
    convenience_yield: float = 0.0  # annualised % of spot
    seasonal: object = None        # SeasonalFactors from commodity_seasonal.py
    storage_model: object = None   # StorageCostModel from commodity_seasonal.py


@dataclass
class FXUnderlying:
    """FX underlying for TRS.

    Forward = spot x df_base / df_quote (covered interest rate parity).
    Performance = (spot_current / spot_initial - 1) x notional.
    Supports quanto adjustment via fx_vol and fx_correlation.
    """
    base_ccy: str   # e.g. "EUR"
    quote_ccy: str  # e.g. "USD"
    spot: float     # base/quote (e.g. 1.08 EUR/USD)
    fx_vol: float = 0.0          # annualised FX vol (for quanto)
    fx_correlation: float = 0.0  # correlation between FX and asset (for quanto)


@dataclass
class XccySpec:
    """Cross-currency specification for bond/loan TRS.

    Asset denominated in asset_currency, funding in funding_currency.
    """
    fx_rate: float          # asset_ccy per funding_ccy at inception
    asset_currency: str
    funding_currency: str
    fx_haircut: float = 0.08  # FX add-on haircut (Basel: 8%)


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

    @classmethod
    def from_dict(cls, d: dict) -> TRSResult:
        p = d.get("params", d)
        return cls(value=p["price"], total_return_leg=p["total_return_leg"],
                   funding_leg=p["funding_leg"], price_return=p["price_return"],
                   income_return=p["income_return"], fva=p["fva"],
                   repo_factor=p["repo_factor"], dpv=p.get("dpv", 0.0))


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
        # Cross-currency
        xccy: XccySpec | None = None,
        # Haircut schedule
        haircut_schedule: list[tuple[date, float]] | None = None,
    ):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if end < start:
            raise ValueError(f"end ({end}) must not be before start ({start})")
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
        self.xccy = xccy
        self.haircut_schedule = haircut_schedule

        # Detect underlying type
        self._underlying_type = self._detect_type()

    def _detect_type(self) -> str:
        if isinstance(self.underlying, (int, float)):
            return "equity"
        if isinstance(self.underlying, CommodityUnderlying):
            return "commodity"
        if isinstance(self.underlying, FXUnderlying):
            return "fx"
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
            return price_multi_period(self, curve, projection_curve)
        _PRICERS = {
            "equity": price_equity_trs,
            "bond": price_bond_trs,
            "loan": price_loan_trs,
            "cln": price_cln_trs,
            "commodity": price_commodity_trs,
            "fx": price_fx_trs,
        }
        pricer = _PRICERS.get(self._underlying_type)
        if pricer is None:
            raise TypeError(f"Unsupported underlying type: {type(self.underlying).__name__}")
        return pricer(self, curve, projection_curve)

    # ---- Tree pricing (equity only) ----

    def price_tree(self, curve, n_steps=100, margin_style="full_csa",
                   mu=1.0, r_b=None, r_c=None) -> TRSResult:
        """Price via trinomial tree (Lou Section 4). Equity only."""
        if self._underlying_type != "equity":
            raise NotImplementedError("Tree pricing only for equity TRS")

        from pricebook.equity.trs_tree import trs_trinomial_tree

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

        from pricebook.fixed_income.bond_forward import repo_financing_factor
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

        from pricebook.equity.trs_tree import trs_tree_xva

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
        """Bump-and-reprice sensitivities (immutable — no state mutation)."""
        base = self.price(curve, projection_curve)

        # Delta via bumped copy
        delta = 0.0
        if self._underlying_type == "equity":
            spot = float(self.underlying)
            bump = spot * 0.01
            trs_up = _trs_with_underlying(self, spot + bump)
            trs_dn = _trs_with_underlying(self, spot - bump)
            delta = (trs_up.price(curve, projection_curve).value
                     - trs_dn.price(curve, projection_curve).value) / (2 * bump)
        elif self._underlying_type == "commodity":
            bump = self.underlying.spot * 0.01
            trs_up = _trs_with_underlying(self, CommodityUnderlying(
                self.underlying.name, self.underlying.spot + bump,
                self.underlying.storage_cost, self.underlying.convenience_yield))
            trs_dn = _trs_with_underlying(self, CommodityUnderlying(
                self.underlying.name, self.underlying.spot - bump,
                self.underlying.storage_cost, self.underlying.convenience_yield))
            delta = (trs_up.price(curve, projection_curve).value
                     - trs_dn.price(curve, projection_curve).value) / (2 * bump)
        elif self._underlying_type == "fx":
            bump = self.underlying.spot * 0.01
            trs_up = _trs_with_underlying(self, FXUnderlying(
                self.underlying.base_ccy, self.underlying.quote_ccy,
                self.underlying.spot + bump))
            trs_dn = _trs_with_underlying(self, FXUnderlying(
                self.underlying.base_ccy, self.underlying.quote_ccy,
                self.underlying.spot - bump))
            delta = (trs_up.price(curve, projection_curve).value
                     - trs_dn.price(curve, projection_curve).value) / (2 * bump)

        # Repo sensitivity via bumped copy
        trs_repo_up = _trs_with_repo_spread(self, self.repo_spread + 0.0001)
        repo_sens = (trs_repo_up.price(curve, projection_curve).value - base.value) / 0.0001

        return {"delta": delta, "repo_sensitivity": repo_sens, "fva": base.fva}

    # ---- Breakeven spread ----

    def breakeven_spread(self, curve, projection_curve=None) -> float:
        """Fair TRS spread that makes NPV = 0 (immutable — no state mutation)."""
        from pricebook.core.solvers import brentq

        def objective(sf):
            trs_bumped = _trs_with_funding_spread(self, sf)
            return trs_bumped.price(curve, projection_curve).value

        return brentq(objective, -0.10, 0.10)

    # ---- Trade/Portfolio integration ----

    def pv_ctx(self, ctx) -> float:
        """Price from PricingContext — compatible with Trade.pv()."""
        curve = ctx.discount_curve
        proj = None
        if ctx.projection_curves:
            proj = next(iter(ctx.projection_curves.values()), None)
        return self.price(curve, proj).value

    # ---- Serialisation ----

    _SERIAL_TYPE = "trs"

    def to_dict(self) -> dict:
        from pricebook.core.serialisable import _serialise_atom
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
    def from_dict(cls, d: dict) -> TotalReturnSwap:
        from pricebook.core.serialisable import from_dict as _s_from_dict
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
                   start=date.fromisoformat(p["start"]), end=date.fromisoformat(p["end"]),
                   repo_spread=p.get("repo_spread", 0.0),
                   haircut=p.get("haircut", 0.0), sigma=p.get("sigma", 0.20))


from pricebook.core.serialisable import _register
_register(TotalReturnSwap)


# ---------------------------------------------------------------------------
# Standalone pricing functions (extracted from TotalReturnSwap)
# ---------------------------------------------------------------------------

def price_equity_trs(trs, curve, projection_curve) -> TRSResult:
    """Equity TRS: Lou (2018) Eq (7).

    V = (M0 rf T + S0) D - St exp((rs-r)(T-t))

    All values in currency units (notional-scaled).
    """
    from pricebook.fixed_income.bond_forward import repo_financing_factor
    from pricebook.equity.dividend_model import Dividend

    spot = float(trs.underlying)
    S_0 = trs.initial_price if trs.initial_price is not None else spot
    if S_0 <= 0:
        raise ValueError(f"Initial price must be positive, got {S_0}")

    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    if T < 1e-5:
        raise ValueError(f"Maturity too short: {T:.6f} years")

    tau = year_fraction(curve.reference_date, trs.end, DayCountConvention.ACT_365_FIXED)
    D = curve.df(trs.end)

    # Funding rate: forward OIS + spread
    fwd_rate = -math.log(curve.df(trs.end) / curve.df(trs.start)) / T
    r_f = fwd_rate + trs.funding.spread

    # Dividends: income (received) and forward adjustment (future)
    income = 0.0
    future_div_pv = 0.0
    for div in trs.dividends:
        if isinstance(div, Dividend):
            if trs.start < div.ex_date <= curve.reference_date:
                income += div.amount
            elif div.ex_date > curve.reference_date:
                future_div_pv += div.amount * curve.df(div.ex_date)

    # Scale factor: number of shares = notional / S_0
    shares = trs.notional / S_0

    # Lou Eq (7): V = (M0 rf T + S0) D - St exp((rs-r) tau)
    # M0 = notional (currency), S0 and St are per-share
    repo_factor = repo_financing_factor(trs.repo_spread, tau)
    funding_leg = (trs.notional * r_f * T + trs.notional) * D
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


def price_bond_trs(trs, curve, projection_curve) -> TRSResult:
    """Bond TRS: Lou (2018) Eq (25) with haircut blending (Eq 19)."""
    from pricebook.fixed_income.bond_forward import repo_financing_factor, blended_repo_rate

    bond = trs.underlying
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    if T < 1e-5:
        raise ValueError(f"Maturity too short: {T:.6f} years")
    if curve.reference_date >= bond.maturity:
        raise ValueError("Bond has matured")

    D = curve.df(trs.end)

    # Current and initial price
    current_dirty = bond.dirty_price(curve)
    initial_dirty = trs.initial_price if trs.initial_price is not None else current_dirty

    # Coupons received
    income = 0.0
    for cf in bond.coupon_leg.cashflows:
        if trs.start < cf.payment_date <= curve.reference_date:
            income += cf.amount
    income_scaled = income / bond.face_value * trs.notional

    # Price return (per 100 face -> scale to notional)
    price_return = (current_dirty - initial_dirty) / 100.0 * trs.notional

    # Funding leg
    fwd_rate = -math.log(D / curve.df(trs.start)) / T
    r_f = fwd_rate + trs.funding.spread
    yf = year_fraction(trs.start, curve.reference_date, trs.funding.day_count)
    funding_leg = trs.notional * r_f * yf

    # Effective haircut: use schedule if available
    effective_haircut = trs.haircut
    if trs.haircut_schedule:
        for d, h in sorted(trs.haircut_schedule):
            if d <= curve.reference_date:
                effective_haircut = h

    # Blended repo rate: r_bar_s = (1-h)rs + h rN (Lou Eq 19)
    # Use fwd_rate as proxy for rN (bank's unsecured ~ OIS + spread)
    r_s = fwd_rate + trs.repo_spread
    r_N = fwd_rate + 0.02  # unsecured funding proxy
    rs_bar = blended_repo_rate(r_s, r_N, effective_haircut)
    rs_bar_spread = rs_bar - fwd_rate

    # FVA using blended rate (Lou Eq 25-26)
    repo_factor = repo_financing_factor(rs_bar_spread, T)
    fva = (repo_factor - 1) * current_dirty / 100.0 * trs.notional

    total_return = price_return + income_scaled
    value = total_return - funding_leg - fva

    # Cross-currency adjustment: convert asset-currency PV to funding currency
    if trs.xccy is not None:
        fx = trs.xccy.fx_rate
        # Total return is in asset currency -> convert
        total_return_funding = total_return / fx
        # Funding leg already in funding currency
        # FX haircut adds to effective haircut
        fx_haircut_cost = trs.notional * trs.xccy.fx_haircut * T * D
        value = total_return_funding - funding_leg - fva - fx_haircut_cost

    return TRSResult(
        value=value, total_return_leg=total_return,
        funding_leg=funding_leg, price_return=price_return,
        income_return=income_scaled, fva=fva, repo_factor=repo_factor,
    )


def price_loan_trs(trs, curve, projection_curve) -> TRSResult:
    """Loan TRS: adapts bond TRS framework to TermLoan cashflows.

    Supports prepayment-adjusted cashflows (CPR/PSA), LSTA settlement,
    credit-adjusted pricing, and market price override.

    Enhancements over basic version:
    - If survival_curve provided: credit-adjusts the total return leg
    - If initial_price provided: uses dealer mark instead of model price
    - Settlement cost uses compound interest (not linear)
    """
    from pricebook.fixed_income.bond_forward import repo_financing_factor

    loan = trs.underlying
    proj = projection_curve or curve
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    if T < 1e-5:
        raise ValueError(f"Maturity too short: {T:.6f} years")

    # Current price: market override or model
    current_price = loan.dirty_price(curve, proj)
    initial_price = trs.initial_price if trs.initial_price is not None else 100.0

    # Get cashflows -- prepayment-adjusted if model specified
    if trs.prepay_model is not None:
        from pricebook.credit.exotic_loan import prepay_adjusted_loan, psa_cpr
        if isinstance(trs.prepay_model, (int, float)):
            cpr = float(trs.prepay_model)
        elif isinstance(trs.prepay_model, tuple) and trs.prepay_model[0] == "PSA":
            speed = trs.prepay_model[1] if len(trs.prepay_model) > 1 else 1.0
            cpr = psa_cpr(12, speed)
        elif trs.prepay_model == "PSA":
            cpr = psa_cpr(12, 1.0)
        else:
            cpr = 0.0
        flows = prepay_adjusted_loan(loan, cpr, proj)
    else:
        flows = loan.cashflows(proj)

    # Interest income received (between start and valuation)
    income = 0.0
    for d, interest, _ in flows:
        if trs.start < d <= curve.reference_date:
            income += interest

    # Credit adjustment: if survival curve provided, weight income by survival
    if trs.survival_curve is not None:
        credit_adj_income = 0.0
        for d, interest, _ in flows:
            if trs.start < d <= curve.reference_date:
                surv = trs.survival_curve.survival(d)
                credit_adj_income += interest * surv
        income = credit_adj_income

    income_scaled = income / loan.notional * trs.notional

    # Price return
    price_return = (current_price - initial_price) / 100.0 * trs.notional

    # LSTA settlement adjustment
    settlement_shift = 0
    if trs.settlement_terms is not None:
        settlement_shift = trs.settlement_terms.settlement_days

    # Funding leg (adjusted for settlement delay)
    yf = year_fraction(trs.start, curve.reference_date, trs.funding.day_count)
    fwd_rate = proj.forward_rate(trs.start, trs.end) if T > 0 else 0.0
    r_f = fwd_rate + trs.funding.spread
    funding_leg = trs.notional * r_f * yf

    # Settlement delay cost: compound interest over settlement period
    if settlement_shift > 0:
        settle_cost = trs.notional * (math.pow(1 + r_f, settlement_shift / 365.0) - 1)
        funding_leg += settle_cost

    # FVA
    repo_factor = repo_financing_factor(trs.repo_spread, T)
    fva = (repo_factor - 1) * current_price / 100.0 * trs.notional

    total_return = price_return + income_scaled
    value = total_return - funding_leg - fva

    return TRSResult(
        value=value, total_return_leg=total_return,
        funding_leg=funding_leg, price_return=price_return,
        income_return=income_scaled, fva=fva, repo_factor=repo_factor,
    )


def price_cln_trs(trs, curve, projection_curve) -> TRSResult:
    """CLN TRS: total return on a credit-linked note.

    The TR receiver gets CLN price changes + coupon income.
    The TR payer finances at floating + spread.
    """
    from pricebook.fixed_income.bond_forward import repo_financing_factor

    cln = trs.underlying
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    if T < 1e-5:
        raise ValueError(f"Maturity too short: {T:.6f} years")

    # CLN pricing requires a survival curve -- use TRS-level, then CLN-level, then error
    survival_curve = trs.survival_curve
    if survival_curve is None and hasattr(cln, '_survival_curve'):
        survival_curve = cln._survival_curve
    if survival_curve is None:
        from pricebook.core.survival_curve import SurvivalCurve
        import warnings
        warnings.warn(
            "No survival curve provided for CLN TRS — using flat hazard=0.02 fallback. "
            "Pass survival_curve to TotalReturnSwap for accurate pricing.",
            stacklevel=2,
        )
        survival_curve = SurvivalCurve.flat(curve.reference_date, 0.02)

    current_price = cln.price_per_100(curve, survival_curve)
    initial_price = trs.initial_price if trs.initial_price is not None else 100.0

    # Coupon income between start and valuation date
    income = 0.0
    for i in range(1, len(cln.schedule)):
        pay_date = cln.schedule[i]
        if trs.start < pay_date <= curve.reference_date:
            yf = year_fraction(cln.schedule[i-1], pay_date, cln.day_count)
            income += cln.notional * cln.coupon_rate * yf
    income_scaled = income / cln.notional * trs.notional

    # Price return
    price_return = (current_price - initial_price) / 100.0 * trs.notional

    # Funding leg
    D = curve.df(trs.end)
    fwd_rate = -math.log(D / curve.df(trs.start)) / T
    r_f = fwd_rate + trs.funding.spread
    yf = year_fraction(trs.start, curve.reference_date, trs.funding.day_count)
    funding_leg = trs.notional * r_f * yf

    # FVA
    repo_factor = repo_financing_factor(trs.repo_spread, T)
    fva = (repo_factor - 1) * current_price / 100.0 * trs.notional

    total_return = price_return + income_scaled
    value = total_return - funding_leg - fva

    return TRSResult(
        value=value, total_return_leg=total_return,
        funding_leg=funding_leg, price_return=price_return,
        income_return=income_scaled, fva=fva, repo_factor=repo_factor,
    )


def price_commodity_trs(trs, curve, projection_curve) -> TRSResult:
    """Commodity TRS: total return on a commodity forward.

    Forward = spot x exp((r - convenience_yield + storage_cost) x T).
    Performance = (forward / initial - 1) x notional.
    Funding = notional x (r + spread) x T x df.
    """
    from pricebook.fixed_income.bond_forward import repo_financing_factor

    comm = trs.underlying  # CommodityUnderlying
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    if T < 1e-5:
        raise ValueError(f"Maturity too short: {T:.6f} years")

    D = curve.df(trs.end)
    fwd_rate = -math.log(D / curve.df(trs.start)) / T

    # Commodity forward: cost-of-carry model
    if comm.storage_model is not None:
        forward = comm.storage_model.implied_forward(comm.spot, fwd_rate, T)
    elif comm.seasonal is not None:
        seasonal_factor = comm.seasonal.factor(trs.end)
        carry = fwd_rate + comm.storage_cost
        forward = comm.spot * seasonal_factor * math.exp(carry * T)
    else:
        carry = fwd_rate - comm.convenience_yield + comm.storage_cost
        forward = comm.spot * math.exp(carry * T)

    initial = trs.initial_price if trs.initial_price is not None else comm.spot
    if initial <= 0:
        raise ValueError(f"Initial price must be positive, got {initial}")

    # Performance leg (in currency units)
    quantity = trs.notional / initial
    price_return = (forward - initial) * quantity

    # Funding leg
    r_f = fwd_rate + trs.funding.spread
    funding_leg = trs.notional * r_f * T * D

    # FVA (repo/financing)
    repo_factor = repo_financing_factor(trs.repo_spread, T)
    fva = (repo_factor - 1) * forward * quantity

    value = price_return - funding_leg - fva

    return TRSResult(
        value=value, total_return_leg=price_return,
        funding_leg=funding_leg, price_return=price_return,
        income_return=0.0, fva=fva, repo_factor=repo_factor,
    )


def price_fx_trs(trs, curve, projection_curve) -> TRSResult:
    """FX TRS: total return on an FX rate.

    Forward = spot x df_base / df_quote (covered interest rate parity).
    Performance = (current_spot / initial_spot - 1) x notional.
    Funding = notional x (r_quote + spread) x T x df.

    The projection_curve is the base-currency discount curve.
    The main curve is the quote-currency (funding) curve.
    """
    from pricebook.fixed_income.bond_forward import repo_financing_factor

    fx = trs.underlying  # FXUnderlying
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    if T < 1e-5:
        raise ValueError(f"Maturity too short: {T:.6f} years")

    # curve = quote currency (e.g. USD), projection_curve = base currency (e.g. EUR)
    quote_curve = curve
    base_curve = projection_curve or curve

    D_quote = quote_curve.df(trs.end)
    D_base = base_curve.df(trs.end)

    # FX forward via covered interest rate parity
    forward = fx.spot * D_base / D_quote

    # Quanto adjustment (Wystup 2006, Eq 1.52):
    #   forward *= exp(-rho x sigma_fx x sigma_asset x T)
    if fx.fx_vol > 0 and fx.fx_correlation != 0 and trs.sigma > 0:
        forward *= math.exp(-fx.fx_correlation * fx.fx_vol * trs.sigma * T)

    initial = trs.initial_price if trs.initial_price is not None else fx.spot
    if initial <= 0:
        raise ValueError(f"Initial spot must be positive, got {initial}")

    # Performance leg: notional in quote currency
    price_return = (forward / initial - 1) * trs.notional

    # Funding leg (in quote currency)
    fwd_rate = -math.log(D_quote / quote_curve.df(trs.start)) / T
    r_f = fwd_rate + trs.funding.spread
    funding_leg = trs.notional * r_f * T * D_quote

    # FVA
    repo_factor = repo_financing_factor(trs.repo_spread, T)
    fva = (repo_factor - 1) * trs.notional * (forward / initial)

    value = price_return - funding_leg - fva

    return TRSResult(
        value=value, total_return_leg=price_return,
        funding_leg=funding_leg, price_return=price_return,
        income_return=0.0, fva=fva, repo_factor=repo_factor,
    )


def price_multi_period(trs, curve, projection_curve) -> TRSResult:
    from pricebook.equity.trs_lou import trs_multi_period

    resets = sorted([trs.start] + trs.reset_dates + [trs.end])
    K = len(resets) - 1
    periods = [year_fraction(resets[i], resets[i + 1], DayCountConvention.ACT_365_FIXED)
                for i in range(K)]
    dfs = [curve.df(resets[i + 1]) for i in range(K)]

    # Forwards at each reset: F_j = spot x df(start) / df(reset_j)
    if trs._underlying_type == "equity":
        spot = float(trs.underlying)
        forwards = [spot * curve.df(trs.start) / curve.df(resets[i])
                    for i in range(K + 1)]
    elif trs._underlying_type == "bond":
        price = trs.underlying.dirty_price(curve)
        forwards = [price * curve.df(trs.start) / curve.df(resets[i])
                    for i in range(K + 1)]
    else:
        forwards = [100.0 * curve.df(trs.start) / curve.df(resets[i])
                    for i in range(K + 1)]

    if forwards[0] <= 0:
        raise ValueError(f"Forward price must be positive, got {forwards[0]}")

    # Funding rates: period-matched forward rate + spread
    funding_rates = []
    for i in range(K):
        if periods[i] < 1e-10:
            raise ValueError(f"Period {i} too short: {periods[i]}")
        fwd_r = -math.log(dfs[i] / curve.df(resets[i])) / periods[i]
        funding_rates.append(fwd_r + trs.funding.spread)

    # Notionals: MTM = F_j (currency), fixed = notional (currency)
    if trs.mtm_reset:
        funding_notionals = [f * trs.notional / forwards[0] for f in forwards[:K]]
    else:
        funding_notionals = [trs.notional] * K

    value = trs_multi_period(
        forwards=forwards, funding_rates=funding_rates,
        funding_notionals=funding_notionals, periods=periods,
        discount_factors=dfs, recovery=trs.recovery,
    )

    from pricebook.fixed_income.bond_forward import repo_financing_factor
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    repo_factor = repo_financing_factor(trs.repo_spread, T)

    return TRSResult(
        value=value, total_return_leg=0.0,
        funding_leg=0.0, price_return=0.0,
        income_return=0.0, fva=0.0, repo_factor=repo_factor,
    )


# ---------------------------------------------------------------------------
# Immutable copy helpers (for bump-and-reprice without state mutation)
# ---------------------------------------------------------------------------

def _trs_with_underlying(trs: TotalReturnSwap, new_underlying) -> TotalReturnSwap:
    """Return a copy with a different underlying (no mutation)."""
    return TotalReturnSwap(
        underlying=new_underlying, notional=trs.notional,
        start=trs.start, end=trs.end, funding=trs.funding,
        dividends=trs.dividends, initial_price=trs.initial_price,
        repo_spread=trs.repo_spread, haircut=trs.haircut,
        credit_curve_name=trs.credit_curve_name, recovery=trs.recovery,
        reset_dates=trs.reset_dates, mtm_reset=trs.mtm_reset,
        sigma=trs.sigma, prepay_model=trs.prepay_model,
        settlement_terms=trs.settlement_terms,
        survival_curve=trs.survival_curve, xccy=trs.xccy,
        haircut_schedule=trs.haircut_schedule,
    )


def _trs_with_repo_spread(trs: TotalReturnSwap, new_spread: float) -> TotalReturnSwap:
    """Return a copy with a different repo spread."""
    return TotalReturnSwap(
        underlying=trs.underlying, notional=trs.notional,
        start=trs.start, end=trs.end, funding=trs.funding,
        dividends=trs.dividends, initial_price=trs.initial_price,
        repo_spread=new_spread, haircut=trs.haircut,
        credit_curve_name=trs.credit_curve_name, recovery=trs.recovery,
        reset_dates=trs.reset_dates, mtm_reset=trs.mtm_reset,
        sigma=trs.sigma, prepay_model=trs.prepay_model,
        settlement_terms=trs.settlement_terms,
        survival_curve=trs.survival_curve, xccy=trs.xccy,
        haircut_schedule=trs.haircut_schedule,
    )


def _trs_with_funding_spread(trs: TotalReturnSwap, new_spread: float) -> TotalReturnSwap:
    """Return a copy with a different funding spread."""
    new_funding = FundingLegSpec(
        spread=new_spread,
        **{k: v for k, v in trs.funding.__dict__.items() if k != "spread"}
    )
    return TotalReturnSwap(
        underlying=trs.underlying, notional=trs.notional,
        start=trs.start, end=trs.end, funding=new_funding,
        dividends=trs.dividends, initial_price=trs.initial_price,
        repo_spread=trs.repo_spread, haircut=trs.haircut,
        credit_curve_name=trs.credit_curve_name, recovery=trs.recovery,
        reset_dates=trs.reset_dates, mtm_reset=trs.mtm_reset,
        sigma=trs.sigma, prepay_model=trs.prepay_model,
        settlement_terms=trs.settlement_terms,
        survival_curve=trs.survival_curve, xccy=trs.xccy,
        haircut_schedule=trs.haircut_schedule,
    )
