"""Index CDS swaption: options on CDS index forward spread.

Dedicated product for CDX/iTraxx index swaptions with proper
Jensen's inequality handling for the forward index spread.

* :class:`IndexCDSSwaptionResult` — pricing result with Greeks.
* :func:`index_cds_swaption_black` — Black-76 on forward index spread.
* :func:`index_cds_swaption_bachelier` — Bachelier (normal) model.
* :func:`index_forward_spread` — annuity-weighted forward index spread.
* :func:`index_swaption_greeks` — delta, gamma, vega, theta.

References:
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*,
    Ch. 8 (Index options), 2008.
    Pedersen, *Valuation of Portfolio Credit Default Swaptions*,
    Lehman Brothers QR, 2003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.models.black76 import _norm_cdf


@dataclass
class IndexForwardResult:
    """Forward index spread and components."""
    forward_spread: float
    index_annuity: float
    portfolio_survival: float
    constituent_forwards: list[float]
    constituent_annuities: list[float]

    def to_dict(self) -> dict:
        return {
            "forward_spread": self.forward_spread,
            "index_annuity": self.index_annuity,
            "portfolio_survival": self.portfolio_survival,
            "n_constituents": len(self.constituent_forwards),
        }


@dataclass
class IndexCDSSwaptionResult:
    """Index CDS swaption pricing result."""
    premium: float
    forward_spread: float
    strike_spread: float
    index_annuity: float
    spread_vol: float
    portfolio_survival: float
    option_type: str
    model: str  # "black76" or "bachelier"
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0

    def to_dict(self) -> dict:
        return vars(self)


# ---- Forward index spread ----

def index_forward_spread(
    discount_curve: DiscountCurve,
    survival_curves: list[SurvivalCurve],
    expiry_date: date,
    maturity_date: date,
    recovery: float = 0.4,
) -> IndexForwardResult:
    """Annuity-weighted forward index spread (Jensen's inequality).

    F_index = Σ(F_i × A_i) / Σ(A_i)

    NOT the simple average of constituent spreads. This respects
    Jensen's inequality: E[f(X)] ≠ f(E[X]) for convex f.

    Args:
        discount_curve: risk-free discount curve.
        survival_curves: one survival curve per constituent.
        expiry_date: swaption expiry / CDS start date.
        maturity_date: CDS maturity date.
        recovery: recovery rate.
    """
    from pricebook.credit.cds import forward_cds_par_spread

    fwd_spreads = []
    fwd_annuities = []
    fwd_survivals = []

    for sc in survival_curves:
        fwd = forward_cds_par_spread(
            discount_curve, sc, expiry_date, maturity_date, recovery,
        )
        fwd_spreads.append(fwd.forward_spread)
        fwd_annuities.append(fwd.risky_annuity)
        fwd_survivals.append(fwd.survival_to_start)

    total_ann = sum(fwd_annuities)
    if total_ann <= 0:
        return IndexForwardResult(0.0, 0.0, 0.0, fwd_spreads, fwd_annuities)

    # Annuity-weighted forward: F_idx = Σ(F_i × A_i) / Σ(A_i)
    idx_fwd = sum(f * a for f, a in zip(fwd_spreads, fwd_annuities)) / total_ann
    avg_ann = total_ann / len(fwd_annuities)
    avg_surv = sum(fwd_survivals) / len(fwd_survivals)

    return IndexForwardResult(
        forward_spread=idx_fwd,
        index_annuity=avg_ann,
        portfolio_survival=avg_surv,
        constituent_forwards=fwd_spreads,
        constituent_annuities=fwd_annuities,
    )


# ---- Black-76 index swaption ----

def index_cds_swaption_black(
    forward_spread: float,
    strike_spread: float,
    spread_vol: float,
    expiry_years: float,
    index_annuity: float,
    portfolio_survival: float,
    notional: float = 10_000_000.0,
    option_type: str = "payer",
) -> IndexCDSSwaptionResult:
    """Index CDS swaption via Black-76 on forward index spread.

    Payer swaption (right to buy index protection):
        V = N × Q(0,T) × A × [F·Φ(d₁) − K·Φ(d₂)]

    Receiver swaption (right to sell index protection):
        V = N × Q(0,T) × A × [K·Φ(−d₂) − F·Φ(−d₁)]

    where F = forward index spread, K = strike, A = index annuity,
    Q(0,T) = portfolio survival to expiry.

    Args:
        forward_spread: annuity-weighted forward index spread.
        strike_spread: strike spread.
        spread_vol: lognormal spread vol.
        expiry_years: time to expiry in years.
        index_annuity: average constituent risky annuity.
        portfolio_survival: average survival probability to expiry.
        notional: index notional.
        option_type: "payer" or "receiver".
    """
    if expiry_years <= 0 or spread_vol <= 0 or forward_spread <= 0 or strike_spread <= 0:
        intrinsic = _intrinsic(forward_spread, strike_spread, option_type)
        premium = max(intrinsic, 0) * notional * portfolio_survival * index_annuity
        return IndexCDSSwaptionResult(
            premium=premium, forward_spread=forward_spread,
            strike_spread=strike_spread, index_annuity=index_annuity,
            spread_vol=spread_vol, portfolio_survival=portfolio_survival,
            option_type=option_type, model="black76",
        )

    sqrt_t = math.sqrt(expiry_years)
    d1 = (math.log(forward_spread / strike_spread) + 0.5 * spread_vol**2 * expiry_years) / (spread_vol * sqrt_t)
    d2 = d1 - spread_vol * sqrt_t

    if option_type == "payer":
        premium = forward_spread * _norm_cdf(d1) - strike_spread * _norm_cdf(d2)
    else:
        premium = strike_spread * _norm_cdf(-d2) - forward_spread * _norm_cdf(-d1)

    premium *= notional * portfolio_survival * index_annuity

    return IndexCDSSwaptionResult(
        premium=premium, forward_spread=forward_spread,
        strike_spread=strike_spread, index_annuity=index_annuity,
        spread_vol=spread_vol, portfolio_survival=portfolio_survival,
        option_type=option_type, model="black76",
    )


# ---- Bachelier (normal) index swaption ----

def index_cds_swaption_bachelier(
    forward_spread: float,
    strike_spread: float,
    normal_vol: float,
    expiry_years: float,
    index_annuity: float,
    portfolio_survival: float,
    notional: float = 10_000_000.0,
    option_type: str = "payer",
) -> IndexCDSSwaptionResult:
    """Index CDS swaption via Bachelier (normal) model.

    For tight spreads or near-zero environments where lognormal
    is inappropriate.

    V_payer = N × Q × A × [(F−K)Φ(d) + σ√T φ(d)]

    where d = (F − K) / (σ√T).

    Args:
        normal_vol: normal (absolute) spread vol.
    """
    if expiry_years <= 0 or normal_vol <= 0:
        intrinsic = _intrinsic(forward_spread, strike_spread, option_type)
        premium = max(intrinsic, 0) * notional * portfolio_survival * index_annuity
        return IndexCDSSwaptionResult(
            premium=premium, forward_spread=forward_spread,
            strike_spread=strike_spread, index_annuity=index_annuity,
            spread_vol=normal_vol, portfolio_survival=portfolio_survival,
            option_type=option_type, model="bachelier",
        )

    sqrt_t = math.sqrt(expiry_years)
    d = (forward_spread - strike_spread) / (normal_vol * sqrt_t)

    if option_type == "payer":
        premium = (forward_spread - strike_spread) * _norm_cdf(d) + normal_vol * sqrt_t * _norm_pdf(d)
    else:
        premium = (strike_spread - forward_spread) * _norm_cdf(-d) + normal_vol * sqrt_t * _norm_pdf(d)

    premium *= notional * portfolio_survival * index_annuity

    return IndexCDSSwaptionResult(
        premium=premium, forward_spread=forward_spread,
        strike_spread=strike_spread, index_annuity=index_annuity,
        spread_vol=normal_vol, portfolio_survival=portfolio_survival,
        option_type=option_type, model="bachelier",
    )


# ---- Greeks ----

def index_swaption_greeks(
    forward_spread: float,
    strike_spread: float,
    spread_vol: float,
    expiry_years: float,
    index_annuity: float,
    portfolio_survival: float,
    notional: float = 10_000_000.0,
    option_type: str = "payer",
    model: str = "black76",
) -> IndexCDSSwaptionResult:
    """Price + Greeks for index CDS swaption.

    Delta: ∂V/∂F (spread delta)
    Gamma: ∂²V/∂F²
    Vega: ∂V/∂σ (per 1% vol)
    Theta: ∂V/∂t (per day)
    """
    price_fn = index_cds_swaption_black if model == "black76" else index_cds_swaption_bachelier

    base = price_fn(forward_spread, strike_spread, spread_vol,
                    expiry_years, index_annuity, portfolio_survival,
                    notional, option_type)

    # Delta: bump forward spread
    ds = forward_spread * 0.001 if forward_spread > 0 else 1e-6
    up = price_fn(forward_spread + ds, strike_spread, spread_vol,
                  expiry_years, index_annuity, portfolio_survival,
                  notional, option_type).premium
    dn = price_fn(forward_spread - ds, strike_spread, spread_vol,
                  expiry_years, index_annuity, portfolio_survival,
                  notional, option_type).premium
    delta = (up - dn) / (2 * ds)

    # Gamma
    gamma = (up - 2 * base.premium + dn) / (ds ** 2)

    # Vega: per 1% vol move
    dv = 0.01
    vol_up = price_fn(forward_spread, strike_spread, spread_vol + dv,
                      expiry_years, index_annuity, portfolio_survival,
                      notional, option_type).premium
    vega = vol_up - base.premium

    # Theta: per day
    dt_day = 1.0 / 365.0
    if expiry_years > dt_day:
        theta_val = price_fn(forward_spread, strike_spread, spread_vol,
                             expiry_years - dt_day, index_annuity,
                             portfolio_survival, notional, option_type).premium
        theta = theta_val - base.premium
    else:
        theta = 0.0

    base.delta = delta
    base.gamma = gamma
    base.vega = vega
    base.theta = theta
    return base


# ---- Full pricing with curves ----

def price_index_cds_swaption(
    discount_curve: DiscountCurve,
    survival_curves: list[SurvivalCurve],
    expiry_date: date,
    maturity_date: date,
    strike_spread: float,
    spread_vol: float,
    notional: float = 10_000_000.0,
    option_type: str = "payer",
    recovery: float = 0.4,
    model: str = "black76",
    compute_greeks: bool = True,
) -> IndexCDSSwaptionResult:
    """Full index CDS swaption pricing from curves.

    Computes forward index spread with Jensen's inequality correction,
    then prices via Black-76 or Bachelier.

    Args:
        discount_curve: risk-free discount curve.
        survival_curves: one per constituent.
        expiry_date: swaption expiry.
        maturity_date: underlying CDS maturity.
        strike_spread: strike spread.
        spread_vol: lognormal or normal vol (per model).
        notional: index notional.
        option_type: "payer" or "receiver".
        recovery: recovery rate.
        model: "black76" or "bachelier".
        compute_greeks: if True, compute delta/gamma/vega/theta.
    """
    ref = discount_curve.reference_date
    dc = DayCountConvention.ACT_365_FIXED
    expiry_years = year_fraction(ref, expiry_date, dc)

    # Forward index spread
    fwd = index_forward_spread(
        discount_curve, survival_curves, expiry_date, maturity_date, recovery,
    )

    if compute_greeks:
        return index_swaption_greeks(
            fwd.forward_spread, strike_spread, spread_vol,
            expiry_years, fwd.index_annuity, fwd.portfolio_survival,
            notional, option_type, model,
        )

    price_fn = index_cds_swaption_black if model == "black76" else index_cds_swaption_bachelier
    return price_fn(
        fwd.forward_spread, strike_spread, spread_vol,
        expiry_years, fwd.index_annuity, fwd.portfolio_survival,
        notional, option_type,
    )


# ---- Helpers ----

def _intrinsic(fwd: float, strike: float, option_type: str) -> float:
    if option_type == "payer":
        return max(fwd - strike, 0)
    return max(strike - fwd, 0)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
