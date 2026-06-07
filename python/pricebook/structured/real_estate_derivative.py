"""Real estate / property index derivatives and valuation tools.

* :func:`property_total_return_swap` — TRS on a property index.
* :func:`property_index_forward`     — Forward price adjusted for illiquidity.
* :func:`property_option`            — Black-76 option on a property index.
* :func:`reit_nav_model`             — REIT net asset value per share.
* :func:`housing_affordability`      — Affordability metrics for residential market.

References:
    Geltner, Miller, Clayton & Eichholtz (2013). *Commercial Real Estate
        Analysis and Investments*, 3rd ed.  OnCourse Learning.
    Fabozzi, Shiller & Tunaru (2010). Property Derivatives for Managing
        European Real-Estate Risk.  European Financial Management 16(1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(r: float, t: float) -> float:
    return math.exp(-r * t)


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Property Total Return Swap
# ---------------------------------------------------------------------------

@dataclass
class PropertySwapResult:
    """Output of :func:`property_total_return_swap`."""
    pv: float
    fixed_leg_pv: float
    floating_leg_pv: float
    break_even_appreciation: float
    notional: float

    def to_dict(self) -> dict:
        return vars(self)


def property_total_return_swap(
    notional: float,
    fixed_rate: float,
    expected_appreciation: float,
    rental_yield: float,
    risk_free_rate: float,
    T: float,
    illiquidity_premium: float = 0.02,
) -> PropertySwapResult:
    """Total Return Swap on a property index.

    The fixed-rate receiver pays ``fixed_rate`` per annum in exchange for the
    total return of the property index (capital appreciation + rental yield).
    An illiquidity premium is added to the floating leg as compensation for
    the reduced tradability of real-estate assets.

    PV from the perspective of the fixed-rate receiver:

    .. math::

        \\text{PV} = N \\cdot [\\text{floating\\_leg\\_pv} - \\text{fixed\\_leg\\_pv}]

    Args:
        notional:              Swap notional.
        fixed_rate:            Annual fixed rate paid by the fixed-rate receiver.
        expected_appreciation: Annual capital appreciation of the property index.
        rental_yield:          Annual rental (income) yield of the index.
        risk_free_rate:        Risk-free discount rate.
        T:                     Swap maturity in years.
        illiquidity_premium:   Extra yield demanded for property illiquidity (default 2%).

    Returns:
        :class:`PropertySwapResult`.
    """
    # Discrete annual cash flows discounted continuously
    n_years = max(int(round(T)), 1)
    fixed_leg_pv = 0.0
    floating_leg_pv = 0.0

    total_return_rate = expected_appreciation + rental_yield + illiquidity_premium

    for t in range(1, n_years + 1):
        df_t = _df(risk_free_rate, t)
        fixed_leg_pv += fixed_rate * notional * df_t
        floating_leg_pv += total_return_rate * notional * df_t

    pv = floating_leg_pv - fixed_leg_pv  # positive = fixed-rate receiver wins

    # Break-even: appreciation at which PV = 0
    break_even_appreciation = fixed_rate - rental_yield - illiquidity_premium

    return PropertySwapResult(
        pv=pv,
        fixed_leg_pv=fixed_leg_pv,
        floating_leg_pv=floating_leg_pv,
        break_even_appreciation=break_even_appreciation,
        notional=notional,
    )


# ---------------------------------------------------------------------------
# Property Index Forward
# ---------------------------------------------------------------------------

def property_index_forward(
    spot_index: float,
    risk_free_rate: float,
    rental_yield: float,
    T: float,
    illiquidity_premium: float = 0.01,
) -> float:
    """Forward price on a property index.

    Analogous to an equity forward but with a rental yield acting as a
    continuous dividend and an additional illiquidity premium that raises the
    cost-of-carry above the risk-free rate:

    .. math::

        F = S_0 \\cdot e^{(r + \\lambda - y) T}

    where :math:`\\lambda` is the illiquidity premium and :math:`y` is the
    rental yield.

    Args:
        spot_index:          Current property index level.
        risk_free_rate:      Continuously-compounded risk-free rate.
        rental_yield:        Continuous rental (income) yield.
        T:                   Forward maturity in years.
        illiquidity_premium: Illiquidity cost-of-carry adjustment (default 1%).

    Returns:
        Forward price of the property index.
    """
    carry = risk_free_rate + illiquidity_premium - rental_yield
    return spot_index * math.exp(carry * T)


# ---------------------------------------------------------------------------
# Property Option
# ---------------------------------------------------------------------------

@dataclass
class PropertyOptionResult:
    """Output of :func:`property_option`."""
    price: float
    delta: float
    gamma: float
    vega: float

    def to_dict(self) -> dict:
        return vars(self)


def property_option(
    spot_index: float,
    strike: float,
    vol: float,
    T: float,
    risk_free_rate: float,
    rental_yield: float,
    option_type: str = "call",
    illiquidity_premium: float = 0.01,
) -> PropertyOptionResult:
    """Black-76 option on a property index.

    The forward is computed including the illiquidity premium; Black-76
    then prices the option on that forward.

    Args:
        spot_index:          Current property index level.
        strike:              Option strike.
        vol:                 Implied volatility of the property index.
        T:                   Time to maturity in years.
        risk_free_rate:      Risk-free rate for discounting.
        rental_yield:        Continuous rental yield (reduces carry).
        option_type:         ``"call"`` or ``"put"``.
        illiquidity_premium: Illiquidity adjustment applied to the forward (default 1%).

    Returns:
        :class:`PropertyOptionResult` with price and first/second-order Greeks.
    """
    F = property_index_forward(spot_index, risk_free_rate, rental_yield, T, illiquidity_premium)
    df = _df(risk_free_rate, T)
    sqrt_T = math.sqrt(T)
    sigma_sqrt_T = vol * sqrt_T

    d1 = (math.log(F / strike) + 0.5 * vol ** 2 * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    phi_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)  # standard normal PDF

    if option_type.lower() == "call":
        price = df * (F * _norm_cdf(d1) - strike * _norm_cdf(d2))
        delta = df * _norm_cdf(d1)
    else:
        price = df * (strike * _norm_cdf(-d2) - F * _norm_cdf(-d1))
        delta = -df * _norm_cdf(-d1)

    gamma = df * phi_d1 / (F * sigma_sqrt_T)
    vega = F * df * phi_d1 * sqrt_T

    return PropertyOptionResult(price=price, delta=delta, gamma=gamma, vega=vega)


# ---------------------------------------------------------------------------
# REIT NAV Model
# ---------------------------------------------------------------------------

def reit_nav_model(
    properties_value: float,
    debt: float,
    cash: float,
    shares: float,
    cap_rate: float,
    discount_rate: float,
) -> float:
    """REIT Net Asset Value (NAV) per share.

    Standard NAV: mark portfolio to market using the investor's discount
    rate vs the portfolio cap rate, then add cash and subtract debt:

    .. math::

        \\text{NAV/share} = \\frac{\\text{NOI} / r + \\text{cash} - \\text{debt}}{N}

    where NOI = properties_value × cap_rate, and r is the discount rate.
    When cap_rate == discount_rate, adjusted value == properties_value.

    Args:
        properties_value: Gross appraised value of the property portfolio.
        debt:             Total debt outstanding.
        cash:             Cash and liquid equivalents held.
        shares:           Shares outstanding.
        cap_rate:         Portfolio net operating income / property value.
        discount_rate:    Investor required return (WACC or equity discount rate).

    Returns:
        NAV per share.
    """
    if shares <= 0:
        raise ValueError("shares must be positive")
    if discount_rate <= 0:
        raise ValueError("discount_rate must be positive")
    # Mark-to-market: NOI / discount_rate (Gordon growth with g=0)
    noi = properties_value * cap_rate
    adjusted_value = noi / discount_rate
    nav = adjusted_value + cash - debt
    return nav / shares


# ---------------------------------------------------------------------------
# Housing Affordability
# ---------------------------------------------------------------------------

@dataclass
class AffordabilityResult:
    """Output of :func:`housing_affordability`."""
    payment_to_income: float
    max_affordable_price: float
    price_to_income: float

    def to_dict(self) -> dict:
        return vars(self)


def housing_affordability(
    median_price: float,
    median_income: float,
    mortgage_rate: float,
    ltv: float = 0.8,
    term_years: int = 30,
) -> AffordabilityResult:
    """Residential housing affordability metrics.

    Computes three standard measures used by central banks and real-estate
    analysts to assess whether housing is affordable for median earners.

    Args:
        median_price:   Median home price in the market.
        median_income:  Gross annual household income (median).
        mortgage_rate:  Annual mortgage interest rate (nominal, monthly compounding).
        ltv:            Loan-to-value ratio (default 80% — 20% down payment).
        term_years:     Mortgage amortisation period in years (default 30).

    Returns:
        :class:`AffordabilityResult`.
    """
    n_months = term_years * 12
    monthly_rate = mortgage_rate / 12.0
    loan_amount = median_price * ltv
    monthly_income = median_income / 12.0

    # Standard annuity payment formula
    if monthly_rate > 1e-10:
        monthly_payment = (
            loan_amount * monthly_rate / (1.0 - (1.0 + monthly_rate) ** (-n_months))
        )
    else:
        monthly_payment = loan_amount / n_months

    payment_to_income = monthly_payment / monthly_income if monthly_income > 0.0 else float("inf")
    price_to_income = median_price / median_income if median_income > 0.0 else float("inf")

    # Maximum price affordable if payment_to_income ratio <= 28% (standard threshold)
    max_payment = 0.28 * monthly_income
    if monthly_rate > 1e-10:
        max_loan = max_payment * (1.0 - (1.0 + monthly_rate) ** (-n_months)) / monthly_rate
    else:
        max_loan = max_payment * n_months
    max_affordable_price = max_loan / ltv if ltv > 0.0 else 0.0

    return AffordabilityResult(
        payment_to_income=payment_to_income,
        max_affordable_price=max_affordable_price,
        price_to_income=price_to_income,
    )
