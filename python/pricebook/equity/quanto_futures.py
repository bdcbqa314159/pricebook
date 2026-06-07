"""Quanto futures pricing and analytics (e.g., CME Nikkei 225 in USD).

A quanto futures contract pays an index denominated in a foreign currency but
settles in domestic currency at a fixed exchange rate.  The quanto drift
correction removes the FX risk from the payoff, embedding a correlation
adjustment in the forward price.

* :class:`QuantoFuturesResult` — fair value, quanto adjustment, domestic
  equivalent and correlation impact.
* :func:`quanto_futures_price` — quanto futures fair value using the
  quanto drift correction.
* :func:`quanto_forward` — domestic investor's quanto-adjusted forward price.
* :func:`implied_correlation` — back out correlation from observed quanto price.
* :func:`quanto_basis` — basis between quanto and FX-converted vanilla futures.
* :class:`CompoQuantoResult` — composite vs. quanto comparison result.
* :func:`compo_vs_quanto` — compare composite (FX-exposed) vs. quanto forwards.

References:
    Hull, J.C., *Options, Futures and Other Derivatives*, Ch. 33, 10th ed.
    Wystup, U., *FX Options and Structured Products*, Wiley, 2006.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class QuantoFuturesResult:
    """Result of a quanto futures pricing calculation.

    Attributes:
        fair_value: Quanto futures fair value in domestic currency units.
        quanto_adjustment: Multiplicative quanto drift correction factor,
            exp(-rho * sigma_S * sigma_FX * T).
        domestic_equivalent: Vanilla (non-quanto) futures price converted at
            spot FX (for comparison purposes).
        fx_correlation_impact: Absolute difference between quanto futures and
            the unadjusted domestic-equivalent futures price.
    """

    fair_value: float
    quanto_adjustment: float
    domestic_equivalent: float
    fx_correlation_impact: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CompoQuantoResult:
    """Comparison of composite vs. quanto forward prices.

    Attributes:
        compo_forward: Forward price for a composite (FX-unhedged) position;
            the investor is exposed to both index and FX movements.
        quanto_forward: Forward price under the quanto measure (FX risk
            eliminated via correlation adjustment).
        difference: ``quanto_forward - compo_forward``.
        break_even_correlation: Correlation at which the two forwards are equal
            (i.e., the quanto adjustment exactly cancels the FX forward).
    """

    compo_forward: float
    quanto_forward: float
    difference: float
    break_even_correlation: float

    def to_dict(self) -> dict:
        return vars(self)


# ---------------------------------------------------------------------------
# Core pricing functions
# ---------------------------------------------------------------------------

def quanto_futures_price(
    spot_index: float,
    rate_domestic: float,
    rate_foreign: float,
    div_yield: float,
    fx_vol: float,
    index_vol: float,
    correlation: float,
    T: float,
) -> QuantoFuturesResult:
    """Fair value of a quanto futures contract.

    Under the domestic risk-neutral measure, the quanto drift correction
    replaces the foreign risk-free rate with an adjusted rate:

    .. math::

        F_Q = S \\times \\exp\\bigl((r_d - q - \\rho\\,\\sigma_S\\,\\sigma_{FX})\\,T\\bigr)

    The term :math:`-\\rho\\,\\sigma_S\\,\\sigma_{FX}` is the *quanto drift
    adjustment* that arises from the covariance between the index and the
    FX rate under the domestic measure.

    Args:
        spot_index: Current index spot level (in foreign currency units).
        rate_domestic: Continuously compounded domestic risk-free rate.
        rate_foreign: Continuously compounded foreign risk-free rate.
        div_yield: Continuous dividend yield of the index.
        fx_vol: Annualised FX volatility (foreign/domestic exchange rate).
        index_vol: Annualised index volatility (in foreign currency terms).
        correlation: Correlation between index returns and FX returns
            (positive means index up when foreign currency strengthens).
        T: Time to delivery in years.

    Returns:
        :class:`QuantoFuturesResult` with the quanto fair value and components.
    """
    quanto_adj_exponent = -correlation * index_vol * fx_vol * T
    quanto_adjustment = math.exp(quanto_adj_exponent)

    # F_Q = S * exp((r_d - q - rho*sigma_S*sigma_FX) * T)
    fair_value = spot_index * math.exp(
        (rate_domestic - div_yield - correlation * index_vol * fx_vol) * T
    )

    # Vanilla index futures at foreign rate, converted at par (FX = 1) for
    # comparison: F_vanilla = S * exp((r_d - q) * T)
    domestic_equivalent = spot_index * math.exp((rate_domestic - div_yield) * T)

    fx_correlation_impact = fair_value - domestic_equivalent

    return QuantoFuturesResult(
        fair_value=fair_value,
        quanto_adjustment=quanto_adjustment,
        domestic_equivalent=domestic_equivalent,
        fx_correlation_impact=fx_correlation_impact,
    )


def quanto_forward(
    spot_index: float,
    fx_spot: float,
    rate_domestic: float,
    rate_foreign: float,
    div_yield: float,
    fx_vol: float,
    index_vol: float,
    correlation: float,
    T: float,
) -> float:
    """Quanto-adjusted forward price for a domestic investor.

    Computes the expected payoff under the domestic risk-neutral (T-forward)
    measure.  The result is denominated in domestic currency units.

    .. math::

        E^d[S_T] = S_0 \\times X_0
            \\times \\exp\\bigl((r_d - q - \\rho\\,\\sigma_S\\,\\sigma_{FX})\\,T\\bigr)

    where :math:`X_0` is the spot FX rate (domestic per foreign).

    Args:
        spot_index: Current index spot level in foreign currency units.
        fx_spot: Spot FX rate (domestic per foreign unit).
        rate_domestic: Continuously compounded domestic risk-free rate.
        rate_foreign: Continuously compounded foreign risk-free rate.
        div_yield: Continuous dividend yield of the index.
        fx_vol: Annualised FX volatility.
        index_vol: Annualised index volatility (in foreign currency terms).
        correlation: Correlation between index returns and FX returns.
        T: Time to delivery in years.

    Returns:
        Expected index value in domestic currency units at time *T*.
    """
    quanto_drift = rate_domestic - div_yield - correlation * index_vol * fx_vol
    return spot_index * fx_spot * math.exp(quanto_drift * T)


def implied_correlation(
    quanto_price: float,
    spot_index: float,
    rate_domestic: float,
    rate_foreign: float,
    div_yield: float,
    fx_vol: float,
    index_vol: float,
    T: float,
) -> float:
    """Back out the implied correlation from an observed quanto futures price.

    Inverts :func:`quanto_futures_price`:

    .. math::

        \\rho = \\frac{r_d - q - \\ln(F_Q / S) / T}{\\sigma_S\\,\\sigma_{FX}}

    Args:
        quanto_price: Observed quanto futures price.
        spot_index: Current index spot level.
        rate_domestic: Continuously compounded domestic risk-free rate.
        rate_foreign: Continuously compounded foreign risk-free rate (unused
            in the formula but kept for interface consistency).
        div_yield: Continuous dividend yield of the index.
        fx_vol: Annualised FX volatility.
        index_vol: Annualised index volatility.
        T: Time to delivery in years.

    Returns:
        Implied correlation (dimensionless, in [-1, 1]).

    Raises:
        ValueError: If *fx_vol*, *index_vol* or *T* is zero.
    """
    if fx_vol == 0.0 or index_vol == 0.0:
        raise ValueError("fx_vol and index_vol must be non-zero.")
    if T <= 0.0:
        raise ValueError("T must be positive.")

    log_moneyness = math.log(quanto_price / spot_index) / T
    rho = (rate_domestic - div_yield - log_moneyness) / (index_vol * fx_vol)
    return rho


def quanto_basis(
    quanto_price: float,
    vanilla_futures_price: float,
    fx_spot: float,
) -> float:
    """Basis between the quanto futures and the FX-converted vanilla futures.

    .. math::

        \\text{basis} = F_Q - F_{\\text{vanilla}} \\times X_0

    A positive basis means the quanto contract trades above the FX-hedged
    vanilla equivalent, implying a negative correlation (index falls when
    foreign currency weakens against domestic).

    Args:
        quanto_price: Quanto futures price in domestic currency units.
        vanilla_futures_price: Vanilla (foreign-currency) futures price.
        fx_spot: Spot FX rate (domestic per foreign unit).

    Returns:
        Basis in domestic currency units.
    """
    return quanto_price - vanilla_futures_price * fx_spot


def compo_vs_quanto(
    spot_index: float,
    fx_spot: float,
    rate_domestic: float,
    rate_foreign: float,
    div_yield: float,
    fx_vol: float,
    index_vol: float,
    correlation: float,
    T: float,
) -> CompoQuantoResult:
    """Compare composite (FX-exposed) vs. quanto (FX-hedged) forwards.

    A *composite* forward keeps full FX exposure; its domestic-currency
    forward price is:

    .. math::

        F_{\\text{compo}} = S_0 \\times X_0
            \\times \\exp\\bigl((r_d - r_f - q)\\,T\\bigr)

    A *quanto* forward removes FX risk via a correlation drift adjustment;
    see :func:`quanto_forward`.

    The *break-even correlation* is the correlation at which both forwards
    coincide:

    .. math::

        \\rho^* = \\frac{r_f}{\\sigma_S\\,\\sigma_{FX}}

    Args:
        spot_index: Current index spot level in foreign currency units.
        fx_spot: Spot FX rate (domestic per foreign unit).
        rate_domestic: Continuously compounded domestic risk-free rate.
        rate_foreign: Continuously compounded foreign risk-free rate.
        div_yield: Continuous dividend yield of the index.
        fx_vol: Annualised FX volatility.
        index_vol: Annualised index volatility.
        correlation: Correlation between index returns and FX returns.
        T: Time to delivery in years.

    Returns:
        :class:`CompoQuantoResult` with the two forwards and analytics.
    """
    # Composite forward: investor converts payoff at prevailing FX rate.
    # Under domestic risk-neutral measure:
    # F_compo = S * X0 * exp((r_d - r_f - q) * T)  [covered interest parity]
    compo_fwd = spot_index * fx_spot * math.exp(
        (rate_domestic - rate_foreign - div_yield) * T
    )

    quanto_fwd = quanto_forward(
        spot_index, fx_spot, rate_domestic, rate_foreign,
        div_yield, fx_vol, index_vol, correlation, T,
    )

    difference = quanto_fwd - compo_fwd

    # Break-even: quanto_drift == (r_d - r_f - q)
    # => -rho* * sigma_S * sigma_FX = -r_f
    # => rho* = r_f / (sigma_S * sigma_FX)
    if index_vol > 0.0 and fx_vol > 0.0:
        break_even_corr = rate_foreign / (index_vol * fx_vol)
    else:
        break_even_corr = float("nan")

    return CompoQuantoResult(
        compo_forward=compo_fwd,
        quanto_forward=quanto_fwd,
        difference=difference,
        break_even_correlation=break_even_corr,
    )
