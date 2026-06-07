"""ETF pricing, creation/redemption arbitrage, and tracking analytics.

Exchange-traded funds (ETFs) are priced via the authorised participant (AP)
arbitrage mechanism: APs can create or redeem shares in-kind against the
underlying basket, keeping the market price tethered to NAV.

* :class:`ETFResult` — NAV, market price, premium/discount and tracking error.
* :func:`etf_nav` — net asset value per share.
* :func:`premium_discount` — market price premium or discount to NAV.
* :class:`ArbResult` — AP arbitrage direction and profitability.
* :func:`creation_redemption_arb` — AP arbitrage profit/loss analysis.
* :func:`tracking_error` — annualised tracking error (volatility of excess return).
* :func:`tracking_difference` — cumulative return gap (total underperformance).
* :func:`synthetic_etf_cost` — total cost of swap-based vs. physical ETF.
* :func:`leveraged_etf_decay` — expected value of leveraged ETF with vol drag.

References:
    Madhavan, A., *Exchange-Traded Funds and the New Dynamics of Investing*,
        Oxford University Press, 2016.
    Ben-David, I., Franzoni, F. & Moussawi, R., "Exchange-Traded Funds",
        *Annual Review of Financial Economics*, 9, 2017.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ETFResult:
    """Summary result for ETF pricing and tracking analytics.

    Attributes:
        nav: Net asset value per share in currency units.
        market_price: Observed market price per share.
        premium_discount: Premium (positive) or discount (negative) as
            percentage of NAV — same units as ``premium_discount()`` function.
        premium_discount_pct: Alias for premium_discount (percentage).
        tracking_error: Annualised tracking error as a decimal fraction
            (std of daily excess returns × sqrt(252)).  ``nan`` if not computed.
    """

    nav: float
    market_price: float
    premium_discount: float
    premium_discount_pct: float
    tracking_error: float = float("nan")

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class ArbResult:
    """Result of an AP creation/redemption arbitrage analysis.

    Attributes:
        direction: ``"create"`` if market premium exceeds round-trip costs,
            ``"redeem"`` if discount exceeds costs, or ``"none"`` if no
            profitable arbitrage exists.
        gross_profit: Gross profit per share before transaction costs.
        net_profit: Net profit per share after all fees and costs.
        is_profitable: ``True`` when ``net_profit > 0``.
    """

    direction: str
    gross_profit: float
    net_profit: float
    is_profitable: bool

    def to_dict(self) -> dict:
        return vars(self)


# ---------------------------------------------------------------------------
# NAV and premium/discount
# ---------------------------------------------------------------------------

def etf_nav(
    holdings: list[float],
    prices: list[float],
    shares_outstanding: float,
    cash: float = 0.0,
    liabilities: float = 0.0,
) -> float:
    """Net asset value (NAV) per share.

    .. math::

        \\text{NAV} = \\frac{\\sum_i h_i \\times p_i + C - L}{N}

    where :math:`h_i` are share holdings per creation unit, :math:`p_i` are
    current market prices, :math:`C` is cash (accrued income, cash component),
    :math:`L` are accrued liabilities, and :math:`N` is shares outstanding.

    Args:
        holdings: Number of shares held for each constituent (len must equal
            len(prices)).
        prices: Current market price for each constituent.
        shares_outstanding: Total ETF shares outstanding.
        cash: Cash and cash-equivalent balance (default 0).
        liabilities: Accrued liabilities such as management fees payable
            (default 0).

    Returns:
        NAV per share in currency units.

    Raises:
        ValueError: If *holdings* and *prices* differ in length or
            *shares_outstanding* is not positive.
    """
    if len(holdings) != len(prices):
        raise ValueError("holdings and prices must have the same length.")
    if shares_outstanding <= 0.0:
        raise ValueError("shares_outstanding must be positive.")

    basket_value = sum(h * p for h, p in zip(holdings, prices))
    return (basket_value + cash - liabilities) / shares_outstanding


def premium_discount(market_price: float, nav: float) -> float:
    """Premium (positive) or discount (negative) to NAV in percent.

    .. math::

        \\text{P/D} = \\frac{\\text{market} - \\text{NAV}}{\\text{NAV}} \\times 100

    Args:
        market_price: Observed ETF market price per share.
        nav: Net asset value per share.

    Returns:
        Premium/discount in percentage points.

    Raises:
        ValueError: If *nav* is zero.
    """
    if nav == 0.0:
        raise ValueError("nav must be non-zero.")
    return (market_price - nav) / nav * 100.0


# ---------------------------------------------------------------------------
# AP creation/redemption arbitrage
# ---------------------------------------------------------------------------

def creation_redemption_arb(
    market_price: float,
    nav: float,
    creation_fee: float,
    redemption_fee: float,
    transaction_costs: float,
) -> ArbResult:
    """Authorised participant creation/redemption arbitrage analysis.

    APs can profitably arbitrage deviations between the ETF market price and
    NAV:

    * **Creation arb** (premium): buy the basket, deliver to fund, receive ETF
      shares, sell at market price.  Profitable when
      ``market_price - nav > creation_fee + transaction_costs``.

    * **Redemption arb** (discount): buy ETF at market, redeem for basket,
      sell basket at NAV.  Profitable when
      ``nav - market_price > redemption_fee + transaction_costs``.

    All fee and cost parameters are in currency units *per share*.

    Args:
        market_price: Observed ETF market price per share.
        nav: Net asset value per share.
        creation_fee: Fee charged by the fund for a creation transaction
            (per share equivalent).
        redemption_fee: Fee charged by the fund for a redemption transaction.
        transaction_costs: Market-impact and brokerage costs for trading the
            basket (per share equivalent).

    Returns:
        :class:`ArbResult` with direction, gross and net profit.
    """
    premium = market_price - nav  # positive → ETF is expensive

    # Creation arb
    gross_create = premium          # buy basket at NAV, sell ETF at market
    net_create = gross_create - creation_fee - transaction_costs

    # Redemption arb
    gross_redeem = -premium         # buy ETF at market, redeem basket at NAV
    net_redeem = gross_redeem - redemption_fee - transaction_costs

    if net_create > 0.0 and net_create >= net_redeem:
        return ArbResult(
            direction="create",
            gross_profit=gross_create,
            net_profit=net_create,
            is_profitable=True,
        )
    elif net_redeem > 0.0:
        return ArbResult(
            direction="redeem",
            gross_profit=gross_redeem,
            net_profit=net_redeem,
            is_profitable=True,
        )
    else:
        # Report the less unprofitable direction for transparency
        if abs(net_create) <= abs(net_redeem):
            return ArbResult(
                direction="none",
                gross_profit=gross_create,
                net_profit=net_create,
                is_profitable=False,
            )
        else:
            return ArbResult(
                direction="none",
                gross_profit=gross_redeem,
                net_profit=net_redeem,
                is_profitable=False,
            )


# ---------------------------------------------------------------------------
# Tracking analytics
# ---------------------------------------------------------------------------

def tracking_error(
    etf_returns: list[float] | np.ndarray,
    index_returns: list[float] | np.ndarray,
) -> float:
    """Annualised tracking error (volatility of daily excess returns).

    .. math::

        \\text{TE} = \\sigma(r_{ETF} - r_{index}) \\times \\sqrt{252}

    Args:
        etf_returns: Daily (or periodic) ETF total returns as decimals.
        index_returns: Daily (or periodic) index total returns as decimals.

    Returns:
        Annualised tracking error as a decimal (multiply by 100 for percent).

    Raises:
        ValueError: If the two series differ in length or have fewer than
            two observations.
    """
    etf_arr = np.asarray(etf_returns, dtype=float)
    idx_arr = np.asarray(index_returns, dtype=float)
    if etf_arr.shape != idx_arr.shape:
        raise ValueError("etf_returns and index_returns must have the same shape.")
    if etf_arr.size < 2:
        raise ValueError("At least two observations are required.")
    excess = etf_arr - idx_arr
    return float(np.std(excess, ddof=1) * math.sqrt(252))


def tracking_difference(
    etf_returns: list[float] | np.ndarray,
    index_returns: list[float] | np.ndarray,
) -> float:
    """Cumulative tracking difference (total return gap) over the sample.

    The tracking difference is the cumulative underperformance of the ETF
    relative to its index.  Over a full year it is approximately equal to
    the total expense ratio plus implicit transaction costs:

    .. math::

        \\text{TD} = \\prod_t (1 + r_{ETF,t}) - \\prod_t (1 + r_{index,t})

    A negative value indicates the ETF underperformed.

    Args:
        etf_returns: Daily (or periodic) ETF total returns as decimals.
        index_returns: Daily (or periodic) index total returns as decimals.

    Returns:
        Cumulative tracking difference as a decimal.

    Raises:
        ValueError: If the two series differ in length.
    """
    etf_arr = np.asarray(etf_returns, dtype=float)
    idx_arr = np.asarray(index_returns, dtype=float)
    if etf_arr.shape != idx_arr.shape:
        raise ValueError("etf_returns and index_returns must have the same shape.")
    etf_cumulative = float(np.prod(1.0 + etf_arr))
    idx_cumulative = float(np.prod(1.0 + idx_arr))
    return etf_cumulative - idx_cumulative


# ---------------------------------------------------------------------------
# Synthetic ETF and leveraged ETF
# ---------------------------------------------------------------------------

def synthetic_etf_cost(
    swap_spread: float,
    expense_ratio: float,
    counterparty_haircut: float,
) -> float:
    """Total annual cost of a synthetic (swap-based) ETF vs. physical.

    A synthetic ETF uses a total return swap to replicate the index.  The
    all-in cost to the investor has three components:

    1. **Swap spread**: the premium paid to the swap counterparty for index
       performance delivery (can be negative, i.e., a rebate, for hard-to-
       borrow indices).
    2. **Expense ratio**: the fund's annual management fee.
    3. **Counterparty haircut**: implicit cost of the collateral posted to
       protect against counterparty default (forgone yield on high-quality
       collateral).

    .. math::

        C_{\\text{total}} = s_{\\text{swap}} + e + h

    All inputs and the return are expressed as annualised decimals.

    Args:
        swap_spread: Annual swap spread paid to the counterparty (decimal).
        expense_ratio: Annual management expense ratio (decimal).
        counterparty_haircut: Annual collateral/haircut cost (decimal).

    Returns:
        Total annual cost as a decimal.
    """
    return swap_spread + expense_ratio + counterparty_haircut


def leveraged_etf_decay(
    daily_leverage: float,
    vol: float,
    T_days: int,
    mu: float = 0.0,
) -> float:
    """Expected value of a daily-reset leveraged ETF showing volatility drag.

    A leveraged ETF that resets its leverage ratio *L* each day exhibits
    path-dependent *volatility decay*.  Under a continuous-time approximation
    the expected gross return over *T* trading days is:

    .. math::

        E[V_T / V_0]
            = \\exp\\!\\left(L\\,\\mu\\,T - \\tfrac{1}{2}\\,L(L-1)\\,\\sigma^2\\,T\\right)

    where :math:`\\mu` is the daily drift of the underlying and :math:`\\sigma`
    is its daily volatility.  The term :math:`-\\tfrac{1}{2} L(L-1)\\sigma^2 T`
    is the *volatility drag*, which is negative for :math:`L > 1` (long-
    leveraged) and :math:`L < 0` (inverse leveraged) and zero for :math:`L=1`.

    Args:
        daily_leverage: Leverage factor (e.g., 2 for 2× long, -1 for 1×
            inverse).  Not restricted to integers.
        vol: Daily volatility of the underlying (decimal, e.g., 0.01 for 1%).
        T_days: Number of trading days.
        mu: Expected daily return of the underlying (decimal).  Default is
            zero, isolating the pure volatility decay.

    Returns:
        Expected value ratio :math:`E[V_T / V_0]` (i.e., starting from 1.0).
    """
    L = daily_leverage
    T = float(T_days)
    drift_component = L * mu * T
    decay_component = -0.5 * L * (L - 1) * vol**2 * T
    return math.exp(drift_component + decay_component)
