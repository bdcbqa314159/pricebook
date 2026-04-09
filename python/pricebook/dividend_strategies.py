"""Dividend strategies: futures, basis, carry trade, roll-down, backtest.

Builds on top of ``dividend_desk`` (which provides implied dividends from
put-call parity, ``DividendSwap``, dividend forwards, dividend risk).

* :class:`DividendFuture` — exchange-traded contract on cumulative
  dividends paid over a reference period (e.g. EUREX dividend futures).
* :func:`dividend_basis` — implied (option) vs traded (swap/future) level.
* :class:`EquityCarryTrade` — long stock + short matched dividend swap.
* :func:`dividend_curve_carry` — carry from holding the implied curve.
* :func:`implied_vs_realised_backtest` — empirical curve accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.discount_curve import DiscountCurve
from pricebook.dividend_desk import DividendSwap
from pricebook.dividend_model import Dividend


# ---- Dividend futures ----

@dataclass
class DividendFuture:
    """Exchange-traded dividend future on an index or single name.

    Settlement is the sum of cash dividends paid by the underlying over
    the reference period. Mark-to-market PV is taken against the entry
    ``fixed_price`` and discounted to the settlement date.

    Attributes:
        reference_start: first ex-date in the reference period.
        reference_end: last ex-date in the reference period.
        settlement_date: cash settlement date (≥ reference_end).
        fixed_price: agreed futures level.
        notional: contract multiplier.
        direction: +1 long, -1 short.
    """
    reference_start: date
    reference_end: date
    settlement_date: date
    fixed_price: float
    notional: float = 1.0
    direction: int = 1

    def settlement_value(self, dividends: list[Dividend]) -> float:
        """Realised dividend total over the reference period."""
        return sum(
            d.amount for d in dividends
            if self.reference_start <= d.ex_date <= self.reference_end
        )

    def pv(
        self,
        dividends: list[Dividend],
        curve: DiscountCurve,
    ) -> float:
        """Mark-to-market PV against the realised dividend assumption.

            PV = direction · (settlement − fixed_price) · notional · df(settlement)
        """
        settle = self.settlement_value(dividends)
        df = curve.df(self.settlement_date)
        return self.direction * (settle - self.fixed_price) * self.notional * df

    def fair_price(self, dividends: list[Dividend]) -> float:
        """Settlement value — i.e. the futures level for which PV = 0."""
        return self.settlement_value(dividends)


# ---- Dividend basis ----

def dividend_basis(option_implied: float, traded: float) -> float:
    """Basis between an option-implied dividend level and a traded one.

        basis = traded − option_implied

    A positive basis means the traded market (swap or future) is rich
    relative to put-call parity implied levels.
    """
    return traded - option_implied


# ---- Equity carry trade ----

@dataclass
class EquityCarryTrade:
    """Long stock + short dividend swap = funded equity carry trade.

    The investor owns ``shares`` of an underlying and is short a matched
    ``DividendSwap`` that pays the realised dividends to the swap
    counterparty. By construction the realised-dividend exposure cancels
    when ``shares == swap.notional``: the investor's P&L comes only from
    the spot move and the swap's fair-vs-fixed differential.
    """
    shares: float
    spot: float
    swap: DividendSwap

    def stock_value(self) -> float:
        return self.shares * self.spot

    def net_dividend_exposure(self) -> float:
        """Sensitivity to realised dividends.

        Long stock receives realised dividends with a unit per share;
        short swap pays them at the swap notional. The trade is
        dividend-neutral when the two notionals match.
        """
        return self.shares - self.swap.notional

    def pv(
        self,
        dividends: list[Dividend],
        curve: DiscountCurve,
    ) -> float:
        """Mark-to-market: stock value − swap value (we are short the swap)."""
        swap_pv = self.swap.pv(dividends, curve)
        return self.stock_value() - swap_pv


# ---- Roll-down ----

def dividend_curve_carry(
    implied_today: list[tuple[date, float]],
    realised_path: list[tuple[date, float]],
) -> float:
    """Carry from selling the implied dividend curve and receiving realised.

    Args:
        implied_today: list of (ex_date, implied dividend) at trade inception.
        realised_path: list of (ex_date, realised dividend) over the holding
            period — same dates as ``implied_today`` (extra dates ignored).

    Returns:
        ``Σ (implied − realised)``. Positive if implieds were higher than
        what materialised — i.e. the trader who sold the curve made money.
    """
    realised_map = {d: amt for d, amt in realised_path}
    return sum(
        implied - realised_map.get(d, 0.0)
        for d, implied in implied_today
    )


# ---- Implied vs realised backtest ----

@dataclass
class BacktestResult:
    """Statistics from an implied-vs-realised dividend backtest.

    Errors are defined as ``implied − realised``: positive bias means
    the implied curve over-estimated the realised dividend on average.
    """
    n: int
    mean_error: float
    mean_abs_error: float
    rmse: float
    bias: float


def implied_vs_realised_backtest(
    pairs: list[tuple[float, float]],
) -> BacktestResult:
    """Compare implied dividends to subsequently realised dividends.

    Args:
        pairs: list of ``(implied, realised)`` dividend amounts.
    """
    n = len(pairs)
    if n == 0:
        return BacktestResult(0, 0.0, 0.0, 0.0, 0.0)
    errors = [implied - realised for implied, realised in pairs]
    mean = sum(errors) / n
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    return BacktestResult(n, mean, mae, rmse, mean)
