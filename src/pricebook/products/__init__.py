"""L2 — product descriptions. Frozen data; nothing here prices itself."""

from pricebook.products.cash import FixedCashflow
from pricebook.products.interest_rate_swap import InterestRateSwap, SwapDirection
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption
from pricebook.products.generalized_variance_swap import CorridorVarianceSwap, GammaSwap
from pricebook.products.realized_variance_option import RealizedVarianceOption
from pricebook.products.discrete_variance_swap import DiscreteVarianceSwap
from pricebook.products.timer_option import TimerOption
from pricebook.products.position import Position
from pricebook.products.trade import Trade, TradeDirection
from pricebook.products.variance_swap import VarianceSwap
from pricebook.products.vix import VixFuture, VixOption
from pricebook.products.volatility_swap import VolatilitySwap
from pricebook.products.nominal_cap import NominalCap
from pricebook.products.inflation import (
    InflationCapFloor,
    InflationLinkedBond,
    YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
)

__all__ = [
    "VarianceSwap", "DiscreteVarianceSwap", "TimerOption", "Trade", "TradeDirection", "Position", "GammaSwap", "CorridorVarianceSwap", "RealizedVarianceOption", "VixFuture", "VixOption",
    "VolatilitySwap",
    "EquityOptionKind",
    "EuropeanEquityOption",
    "NominalCap",
    "FixedCashflow",
    "InterestRateSwap",
    "SwapDirection", "InflationCapFloor", "InflationLinkedBond", "YearOnYearInflationSwap",
    "ZeroCouponInflationSwap",
]
