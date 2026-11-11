"""L2 — product descriptions. Frozen data; nothing here prices itself."""

from pricebook.products.cash import FixedCashflow
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption
from pricebook.products.variance_swap import VarianceSwap
from pricebook.products.nominal_cap import NominalCap
from pricebook.products.inflation import (
    InflationCapFloor,
    InflationLinkedBond,
    YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
)

__all__ = [
    "VarianceSwap",
    "EquityOptionKind",
    "EuropeanEquityOption",
    "NominalCap",
    "FixedCashflow", "InflationCapFloor", "InflationLinkedBond", "YearOnYearInflationSwap",
    "ZeroCouponInflationSwap",
]
