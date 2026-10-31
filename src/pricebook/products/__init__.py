"""L2 — product descriptions. Frozen data; nothing here prices itself."""

from pricebook.products.cash import FixedCashflow
from pricebook.products.inflation import (
    InflationLinkedBond,
    YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
)

__all__ = [
    "FixedCashflow", "InflationLinkedBond", "YearOnYearInflationSwap",
    "ZeroCouponInflationSwap",
]
