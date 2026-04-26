"""Common result interface for structured product instruments.

All instrument result dataclasses should implement:
- `.price` property: the primary valuation metric
- `.to_dict()`: flat dictionary for risk systems

    from pricebook.instrument_result import InstrumentResult

    isinstance(result, InstrumentResult)  # True for all product results
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class InstrumentResult(Protocol):
    """Protocol for instrument pricing results.

    Every structured product result should expose:
    - price: the primary PV or normalised value
    - to_dict: flat dictionary for downstream consumption (risk, reporting)
    """

    @property
    def price(self) -> float:
        ...

    def to_dict(self) -> dict[str, float]:
        ...
