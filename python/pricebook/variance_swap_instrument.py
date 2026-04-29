"""Variance swap instrument with serialisation.

Wraps existing `variance_swap_pv()` and `fair_variance()` from
`variance_swap.py` in a proper instrument class.

    from pricebook.variance_swap_instrument import VarianceSwapOption

    vs = VarianceSwapOption(strike_vol=0.20, notional_vega=100_000,
                            maturity=date(2027,4,28))
    result = vs.price(curve=ois, realised_vol=0.22)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.variance_swap import variance_swap_pv
from pricebook.serialisable import _register, _serialise_atom


@dataclass
class VarianceSwapResult:
    price: float
    fair_vol: float
    strike_vol: float
    pnl: float

    def to_dict(self) -> dict:
        return {"price": self.price, "fair_vol": self.fair_vol,
                "strike_vol": self.strike_vol, "pnl": self.pnl}


class VarianceSwapOption:
    """Variance swap: pays notional × (realised_var - strike_var).

    The strike is quoted in vol terms (strike_vol²  = strike_variance).
    Settlement at maturity: N_vega × (σ²_realised - K²) / (2K).

    Args:
        strike_vol: strike volatility (e.g. 0.20 for 20%).
        notional_vega: vega notional (PnL per 1 vol point move).
        maturity: expiry date.
    """

    _SERIAL_TYPE = "variance_swap"

    def __init__(
        self,
        strike_vol: float,
        notional_vega: float = 100_000.0,
        maturity: date = date(2027, 4, 28),
    ):
        self.strike_vol = strike_vol
        self.notional_vega = notional_vega
        self.maturity = maturity

    def price(
        self,
        curve: DiscountCurve,
        realised_vol: float | None = None,
        fair_vol: float | None = None,
    ) -> VarianceSwapResult:
        """Price the variance swap.

        Args:
            curve: discount curve.
            realised_vol: actual realised vol (for MTM). If None, uses fair_vol.
            fair_vol: market-implied fair vol. If None, uses strike_vol.
        """
        ref = curve.reference_date
        T = year_fraction(ref, self.maturity, DayCountConvention.ACT_365_FIXED)
        df = curve.df(self.maturity)

        fv = fair_vol if fair_vol is not None else self.strike_vol
        rv = realised_vol if realised_vol is not None else fv

        fair_var = fv ** 2
        strike_var = self.strike_vol ** 2

        pv_result = variance_swap_pv(fair_var, strike_var, self.notional_vega, T, df)
        pv = pv_result.pv
        pnl = self.notional_vega * (rv ** 2 - strike_var) / (2 * self.strike_vol) * df

        return VarianceSwapResult(price=pv, fair_vol=fv,
                                   strike_vol=self.strike_vol, pnl=pnl)

    def pv_ctx(self, ctx) -> float:
        return self.price(ctx.discount_curve).price

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "strike_vol": self.strike_vol,
            "notional_vega": self.notional_vega,
            "maturity": self.maturity.isoformat(),
        }}

    @classmethod
    def from_dict(cls, d: dict) -> VarianceSwapOption:
        p = d["params"]
        return cls(strike_vol=p["strike_vol"],
                   notional_vega=p.get("notional_vega", 100_000.0),
                   maturity=date.fromisoformat(p["maturity"]))


_register(VarianceSwapOption)
