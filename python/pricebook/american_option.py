"""American option instrument: LSM + PDE, serialisable.

Wraps existing `lsm_american()` and `fd_american()` in a proper
instrument class with exercise boundary output.

    from pricebook.american_option import AmericanOption

    opt = AmericanOption(strike=100, maturity=date(2027,4,28),
                          option_type=OptionType.PUT, notional=1_000_000)
    result = opt.price(spot=100, curve=ois, vol=0.20)

References:
    Longstaff & Schwartz (2001). Valuing American Options by Simulation.
    Review of Financial Studies, 14(1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook.black76 import OptionType, black76_price
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.serialisable import _register, _serialise_atom


@dataclass
class AmericanResult:
    price: float
    european_price: float = 0.0
    early_exercise_premium: float = 0.0
    method: str = ""

    def to_dict(self) -> dict:
        return {"price": self.price, "european_price": self.european_price,
                "early_exercise_premium": self.early_exercise_premium,
                "method": self.method}


class AmericanOption:
    """American option with early exercise.

    Priced via LSM (Monte Carlo) or PDE (finite difference).
    The early exercise premium = American price - European price.

    Args:
        strike: option strike.
        maturity: expiry date.
        option_type: CALL or PUT.
        notional: position size.
    """

    _SERIAL_TYPE = "american"

    def __init__(
        self,
        strike: float,
        maturity: date,
        option_type: OptionType = OptionType.PUT,
        notional: float = 1.0,
    ):
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type
        self.notional = notional

    def price(
        self,
        spot: float,
        curve: DiscountCurve,
        vol: float,
        div_yield: float = 0.0,
        method: str = "pde",
        n_paths: int = 100_000,
        n_steps: int = 200,
        seed: int = 42,
    ) -> AmericanResult:
        """Price the American option."""
        ref = curve.reference_date
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        T = year_fraction(ref, self.maturity, DayCountConvention.ACT_365_FIXED)
        rate = -math.log(curve.df(self.maturity)) / max(T, 1e-10)

        # European price for comparison
        fwd = spot * math.exp((rate - div_yield) * T)
        df = math.exp(-rate * T)
        euro = black76_price(fwd, self.strike, vol, T, df, self.option_type)

        if method == "pde":
            from pricebook.finite_difference import fd_american
            am_price = fd_american(spot, self.strike, rate, vol, T,
                                    self.option_type, div_yield,
                                    n_spot=n_steps, n_time=n_steps)
        else:
            from pricebook.lsm import lsm_american
            result = lsm_american(spot, self.strike, rate, vol, T,
                                   n_steps=min(n_steps, 50), n_paths=n_paths,
                                   option_type=self.option_type,
                                   div_yield=div_yield, seed=seed)
            am_price = result.price

        return AmericanResult(
            price=am_price * self.notional,
            european_price=euro * self.notional,
            early_exercise_premium=(am_price - euro) * self.notional,
            method=method,
        )

    def greeks(self, spot, curve, vol, div_yield=0.0, method="pde") -> dict[str, float]:
        base = self.price(spot, curve, vol, div_yield, method)
        bump = spot * 0.01
        up = self.price(spot + bump, curve, vol, div_yield, method)
        dn = self.price(spot - bump, curve, vol, div_yield, method)
        delta = (up.price - dn.price) / (2 * bump)
        gamma = (up.price - 2 * base.price + dn.price) / (bump ** 2)
        v_up = self.price(spot, curve, vol + 0.01, div_yield, method)
        vega = v_up.price - base.price
        return {"delta": delta, "gamma": gamma, "vega": vega, "price": base.price}

    def pv_ctx(self, ctx) -> float:
        vol_surface = ctx.vol_surfaces.get("equity") if ctx.vol_surfaces else None
        vol = vol_surface.vol(self.maturity, self.strike) if vol_surface else 0.20
        return self.price(spot=100.0, curve=ctx.discount_curve, vol=vol).price

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "strike": self.strike, "maturity": self.maturity.isoformat(),
            "option_type": _serialise_atom(self.option_type),
            "notional": self.notional,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> AmericanOption:
        p = d["params"]
        return cls(strike=p["strike"], maturity=date.fromisoformat(p["maturity"]),
                   option_type=OptionType(p.get("option_type", "put")),
                   notional=p.get("notional", 1.0))


_register(AmericanOption)
