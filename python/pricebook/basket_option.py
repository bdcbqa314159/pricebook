"""Basket option instrument: multi-asset, serialisable.

Wraps CorrelatedGBM for multi-asset payoffs: weighted basket,
best-of, worst-of.

    from pricebook.basket_option import BasketOption

    opt = BasketOption(strikes=[100], weights=[0.5, 0.3, 0.2],
                        maturity=date(2027,4,28), n_assets=3)
    result = opt.price_mc(spots=[100, 50, 200], vols=[0.20, 0.25, 0.30],
                           corr_matrix=..., curve=ois)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from pricebook.black76 import OptionType
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.multi_asset_mc import CorrelatedGBM, MultiAssetResult
from pricebook.serialisable import _register, _serialise_atom


class BasketOption:
    """Basket option: payoff on weighted combination of multiple assets.

    payoff_types:
    - "basket": max(Σ w_i S_i(T) - K, 0)
    - "best_of": max(max_i(S_i(T)/S_i(0)) - K, 0)
    - "worst_of": max(min_i(S_i(T)/S_i(0)) - K, 0)

    Args:
        strike: option strike.
        maturity: expiry date.
        weights: asset weights (for "basket" payoff).
        payoff_type: "basket", "best_of", "worst_of".
        option_type: CALL or PUT.
        notional: position size.
        n_assets: number of underlying assets.
    """

    _SERIAL_TYPE = "basket_option"

    def __init__(
        self,
        strike: float,
        maturity: date,
        weights: list[float] | None = None,
        payoff_type: str = "basket",
        option_type: OptionType = OptionType.CALL,
        notional: float = 1.0,
        n_assets: int = 2,
    ):
        self.strike = strike
        self.maturity = maturity
        self.weights = weights
        self.payoff_type = payoff_type
        self.option_type = option_type
        self.notional = notional
        self.n_assets = n_assets

    def price_mc(
        self,
        spots: list[float],
        vols: list[float],
        corr_matrix: list[list[float]],
        curve: DiscountCurve,
        div_yields: list[float] | float = 0.0,
        n_paths: int = 100_000,
        seed: int = 42,
    ) -> MultiAssetResult:
        """Price via correlated MC."""
        ref = curve.reference_date
        T = year_fraction(ref, self.maturity, DayCountConvention.ACT_365_FIXED)
        rate = -math.log(curve.df(self.maturity)) / max(T, 1e-10)
        df = math.exp(-rate * T)

        gen = CorrelatedGBM(spots=spots, vols=vols, corr_matrix=corr_matrix,
                             rates=rate, div_yields=div_yields)
        terminals = gen.terminal(T, n_paths, seed)  # (n_assets, n_paths)

        if self.payoff_type == "basket":
            w = np.array(self.weights if self.weights else [1.0/len(spots)] * len(spots))
            basket = (w[:, np.newaxis] * terminals).sum(axis=0)
            if self.option_type == OptionType.CALL:
                payoffs = np.maximum(basket - self.strike, 0.0)
            else:
                payoffs = np.maximum(self.strike - basket, 0.0)

        elif self.payoff_type == "best_of":
            returns = terminals / np.array(spots)[:, np.newaxis]
            best = returns.max(axis=0)
            if self.option_type == OptionType.CALL:
                payoffs = np.maximum(best - self.strike, 0.0)
            else:
                payoffs = np.maximum(self.strike - best, 0.0)

        elif self.payoff_type == "worst_of":
            returns = terminals / np.array(spots)[:, np.newaxis]
            worst = returns.min(axis=0)
            if self.option_type == OptionType.CALL:
                payoffs = np.maximum(worst - self.strike, 0.0)
            else:
                payoffs = np.maximum(self.strike - worst, 0.0)
        else:
            raise ValueError(f"Unknown payoff_type: {self.payoff_type}")

        discounted = df * payoffs * self.notional
        price = float(discounted.mean())
        std_err = float(discounted.std(ddof=1) / math.sqrt(n_paths))

        return MultiAssetResult(price=price, std_error=std_err,
                                 n_paths=n_paths, n_assets=len(spots))

    def pv_ctx(self, ctx) -> float:
        # Simplified: needs spots and vols from context
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "strike": self.strike, "maturity": self.maturity.isoformat(),
            "weights": self.weights, "payoff_type": self.payoff_type,
            "option_type": _serialise_atom(self.option_type),
            "notional": self.notional, "n_assets": self.n_assets,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> BasketOption:
        p = d["params"]
        return cls(strike=p["strike"], maturity=date.fromisoformat(p["maturity"]),
                   weights=p.get("weights"), payoff_type=p.get("payoff_type", "basket"),
                   option_type=OptionType(p.get("option_type", "call")),
                   notional=p.get("notional", 1.0), n_assets=p.get("n_assets", 2))


_register(BasketOption)
