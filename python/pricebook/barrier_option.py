"""Barrier option instrument: knock-out and knock-in with PDE + MC.

Wraps existing `fd_barrier_knockout/knockin` from `finite_difference.py`
in a proper instrument class with serialisation.

    from pricebook.barrier_option import BarrierOption

    opt = BarrierOption(strike=100, barrier=120, barrier_type="up_out",
                        maturity=date(2027,4,28), notional=1_000_000)
    result = opt.price(spot=100, curve=ois, vol=0.20)

References:
    Merton (1973). Theory of rational option pricing. Bell J. Econ.
    Rubinstein & Reiner (1991). Breaking down the barriers. Risk.
    Hull, *OFOD*, 11th ed., Ch. 26.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from pricebook.black76 import OptionType, black76_price
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.serialisable import _register, _serialise_atom


# ---- Barrier types ----

BARRIER_TYPES = {"up_out", "up_in", "down_out", "down_in"}


@dataclass
class BarrierResult:
    """Barrier option pricing result."""
    price: float
    vanilla_price: float    # price without barrier (for comparison)
    method: str             # "pde" or "mc"
    barrier_hit_prob: float = 0.0

    def to_dict(self) -> dict:
        return {"price": self.price, "vanilla_price": self.vanilla_price,
                "method": self.method, "barrier_hit_prob": self.barrier_hit_prob}


class BarrierOption:
    """European barrier option — knock-out or knock-in.

    Barrier types:
    - up_out: dies if spot hits barrier from below (barrier > spot)
    - up_in: activates if spot hits barrier from below
    - down_out: dies if spot hits barrier from above (barrier < spot)
    - down_in: activates if spot hits barrier from above

    Parity: knock_in + knock_out = vanilla (same strike/maturity).

    Args:
        strike: option strike.
        barrier: barrier level.
        barrier_type: "up_out", "up_in", "down_out", "down_in".
        maturity: expiry date.
        option_type: CALL or PUT.
        notional: position size.
        rebate: cash paid if barrier is hit (knock-out) or not hit (knock-in).
    """

    _SERIAL_TYPE = "barrier"

    def __init__(
        self,
        strike: float,
        barrier: float,
        barrier_type: str,
        maturity: date,
        option_type: OptionType = OptionType.CALL,
        notional: float = 1.0,
        rebate: float = 0.0,
    ):
        if barrier_type not in BARRIER_TYPES:
            raise ValueError(f"barrier_type must be one of {BARRIER_TYPES}, got '{barrier_type}'")
        self.strike = strike
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.maturity = maturity
        self.option_type = option_type
        self.notional = notional
        self.rebate = rebate

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
    ) -> BarrierResult:
        """Price the barrier option.

        Args:
            spot: current underlying price.
            curve: discount curve.
            vol: flat volatility.
            method: "pde" (finite difference) or "mc" (Monte Carlo).
        """
        ref = curve.reference_date
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        T = year_fraction(ref, self.maturity, DayCountConvention.ACT_365_FIXED)
        rate = -math.log(curve.df(self.maturity)) / max(T, 1e-10)

        # Vanilla price for comparison
        fwd = spot * math.exp((rate - div_yield) * T)
        df = math.exp(-rate * T)
        vanilla = black76_price(fwd, self.strike, vol, T, df, self.option_type)

        if method == "pde":
            price = self._price_pde(spot, rate, vol, T, div_yield, n_steps)
        else:
            price, hit_prob = self._price_mc(spot, rate, vol, T, div_yield,
                                              n_paths, n_steps, seed)
            return BarrierResult(
                price=price * self.notional,
                vanilla_price=vanilla * self.notional,
                method="mc",
                barrier_hit_prob=hit_prob,
            )

        return BarrierResult(
            price=price * self.notional,
            vanilla_price=vanilla * self.notional,
            method="pde",
        )

    def _price_pde(self, spot, rate, vol, T, div_yield, n_steps) -> float:
        """Price via finite difference PDE."""
        from pricebook.finite_difference import fd_barrier_knockout, fd_barrier_knockin

        is_knockout = "out" in self.barrier_type
        is_up = "up" in self.barrier_type

        lower = self.barrier if not is_up else None
        upper = self.barrier if is_up else None

        if is_knockout:
            return fd_barrier_knockout(
                spot=spot, strike=self.strike, rate=rate, vol=vol, T=T,
                option_type=self.option_type,
                barrier_lower=lower, barrier_upper=upper,
                n_spot=n_steps, n_time=n_steps,
            )
        else:
            return fd_barrier_knockin(
                spot=spot, strike=self.strike, rate=rate, vol=vol, T=T,
                option_type=self.option_type,
                barrier_lower=lower, barrier_upper=upper,
                n_spot=n_steps, n_time=n_steps,
            )

    def _price_mc(self, spot, rate, vol, T, div_yield,
                   n_paths, n_steps, seed) -> tuple[float, float]:
        """Price via MC with barrier monitoring at each step."""
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)

        S = np.full(n_paths, spot, dtype=float)
        survived = np.ones(n_paths, dtype=bool)    # knock-out: alive if never hit
        activated = np.zeros(n_paths, dtype=bool)   # knock-in: active if ever hit

        is_knockout = "out" in self.barrier_type
        is_up = "up" in self.barrier_type

        # Barrier MC with discrete monitoring
        # Note: discrete monitoring underestimates crossing probability vs continuous.
        # Brownian bridge correction (Broadie, Glasserman & Kou 1997) improves accuracy.
        for step in range(n_steps):
            Z = rng.standard_normal(n_paths)
            S = S * np.exp((rate - div_yield - 0.5 * vol**2) * dt + vol * sqrt_dt * Z)

            if is_up:
                hit = S >= self.barrier
            else:
                hit = S <= self.barrier

            survived &= ~hit       # knock-out: die on hit
            activated |= hit       # knock-in: activate on hit

        df = math.exp(-rate * T)

        if self.option_type == OptionType.CALL:
            payoff = np.maximum(S - self.strike, 0.0)
        else:
            payoff = np.maximum(self.strike - S, 0.0)

        if is_knockout:
            payoff = payoff * survived
        else:
            payoff = payoff * activated  # knock-in: pay only if barrier was hit

        price = float(df * payoff.mean())
        hit_prob = float((~survived if is_knockout else activated).mean())

        return price, hit_prob

    def greeks(
        self,
        spot: float,
        curve: DiscountCurve,
        vol: float,
        div_yield: float = 0.0,
        method: str = "pde",
    ) -> dict[str, float]:
        """Bump-and-reprice Greeks."""
        base = self.price(spot, curve, vol, div_yield, method)
        bump_s = spot * 0.01

        up = self.price(spot + bump_s, curve, vol, div_yield, method)
        dn = self.price(spot - bump_s, curve, vol, div_yield, method)
        delta = (up.price - dn.price) / (2 * bump_s)
        gamma = (up.price - 2 * base.price + dn.price) / (bump_s ** 2)

        v_up = self.price(spot, curve, vol + 0.01, div_yield, method)
        vega = v_up.price - base.price

        return {"delta": delta, "gamma": gamma, "vega": vega, "price": base.price}

    def pv_ctx(self, ctx) -> float:
        vol_surface = ctx.vol_surfaces.get("equity") if ctx.vol_surfaces else None
        vol = vol_surface.vol(self.maturity, self.strike) if vol_surface else 0.20
        return self.price(spot=100.0, curve=ctx.discount_curve, vol=vol).price

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "strike": self.strike, "barrier": self.barrier,
            "barrier_type": self.barrier_type,
            "maturity": self.maturity.isoformat(),
            "option_type": _serialise_atom(self.option_type),
            "notional": self.notional, "rebate": self.rebate,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> BarrierOption:
        p = d["params"]
        return cls(
            strike=p["strike"], barrier=p["barrier"],
            barrier_type=p["barrier_type"],
            maturity=date.fromisoformat(p["maturity"]),
            option_type=OptionType(p.get("option_type", "call")),
            notional=p.get("notional", 1.0),
            rebate=p.get("rebate", 0.0),
        )


_register(BarrierOption)
