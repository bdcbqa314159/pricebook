"""Pricing models: pluggable model abstraction for instruments.

Separates the **model** (how to price) from the **instrument** (what to price).
Each model wraps existing pricing functions behind a uniform interface.

    from pricebook.models.models import Black76Model, BachelierModel, SABRModel
    from pricebook.models.models import BSModel, HestonModel

    # IR options (swaptions, caps/floors)
    model = Black76Model(vol=0.20)
    pv = swaption.price(model, curve)

    # Equity options
    model = HestonModel(HestonParams(v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7))
    pv = price_european(model, spot=100, strike=100, rate=0.04, T=1.0)

Two protocols:
    IROptionModel  — prices on (forward, strike, annuity, T)
    EquityOptionModel — prices on (spot, strike, rate, T, div_yield)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pricebook.models.black76 import OptionType


# ═══════════════════════════════════════════════════════════════
# Protocols
# ═══════════════════════════════════════════════════════════════

@runtime_checkable
class IROptionModel(Protocol):
    """Model that can price IR options (swaptions, caps/floors).

    Works on forward rate + annuity (Black-76-style inputs).
    The annuity already contains discounting, so df=1 internally.
    """

    def price_ir_option(
        self,
        forward: float,
        strike: float,
        annuity: float,
        time_to_expiry: float,
        option_type: OptionType,
    ) -> float: ...


@runtime_checkable
class EquityOptionModel(Protocol):
    """Model that can price European equity/FX options."""

    def price_european(
        self,
        spot: float,
        strike: float,
        rate: float,
        time_to_expiry: float,
        option_type: OptionType,
        div_yield: float = 0.0,
    ) -> float: ...


# ═══════════════════════════════════════════════════════════════
# Params dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SABRParams:
    """SABR model parameters."""
    alpha: float    # initial vol level
    beta: float     # CEV exponent (0=normal, 1=lognormal)
    rho: float      # correlation (vol vs forward)
    nu: float       # vol-of-vol



    def to_dict(self) -> dict:
        return vars(self)
# Reuse HestonParams from slv.py to avoid duplication
from pricebook.options.slv import HestonParams


# ═══════════════════════════════════════════════════════════════
# IR Option Models
# ═══════════════════════════════════════════════════════════════

class Black76Model:
    """Lognormal (Black-76) model for IR options.

    Args:
        vol: lognormal (Black) volatility.
    """

    def __init__(self, vol: float):
        self.vol = vol

    def price_ir_option(self, forward, strike, annuity, T, option_type):
        from pricebook.models.black76 import black76_price
        return annuity * black76_price(forward, strike, self.vol, T,
                                       df=1.0, option_type=option_type)

    def greeks_ir_option(self, forward, strike, annuity, T, option_type):
        from pricebook.models.black76 import (black76_price, black76_delta,
                                        black76_gamma, black76_vega, black76_theta)
        from pricebook.risk.greeks import Greeks
        return Greeks(
            price=annuity * black76_price(forward, strike, self.vol, T, 1.0, option_type),
            delta=annuity * black76_delta(forward, strike, self.vol, T, 1.0, option_type),
            gamma=annuity * black76_gamma(forward, strike, self.vol, T, 1.0),
            vega=annuity * black76_vega(forward, strike, self.vol, T, 1.0),
            theta=annuity * black76_theta(forward, strike, self.vol, T, 1.0, option_type),
        )

    @classmethod
    def from_context(cls, ctx, expiry, strike=None):
        """Extract vol from context's IR vol surface."""
        vs = ctx.get_vol_surface("ir")
        vol = vs.vol(expiry, strike)
        return cls(vol=vol)

    def __repr__(self):
        return f"Black76Model(vol={self.vol:.4f})"


class BachelierModel:
    """Normal (Bachelier) model for IR options.

    Args:
        vol_normal: normal volatility (in rate units, e.g. 0.005 = 50bp).
    """

    def __init__(self, vol_normal: float):
        self.vol_normal = vol_normal

    def price_ir_option(self, forward, strike, annuity, T, option_type):
        from pricebook.models.black76 import bachelier_price
        return annuity * bachelier_price(forward, strike, self.vol_normal, T,
                                         df=1.0, option_type=option_type)

    def greeks_ir_option(self, forward, strike, annuity, T, option_type):
        from pricebook.models.black76 import (bachelier_price, bachelier_delta,
                                        bachelier_gamma, bachelier_vega, bachelier_theta)
        from pricebook.risk.greeks import Greeks
        return Greeks(
            price=annuity * bachelier_price(forward, strike, self.vol_normal, T, 1.0, option_type),
            delta=annuity * bachelier_delta(forward, strike, self.vol_normal, T, 1.0, option_type),
            gamma=annuity * bachelier_gamma(forward, strike, self.vol_normal, T, 1.0),
            vega=annuity * bachelier_vega(forward, strike, self.vol_normal, T, 1.0),
            theta=annuity * bachelier_theta(forward, strike, self.vol_normal, T, 1.0, option_type),
        )

    def __repr__(self):
        return f"BachelierModel(vol_normal={self.vol_normal:.6f})"


class SABRModel:
    """SABR smile model for IR options.

    Computes Black implied vol via Hagan approximation, then prices with Black-76.

    Args:
        params: SABRParams(alpha, beta, rho, nu).
    """

    def __init__(self, params: SABRParams):
        self.params = params

    def price_ir_option(self, forward, strike, annuity, T, option_type):
        from pricebook.options.sabr import sabr_implied_vol
        from pricebook.models.black76 import black76_price

        vol = sabr_implied_vol(
            forward, strike, T,
            self.params.alpha, self.params.beta,
            self.params.rho, self.params.nu,
        )
        return annuity * black76_price(forward, strike, vol, T,
                                       df=1.0, option_type=option_type)

    def greeks_ir_option(self, forward, strike, annuity, T, option_type):
        """SABR greeks via composite: SABR implied vol → Black-76 greeks.

        This is the standard approximation — it ignores ∂σ_impl/∂F
        (the SABR delta adjustment). For most practical purposes this is
        sufficient; the full Hagan ∂σ/∂F correction is a Phase 2 enhancement.
        """
        from pricebook.options.sabr import sabr_implied_vol
        from pricebook.models.black76 import (black76_price, black76_delta,
                                        black76_gamma, black76_vega, black76_theta)
        from pricebook.risk.greeks import Greeks

        vol = sabr_implied_vol(
            forward, strike, T,
            self.params.alpha, self.params.beta,
            self.params.rho, self.params.nu,
        )
        return Greeks(
            price=annuity * black76_price(forward, strike, vol, T, 1.0, option_type),
            delta=annuity * black76_delta(forward, strike, vol, T, 1.0, option_type),
            gamma=annuity * black76_gamma(forward, strike, vol, T, 1.0),
            vega=annuity * black76_vega(forward, strike, vol, T, 1.0),
            theta=annuity * black76_theta(forward, strike, vol, T, 1.0, option_type),
        )

    @classmethod
    def from_atm(cls, atm_vol: float, beta: float = 0.5,
                 rho: float = -0.2, nu: float = 0.3) -> SABRModel:
        """Quick SABR from ATM vol with default smile params.

        This is an approximation, not a calibration. For proper calibration,
        construct SABRParams via sabr_calibrate() from sabr.py.
        """
        return cls(SABRParams(alpha=atm_vol, beta=beta, rho=rho, nu=nu))

    def __repr__(self):
        p = self.params
        return f"SABRModel(alpha={p.alpha:.4f}, beta={p.beta}, rho={p.rho}, nu={p.nu})"


class HullWhiteModel:
    """Hull-White analytical swaption model.

    Uses the Rebonato approximation to compute an HW-implied Black vol,
    then prices with Black-76. This is an analytical approximation —
    for exact tree-based pricing, use the callable_bond module directly.

    The HW-implied vol for a swaption expiring at T into a swap with
    payment dates {T_1, ..., T_n} is:

        sigma_sw^2 = (1/T) * [sum_i w_i * B(0, T_i)]^2 * integral

    where B(s, t) = (1 - e^{-a(t-s)}) / a is the HW bond sensitivity.

    Simplified here as: sigma_sw ≈ sigma * B(0, T) / T * sum(w_i * B(0, T_i))

    Args:
        hw: HullWhite model instance (with a, sigma, curve).
    """

    def __init__(self, hw):
        self.hw = hw

    def price_swaption(self, swaption, curve, projection_curve=None,
                       valuation_date=None):
        """Price a swaption via HW analytical approximation."""
        if valuation_date is None:
            valuation_date = curve.reference_date

        expiry_years = (swaption.expiry - valuation_date).days / 365.25

        if expiry_years <= 0:
            fwd = swaption.forward_swap_rate(curve, projection_curve)
            ann = swaption.annuity(curve)
            if swaption.swaption_type.value == "payer":
                return ann * max(fwd - swaption.strike, 0.0)
            return ann * max(swaption.strike - fwd, 0.0)

        a, sigma = self.hw.a, self.hw.sigma
        fwd = swaption.forward_swap_rate(curve, projection_curve)
        ann = swaption.annuity(curve)

        # Integrated HW vol: sqrt((1 - e^{-2aT}) / (2a))
        if a > 1e-10:
            integrated = math.sqrt((1 - math.exp(-2 * a * expiry_years)) / (2 * a))
        else:
            integrated = math.sqrt(expiry_years)

        # Approximate swaption vol (Rebonato-style)
        hw_vol = sigma * integrated / math.sqrt(expiry_years) if expiry_years > 0 else sigma

        from pricebook.models.black76 import black76_price
        opt_type = OptionType.CALL if swaption.swaption_type.value == "payer" else OptionType.PUT
        return ann * black76_price(fwd, swaption.strike, hw_vol, expiry_years,
                                   df=1.0, option_type=opt_type)

    def __repr__(self):
        return f"HullWhiteModel(a={self.hw.a}, sigma={self.hw.sigma})"


# ═══════════════════════════════════════════════════════════════
# Equity Option Models
# ═══════════════════════════════════════════════════════════════

class BSModel:
    """Black-Scholes model for equity/FX options.

    Args:
        vol: lognormal volatility.
    """

    def __init__(self, vol: float):
        self.vol = vol

    def price_european(self, spot, strike, rate, T, option_type, div_yield=0.0):
        from pricebook.options.equity_option import equity_option_price
        return equity_option_price(spot, strike, rate, self.vol, T, option_type, div_yield)

    def greeks_european(self, spot, strike, rate, T, option_type, div_yield=0.0):
        from pricebook.options.equity_option import equity_greeks
        return equity_greeks(spot, strike, rate, self.vol, T, option_type, div_yield)

    def __repr__(self):
        return f"BSModel(vol={self.vol:.4f})"


class HestonModel:
    """Heston stochastic vol model for equity options.

    Uses the semi-analytical characteristic function approach.

    Args:
        params: HestonParams(v0, kappa, theta, xi, rho).
    """

    def __init__(self, params: HestonParams):
        self.params = params

    def price_european(self, spot, strike, rate, T, option_type, div_yield=0.0):
        from pricebook.options.heston import heston_price
        p = self.params
        return heston_price(spot, strike, rate, T, p.v0, p.kappa, p.theta,
                           p.xi, p.rho, option_type, div_yield)

    def __repr__(self):
        p = self.params
        return f"HestonModel(v0={p.v0}, kappa={p.kappa}, theta={p.theta}, xi={p.xi}, rho={p.rho})"


class MCEquityModel:
    """Monte Carlo model for equity options using any ProcessSpec.

    Bridges the MC engine to the EquityOptionModel protocol.

    Args:
        process: a ProcessSpec from mc_processes.py.
        n_paths: number of MC paths.
        n_steps: number of time steps.
        seed: random seed.
    """

    def __init__(self, process, n_paths: int = 100_000, n_steps: int = 200,
                 seed: int | None = 42, process_spec: dict | None = None):
        self.process = process
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self._process_spec = process_spec

    def price_european(self, spot, strike, rate, T, option_type, div_yield=0.0):
        from pricebook.models.mc_engine import MCEngine, TimeGrid
        import numpy as np

        grid = TimeGrid.uniform(T, self.n_steps)
        engine = MCEngine(self.process, grid, self.n_paths, seed=self.seed)
        paths = engine.generate_paths()
        S_T = paths[:, -1]
        df = math.exp(-rate * T)

        if option_type == OptionType.CALL:
            payoff = np.maximum(S_T - strike, 0.0)
        else:
            payoff = np.maximum(strike - S_T, 0.0)

        return float(df * payoff.mean())

    def __repr__(self):
        return f"MCEquityModel(process={type(self.process).__name__}, paths={self.n_paths})"


# ═══════════════════════════════════════════════════════════════
# Convenience: standalone equity pricing with model
# ═══════════════════════════════════════════════════════════════

def price_european(
    model: EquityOptionModel,
    spot: float,
    strike: float,
    rate: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """Price a European option using any EquityOptionModel."""
    return model.price_european(spot, strike, rate, T, option_type, div_yield)


# ═══════════════════════════════════════════════════════════════
# Serialisation
# ═══════════════════════════════════════════════════════════════

def _model_to_dict(model) -> dict:
    """Serialise any model to a dict."""
    if isinstance(model, Black76Model):
        return {"type": "black76", "vol": model.vol}
    if isinstance(model, BachelierModel):
        return {"type": "bachelier", "vol_normal": model.vol_normal}
    if isinstance(model, SABRModel):
        p = model.params
        return {"type": "sabr", "alpha": p.alpha, "beta": p.beta,
                "rho": p.rho, "nu": p.nu}
    if isinstance(model, HullWhiteModel):
        hw = model.hw
        d = {"type": "hull_white", "a": hw.a, "sigma": hw.sigma}
        if hasattr(hw, 'curve') and hw.curve is not None:
            d["curve"] = hw.curve.to_dict()
        return d
    if isinstance(model, BSModel):
        return {"type": "bs", "vol": model.vol}
    if isinstance(model, HestonModel):
        p = model.params
        return {"type": "heston", "v0": p.v0, "kappa": p.kappa,
                "theta": p.theta, "xi": p.xi, "rho": p.rho}
    if isinstance(model, MCEquityModel):
        d = {"type": "mc_equity", "n_paths": model.n_paths,
             "n_steps": model.n_steps, "seed": model.seed}
        # Store process construction info if available
        if hasattr(model, '_process_spec'):
            d["process_spec"] = model._process_spec
        return d
    raise TypeError(f"unsupported model type: {type(model).__name__}")


def _model_from_dict(d: dict):
    """Deserialise a model from a dict."""
    t = d["type"]
    if t == "black76":
        return Black76Model(vol=d["vol"])
    if t == "bachelier":
        return BachelierModel(vol_normal=d["vol_normal"])
    if t == "sabr":
        return SABRModel(SABRParams(d["alpha"], d["beta"], d["rho"], d["nu"]))
    if t == "hull_white":
        from pricebook.models.hull_white import HullWhite
        curve = None
        if "curve" in d:
            from pricebook.core.discount_curve import DiscountCurve
            curve = DiscountCurve.from_dict(d["curve"])
        return HullWhiteModel(HullWhite(a=d["a"], sigma=d["sigma"], curve=curve))
    if t == "bs":
        return BSModel(vol=d["vol"])
    if t == "heston":
        return HestonModel(HestonParams(d["v0"], d["kappa"], d["theta"],
                                         d["xi"], d["rho"]))
    if t == "mc_equity":
        spec = d.get("process_spec")
        if spec is None:
            raise ValueError("mc_equity model requires process_spec for deserialisation")
        from pricebook.models.models import mc_processes as _mp
        proc_fn = getattr(_mp, spec["name"], None)
        if proc_fn is None:
            raise ValueError(f"unknown process: {spec['name']!r}")
        process = proc_fn(**spec["args"])
        return MCEquityModel(process, d.get("n_paths", 100_000),
                             d.get("n_steps", 200), d.get("seed", 42),
                             process_spec=spec)
    raise ValueError(f"unknown model type: {t!r}")


# Bind to_dict on each model class
Black76Model.to_dict = lambda self: _model_to_dict(self)
BachelierModel.to_dict = lambda self: _model_to_dict(self)
SABRModel.to_dict = lambda self: _model_to_dict(self)
HullWhiteModel.to_dict = lambda self: _model_to_dict(self)
BSModel.to_dict = lambda self: _model_to_dict(self)
HestonModel.to_dict = lambda self: _model_to_dict(self)
MCEquityModel.to_dict = lambda self: _model_to_dict(self)
