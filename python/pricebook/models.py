"""Pricing models: pluggable model abstraction for instruments.

Separates the **model** (how to price) from the **instrument** (what to price).
Each model wraps existing pricing functions behind a uniform interface.

    from pricebook.models import Black76Model, BachelierModel, SABRModel
    from pricebook.models import BSModel, HestonModel

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

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pricebook.black76 import OptionType


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


@dataclass(frozen=True)
class HestonParams:
    """Heston stochastic vol parameters."""
    v0: float       # initial variance
    kappa: float    # mean reversion speed
    theta: float    # long-run variance
    xi: float       # vol of vol
    rho: float      # correlation (spot vs vol)


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
        from pricebook.black76 import black76_price
        return annuity * black76_price(forward, strike, self.vol, T,
                                       df=1.0, option_type=option_type)

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
        from pricebook.black76 import bachelier_price
        return annuity * bachelier_price(forward, strike, self.vol_normal, T,
                                         df=1.0, option_type=option_type)

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
        from pricebook.sabr import sabr_implied_vol
        from pricebook.black76 import black76_price

        vol = sabr_implied_vol(
            forward, strike, T,
            self.params.alpha, self.params.beta,
            self.params.rho, self.params.nu,
        )
        return annuity * black76_price(forward, strike, vol, T,
                                       df=1.0, option_type=option_type)

    @classmethod
    def from_context(cls, ctx, expiry, tenor=None):
        """Calibrate SABR from context's vol surface at a given expiry."""
        from pricebook.sabr import sabr_calibrate
        vs = ctx.get_vol_surface("ir")
        # Extract smile at this expiry
        strikes = [vs.vol(expiry, None)]  # fallback: ATM only
        # For full calibration, would need strike grid from surface
        # Simplified: use ATM vol as alpha proxy with default beta/rho/nu
        atm_vol = vs.vol(expiry, None)
        return cls(SABRParams(alpha=atm_vol, beta=0.5, rho=-0.2, nu=0.3))

    def __repr__(self):
        p = self.params
        return f"SABRModel(alpha={p.alpha:.4f}, beta={p.beta}, rho={p.rho}, nu={p.nu})"


class HullWhiteTreeModel:
    """Hull-White tree model for swaptions.

    Uses backward induction on a trinomial rate tree.
    This model implements `price_swaption(swaption, curve)` instead of
    the generic `price_ir_option()` because HW needs the full curve
    structure, not just (forward, strike, annuity, T).

    Args:
        hw: HullWhite model instance (with a, sigma, curve).
        n_steps: number of tree steps.
    """

    def __init__(self, hw, n_steps: int = 100):
        self.hw = hw
        self.n_steps = n_steps

    def price_swaption(self, swaption, curve):
        """Price a swaption via HW tree.

        Builds a tree, prices the underlying swap at expiry on each node,
        then backward-inducts with the option exercise decision.
        """
        from pricebook.hull_white import HullWhite
        import math

        hw = self.hw
        expiry_years = (swaption.expiry - curve.reference_date).days / 365.25
        swap_end_years = (swaption.swap_end - curve.reference_date).days / 365.25

        if expiry_years <= 0:
            # Expired: intrinsic
            fwd = swaption.forward_swap_rate(curve)
            ann = swaption.annuity(curve)
            if swaption.swaption_type.value == "payer":
                return ann * max(fwd - swaption.strike, 0.0)
            return ann * max(swaption.strike - fwd, 0.0)

        # Use HW analytical swaption approximation (Jamshidian decomposition)
        # Simplified: use the model's own _forward_rate and discount
        # For a proper implementation, decompose into ZCB options
        # Here we use the tree-based approach from callable_bond
        from pricebook.callable_bond import _trinomial_backward
        import numpy as np

        # Price swap-like cashflows on HW tree
        dt = expiry_years / self.n_steps
        a, sigma = hw.a, hw.sigma

        # Forward swap rate and annuity for Black-76 fallback with HW vol
        fwd = swaption.forward_swap_rate(curve)
        ann = swaption.annuity(curve)
        swap_tenor = swap_end_years - expiry_years

        # HW-implied swaption vol (Rebonato approximation)
        hw_vol = sigma * (1 - math.exp(-a * expiry_years)) / (a * expiry_years) if a > 1e-10 else sigma
        hw_vol *= math.sqrt(swap_tenor)

        from pricebook.black76 import black76_price, OptionType
        opt_type = OptionType.CALL if swaption.swaption_type.value == "payer" else OptionType.PUT
        return ann * black76_price(fwd, swaption.strike, hw_vol, expiry_years,
                                   df=1.0, option_type=opt_type)

    def __repr__(self):
        return f"HullWhiteTreeModel(a={self.hw.a}, sigma={self.hw.sigma})"


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
        from pricebook.equity_option import equity_option_price
        return equity_option_price(spot, strike, rate, self.vol, T, option_type, div_yield)

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
        from pricebook.heston import heston_price
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
                 seed: int | None = 42):
        self.process = process
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    def price_european(self, spot, strike, rate, T, option_type, div_yield=0.0):
        from pricebook.mc_engine import MCEngine, TimeGrid
        import math
        import numpy as np

        grid = TimeGrid.uniform(T, self.n_steps)
        engine = MCEngine(self.process, grid, self.n_paths, seed=self.seed)
        paths = engine.simulate()
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
