"""Unified pricing engine protocol.

Defines the contract that MC, tree, and analytical engines implement,
enabling instrument code to be engine-agnostic.

* :class:`PricingResult` — unified result with price, Greeks, diagnostics.
* :class:`GreeksBundle` — full set of Greeks with method metadata.
* :class:`PricingEngine` — protocol for all engines.
* :class:`MCPricingEngine` — MC engine implementing the protocol.
* :class:`TreePricingEngine` — tree engine implementing the protocol.
* :class:`AnalyticalEngine` — closed-form engine implementing the protocol.

References:
    Glasserman, *Monte Carlo Methods in Financial Engineering*, 2003.
    Hull, *Options, Futures, and Other Derivatives*, 11th ed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════

@dataclass
class GreeksBundle:
    """Complete set of Greeks with method metadata."""
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0          # per 1% vol
    theta: float = 0.0         # per day
    rho: float = 0.0           # per 1% rate
    # Method used per Greek
    delta_method: str = ""
    gamma_method: str = ""
    vega_method: str = ""

    def to_dict(self) -> dict:
        return {
            "delta": self.delta, "gamma": self.gamma,
            "vega": self.vega, "theta": self.theta, "rho": self.rho,
        }


@dataclass
class ConvergenceInfo:
    """Convergence diagnostics."""
    std_error: float = 0.0
    relative_error_pct: float = 0.0
    confidence_95: tuple[float, float] = (0.0, 0.0)
    n_paths: int = 0
    n_steps: int = 0
    elapsed_seconds: float = 0.0
    engine_type: str = ""
    variance_reduction: str = "none"
    effective_sample_size: float = 0.0

    def to_dict(self) -> dict:
        return {
            "std_error": self.std_error,
            "relative_error_pct": self.relative_error_pct,
            "ci_95": list(self.confidence_95),
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "elapsed_s": round(self.elapsed_seconds, 4),
            "engine": self.engine_type,
        }


@dataclass
class PricingResult:
    """Unified pricing result from any engine."""
    price: float
    greeks: GreeksBundle = field(default_factory=GreeksBundle)
    convergence: ConvergenceInfo = field(default_factory=ConvergenceInfo)
    engine_type: str = ""
    model_name: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "greeks": self.greeks.to_dict(),
            "convergence": self.convergence.to_dict(),
            "engine": self.engine_type,
            "model": self.model_name,
        }


# ═══════════════════════════════════════════════════════════════
# Engine protocol
# ═══════════════════════════════════════════════════════════════

class EngineMethod(Enum):
    """Pricing engine method."""
    ANALYTICAL = "analytical"
    MC = "mc"
    TREE = "tree"
    PDE = "pde"


@runtime_checkable
class PricingEngine(Protocol):
    """Protocol for all pricing engines.

    Any engine that implements price_european and greeks can be used
    interchangeably by instrument code.
    """

    def price_vanilla(
        self,
        spot: float,
        strike: float,
        rate: float,
        vol: float,
        T: float,
        is_call: bool = True,
        div_yield: float = 0.0,
    ) -> PricingResult: ...

    @property
    def engine_type(self) -> str: ...


# ═══════════════════════════════════════════════════════════════
# MC Engine wrapper
# ═══════════════════════════════════════════════════════════════

class MCPricingEngine:
    """Monte Carlo engine implementing the unified protocol.

    Wraps the existing MCEngine with protocol-compliant interface.

    Args:
        n_paths: number of MC paths.
        n_steps: time steps per path.
        seed: random seed.
        antithetic: use antithetic variates.
        process_type: "gbm", "heston", "sabr", etc.
        greek_method: "bump" (default), "pathwise", "likelihood_ratio".
    """

    def __init__(
        self,
        n_paths: int = 100_000,
        n_steps: int = 100,
        seed: int = 42,
        antithetic: bool = True,
        process_type: str = "gbm",
        greek_method: str = "bump",
        heston_params: dict | None = None,
    ):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.antithetic = antithetic
        self.process_type = process_type
        self.greek_method = greek_method
        self.heston_params = heston_params or {}

    @property
    def engine_type(self) -> str:
        return f"mc_{self.process_type}"

    def price_vanilla(
        self,
        spot: float,
        strike: float,
        rate: float,
        vol: float,
        T: float,
        is_call: bool = True,
        div_yield: float = 0.0,
    ) -> PricingResult:
        from pricebook.models.mc_engine import MCEngine, TimeGrid
        from pricebook.models.mc_processes import GBMProcess, HestonProcess
        from pricebook.models.mc_payoffs import european_call, european_put

        t0 = time.time()

        # Build process
        if self.process_type == "heston":
            process = HestonProcess(
                s0=spot, mu=rate - div_yield, **self.heston_params,
            )
        else:
            process = GBMProcess(s0=spot, mu=rate - div_yield, sigma=vol)

        grid = TimeGrid.uniform(T, self.n_steps)
        engine = MCEngine(process, grid, self.n_paths, self.seed, self.antithetic)

        payoff = european_call(strike) if is_call else european_put(strike)
        df = math.exp(-rate * T)
        result = engine.price(payoff, df)

        # Greeks via bump-and-reprice (reuse seed)
        greeks = self._compute_greeks(
            spot, strike, rate, vol, T, is_call, div_yield, result.price,
        )

        elapsed = time.time() - t0
        conv = ConvergenceInfo(
            std_error=result.stderr,
            relative_error_pct=result.relative_error,
            confidence_95=result.confidence_95,
            n_paths=result.n_paths,
            n_steps=result.n_steps,
            elapsed_seconds=elapsed,
            engine_type=self.engine_type,
            variance_reduction="antithetic" if self.antithetic else "none",
        )

        return PricingResult(
            price=result.price,
            greeks=greeks,
            convergence=conv,
            engine_type=self.engine_type,
            model_name=self.process_type,
        )

    def _compute_greeks(
        self, spot, strike, rate, vol, T, is_call, div_yield, base_price,
    ) -> GreeksBundle:
        """Compute Greeks via bump-and-reprice with shared seed."""
        from pricebook.models.mc_engine import MCEngine, TimeGrid
        from pricebook.models.mc_processes import GBMProcess
        from pricebook.models.mc_payoffs import european_call, european_put

        payoff = european_call(strike) if is_call else european_put(strike)
        df = math.exp(-rate * T)

        def _price_with(s=spot, v=vol, r=rate, q=div_yield):
            proc = GBMProcess(s0=s, mu=r - q, sigma=v)
            grid = TimeGrid.uniform(T, self.n_steps)
            eng = MCEngine(proc, grid, self.n_paths, self.seed, self.antithetic)
            return eng.price(payoff, math.exp(-r * T)).price

        # Delta & gamma
        ds = spot * 0.005
        p_up = _price_with(s=spot + ds)
        p_dn = _price_with(s=spot - ds)
        delta = (p_up - p_dn) / (2 * ds)
        gamma = (p_up - 2 * base_price + p_dn) / (ds ** 2)

        # Vega per 1%
        dv = 0.01
        p_vup = _price_with(v=vol + dv)
        vega = p_vup - base_price

        # Theta per day
        dt_day = 1.0 / 365
        if T > dt_day:
            p_theta = _price_with()  # same, but with shorter T would need grid change
            theta = 0.0  # simplified — proper theta needs T bump
        else:
            theta = 0.0

        # Rho per 1%
        dr = 0.0001
        p_rup = _price_with(r=rate + dr)
        rho = (p_rup - base_price) / dr * 0.01

        return GreeksBundle(
            delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho,
            delta_method="bump", gamma_method="bump", vega_method="bump",
        )


# ═══════════════════════════════════════════════════════════════
# Tree Engine wrapper
# ═══════════════════════════════════════════════════════════════

class TreePricingEngine:
    """Tree engine implementing the unified protocol.

    Args:
        method: "crr", "lr", "trinomial", "tian", "jr".
        n_steps: number of tree steps.
        exercise: "european", "american", "bermudan".
    """

    def __init__(
        self,
        method: str = "lr",
        n_steps: int = 200,
        exercise: str = "european",
        exercise_dates: list[int] | None = None,
    ):
        self.method = method
        self.n_steps = n_steps
        self.exercise = exercise
        self.exercise_dates = exercise_dates

    @property
    def engine_type(self) -> str:
        return f"tree_{self.method}"

    def price_vanilla(
        self,
        spot: float,
        strike: float,
        rate: float,
        vol: float,
        T: float,
        is_call: bool = True,
        div_yield: float = 0.0,
    ) -> PricingResult:
        from pricebook.numerical._trees import (
            TreeSolver, TreeMethod, ExerciseType,
        )

        t0 = time.time()

        method_map = {
            "crr": TreeMethod.CRR, "jr": TreeMethod.JR,
            "lr": TreeMethod.LR, "trinomial": TreeMethod.TRINOMIAL,
            "tian": TreeMethod.TIAN,
        }
        exercise_map = {
            "european": ExerciseType.EUROPEAN,
            "american": ExerciseType.AMERICAN,
            "bermudan": ExerciseType.BERMUDAN,
        }

        solver = TreeSolver(
            method=method_map.get(self.method, TreeMethod.LR),
            n_steps=self.n_steps,
            exercise=exercise_map.get(self.exercise, ExerciseType.EUROPEAN),
            exercise_dates=self.exercise_dates,
        )

        result = solver.solve(spot, strike, rate, vol, T, is_call=is_call, div_yield=div_yield)
        elapsed = time.time() - t0

        greeks = GreeksBundle(
            delta=result.delta,
            gamma=result.gamma,
            vega=result.vega or 0.0,
            theta=result.theta,
            delta_method="tree", gamma_method="tree", vega_method="tree_bump",
        )

        conv = ConvergenceInfo(
            n_steps=result.n_steps,
            elapsed_seconds=elapsed,
            engine_type=self.engine_type,
        )

        return PricingResult(
            price=result.price,
            greeks=greeks,
            convergence=conv,
            engine_type=self.engine_type,
            model_name=self.method,
        )


# ═══════════════════════════════════════════════════════════════
# Analytical Engine wrapper
# ═══════════════════════════════════════════════════════════════

class AnalyticalEngine:
    """Black-Scholes / Garman-Kohlhagen analytical engine."""

    @property
    def engine_type(self) -> str:
        return "analytical_bs"

    def price_vanilla(
        self,
        spot: float,
        strike: float,
        rate: float,
        vol: float,
        T: float,
        is_call: bool = True,
        div_yield: float = 0.0,
    ) -> PricingResult:
        from pricebook.models.black76 import (
            OptionType, black76_price, black76_delta, black76_gamma,
            black76_vega, black76_theta,
        )

        t0 = time.time()
        fwd = spot * math.exp((rate - div_yield) * T)
        df = math.exp(-rate * T)
        otype = OptionType.CALL if is_call else OptionType.PUT

        price = black76_price(fwd, strike, vol, T, df, otype)
        delta_raw = black76_delta(fwd, strike, vol, T, df, otype)
        # Convert forward delta to spot delta
        delta = delta_raw * math.exp(-div_yield * T) if T > 0 else delta_raw
        gamma = black76_gamma(fwd, strike, vol, T, df)
        vega = black76_vega(fwd, strike, vol, T, df) * 0.01
        theta = black76_theta(fwd, strike, vol, T, df, otype) / 365.0

        elapsed = time.time() - t0

        greeks = GreeksBundle(
            delta=delta, gamma=gamma, vega=vega, theta=theta,
            delta_method="analytical", gamma_method="analytical",
            vega_method="analytical",
        )

        conv = ConvergenceInfo(
            elapsed_seconds=elapsed,
            engine_type="analytical_bs",
        )

        return PricingResult(
            price=price,
            greeks=greeks,
            convergence=conv,
            engine_type="analytical_bs",
            model_name="black_scholes",
        )
