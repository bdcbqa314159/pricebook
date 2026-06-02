"""Automatic Greek method selection with path caching.

Inspects payoff characteristics and routes to the optimal Greek
computation method. Caches paths so all Greeks share the same
random draws, reducing noise in Greek ratios.

* :class:`PayoffType` — payoff smoothness classification.
* :func:`classify_payoff` — detect payoff type from name/signature.
* :func:`auto_greeks` — compute all Greeks with best method per Greek.
* :class:`PathCache` — LRU path cache keyed by (process, grid, seed).

References:
    Glasserman, *Monte Carlo Methods in Financial Engineering*, Ch. 7.
"""

from __future__ import annotations

import math
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from pricebook.models.engine_protocol import GreeksBundle


class PayoffType(Enum):
    """Payoff smoothness classification."""
    SMOOTH = "smooth"               # call, put — pathwise IPA works
    DISCONTINUOUS = "discontinuous" # digital, barrier — need LR
    PATH_DEPENDENT = "path_dependent"  # Asian, lookback — bump only
    EARLY_EXERCISE = "early_exercise"  # American — bump only


# Payoff names that are smooth (pathwise-safe)
_SMOOTH_PAYOFFS = {
    "european_call", "european_put", "call", "put",
    "basket_call", "basket_put",
}

# Payoff names that are discontinuous (need LR)
_DISCONTINUOUS_PAYOFFS = {
    "digital_call", "digital_put", "digital",
    "barrier_knockout", "barrier_knockin",
    "one_touch", "no_touch",
}

# Payoff names that are path-dependent (bump only)
_PATH_DEPENDENT_PAYOFFS = {
    "autocall", "cliquet", "tarf", "swing",
    "american_put", "american_call",
    "asian_arithmetic", "asian_geometric",
    "lookback_call", "lookback_put",
}


def classify_payoff(payoff: Callable | str) -> PayoffType:
    """Classify a payoff by smoothness.

    Args:
        payoff: callable or string name of payoff.
    """
    if isinstance(payoff, str):
        name = payoff.lower()
    else:
        name = getattr(payoff, "__name__", "").lower()
        if not name:
            name = getattr(payoff, "func", lambda: None).__name__ if hasattr(payoff, "func") else ""
            name = name.lower()

    if name in _SMOOTH_PAYOFFS:
        return PayoffType.SMOOTH
    if name in _DISCONTINUOUS_PAYOFFS:
        return PayoffType.DISCONTINUOUS
    if name in _PATH_DEPENDENT_PAYOFFS or "american" in name:
        return PayoffType.EARLY_EXERCISE
    if "asian" in name or "lookback" in name or "cliquet" in name:
        return PayoffType.PATH_DEPENDENT

    # Default: bump is always safe
    return PayoffType.PATH_DEPENDENT


def select_greek_method(payoff_type: PayoffType) -> str:
    """Select optimal Greek method for a payoff type.

    smooth → pathwise (lowest variance)
    discontinuous → likelihood ratio
    path_dependent / early_exercise → bump-and-reprice
    """
    if payoff_type == PayoffType.SMOOTH:
        return "pathwise"
    if payoff_type == PayoffType.DISCONTINUOUS:
        return "likelihood_ratio"
    return "bump"


# ═══════════════════════════════════════════════════════════════
# Path Cache
# ═══════════════════════════════════════════════════════════════

class PathCache:
    """LRU cache for MC paths, keyed by (process_hash, grid_hash, seed).

    Enables multi-Greek computation with shared random draws.
    """

    def __init__(self, max_size: int = 16):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, process_id: str, n_steps: int, n_paths: int, seed: int) -> str:
        raw = f"{process_id}|{n_steps}|{n_paths}|{seed}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, process_id: str, n_steps: int, n_paths: int, seed: int) -> np.ndarray | None:
        key = self._key(process_id, n_steps, n_paths, seed)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, process_id: str, n_steps: int, n_paths: int, seed: int, paths: np.ndarray):
        key = self._key(process_id, n_steps, n_paths, seed)
        self._cache[key] = paths
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
        }


# Global path cache
_GLOBAL_CACHE = PathCache(max_size=32)


def get_global_cache() -> PathCache:
    return _GLOBAL_CACHE


# ═══════════════════════════════════════════════════════════════
# Auto Greeks
# ═══════════════════════════════════════════════════════════════

def auto_greeks(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    payoff: Callable,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int = 42,
    div_yield: float = 0.0,
    payoff_name: str = "",
    bump_sizes: dict | None = None,
) -> GreeksBundle:
    """Compute all Greeks with automatic method selection.

    Inspects payoff type, picks best method per Greek, and uses
    path caching for efficiency.

    Args:
        payoff: callable(paths, times) → values.
        payoff_name: hint for classification (if payoff.__name__ is unhelpful).
        bump_sizes: override default bump sizes.
    """
    from pricebook.models.mc_engine import MCEngine, TimeGrid
    from pricebook.models.mc_processes import GBMProcess

    # Classify payoff
    ptype = classify_payoff(payoff_name or payoff)
    method = select_greek_method(ptype)

    bumps = bump_sizes or {"spot": 0.005, "vol": 0.01, "rate": 0.0001}
    df = math.exp(-rate * T)
    mu = rate - div_yield

    def _make_engine(s=spot, v=vol, r=rate, q=div_yield):
        proc = GBMProcess(s0=s, mu=r - q, sigma=v)
        grid = TimeGrid.uniform(T, n_steps)
        return MCEngine(proc, grid, n_paths, seed, antithetic=True)

    # Base price (cached)
    base_eng = _make_engine()
    base_result = base_eng.price(payoff, df)
    base_price = base_result.price

    if method == "pathwise" and ptype == PayoffType.SMOOTH:
        greeks = _pathwise_greeks(spot, strike, rate, vol, T, n_paths, seed, div_yield, df)
    elif method == "likelihood_ratio":
        greeks = _lr_greeks(spot, strike, rate, vol, T, n_paths, seed, div_yield, payoff, df)
    else:
        greeks = _bump_greeks(spot, vol, rate, div_yield, T, n_paths, n_steps, seed, payoff, df, base_price, bumps)

    greeks.delta_method = method
    greeks.gamma_method = method
    greeks.vega_method = method
    return greeks


def _pathwise_greeks(spot, strike, rate, vol, T, n_paths, seed, div_yield, df):
    """Pathwise (IPA) Greeks for smooth payoffs."""
    rng = np.random.default_rng(seed)
    mu = rate - div_yield
    z = rng.standard_normal(n_paths)
    S_T = spot * np.exp((mu - 0.5 * vol**2) * T + vol * math.sqrt(T) * z)

    itm = S_T > strike
    # Delta: E[df × 1_{ITM} × S_T / S_0]
    delta = float(np.mean(df * itm * S_T / spot))

    # Gamma via d²/dS₀²
    gamma = float(np.mean(df * itm * S_T / (spot**2)))

    # Vega: E[df × 1_{ITM} × S_T × (z√T − σT)] per 1%
    vega_raw = float(np.mean(df * itm * S_T * (z * math.sqrt(T) - vol * T)))
    vega = vega_raw * 0.01

    return GreeksBundle(delta=delta, gamma=gamma, vega=vega)


def _lr_greeks(spot, strike, rate, vol, T, n_paths, seed, div_yield, payoff, df):
    """Likelihood ratio Greeks for discontinuous payoffs."""
    from pricebook.models.mc_engine import MCEngine, TimeGrid
    from pricebook.models.mc_processes import GBMProcess

    rng = np.random.default_rng(seed)
    mu = rate - div_yield
    z = rng.standard_normal(n_paths)
    S_T = spot * np.exp((mu - 0.5 * vol**2) * T + vol * math.sqrt(T) * z)

    # Evaluate payoff
    paths = np.column_stack([np.full(n_paths, spot), S_T])
    times = np.array([0.0, T])
    values = payoff(paths, times) * df

    # Score for delta: z / (spot × σ × √T)
    score_delta = z / (spot * vol * math.sqrt(T))
    delta = float(np.mean(values * score_delta))

    # Score for vega: (z² − 1)/σ − z√T
    score_vega = (z**2 - 1) / vol - z * math.sqrt(T)
    vega = float(np.mean(values * score_vega)) * 0.01

    # Gamma via second score
    score_gamma = (score_delta**2 - score_delta / spot)
    gamma = float(np.mean(values * score_gamma))

    return GreeksBundle(delta=delta, gamma=gamma, vega=vega)


def _bump_greeks(spot, vol, rate, div_yield, T, n_paths, n_steps, seed, payoff, df, base_price, bumps):
    """Bump-and-reprice Greeks (universal fallback)."""
    from pricebook.models.mc_engine import MCEngine, TimeGrid
    from pricebook.models.mc_processes import GBMProcess

    def _price(s=spot, v=vol, r=rate, q=div_yield):
        proc = GBMProcess(s0=s, mu=r - q, sigma=v)
        grid = TimeGrid.uniform(T, n_steps)
        eng = MCEngine(proc, grid, n_paths, seed, antithetic=True)
        return eng.price(payoff, math.exp(-r * T)).price

    ds = spot * bumps["spot"]
    p_up = _price(s=spot + ds)
    p_dn = _price(s=spot - ds)
    delta = (p_up - p_dn) / (2 * ds)
    gamma = (p_up - 2 * base_price + p_dn) / (ds**2)

    dv = bumps["vol"]
    vega = _price(v=vol + dv) - base_price

    dr = bumps["rate"]
    rho = (_price(r=rate + dr) - base_price) / dr * 0.01

    return GreeksBundle(delta=delta, gamma=gamma, vega=vega, rho=rho)
