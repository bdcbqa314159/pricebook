"""Engine comparison and validation.

Price the same instrument via MC, tree, and analytical. Report
convergence, Greek agreement, and compute time.

* :class:`ComparisonResult` — side-by-side engine results.
* :func:`compare_engines` — run all engines on same instrument.
* :func:`validate_greeks` — check Greek consistency across engines.

References:
    Hull, *Options, Futures, and Other Derivatives*, 11th ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pricebook.models.engine_protocol import (
    PricingResult, MCPricingEngine, TreePricingEngine, AnalyticalEngine,
)


@dataclass
class EngineComparison:
    """Single engine result within a comparison."""
    engine_name: str
    price: float
    delta: float
    gamma: float
    vega: float
    elapsed_seconds: float
    std_error: float = 0.0

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class ComparisonResult:
    """Side-by-side comparison of pricing engines."""
    engines: list[EngineComparison]
    price_spread: float         # max - min price
    price_spread_pct: float     # as % of mean
    delta_spread: float
    greeks_consistent: bool     # all within tolerance
    fastest_engine: str
    most_accurate: str          # lowest std error or exact

    def to_dict(self) -> dict:
        return {
            "n_engines": len(self.engines),
            "price_spread": self.price_spread,
            "price_spread_pct": self.price_spread_pct,
            "greeks_consistent": self.greeks_consistent,
            "fastest": self.fastest_engine,
            "most_accurate": self.most_accurate,
            "engines": [e.to_dict() for e in self.engines],
        }


def compare_engines(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    is_call: bool = True,
    div_yield: float = 0.0,
    mc_paths: int = 200_000,
    tree_steps: int = 500,
    tolerance_pct: float = 1.0,
) -> ComparisonResult:
    """Run all engines on the same European vanilla option.

    Args:
        tolerance_pct: max price difference (%) to consider consistent.
    """
    results = []

    # Analytical
    analytical = AnalyticalEngine()
    r_a = analytical.price_vanilla(spot, strike, rate, vol, T, is_call, div_yield)
    results.append(EngineComparison(
        "analytical", r_a.price, r_a.greeks.delta, r_a.greeks.gamma,
        r_a.greeks.vega, r_a.convergence.elapsed_seconds,
    ))

    # Tree (LR — fast convergence)
    tree = TreePricingEngine(method="lr", n_steps=tree_steps)
    r_t = tree.price_vanilla(spot, strike, rate, vol, T, is_call, div_yield)
    results.append(EngineComparison(
        "tree_lr", r_t.price, r_t.greeks.delta, r_t.greeks.gamma,
        r_t.greeks.vega, r_t.convergence.elapsed_seconds,
    ))

    # MC
    mc = MCPricingEngine(n_paths=mc_paths, n_steps=100, antithetic=True)
    r_m = mc.price_vanilla(spot, strike, rate, vol, T, is_call, div_yield)
    results.append(EngineComparison(
        "mc_gbm", r_m.price, r_m.greeks.delta, r_m.greeks.gamma,
        r_m.greeks.vega, r_m.convergence.elapsed_seconds,
        r_m.convergence.std_error,
    ))

    # Analysis
    prices = [r.price for r in results]
    deltas = [r.delta for r in results]
    spread = max(prices) - min(prices)
    mean_price = sum(prices) / len(prices) if prices else 0
    spread_pct = spread / abs(mean_price) * 100 if mean_price != 0 else 0

    delta_spread = max(deltas) - min(deltas)
    greeks_ok = spread_pct < tolerance_pct and delta_spread < 0.05

    times = [r.elapsed_seconds for r in results]
    fastest = results[times.index(min(times))].engine_name

    # Most accurate: analytical is exact, then tree, then MC
    most_accurate = "analytical"

    return ComparisonResult(
        engines=results,
        price_spread=spread,
        price_spread_pct=spread_pct,
        delta_spread=delta_spread,
        greeks_consistent=greeks_ok,
        fastest_engine=fastest,
        most_accurate=most_accurate,
    )


def validate_greeks(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    is_call: bool = True,
    tolerance: float = 0.02,
) -> dict:
    """Check Greek consistency across engines.

    Returns dict with pass/fail per Greek and max deviation.
    """
    analytical = AnalyticalEngine()
    r_a = analytical.price_vanilla(spot, strike, rate, vol, T, is_call)

    tree = TreePricingEngine(method="lr", n_steps=500)
    r_t = tree.price_vanilla(spot, strike, rate, vol, T, is_call)

    checks = {}
    for greek_name in ["delta", "gamma", "vega"]:
        val_a = getattr(r_a.greeks, greek_name)
        val_t = getattr(r_t.greeks, greek_name)
        diff = abs(val_a - val_t)
        scale = abs(val_a) if abs(val_a) > 1e-10 else 1.0
        rel_diff = diff / scale
        checks[greek_name] = {
            "analytical": val_a,
            "tree": val_t,
            "abs_diff": diff,
            "rel_diff": rel_diff,
            "pass": rel_diff < tolerance,
        }

    checks["all_pass"] = all(c["pass"] for c in checks.values() if isinstance(c, dict))
    return checks
