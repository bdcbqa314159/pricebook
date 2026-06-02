"""Unified engine registry: one function, any instrument, best engine.

* :func:`price` — auto-select best engine and price.
* :func:`register_engine` — register a custom engine.
* :func:`list_engines` — list available engines.
* :func:`engine_recommendation` — suggest engine for instrument type.

References:
    Internal pricebook architecture documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pricebook.models.engine_protocol import (
    PricingResult, PricingEngine,
    MCPricingEngine, TreePricingEngine, AnalyticalEngine,
)


class InstrumentType(Enum):
    """Instrument classification for engine selection."""
    EUROPEAN_VANILLA = "european_vanilla"
    AMERICAN_VANILLA = "american_vanilla"
    BERMUDAN = "bermudan"
    BARRIER = "barrier"
    ASIAN = "asian"
    LOOKBACK = "lookback"
    DIGITAL = "digital"
    AUTOCALLABLE = "autocallable"
    CLIQUET = "cliquet"
    CALLABLE_BOND = "callable_bond"
    BERMUDAN_SWAPTION = "bermudan_swaption"
    CDS_SWAPTION = "cds_swaption"
    BASKET = "basket"
    QUANTO = "quanto"


# Engine recommendations per instrument type
_RECOMMENDATIONS: dict[InstrumentType, list[str]] = {
    InstrumentType.EUROPEAN_VANILLA: ["analytical", "tree_lr", "mc"],
    InstrumentType.AMERICAN_VANILLA: ["tree_lr", "tree_crr", "lsm_on_tree"],
    InstrumentType.BERMUDAN: ["tree_trinomial", "lsm_on_tree"],
    InstrumentType.BARRIER: ["tree_adaptive", "mc", "pde"],
    InstrumentType.ASIAN: ["mc", "non_recombining_tree"],
    InstrumentType.LOOKBACK: ["mc"],
    InstrumentType.DIGITAL: ["analytical", "mc_lr"],
    InstrumentType.AUTOCALLABLE: ["mc"],
    InstrumentType.CLIQUET: ["mc"],
    InstrumentType.CALLABLE_BOND: ["tree_hw", "tree_bdt"],
    InstrumentType.BERMUDAN_SWAPTION: ["tree_hw", "lsm_mc"],
    InstrumentType.CDS_SWAPTION: ["tree_hazard", "mc"],
    InstrumentType.BASKET: ["mc_correlated"],
    InstrumentType.QUANTO: ["mc", "analytical_quanto"],
}

# Custom engine registry
_CUSTOM_ENGINES: dict[str, type] = {}


def register_engine(name: str, engine_class: type):
    """Register a custom engine.

    Args:
        name: engine identifier.
        engine_class: class implementing PricingEngine protocol.
    """
    _CUSTOM_ENGINES[name] = engine_class


def list_engines() -> list[str]:
    """List all available engine names."""
    built_in = ["analytical", "tree_lr", "tree_crr", "tree_trinomial",
                "tree_tian", "mc_gbm", "mc_heston", "lsm_on_tree",
                "stoch_vol_tree", "tree_bdt"]
    return built_in + list(_CUSTOM_ENGINES.keys())


def engine_recommendation(instrument_type: InstrumentType) -> list[str]:
    """Suggest engines for an instrument type, ordered by preference."""
    return _RECOMMENDATIONS.get(instrument_type, ["mc"])


def _build_engine(engine_name: str, **kwargs) -> PricingEngine:
    """Build an engine by name."""
    if engine_name in _CUSTOM_ENGINES:
        return _CUSTOM_ENGINES[engine_name](**kwargs)

    if engine_name == "analytical":
        return AnalyticalEngine()
    if engine_name.startswith("tree"):
        method = engine_name.replace("tree_", "") if "_" in engine_name else "lr"
        return TreePricingEngine(
            method=method,
            n_steps=kwargs.get("n_steps", 200),
            exercise=kwargs.get("exercise", "european"),
        )
    if engine_name.startswith("mc"):
        process = engine_name.replace("mc_", "") if "_" in engine_name else "gbm"
        return MCPricingEngine(
            n_paths=kwargs.get("n_paths", 100_000),
            n_steps=kwargs.get("n_steps", 100),
            process_type=process,
            antithetic=True,
        )

    return AnalyticalEngine()


def price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    is_call: bool = True,
    div_yield: float = 0.0,
    engine: str = "auto",
    instrument_type: InstrumentType = InstrumentType.EUROPEAN_VANILLA,
    **kwargs,
) -> PricingResult:
    """Price with auto-selected or specified engine.

    Args:
        engine: "auto" (picks best), or specific engine name.
        instrument_type: used for auto-selection.
        **kwargs: passed to engine constructor.

    Returns:
        PricingResult with price, Greeks, and convergence info.
    """
    if engine == "auto":
        candidates = engine_recommendation(instrument_type)
        engine_name = candidates[0]
    else:
        engine_name = engine

    # Handle exercise type
    if instrument_type == InstrumentType.AMERICAN_VANILLA:
        kwargs.setdefault("exercise", "american")

    eng = _build_engine(engine_name, **kwargs)
    return eng.price_vanilla(spot, strike, rate, vol, T, is_call, div_yield)
