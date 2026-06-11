"""Numerical method recommendation map.

Given instrument features, recommend the best SDE scheme, PDE method,
Fourier approach, or hybrid combination.

* :func:`recommend` — recommend numerical method for an instrument.
* :func:`compare_methods` — price via all applicable methods, report.

References:
    Glasserman, *Monte Carlo Methods in Financial Engineering*, 2003.
    Duffy, *Finite Difference Methods in Financial Engineering*, 2006.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Feature(Enum):
    """Instrument features that affect method selection."""
    EUROPEAN = "european"
    AMERICAN = "american"
    BERMUDAN = "bermudan"
    BARRIER = "barrier"
    ASIAN = "asian"
    LOOKBACK = "lookback"
    DIGITAL = "digital"
    PATH_DEPENDENT = "path_dependent"
    EARLY_EXERCISE = "early_exercise"
    STOCHASTIC_VOL = "stochastic_vol"
    LOCAL_VOL = "local_vol"
    JUMPS = "jumps"
    MULTI_ASSET = "multi_asset"
    HIGH_DIMENSION = "high_dimension"  # > 3 factors


@dataclass
class MethodRecommendation:
    """Recommended numerical method with reasoning."""
    primary: str                # recommended method
    alternatives: list[str]     # fallback methods
    sde_scheme: str             # recommended SDE discretisation
    pde_method: str             # recommended PDE scheme
    fourier_method: str         # recommended Fourier approach
    reason: str                 # human-readable explanation
    features: list[str]         # detected features

    def to_dict(self) -> dict:
        return dict(vars(self))


# ═══════════════════════════════════════════════════════════════
# Method selection rules
# ═══════════════════════════════════════════════════════════════

_RULES = {
    # (feature_set) → (primary, alternatives, sde, pde, fourier, reason)

    frozenset(): (
        "analytical", ["tree_lr", "cos"], "exact_gbm", "none", "cos",
        "European vanilla: analytical Black-Scholes is exact"
    ),
    frozenset([Feature.EUROPEAN]): (
        "analytical", ["cos", "tree_lr"], "exact_gbm", "none", "cos",
        "European vanilla: analytical or COS (exponential convergence)"
    ),
    frozenset([Feature.AMERICAN]): (
        "tree_lr", ["pde_cn", "lsm"], "exact_gbm", "cn_rannacher", "cos_bermudan",
        "American: LR tree (fast, accurate), PDE with Rannacher, or LSM"
    ),
    frozenset([Feature.BARRIER]): (
        "pde_cn", ["mc_bridge", "tree_adaptive"], "gbm_bridge", "cn_sinh", "none",
        "Barrier: PDE with sinh grid near barrier, or MC with Brownian bridge"
    ),
    frozenset([Feature.ASIAN]): (
        "mc_antithetic", ["mc_control_variate"], "euler", "none", "none",
        "Asian: MC with control variate (geometric Asian as control)"
    ),
    frozenset([Feature.STOCHASTIC_VOL]): (
        "cos_heston", ["pde_adi", "mc_qe"], "qe_heston", "craig_sneyd", "cos",
        "Stochastic vol: COS with Heston CF (fast), ADI PDE, or QE MC"
    ),
    frozenset([Feature.STOCHASTIC_VOL, Feature.AMERICAN]): (
        "mc_lsm", ["pde_adi_penalty"], "qe_heston", "adi_penalty", "cos_bermudan",
        "American + stoch vol: LSM with QE, or ADI with penalty"
    ),
    frozenset([Feature.JUMPS]): (
        "cos", ["fft", "mc_jump"], "jump_diffusion", "pide_splitting", "cos",
        "Jumps: COS method (handles any CF), PIDE via operator splitting"
    ),
    frozenset([Feature.LOCAL_VOL]): (
        "pde_local_vol", ["mc_local_vol"], "euler_local", "cn_local_vol", "none",
        "Local vol: PDE with σ(S,t) coefficients, or MC with local vol"
    ),
    frozenset([Feature.MULTI_ASSET]): (
        "mc_correlated", ["pde_adi_2d", "fft_2d"], "correlated_gbm", "adi_2d", "fft_2d",
        "Multi-asset: MC with Cholesky (scalable), 2D ADI, or 2D FFT"
    ),
    frozenset([Feature.HIGH_DIMENSION]): (
        "mc_sobol", ["mc_mlmc"], "euler", "none", "none",
        "High-dim: MC with Sobol QMC (only feasible method for d>3)"
    ),
    frozenset([Feature.DIGITAL]): (
        "cos", ["mc_lr", "pde_cn"], "exact_gbm", "cn", "cos",
        "Digital: COS (smooth CF), or MC with likelihood ratio Greeks"
    ),
    frozenset([Feature.PATH_DEPENDENT]): (
        "mc_antithetic", ["mc_mlmc"], "euler", "none", "none",
        "Path-dependent: MC is the natural choice"
    ),
}


def recommend(features: list[Feature] | list[str]) -> MethodRecommendation:
    """Recommend numerical method for given instrument features.

    Args:
        features: list of Feature enums or string names.

    Returns:
        MethodRecommendation with primary method, alternatives, and reasoning.
    """
    # Convert strings to Feature enums
    feat_set = set()
    for f in features:
        if isinstance(f, str):
            try:
                feat_set.add(Feature(f))
            except ValueError:
                pass
        else:
            feat_set.add(f)

    # Try exact match first
    frozen = frozenset(feat_set)
    if frozen in _RULES:
        primary, alts, sde, pde, fourier, reason = _RULES[frozen]
    else:
        # Find best partial match (most features matched)
        best_match = None
        best_count = -1
        for rule_features, rule_data in _RULES.items():
            if rule_features.issubset(frozen):
                count = len(rule_features)
                if count > best_count:
                    best_count = count
                    best_match = rule_data

        if best_match:
            primary, alts, sde, pde, fourier, reason = best_match
        else:
            primary, alts, sde, pde, fourier, reason = (
                "mc_antithetic", ["pde_cn", "tree_lr"], "euler", "cn", "cos",
                "Default: MC (universal), PDE (1D/2D), or tree"
            )

    return MethodRecommendation(
        primary=primary,
        alternatives=alts,
        sde_scheme=sde,
        pde_method=pde,
        fourier_method=fourier,
        reason=reason,
        features=[f.value if isinstance(f, Feature) else str(f) for f in feat_set],
    )


def compare_methods(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    is_call: bool = True,
    features: list[Feature] | None = None,
) -> dict:
    """Price via all applicable methods, report agreement.

    Uses the recommendation to select methods, then runs each.
    """
    rec = recommend(features or [Feature.EUROPEAN])

    results = {}

    # Analytical (always try)
    try:
        from pricebook.models.engine_protocol import AnalyticalEngine
        r = AnalyticalEngine().price_vanilla(spot, strike, rate, vol, T, is_call)
        results["analytical"] = r.price
    except Exception:
        pass

    # COS
    try:
        from pricebook.models.cos_method import cos_price, bs_char_func
        from pricebook.models.black76 import OptionType
        cf = bs_char_func(rate, 0.0, vol, T)
        otype = OptionType.CALL if is_call else OptionType.PUT
        results["cos"] = cos_price(cf, spot, strike, rate, T, otype)
    except Exception:
        pass

    # PDE
    try:
        from pricebook.models.pde_protocol import pde_price
        r = pde_price(spot, strike, vol, rate, T, is_call)
        results["pde"] = r.price
    except Exception:
        pass

    # Tree
    try:
        from pricebook.models.engine_protocol import TreePricingEngine
        r = TreePricingEngine(n_steps=200).price_vanilla(spot, strike, rate, vol, T, is_call)
        results["tree"] = r.price
    except Exception:
        pass

    prices = list(results.values())
    spread = max(prices) - min(prices) if prices else 0
    mean = sum(prices) / len(prices) if prices else 0

    return {
        "recommendation": rec.to_dict(),
        "prices": results,
        "spread": spread,
        "spread_pct": spread / abs(mean) * 100 if mean != 0 else 0,
        "consistent": spread / abs(mean) < 0.02 if mean != 0 else True,
    }
