"""Vol model comparison: SABR vs SVI vs LV vs Heston on same data.

* :func:`compare_models` — price same option under 4 models, report dispersion.
* :func:`model_risk_quantification` — max spread, mean spread, worst model.
* :func:`model_selection_guide` — recommend model per product type.

References:
    Gatheral, *The Volatility Surface*, Wiley, 2006.
    Rebonato, *Volatility and Correlation*, Wiley, 2004.
    Bergomi, *Stochastic Volatility Modeling*, CRC Press, 2016.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import black76_price, OptionType
from pricebook.sabr import sabr_implied_vol


@dataclass
class ModelPriceEntry:
    """Price from one model."""
    model_name: str
    price: float
    implied_vol: float


@dataclass
class ModelComparisonResult:
    """Comparison across models."""
    entries: list[ModelPriceEntry]
    mean_price: float
    price_spread: float         # max − min
    spread_pct: float           # spread / mean × 100
    n_models: int
    best_model: str             # closest to mean (proxy for "most robust")


def compare_models(
    forward: float,
    strike: float,
    T: float,
    df: float,
    atm_vol: float,
    sabr_alpha: float = 0.0,
    sabr_rho: float = 0.0,
    sabr_nu: float = 0.0,
    sabr_beta: float = 1.0,
    svi_a: float = 0.0,
    svi_b: float = 0.0,
    svi_rho_svi: float = 0.0,
    svi_m: float = 0.0,
    svi_sigma: float = 0.0,
    local_vol: float | None = None,
    heston_v0: float = 0.0,
    heston_kappa: float = 0.0,
    heston_theta: float = 0.0,
    heston_xi: float = 0.0,
    heston_rho: float = 0.0,
    is_call: bool = True,
) -> ModelComparisonResult:
    """Price the same option under up to 4 models, report dispersion.

    Models included only if their parameters are provided (non-zero).
    Always includes flat-vol (Black-76) as the baseline.
    """
    opt = OptionType.CALL if is_call else OptionType.PUT
    entries = []

    # 1. Black-76 (flat vol)
    bs_price = black76_price(forward, strike, atm_vol, T, df, opt)
    entries.append(ModelPriceEntry("black76", float(bs_price), atm_vol))

    # 2. SABR
    if sabr_alpha > 0 and sabr_nu > 0:
        try:
            sabr_vol = sabr_implied_vol(forward, strike, T, sabr_alpha,
                                          sabr_beta, sabr_rho, sabr_nu)
            sabr_price = black76_price(forward, strike, sabr_vol, T, df, opt)
            entries.append(ModelPriceEntry("sabr", float(sabr_price), float(sabr_vol)))
        except (ValueError, ZeroDivisionError):
            pass

    # 3. SVI
    if svi_b > 0 and svi_sigma > 0:
        k = math.log(strike / forward)
        w = svi_a + svi_b * (svi_rho_svi * (k - svi_m)
                              + math.sqrt((k - svi_m)**2 + svi_sigma**2))
        svi_vol = math.sqrt(max(w / T, 1e-10))
        svi_price = black76_price(forward, strike, svi_vol, T, df, opt)
        entries.append(ModelPriceEntry("svi", float(svi_price), float(svi_vol)))

    # 4. Local vol (if provided as scalar proxy)
    if local_vol is not None and local_vol > 0:
        lv_price = black76_price(forward, strike, local_vol, T, df, opt)
        entries.append(ModelPriceEntry("local_vol", float(lv_price), float(local_vol)))

    # 5. Heston (via MC-free approximation: use Heston ATM vol proxy)
    if heston_v0 > 0 and heston_kappa > 0:
        # Heston ATM vol ≈ √(θ + (v₀ − θ)×(1−e^{−κT})/(κT))
        if heston_kappa * T > 1e-6:
            e = math.exp(-heston_kappa * T)
            heston_avg_var = heston_theta + (heston_v0 - heston_theta) * (1 - e) / (heston_kappa * T)
        else:
            heston_avg_var = heston_v0
        heston_vol = math.sqrt(max(heston_avg_var, 1e-10))
        heston_price = black76_price(forward, strike, heston_vol, T, df, opt)
        entries.append(ModelPriceEntry("heston", float(heston_price), float(heston_vol)))

    prices = [e.price for e in entries]
    mean_p = float(np.mean(prices))
    spread = float(max(prices) - min(prices))
    spread_pct = spread / max(mean_p, 1e-10) * 100

    # Best = closest to mean
    best = min(entries, key=lambda e: abs(e.price - mean_p))

    return ModelComparisonResult(
        entries=entries,
        mean_price=mean_p,
        price_spread=spread,
        spread_pct=float(spread_pct),
        n_models=len(entries),
        best_model=best.model_name,
    )


@dataclass
class ModelRiskResult:
    """Model risk quantification across a strike grid."""
    max_spread_pct: float
    mean_spread_pct: float
    worst_strike: float
    worst_spread_pct: float
    n_strikes: int


def model_risk_quantification(
    forward: float,
    T: float,
    df: float,
    strikes: list[float],
    atm_vol: float,
    sabr_params: tuple[float, float, float, float] | None = None,
    svi_params: tuple[float, float, float, float, float] | None = None,
    heston_params: tuple[float, float, float, float, float] | None = None,
) -> ModelRiskResult:
    """Model risk across a strike grid.

    Args:
        sabr_params: (alpha, beta, rho, nu) or None.
        svi_params: (a, b, rho, m, sigma) or None.
        heston_params: (v0, kappa, theta, xi, rho) or None.
    """
    spreads = []
    worst_k = strikes[0]
    worst_sp = 0.0

    for K in strikes:
        kw = dict(forward=forward, strike=K, T=T, df=df, atm_vol=atm_vol)
        if sabr_params:
            kw.update(sabr_alpha=sabr_params[0], sabr_beta=sabr_params[1],
                      sabr_rho=sabr_params[2], sabr_nu=sabr_params[3])
        if svi_params:
            kw.update(svi_a=svi_params[0], svi_b=svi_params[1],
                      svi_rho_svi=svi_params[2], svi_m=svi_params[3],
                      svi_sigma=svi_params[4])
        if heston_params:
            kw.update(heston_v0=heston_params[0], heston_kappa=heston_params[1],
                      heston_theta=heston_params[2], heston_xi=heston_params[3],
                      heston_rho=heston_params[4])

        result = compare_models(**kw)
        spreads.append(result.spread_pct)
        if result.spread_pct > worst_sp:
            worst_sp = result.spread_pct
            worst_k = K

    return ModelRiskResult(
        max_spread_pct=float(max(spreads)),
        mean_spread_pct=float(np.mean(spreads)),
        worst_strike=float(worst_k),
        worst_spread_pct=float(worst_sp),
        n_strikes=len(strikes),
    )


@dataclass
class ModelGuideResult:
    """Model selection recommendation."""
    product: str
    recommended: str
    reason: str
    alternatives: list[str]


def model_selection_guide(product_type: str) -> ModelGuideResult:
    """Recommend best vol model per product type.

    Based on industry practice and mathematical properties.
    """
    guides = {
        "vanilla_equity": ModelGuideResult(
            "vanilla_equity", "sabr", "SABR captures equity skew well with β=1",
            ["svi", "local_vol"]),
        "vanilla_fx": ModelGuideResult(
            "vanilla_fx", "sabr", "SABR with β=0.5 standard for FX smile",
            ["vanna_volga", "local_vol"]),
        "vanilla_rates": ModelGuideResult(
            "vanilla_rates", "sabr", "SABR with shifted strike for negative rates",
            ["normal_sabr", "local_vol"]),
        "barrier": ModelGuideResult(
            "barrier", "local_vol", "LV critical for barrier pricing (forward smile)",
            ["slv", "heston"]),
        "variance_swap": ModelGuideResult(
            "variance_swap", "model_free", "Demeterfi replication is model-free",
            ["heston", "local_vol"]),
        "autocallable": ModelGuideResult(
            "autocallable", "slv", "SLV needed for forward smile + vol dynamics",
            ["local_vol", "heston"]),
        "basket": ModelGuideResult(
            "basket", "local_vol", "Multi-asset LV with correlation",
            ["heston", "sabr"]),
        "cliquet": ModelGuideResult(
            "cliquet", "heston", "Heston captures vol dynamics (forward-starting)",
            ["slv", "rough_vol"]),
    }
    return guides.get(product_type, ModelGuideResult(
        product_type, "sabr", "SABR is the default for most vanilla products",
        ["local_vol", "heston"]))
