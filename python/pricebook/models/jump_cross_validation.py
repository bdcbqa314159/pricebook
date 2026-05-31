"""Cross-validation framework: COS vs MC vs FFT for all jump models.

Systematic comparison of pricing methods to verify characteristic function
implementations are correct and consistent.

    from pricebook.models.jump_cross_validation import (
        cross_validate_model, cross_validate_all,
    )

    result = cross_validate_model("merton", spot=100, rate=0.05, T=1.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.black76 import OptionType
from pricebook.models.cos_method import cos_price
from pricebook.models.char_func_protocol import (
    merton_char_func, vg_char_func, kou_char_func, bates_char_func,
)
from pricebook.models.levy_processes import nig_char_func, cgmy_char_func


# ═══════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════

@dataclass
class StrikeResult:
    """Per-strike cross-validation result."""
    strike: float
    moneyness: float
    cos_price: float
    mc_price: float
    fft_price: float | None
    cos_mc_diff: float
    cos_mc_pct: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CrossValResult:
    """Cross-validation result for a single model."""
    model_name: str
    spot: float
    rate: float
    T: float
    strikes: list[StrikeResult]
    max_cos_mc_pct: float
    mean_cos_mc_pct: float
    cos_fft_max_diff: float | None

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "max_cos_mc_pct": self.max_cos_mc_pct,
            "mean_cos_mc_pct": self.mean_cos_mc_pct,
            "n_strikes": len(self.strikes),
        }


# ═══════════════════════════════════════════════════════════════
# Default model parameters
# ═══════════════════════════════════════════════════════════════

_DEFAULT_PARAMS = {
    "merton": {"rate": 0.05, "sigma": 0.20, "lam": 1.0, "mu_j": -0.10, "sigma_j": 0.15},
    "vg": {"sigma": 0.20, "nu": 0.25, "theta": -0.14, "rate": 0.05},
    "kou": {"rate": 0.05, "sigma": 0.20, "lam": 1.0, "p": 0.6, "eta1": 8.0, "eta2": 5.0},
    "nig": {"alpha": 15.0, "beta": -5.0, "delta": 0.5, "rate": 0.05},
    "cgmy": {"C": 1.0, "G": 5.0, "M": 10.0, "Y": 0.5, "rate": 0.05},
    "bates": {"rate": 0.05, "v0": 0.04, "kappa": 1.5, "theta": 0.04,
              "xi": 0.3, "rho": -0.7, "lam": 0.5, "mu_j": -0.05, "sigma_j": 0.1},
}


def _build_char_func(model_name: str, T: float, params: dict | None = None):
    """Build characteristic function for a model."""
    p = params or _DEFAULT_PARAMS[model_name]

    if model_name == "merton":
        return merton_char_func(p["rate"], p["sigma"], p["lam"], p["mu_j"], p["sigma_j"], T)
    elif model_name == "vg":
        return vg_char_func(p["rate"], p["sigma"], p["nu"], p["theta"], T)
    elif model_name == "kou":
        return kou_char_func(p["rate"], p["sigma"], T, p["lam"], p["p"], p["eta1"], p["eta2"])
    elif model_name == "nig":
        return nig_char_func(p["rate"], p["alpha"], p["beta"], p["delta"], T)
    elif model_name == "cgmy":
        return cgmy_char_func(p["rate"], p["C"], p["G"], p["M"], p["Y"], T)
    elif model_name == "bates":
        return bates_char_func(p["rate"], p["v0"], p["kappa"], p["theta"],
                                p["xi"], p["rho"], p["lam"], p["mu_j"], p["sigma_j"], T)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _mc_price_model(model_name: str, spot: float, strike: float,
                     rate: float, T: float, n_paths: int, seed: int,
                     params: dict | None = None) -> float:
    """Price a call option via MC simulation for a given model."""
    p = params or _DEFAULT_PARAMS.get(model_name, {})

    if model_name == "merton":
        from pricebook.models.jump_process import MertonJumpDiffusion
        mjd = MertonJumpDiffusion(p["rate"], p["sigma"], p["lam"], p["mu_j"], p["sigma_j"])
        st = mjd.terminal(spot, T, n_paths, seed)
    elif model_name == "vg":
        from pricebook.models.jump_process import VarianceGammaProcess
        vg = VarianceGammaProcess(p["sigma"], p["theta"], p["nu"])
        st = vg.terminal(spot, p["rate"], T, n_paths, seed)
    elif model_name == "nig":
        from pricebook.models.levy_processes import NIGProcess
        nig = NIGProcess(p["alpha"], p["beta"], p["delta"])
        st = nig.terminal(spot, p["rate"], T, n_paths, seed)
    elif model_name == "cgmy":
        from pricebook.models.levy_processes import CGMYProcess
        cgmy = CGMYProcess(p["C"], p["G"], p["M"], p["Y"])
        st = cgmy.terminal(spot, p["rate"], T, n_paths, seed)
    elif model_name == "kou":
        # Kou MC via compound Poisson with double-exponential jumps
        rng = np.random.default_rng(seed)
        sigma, lam = p["sigma"], p["lam"]
        p_up, eta1, eta2 = p["p"], p["eta1"], p["eta2"]
        zeta = p_up * eta1 / (eta1 - 1) + (1 - p_up) * eta2 / (eta2 + 1) - 1
        drift = (rate - lam * zeta - 0.5 * sigma**2) * T
        diffusion = sigma * math.sqrt(T) * rng.standard_normal(n_paths)
        N = rng.poisson(lam * T, size=n_paths)
        N_max = max(int(N.max()), 1)
        # Double-exponential jumps
        U = rng.random((n_paths, N_max))
        E1 = rng.exponential(1.0 / eta1, (n_paths, N_max))  # up
        E2 = rng.exponential(1.0 / eta2, (n_paths, N_max))  # down
        jumps = np.where(U < p_up, E1, -E2)
        mask = np.arange(N_max) < N[:, None]
        jump_sum = (jumps * mask).sum(axis=1)
        st = spot * np.exp(drift + diffusion + jump_sum)
    elif model_name == "bates":
        # Bates via mc_migrate
        from pricebook.models.mc_migrate import bates_paths
        S, _v = bates_paths(
            spot=spot, v0=p["v0"], rate=p["rate"],
            kappa=p["kappa"], theta=p["theta"], xi=p["xi"], rho=p["rho"],
            jump_intensity=p["lam"], jump_mean=p["mu_j"], jump_vol=p["sigma_j"],
            T=T, n_steps=max(int(T * 252), 50), n_paths=n_paths, seed=seed,
        )
        st = S[:, -1]
    else:
        return float("nan")

    return math.exp(-rate * T) * float(np.maximum(st - strike, 0).mean())


# ═══════════════════════════════════════════════════════════════
# Core cross-validation
# ═══════════════════════════════════════════════════════════════

def cross_validate_model(
    model_name: str,
    spot: float = 100.0,
    rate: float = 0.05,
    T: float = 1.0,
    moneyness_range: list[float] | None = None,
    n_mc_paths: int = 200_000,
    seed: int = 42,
    params: dict | None = None,
) -> CrossValResult:
    """Cross-validate COS vs MC pricing for a jump model.

    Args:
        model_name: one of "merton", "vg", "kou", "nig", "cgmy", "bates".
        moneyness_range: list of K/S ratios (default: 0.85 to 1.15).
        n_mc_paths: number of MC paths.
        params: override default model parameters.

    Returns:
        CrossValResult with per-strike COS vs MC comparison.
    """
    if moneyness_range is None:
        moneyness_range = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]

    phi = _build_char_func(model_name, T, params)
    strikes = [spot * m for m in moneyness_range]

    # FFT pricing (if available)
    fft_prices = {}
    try:
        from pricebook.models.fft_pricing import carr_madan_fft
        fft_result = carr_madan_fft(phi, spot, rate, T)
        # Interpolate FFT prices at our strikes
        for k in strikes:
            idx = np.searchsorted(fft_result.strikes, k)
            if 0 < idx < len(fft_result.strikes):
                # Linear interpolation
                k1, k2 = fft_result.strikes[idx-1], fft_result.strikes[idx]
                p1, p2 = fft_result.prices[idx-1], fft_result.prices[idx]
                fft_prices[k] = p1 + (p2 - p1) * (k - k1) / (k2 - k1)
    except Exception:
        pass

    results = []
    # All models now support MC
    has_mc = model_name in ("merton", "vg", "nig", "kou", "cgmy", "bates")

    for i, k in enumerate(strikes):
        cos_p = cos_price(phi, spot, k, rate, T, OptionType.CALL, N=256)

        if has_mc:
            mc_p = _mc_price_model(model_name, spot, k, rate, T, n_mc_paths, seed,
                                    params or _DEFAULT_PARAMS.get(model_name))
        else:
            mc_p = float("nan")

        fft_p = fft_prices.get(k)

        diff = abs(cos_p - mc_p) if not math.isnan(mc_p) else 0
        pct = diff / max(cos_p, 1e-10) * 100 if not math.isnan(mc_p) and cos_p > 0.01 else 0

        results.append(StrikeResult(
            strike=k, moneyness=moneyness_range[i],
            cos_price=cos_p, mc_price=mc_p, fft_price=fft_p,
            cos_mc_diff=diff, cos_mc_pct=pct,
        ))

    valid_pcts = [r.cos_mc_pct for r in results if not math.isnan(r.mc_price)]
    max_pct = max(valid_pcts) if valid_pcts else 0.0
    mean_pct = sum(valid_pcts) / len(valid_pcts) if valid_pcts else 0.0

    cos_fft_max = None
    if fft_prices:
        diffs = [abs(r.cos_price - r.fft_price) for r in results if r.fft_price is not None]
        cos_fft_max = max(diffs) if diffs else None

    return CrossValResult(
        model_name=model_name, spot=spot, rate=rate, T=T,
        strikes=results, max_cos_mc_pct=max_pct, mean_cos_mc_pct=mean_pct,
        cos_fft_max_diff=cos_fft_max,
    )


def cross_validate_all(
    spot: float = 100.0,
    rate: float = 0.05,
    T: float = 1.0,
    models: list[str] | None = None,
    n_mc_paths: int = 100_000,
    seed: int = 42,
) -> list[CrossValResult]:
    """Cross-validate all jump models.

    Returns list of CrossValResult sorted by mean COS/MC difference.
    """
    if models is None:
        models = ["merton", "vg", "kou", "nig", "cgmy", "bates"]

    results = []
    for m in models:
        r = cross_validate_model(m, spot, rate, T, n_mc_paths=n_mc_paths, seed=seed)
        results.append(r)

    return sorted(results, key=lambda r: r.mean_cos_mc_pct)
