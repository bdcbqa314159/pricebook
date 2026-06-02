"""Declarative MC configuration and factory.

Encapsulates all MC engine settings in a single config object.
Mode switching = swap config, everything else stays the same.

* :class:`MCConfig` — full MC configuration.
* :class:`VarianceReduction` — variance reduction method enum.
* :class:`GreekMethod` — Greek computation method enum.
* :func:`mc_pricer_from_config` — build pricer from config.
* :func:`preset_configs` — standard configurations for common use cases.

References:
    Glasserman, *Monte Carlo Methods in Financial Engineering*, 2003.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VarianceReduction(Enum):
    """Variance reduction technique."""
    NONE = "none"
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"
    STRATIFIED = "stratified"
    IMPORTANCE = "importance_sampling"
    CONDITIONAL = "conditional_mc"
    SOBOL = "sobol"


class GreekMethod(Enum):
    """Greek computation method."""
    BUMP = "bump"                   # finite difference (any payoff)
    PATHWISE = "pathwise"           # IPA (smooth payoffs only)
    LIKELIHOOD_RATIO = "lr"         # score function (any payoff)
    AUTO = "auto"                   # auto-select based on payoff


class ProcessType(Enum):
    """Stochastic process type."""
    GBM = "gbm"
    HESTON = "heston"
    SABR = "sabr"
    BATES = "bates"
    CIR = "cir"
    HULL_WHITE = "hull_white"
    ROUGH_BERGOMI = "rough_bergomi"
    SLV = "slv"
    VARIANCE_GAMMA = "vg"
    JUMP_DIFFUSION = "jump_diffusion"
    CEV = "cev"
    CORRELATED_GBM = "correlated_gbm"


class Discretisation(Enum):
    """SDE discretisation scheme."""
    EULER = "euler"
    MILSTEIN = "milstein"
    EXACT = "exact"
    QE_HESTON = "qe_heston"        # Andersen quadratic-exponential


@dataclass
class MCConfig:
    """Complete MC engine configuration.

    All settings in one place. Swap this object to switch modes.

    Attributes:
        process: stochastic process type.
        n_paths: number of simulation paths.
        n_steps: time discretisation steps.
        seed: random seed for reproducibility.
        variance_reduction: VR technique.
        greek_method: how to compute Greeks.
        discretisation: SDE stepping scheme.
        quasi_random: use Sobol sequences instead of pseudo-random.
        antithetic: use antithetic variates (ignored if quasi_random).
        process_params: extra parameters for the process (e.g. Heston kappa, theta).
        bump_sizes: custom bump sizes for finite-difference Greeks.
    """
    process: ProcessType = ProcessType.GBM
    n_paths: int = 100_000
    n_steps: int = 100
    seed: int = 42
    variance_reduction: VarianceReduction = VarianceReduction.ANTITHETIC
    greek_method: GreekMethod = GreekMethod.BUMP
    discretisation: Discretisation = Discretisation.EULER
    quasi_random: bool = False
    antithetic: bool = True
    process_params: dict = field(default_factory=dict)
    bump_sizes: dict = field(default_factory=lambda: {
        "spot": 0.005,     # 0.5% of spot
        "vol": 0.01,       # 1 vol point
        "rate": 0.0001,    # 1bp
        "time": 1.0 / 365, # 1 day
    })

    def to_dict(self) -> dict:
        return {
            "process": self.process.value,
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "seed": self.seed,
            "variance_reduction": self.variance_reduction.value,
            "greek_method": self.greek_method.value,
            "discretisation": self.discretisation.value,
            "quasi_random": self.quasi_random,
        }

    def with_overrides(self, **kwargs) -> MCConfig:
        """Create a copy with specific fields overridden."""
        d = {
            "process": self.process, "n_paths": self.n_paths,
            "n_steps": self.n_steps, "seed": self.seed,
            "variance_reduction": self.variance_reduction,
            "greek_method": self.greek_method,
            "discretisation": self.discretisation,
            "quasi_random": self.quasi_random,
            "antithetic": self.antithetic,
            "process_params": dict(self.process_params),
            "bump_sizes": dict(self.bump_sizes),
        }
        d.update(kwargs)
        return MCConfig(**d)


# ═══════════════════════════════════════════════════════════════
# Preset configurations
# ═══════════════════════════════════════════════════════════════

def preset_configs() -> dict[str, MCConfig]:
    """Standard MC configurations for common use cases."""
    return {
        "fast": MCConfig(
            n_paths=10_000, n_steps=50,
            variance_reduction=VarianceReduction.ANTITHETIC,
            greek_method=GreekMethod.BUMP,
        ),
        "production": MCConfig(
            n_paths=500_000, n_steps=200,
            variance_reduction=VarianceReduction.ANTITHETIC,
            greek_method=GreekMethod.AUTO,
        ),
        "high_precision": MCConfig(
            n_paths=1_000_000, n_steps=500,
            variance_reduction=VarianceReduction.ANTITHETIC,
            greek_method=GreekMethod.AUTO,
            quasi_random=True,
        ),
        "heston": MCConfig(
            process=ProcessType.HESTON, n_paths=200_000, n_steps=200,
            discretisation=Discretisation.QE_HESTON,
            variance_reduction=VarianceReduction.ANTITHETIC,
        ),
        "exotic": MCConfig(
            n_paths=200_000, n_steps=252,
            variance_reduction=VarianceReduction.ANTITHETIC,
            greek_method=GreekMethod.BUMP,
        ),
        "xva": MCConfig(
            n_paths=50_000, n_steps=100,
            variance_reduction=VarianceReduction.ANTITHETIC,
            greek_method=GreekMethod.BUMP,
        ),
    }


# ═══════════════════════════════════════════════════════════════
# Factory: build pricer from config
# ═══════════════════════════════════════════════════════════════

def build_process_from_config(
    config: MCConfig,
    spot: float,
    rate: float,
    vol: float,
    div_yield: float = 0.0,
):
    """Build a ProcessSpec from MCConfig.

    Returns a ProcessSpec ready for MCEngine.
    """
    from pricebook.models.mc_processes import (
        GBMProcess, HestonProcess, SABRProcess,
        BatesProcess, CIRProcess, OUProcess,
        CEVProcess,
    )

    mu = rate - div_yield
    params = config.process_params

    if config.process == ProcessType.GBM:
        return GBMProcess(s0=spot, mu=mu, sigma=vol)
    elif config.process == ProcessType.HESTON:
        return HestonProcess(
            s0=spot, mu=mu,
            v0=params.get("v0", vol**2),
            kappa=params.get("kappa", 2.0),
            theta=params.get("theta", vol**2),
            xi=params.get("xi", 0.3),
            rho=params.get("rho", -0.7),
        )
    elif config.process == ProcessType.SABR:
        return SABRProcess(
            f0=spot, alpha=params.get("alpha", vol),
            beta=params.get("beta", 0.5),
            rho=params.get("rho", -0.3),
            nu=params.get("nu", 0.4),
        )
    elif config.process == ProcessType.CEV:
        return CEVProcess(
            s0=spot, mu=mu, sigma=vol,
            beta=params.get("beta", 0.5),
        )
    elif config.process == ProcessType.CIR:
        return CIRProcess(
            x0=spot, kappa=params.get("kappa", 1.0),
            theta=params.get("theta", spot),
            sigma=params.get("sigma", vol),
        )
    else:
        # Default to GBM
        return GBMProcess(s0=spot, mu=mu, sigma=vol)


def mc_pricer_from_config(config: MCConfig):
    """Build an MCPricingEngine from configuration.

    Returns an engine that implements the PricingEngine protocol.
    """
    from pricebook.models.engine_protocol import MCPricingEngine

    return MCPricingEngine(
        n_paths=config.n_paths,
        n_steps=config.n_steps,
        seed=config.seed,
        antithetic=config.antithetic or config.variance_reduction == VarianceReduction.ANTITHETIC,
        process_type=config.process.value,
        greek_method=config.greek_method.value,
        heston_params=config.process_params if config.process == ProcessType.HESTON else {},
    )
