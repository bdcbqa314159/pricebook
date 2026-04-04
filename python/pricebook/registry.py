"""
Registry for numerical methods.

Factory functions with string-based lookup for swapping implementations.

    from pricebook.registry import get_solver, get_tree_pricer

    solver = get_solver("newton")
    result = solver(f, fprime, x0=0.5)

    tree_price = get_tree_pricer("trinomial")(spot=100, strike=105, ...)
"""

from __future__ import annotations

from pricebook import solvers
from pricebook import quadrature
from pricebook import binomial_tree
from pricebook import binomial_jr_lr
from pricebook import trinomial_tree
from pricebook import finite_difference
from pricebook import adi
from pricebook import mc_pricer
from pricebook import mc_advanced
from pricebook import lsm


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

_SOLVERS = {
    "brent": solvers.brentq,
    "brentq": solvers.brentq,
    "newton": solvers.newton,
    "secant": solvers.secant,
    "halley": solvers.halley,
    "itp": solvers.itp,
}


def get_solver(name: str):
    """Get a root-finding solver by name."""
    if name not in _SOLVERS:
        raise KeyError(f"Unknown solver '{name}'. Available: {list(_SOLVERS.keys())}")
    return _SOLVERS[name]


def list_solvers() -> list[str]:
    return list(_SOLVERS.keys())


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

_INTEGRATORS = {
    "gauss_legendre": quadrature.gauss_legendre,
    "gauss_laguerre": quadrature.gauss_laguerre,
    "gauss_hermite": quadrature.gauss_hermite,
    "adaptive_simpson": quadrature.adaptive_simpson,
}


def get_integrator(name: str):
    """Get a numerical integrator by name."""
    if name not in _INTEGRATORS:
        raise KeyError(f"Unknown integrator '{name}'. Available: {list(_INTEGRATORS.keys())}")
    return _INTEGRATORS[name]


def list_integrators() -> list[str]:
    return list(_INTEGRATORS.keys())


# ---------------------------------------------------------------------------
# Tree pricers
# ---------------------------------------------------------------------------

_TREE_EUROPEAN = {
    "binomial": binomial_tree.binomial_european,
    "crr": binomial_tree.binomial_european,
    "jr": binomial_jr_lr.jr_european,
    "jarrow_rudd": binomial_jr_lr.jr_european,
    "lr": binomial_jr_lr.lr_european,
    "leisen_reimer": binomial_jr_lr.lr_european,
    "trinomial": trinomial_tree.trinomial_european,
}

_TREE_AMERICAN = {
    "binomial": binomial_tree.binomial_american,
    "crr": binomial_tree.binomial_american,
    "jr": binomial_jr_lr.jr_american,
    "jarrow_rudd": binomial_jr_lr.jr_american,
    "lr": binomial_jr_lr.lr_american,
    "leisen_reimer": binomial_jr_lr.lr_american,
    "trinomial": trinomial_tree.trinomial_american,
}


def get_tree_european(name: str):
    """Get a European tree pricer by name."""
    if name not in _TREE_EUROPEAN:
        raise KeyError(f"Unknown tree '{name}'. Available: {list(_TREE_EUROPEAN.keys())}")
    return _TREE_EUROPEAN[name]


def get_tree_american(name: str):
    """Get an American tree pricer by name."""
    if name not in _TREE_AMERICAN:
        raise KeyError(f"Unknown tree '{name}'. Available: {list(_TREE_AMERICAN.keys())}")
    return _TREE_AMERICAN[name]


# ---------------------------------------------------------------------------
# PDE pricers
# ---------------------------------------------------------------------------

_PDE = {
    "european": finite_difference.fd_european,
    "american": finite_difference.fd_american,
    "barrier_knockout": finite_difference.fd_barrier_knockout,
    "barrier_knockin": finite_difference.fd_barrier_knockin,
    "heston": adi.heston_pde,
}


def get_pde_pricer(name: str):
    """Get a PDE pricer by name."""
    if name not in _PDE:
        raise KeyError(f"Unknown PDE pricer '{name}'. Available: {list(_PDE.keys())}")
    return _PDE[name]


# ---------------------------------------------------------------------------
# MC pricers
# ---------------------------------------------------------------------------

_MC = {
    "european": mc_pricer.mc_european,
    "stratified": mc_advanced.mc_stratified,
    "importance": mc_advanced.mc_importance,
    "mlmc": mc_advanced.mc_mlmc,
    "lsm": lsm.lsm_american,
}


def get_mc_pricer(name: str):
    """Get a Monte Carlo pricer by name."""
    if name not in _MC:
        raise KeyError(f"Unknown MC pricer '{name}'. Available: {list(_MC.keys())}")
    return _MC[name]


def list_mc_pricers() -> list[str]:
    return list(_MC.keys())


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

from functools import partial  # noqa: E402
from pricebook import optimization  # noqa: E402

_OPTIMIZERS = {
    "nelder_mead": partial(optimization.minimize, method="nelder_mead"),
    "bfgs": partial(optimization.minimize, method="bfgs"),
    "l_bfgs_b": partial(optimization.minimize, method="l_bfgs_b"),
    "differential_evolution": partial(optimization.minimize, method="differential_evolution"),
    "basin_hopping": partial(optimization.minimize, method="basin_hopping"),
}


def get_optimizer(name: str):
    """Get an optimizer by name. Returns a callable(objective, x0, **kwargs) → OptimizerResult."""
    if name not in _OPTIMIZERS:
        raise KeyError(f"Unknown optimizer '{name}'. Available: {list(_OPTIMIZERS.keys())}")
    return _OPTIMIZERS[name]


def list_optimizers() -> list[str]:
    return list(_OPTIMIZERS.keys())


# ---------------------------------------------------------------------------
# ODE solvers
# ---------------------------------------------------------------------------

from pricebook import ode  # noqa: E402
from pricebook import cos_method  # noqa: E402
from pricebook import aad_pricing  # noqa: E402
from pricebook import risk  # noqa: E402

_ODE_SOLVERS = {
    "rk4": ode.rk4,
    "rk45": ode.rk45,
    "bdf": ode.bdf,
}


def get_ode_solver(name: str):
    """Get an ODE solver by name."""
    if name not in _ODE_SOLVERS:
        raise KeyError(f"Unknown ODE solver '{name}'. Available: {list(_ODE_SOLVERS.keys())}")
    return _ODE_SOLVERS[name]


def list_ode_solvers() -> list[str]:
    return list(_ODE_SOLVERS.keys())


# ---------------------------------------------------------------------------
# Spectral / COS pricers
# ---------------------------------------------------------------------------

_PRICERS = {
    "cos_bs": lambda **kw: cos_method.cos_price(
        cos_method.bs_char_func(kw["rate"], kw.get("div_yield", 0.0), kw["vol"], kw["T"]),
        kw["spot"], kw["strike"], kw["rate"], kw["T"],
    ),
    "cos_heston": lambda **kw: cos_method.cos_price(
        cos_method.heston_char_func_cos(
            kw["rate"], kw.get("div_yield", 0.0), kw["v0"], kw["kappa"],
            kw["theta"], kw["xi"], kw["rho"], kw["T"],
        ),
        kw["spot"], kw["strike"], kw["rate"], kw["T"],
    ),
}


def get_pricer(name: str):
    """Get a pricer by name (e.g. 'cos_bs', 'cos_heston')."""
    if name not in _PRICERS:
        raise KeyError(f"Unknown pricer '{name}'. Available: {list(_PRICERS.keys())}")
    return _PRICERS[name]


def list_pricers() -> list[str]:
    return list(_PRICERS.keys())


# ---------------------------------------------------------------------------
# Greek engines
# ---------------------------------------------------------------------------

_GREEK_ENGINES = {
    "aad": {
        "black_scholes": aad_pricing.aad_black_scholes,
        "swap": aad_pricing.aad_swap_pv,
        "cds": aad_pricing.aad_cds_pv,
    },
    "bump": {
        "dv01": risk.dv01_curve,
        "key_rate": risk.key_rate_durations,
    },
}


def get_greek_engine(name: str) -> dict:
    """Get a Greek computation engine by name ('aad' or 'bump')."""
    if name not in _GREEK_ENGINES:
        raise KeyError(f"Unknown Greek engine '{name}'. Available: {list(_GREEK_ENGINES.keys())}")
    return _GREEK_ENGINES[name]


def list_greek_engines() -> list[str]:
    return list(_GREEK_ENGINES.keys())
