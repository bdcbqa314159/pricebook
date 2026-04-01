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
from pricebook import trinomial_tree
from pricebook import finite_difference
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
    "trinomial": trinomial_tree.trinomial_european,
}

_TREE_AMERICAN = {
    "binomial": binomial_tree.binomial_american,
    "crr": binomial_tree.binomial_american,
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

from pricebook import optimization  # noqa: E402

_OPTIMIZERS = {
    "nelder_mead": "nelder_mead",
    "bfgs": "bfgs",
    "l_bfgs_b": "l_bfgs_b",
    "differential_evolution": "differential_evolution",
    "basin_hopping": "basin_hopping",
}


def get_optimizer(name: str):
    """Get an optimizer by name. Returns a partial of optimization.minimize."""
    if name not in _OPTIMIZERS:
        raise KeyError(f"Unknown optimizer '{name}'. Available: {list(_OPTIMIZERS.keys())}")
    method = _OPTIMIZERS[name]

    def _opt(objective, x0, **kwargs):
        return optimization.minimize(objective, x0, method=method, **kwargs)

    return _opt


def list_optimizers() -> list[str]:
    return list(_OPTIMIZERS.keys())


# ---------------------------------------------------------------------------
# ODE solvers
# ---------------------------------------------------------------------------

from pricebook import ode  # noqa: E402

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
