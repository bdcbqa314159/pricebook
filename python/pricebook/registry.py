"""
Registry for numerical methods.

Factory functions with string-based lookup for swapping implementations.

    from pricebook.registry import get_solver, get_tree_pricer

    solver = get_solver("newton")
    result = solver(f, fprime, x0=0.5)

    tree_price = get_tree_pricer("trinomial")(spot=100, strike=105, ...)
"""

from __future__ import annotations

from pricebook.core import solvers
from pricebook.numerical._integrate import integrate, IntegrationMethod
from pricebook.numerical._trees import solve_tree, TreeMethod, ExerciseType
from pricebook.models import finite_difference
from pricebook.models import adi
from pricebook.models import mc_pricer
from pricebook.models import mc_advanced
from pricebook.models import lsm


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

def _make_integrator(method):
    def integrator(f, a, b, **kwargs):
        return integrate(f, a, b, method=method, **kwargs)
    return integrator

_INTEGRATORS = {
    "gauss_legendre": _make_integrator(IntegrationMethod.GAUSS_LEGENDRE),
    "gauss_laguerre": _make_integrator(IntegrationMethod.GAUSS_LAGUERRE),
    "gauss_hermite": _make_integrator(IntegrationMethod.GAUSS_HERMITE),
    "adaptive_simpson": _make_integrator(IntegrationMethod.SIMPSON),
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

def _make_tree_pricer(method, exercise):
    def pricer(spot, strike, rate, vol, T, n_steps=200, option_type=None, div_yield=0.0):
        is_call = option_type is None or str(getattr(option_type, 'value', option_type)).lower() != "put"
        r = solve_tree(spot, strike, rate, vol, T, method=method, n_steps=n_steps,
                        exercise=exercise, is_call=is_call, div_yield=div_yield)
        return r.price
    return pricer

_TREE_EUROPEAN = {
    "binomial": _make_tree_pricer(TreeMethod.CRR, ExerciseType.EUROPEAN),
    "crr": _make_tree_pricer(TreeMethod.CRR, ExerciseType.EUROPEAN),
    "jr": _make_tree_pricer(TreeMethod.JR, ExerciseType.EUROPEAN),
    "jarrow_rudd": _make_tree_pricer(TreeMethod.JR, ExerciseType.EUROPEAN),
    "lr": _make_tree_pricer(TreeMethod.LR, ExerciseType.EUROPEAN),
    "leisen_reimer": _make_tree_pricer(TreeMethod.LR, ExerciseType.EUROPEAN),
    "trinomial": _make_tree_pricer(TreeMethod.TRINOMIAL, ExerciseType.EUROPEAN),
}

_TREE_AMERICAN = {
    "binomial": _make_tree_pricer(TreeMethod.CRR, ExerciseType.AMERICAN),
    "crr": _make_tree_pricer(TreeMethod.CRR, ExerciseType.AMERICAN),
    "jr": _make_tree_pricer(TreeMethod.JR, ExerciseType.AMERICAN),
    "jarrow_rudd": _make_tree_pricer(TreeMethod.JR, ExerciseType.AMERICAN),
    "lr": _make_tree_pricer(TreeMethod.LR, ExerciseType.AMERICAN),
    "leisen_reimer": _make_tree_pricer(TreeMethod.LR, ExerciseType.AMERICAN),
    "trinomial": _make_tree_pricer(TreeMethod.TRINOMIAL, ExerciseType.AMERICAN),
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
from pricebook.statistics import optimization  # noqa: E402

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

from pricebook.numerical._ode import solve_ode, ODEMethod  # noqa: E402
from pricebook.models import cos_method  # noqa: E402
from pricebook.curves import aad_pricing  # noqa: E402
from pricebook.risk import risk  # noqa: E402


def _make_ode_solver(method):
    def solver(f, t_span, y0, **kwargs):
        return solve_ode(f, t_span, y0, method=method, **kwargs)
    return solver

_ODE_SOLVERS = {
    "rk4": _make_ode_solver(ODEMethod.RK4),
    "rk45": _make_ode_solver(ODEMethod.RK45),
    "bdf": _make_ode_solver(ODEMethod.BDF),
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
