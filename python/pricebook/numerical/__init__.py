"""Pricebook numerical methods: self-contained toolkit.

Clean interface over scipy/numpy — users never import scipy directly.

    from pricebook.numerical import Normal, StudentT, LogNormal
    from pricebook.numerical import expm, qr, cholesky, gmres
    from pricebook.numerical import minimize, linprog, qp
    from pricebook.numerical import rk45, bdf
"""

# Distributions
from pricebook.numerical._distributions import (
    Normal, StudentT, LogNormal, Uniform, Exponential,
)

# Linear algebra
from pricebook.numerical._linalg import (
    qr, cholesky, lu, expm, logm, sqrtm,
    solve, lstsq, gmres, bicgstab,
    sylvester, lyapunov,
    cond, rank, is_positive_definite,
)

# ODE integrators
from pricebook.numerical._ode import (
    euler, rk4, rk45, bdf, adams,
)

__all__ = [
    # Distributions
    "Normal", "StudentT", "LogNormal", "Uniform", "Exponential",
    # Linear algebra
    "qr", "cholesky", "lu", "expm", "logm", "sqrtm",
    "solve", "lstsq", "gmres", "bicgstab",
    "sylvester", "lyapunov",
    "cond", "rank", "is_positive_definite",
    # ODE
    "euler", "rk4", "rk45", "bdf", "adams",
]
