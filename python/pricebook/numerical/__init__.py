"""Pricebook numerical methods: self-contained toolkit.

Clean interface over scipy/numpy — users never import scipy directly.

    from pricebook.numerical import Normal, StudentT, LogNormal
    from pricebook.numerical import expm, qr, cholesky, gmres
    from pricebook.numerical import minimize, linprog, qp
    from pricebook.numerical import rk45, bdf, euler
    from pricebook.numerical import gauss_jacobi, tanh_sinh
    from pricebook.numerical import bilinear, bicubic, rbf_interpolate
    from pricebook.numerical import bisection, find_root
    from pricebook.numerical import qe_heston_step, multilevel_mc
    from pricebook.numerical import tree_greeks, binomial_2d
    from pricebook.numerical import fractional_fft, hilbert_transform, CharacteristicFunction
    from pricebook.numerical import TemperedDistribution, dirac_delta, sobolev_norm
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

# Optimisation
from pricebook.numerical._optimize import (
    minimize, linprog, qp, interior_point,
    proximal_gradient, projection_simplex, projection_l1_ball, soft_threshold,
)

# Quadrature
from pricebook.numerical._quadrature import (
    gauss_jacobi, tanh_sinh, clenshaw_curtis,
)

# 2D Interpolation
from pricebook.numerical._interpolation import (
    bilinear, bicubic, rbf_interpolate,
)

# Root finding
from pricebook.numerical._rootfinding import (
    bisection, find_root,
)

# MC improvements
from pricebook.numerical._mc import (
    qe_heston_step, antithetic_paths, multilevel_mc,
)

# Tree improvements
from pricebook.numerical._trees import (
    tree_greeks, binomial_2d,
)

# Fourier
from pricebook.numerical._fourier import (
    fractional_fft, hilbert_transform, wavelet_transform, CharacteristicFunction,
)

# PDE
from pricebook.numerical._pde import (
    PDESolver1D, PDEMethod, PDEResult, GridType, BoundaryCondition,
    solve_bs_pde, build_grid, extract_greeks,
)

# Distribution theory
from pricebook.numerical._distributions_theory import (
    SchwartzTestFunction, TemperedDistribution,
    dirac_delta, heaviside, regular,
    sobolev_norm,
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
    # Optimisation
    "minimize", "linprog", "qp", "interior_point",
    "proximal_gradient", "projection_simplex", "projection_l1_ball", "soft_threshold",
    # Quadrature
    "gauss_jacobi", "tanh_sinh", "clenshaw_curtis",
    # Interpolation
    "bilinear", "bicubic", "rbf_interpolate",
    # Root finding
    "bisection", "find_root",
    # MC
    "qe_heston_step", "antithetic_paths", "multilevel_mc",
    # Trees
    "tree_greeks", "binomial_2d",
    # Fourier
    "fractional_fft", "hilbert_transform", "wavelet_transform", "CharacteristicFunction",
    # PDE
    "PDESolver1D", "PDEMethod", "PDEResult", "GridType", "solve_bs_pde",
    # Distribution theory
    "SchwartzTestFunction", "TemperedDistribution",
    "dirac_delta", "heaviside", "regular", "sobolev_norm",
]
