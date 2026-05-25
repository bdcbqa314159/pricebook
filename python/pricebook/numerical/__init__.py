"""Pricebook numerical methods: self-contained toolkit.

Clean interface over scipy/numpy — users never import scipy directly.

    from pricebook.numerical import Normal, StudentT, LogNormal
    from pricebook.numerical import expm, qr, cholesky, gmres
    from pricebook.numerical import minimize, OptimMethod, linprog, qp
    from pricebook.numerical import solve_ode, ODEMethod
    from pricebook.numerical import integrate, IntegrationMethod
    from pricebook.numerical import bilinear, bicubic, rbf_interpolate, InterpMethod2D
    from pricebook.numerical import find_root, RootMethod
    from pricebook.numerical import qe_heston_step, multilevel_mc, MCVarianceReduction
    from pricebook.numerical import solve_tree, TreeMethod
    from pricebook.numerical import fractional_fft, hilbert_transform, CharacteristicFunction
    from pricebook.numerical import TemperedDistribution, dirac_delta, sobolev_norm
"""

# Distributions
from pricebook.numerical._distributions import (
    Normal, StudentT, LogNormal, Uniform, Exponential,
)

# Linear algebra
from pricebook.numerical._linalg import (
    DecompMethod, IterativeMethod,
    QRResult, SVDResult, LUResult, IterativeSolveResult,
    qr, cholesky, lu, lu_full, svd, decompose,
    expm, logm, sqrtm,
    solve, lstsq, gmres, bicgstab, iterative_solve,
    sylvester, lyapunov,
    cond, rank, is_positive_definite,
)

# ODE integrators
from pricebook.numerical._ode import (
    ODESolver, ODEMethod, ODEResult,
    solve_ode, solve_backward, solve_riccati, solve_system,
)

# Optimisation
from pricebook.numerical._optimize import (
    OptimMethod, OptimizeResult, LPResult, QPResult,
    minimize, linprog, qp, interior_point,
    proximal_gradient, projection_simplex, projection_l1_ball, soft_threshold,
)

# Integration
from pricebook.numerical._integrate import (
    integrate, IntegrationMethod, IntegrationResult,
    integrate_2d, integrate_semi_infinite,
)

# 2D Interpolation
from pricebook.numerical._interpolation import (
    InterpMethod2D, RBFKernel, RBFResult,
    bilinear, bicubic, interpolate_2d, rbf_interpolate,
)

# Root finding
from pricebook.numerical._rootfinding import (
    RootMethod, RootResult,
    bisection, find_root,
)

# MC improvements
from pricebook.numerical._mc import (
    MCVarianceReduction, MCDiscrMethod, MLMCResult,
    qe_heston_step, antithetic_paths, multilevel_mc,
)

# Trees
from pricebook.numerical._trees import (
    TreeSolver, TreeMethod, ExerciseType, BarrierType, TreeResult,
    solve_tree, solve_tree_2d,
)

# Fourier
from pricebook.numerical._fourier import (
    FourierMethod, WaveletType, WaveletResult, CFResult,
    fractional_fft, hilbert_transform, wavelet_transform, CharacteristicFunction,
)

# PDE
from pricebook.numerical._pde import (
    PDESolver1D, PDEMethod, PDEResult, GridType, BoundaryCondition,
    solve_bs_pde, build_grid, extract_greeks,
)

# Graph
from pricebook.numerical._graph import (
    ShortestPathResult, MSTResult, MaxFlowResult,
    dijkstra, dijkstra_full, shortest_path,
    minimum_spanning_tree, minimum_spanning_tree_full,
    max_flow, max_flow_full, connected_components,
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
    "DecompMethod", "IterativeMethod",
    "QRResult", "SVDResult", "LUResult", "IterativeSolveResult",
    "qr", "cholesky", "lu", "lu_full", "svd", "decompose",
    "expm", "logm", "sqrtm",
    "solve", "lstsq", "gmres", "bicgstab", "iterative_solve",
    "sylvester", "lyapunov",
    "cond", "rank", "is_positive_definite",
    # ODE
    "ODESolver", "ODEMethod", "ODEResult",
    "solve_ode", "solve_backward", "solve_riccati", "solve_system",
    # Optimisation
    "OptimMethod", "OptimizeResult", "LPResult", "QPResult",
    "minimize", "linprog", "qp", "interior_point",
    "proximal_gradient", "projection_simplex", "projection_l1_ball", "soft_threshold",
    # Integration
    "integrate", "IntegrationMethod", "IntegrationResult",
    "integrate_2d", "integrate_semi_infinite",
    # Interpolation
    "InterpMethod2D", "RBFKernel", "RBFResult",
    "bilinear", "bicubic", "interpolate_2d", "rbf_interpolate",
    # Root finding
    "RootMethod", "RootResult",
    "bisection", "find_root",
    # MC
    "MCVarianceReduction", "MCDiscrMethod", "MLMCResult",
    "qe_heston_step", "antithetic_paths", "multilevel_mc",
    # Trees
    "TreeSolver", "TreeMethod", "ExerciseType", "BarrierType", "TreeResult",
    "solve_tree", "solve_tree_2d",
    # Fourier
    "FourierMethod", "WaveletType", "WaveletResult", "CFResult",
    "fractional_fft", "hilbert_transform", "wavelet_transform", "CharacteristicFunction",
    # PDE
    "PDESolver1D", "PDEMethod", "PDEResult", "GridType", "BoundaryCondition",
    "solve_bs_pde", "build_grid", "extract_greeks",
    # Graph
    "ShortestPathResult", "MSTResult", "MaxFlowResult",
    "dijkstra", "dijkstra_full", "shortest_path",
    "minimum_spanning_tree", "minimum_spanning_tree_full",
    "max_flow", "max_flow_full", "connected_components",
    # Distribution theory
    "SchwartzTestFunction", "TemperedDistribution",
    "dirac_delta", "heaviside", "regular", "sobolev_norm",
]
