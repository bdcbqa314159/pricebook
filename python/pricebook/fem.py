"""Finite Element Method (1D) for option pricing PDE.

Galerkin FEM with linear (P1) and quadratic (P2) elements.
Uses sparse matrix assembly from pricebook.sparse.
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.sparse import SparseMatrix, sparse_solve


# ---------------------------------------------------------------------------
# Element matrices (reference element [0, 1])
# ---------------------------------------------------------------------------


def _p1_mass(h: float) -> np.ndarray:
    """2x2 mass matrix for linear element of width h."""
    return h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])


def _p1_stiffness(h: float) -> np.ndarray:
    """2x2 stiffness matrix for linear element of width h."""
    return 1.0 / h * np.array([[1.0, -1.0], [-1.0, 1.0]])


def _p2_mass(h: float) -> np.ndarray:
    """3x3 mass matrix for quadratic element of width h."""
    return h / 30.0 * np.array([
        [4.0, 2.0, -1.0],
        [2.0, 16.0, 2.0],
        [-1.0, 2.0, 4.0],
    ])


def _p2_stiffness(h: float) -> np.ndarray:
    """3x3 stiffness matrix for quadratic element of width h."""
    return 1.0 / (3.0 * h) * np.array([
        [7.0, -8.0, 1.0],
        [-8.0, 16.0, -8.0],
        [1.0, -8.0, 7.0],
    ])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_p1(nodes: np.ndarray) -> tuple[SparseMatrix, SparseMatrix]:
    """Assemble global mass (M) and stiffness (K) for P1 elements.

    Args:
        nodes: sorted array of node positions (n nodes, n-1 elements).

    Returns:
        (M, K) as SparseMatrix objects.
    """
    n = len(nodes)
    M = SparseMatrix(n, n)
    K = SparseMatrix(n, n)

    for e in range(n - 1):
        h = nodes[e + 1] - nodes[e]
        me = _p1_mass(h)
        ke = _p1_stiffness(h)
        M.add_dense(e, e, me)
        K.add_dense(e, e, ke)

    return M, K


def assemble_p2(nodes: np.ndarray) -> tuple[SparseMatrix, SparseMatrix, np.ndarray]:
    """Assemble global mass (M) and stiffness (K) for P2 elements.

    Midpoints are added between consecutive nodes.

    Args:
        nodes: sorted array of element boundary positions.

    Returns:
        (M, K, all_nodes) where all_nodes includes midpoints.
    """
    n_elem = len(nodes) - 1
    # Global nodes: boundaries + midpoints
    all_nodes = []
    for e in range(n_elem):
        all_nodes.append(nodes[e])
        all_nodes.append(0.5 * (nodes[e] + nodes[e + 1]))
    all_nodes.append(nodes[-1])
    all_nodes = np.array(all_nodes)

    n_total = len(all_nodes)
    M = SparseMatrix(n_total, n_total)
    K = SparseMatrix(n_total, n_total)

    for e in range(n_elem):
        h = nodes[e + 1] - nodes[e]
        me = _p2_mass(h)
        ke = _p2_stiffness(h)
        # Global indices: 2*e, 2*e+1, 2*e+2
        idx = 2 * e
        M.add_dense(idx, idx, me)
        K.add_dense(idx, idx, ke)

    return M, K, all_nodes


# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------


def _apply_dirichlet(A: SparseMatrix, b: np.ndarray,
                     bc: dict[int, float]) -> tuple[SparseMatrix, np.ndarray]:
    """Apply Dirichlet boundary conditions by modifying the system."""
    A_dense = A.to_dense()
    b = b.copy()
    n = A_dense.shape[0]

    for idx, val in bc.items():
        b -= A_dense[:, idx] * val
        A_dense[idx, :] = 0.0
        A_dense[:, idx] = 0.0
        A_dense[idx, idx] = 1.0
        b[idx] = val

    return SparseMatrix.from_scipy(
        __import__("scipy").sparse.csr_matrix(A_dense)
    ), b


def fem_heat_cn(
    nodes: np.ndarray,
    u0: np.ndarray,
    dt: float,
    n_steps: int,
    bc_left: float,
    bc_right: float,
    order: int = 1,
) -> np.ndarray:
    """Solve the heat equation u_t = u_xx via FEM + Crank-Nicolson.

    Args:
        nodes: spatial mesh (for P1) or element boundaries (for P2).
        u0: initial condition at all nodes.
        dt: time step.
        n_steps: number of time steps.
        bc_left, bc_right: Dirichlet boundary values.
        order: 1 for P1, 2 for P2 elements.

    Returns:
        Solution vector at final time.
    """
    if order == 1:
        M, K = assemble_p1(nodes)
        all_nodes = nodes
    else:
        M, K, all_nodes = assemble_p2(nodes)

    n = len(all_nodes)
    u = u0.copy()
    bc = {0: bc_left, n - 1: bc_right}

    M_dense = M.to_dense()
    K_dense = K.to_dense()

    # Crank-Nicolson: (M + 0.5*dt*K) u^{n+1} = (M - 0.5*dt*K) u^n
    A_lhs = M_dense + 0.5 * dt * K_dense
    A_rhs = M_dense - 0.5 * dt * K_dense

    for _ in range(n_steps):
        rhs = A_rhs @ u

        # Apply BCs
        for idx, val in bc.items():
            rhs -= A_lhs[:, idx] * val
            rhs[idx] = val

        A_bc = A_lhs.copy()
        for idx in bc:
            A_bc[idx, :] = 0.0
            A_bc[:, idx] = 0.0
            A_bc[idx, idx] = 1.0

        u = np.linalg.solve(A_bc, rhs)

    return u


# ---------------------------------------------------------------------------
# Black-Scholes FEM pricer
# ---------------------------------------------------------------------------


def fem_bs_european(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_spatial: int = 100,
    n_time: int = 100,
    x_range: float = 4.0,
    order: int = 1,
    is_call: bool = True,
) -> float:
    """European option price via FEM on the transformed BS PDE.

    Uses the standard substitution that converts BS to the heat equation:
        x = ln(S/K), tau = 0.5*sigma^2*(T-t), v = e^{alpha*x + beta*tau} * u
    where alpha = -(r/sigma^2 - 0.5), beta = -(alpha+1)^2.

    Then u_tau = u_xx (pure heat equation), solved with FEM + CN.
    """
    sigma2 = vol * vol
    alpha = -(rate / sigma2 - 0.5)
    beta = -(alpha + 1.0) ** 2
    tau_max = 0.5 * sigma2 * T
    d_tau = tau_max / n_time

    x_min = -x_range * vol * math.sqrt(T)
    x_max = x_range * vol * math.sqrt(T)
    nodes = np.linspace(x_min, x_max, n_spatial)

    if order == 1:
        M, K = assemble_p1(nodes)
        all_nodes = nodes
    else:
        M, K, all_nodes = assemble_p2(nodes)

    n = len(all_nodes)

    # Terminal payoff in original variables, then transform to u
    S_nodes = strike * np.exp(all_nodes)
    if is_call:
        payoff = np.maximum(S_nodes - strike, 0.0)
    else:
        payoff = np.maximum(strike - S_nodes, 0.0)

    # Transform: u(x, 0) = payoff(x) * exp(-alpha*x) / strike
    u = payoff / strike * np.exp(-alpha * all_nodes)

    M_d = M.to_dense()
    K_d = K.to_dense()

    # CN for heat equation: (M + 0.5*dtau*K) u^{n+1} = (M - 0.5*dtau*K) u^n
    A_lhs = M_d + 0.5 * d_tau * K_d
    A_rhs = M_d - 0.5 * d_tau * K_d

    for step in range(n_time):
        tau = (step + 1) * d_tau

        # Boundary conditions for the transformed variable
        if is_call:
            bc_left = 0.0
            bc_right = math.exp(alpha * all_nodes[-1] + beta * tau) * \
                       max(math.exp(all_nodes[-1]) - math.exp(-rate * T + (rate + 0.5*sigma2) * (T - tau / (0.5*sigma2))), 0.0)
            # Simplified: for large x, call ≈ S - K*e^{-rT} → u ≈ e^{(1-alpha)*x + beta*tau}
            bc_right = math.exp((1.0 - alpha) * all_nodes[-1] + beta * tau)
        else:
            bc_left = math.exp((1.0 - alpha) * all_nodes[0] + beta * tau) * 0.0
            # For large negative x, put ≈ K*e^{-rT} → u ≈ e^{-alpha*x + beta*tau} * e^{-rT}
            bc_left = math.exp(-alpha * all_nodes[0] + beta * tau) * math.exp(-rate * T)
            bc_right = 0.0

        rhs = A_rhs @ u
        bc = {0: bc_left, n - 1: bc_right}

        for idx, val in bc.items():
            rhs -= A_lhs[:, idx] * val
            rhs[idx] = val

        A_bc = A_lhs.copy()
        for idx in bc:
            A_bc[idx, :] = 0.0
            A_bc[:, idx] = 0.0
            A_bc[idx, idx] = 1.0

        u = np.linalg.solve(A_bc, rhs)

    # Transform back: price = strike * exp(alpha*x + beta*tau_max) * u(x)
    prices = strike * np.exp(alpha * all_nodes + beta * tau_max) * u

    x_target = math.log(spot / strike)
    return float(np.interp(x_target, all_nodes, prices))
