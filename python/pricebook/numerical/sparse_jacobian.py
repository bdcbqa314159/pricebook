"""Sparse Jacobian computation via graph colouring.

Exploits banded/block sparsity structure to reduce O(n²) finite
difference Jacobian to O(n) or O(bandwidth) evaluations.

* :func:`sparse_jacobian` — Jacobian exploiting sparsity pattern.
* :func:`banded_jacobian` — Jacobian for banded systems.
* :func:`detect_sparsity` — detect sparsity pattern from function.
* :func:`greedy_colouring` — distance-1 graph colouring for grouping.

References:
    Curtis, Powell & Reid, *On the Estimation of Sparse Jacobian Matrices*,
    IMA JNA, 1974.
    Coleman & Moré, *Estimation of Sparse Jacobian Matrices*, MPC, 1983.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SparseJacobianResult:
    """Sparse Jacobian result."""
    J: np.ndarray               # (m, n) Jacobian matrix
    n_evaluations: int          # function evaluations used
    n_colors: int               # groups used
    sparsity_ratio: float       # fraction of non-zero entries
    full_evaluations: int       # what dense FD would need

    def to_dict(self) -> dict:
        return {
            "n_evaluations": self.n_evaluations,
            "n_colors": self.n_colors,
            "sparsity_ratio": self.sparsity_ratio,
            "speedup": self.full_evaluations / max(self.n_evaluations, 1),
        }


def greedy_colouring(
    sparsity: np.ndarray,
) -> list[list[int]]:
    """Distance-1 graph colouring for column grouping.

    Two columns can be grouped (same colour) if they don't share
    any non-zero row. This allows simultaneous perturbation.

    Args:
        sparsity: (m, n) boolean matrix — True where J[i,j] ≠ 0.

    Returns:
        List of groups, each group is a list of column indices.
    """
    m, n = sparsity.shape
    colors = [-1] * n  # uncoloured
    groups: list[list[int]] = []

    for col in range(n):
        # Find forbidden colours: colours of columns that share a non-zero row
        forbidden = set()
        for row in range(m):
            if sparsity[row, col]:
                for other_col in range(n):
                    if other_col != col and sparsity[row, other_col] and colors[other_col] >= 0:
                        forbidden.add(colors[other_col])

        # Assign smallest available colour
        colour = 0
        while colour in forbidden:
            colour += 1

        colors[col] = colour

        # Extend groups list
        while len(groups) <= colour:
            groups.append([])
        groups[colour].append(col)

    return groups


def sparse_jacobian(
    f,
    x: np.ndarray,
    sparsity: np.ndarray,
    h: float = 1e-7,
) -> SparseJacobianResult:
    """Compute Jacobian exploiting known sparsity pattern.

    Uses graph colouring to group columns that can be perturbed
    simultaneously (no overlapping non-zero rows).

    Cost: n_colours × f evaluations instead of n × f evaluations.
    For banded Jacobians: n_colours = 2 × bandwidth + 1.

    Args:
        f: callable(x) → array (m,).
        x: evaluation point (n,).
        sparsity: (m, n) boolean sparsity pattern.
        h: finite difference step size.
    """
    n = len(x)
    f0 = f(x)
    m = len(f0)

    # Group columns by graph colouring
    groups = greedy_colouring(sparsity)
    n_colors = len(groups)

    J = np.zeros((m, n))
    n_evals = 1  # f0

    for group in groups:
        # Perturb all columns in group simultaneously
        x_pert = x.copy()
        for col in group:
            x_pert[col] += h

        f_pert = f(x_pert)
        n_evals += 1

        # Extract individual column contributions
        df = f_pert - f0
        for col in group:
            for row in range(m):
                if sparsity[row, col]:
                    J[row, col] = df[row] / h

    nnz = int(np.count_nonzero(sparsity))
    total = m * n

    return SparseJacobianResult(
        J=J,
        n_evaluations=n_evals,
        n_colors=n_colors,
        sparsity_ratio=nnz / max(total, 1),
        full_evaluations=n + 1,
    )


def banded_jacobian(
    f,
    x: np.ndarray,
    bandwidth: int,
    h: float = 1e-7,
) -> SparseJacobianResult:
    """Jacobian for banded systems (tridiagonal, pentadiagonal, etc.).

    A banded Jacobian with bandwidth p has at most 2p+1 non-zeros
    per row. Graph colouring needs exactly 2p+1 colours.

    Cost: (2p+1) evaluations instead of n.
    For tridiagonal (p=1): 3 evaluations instead of n.

    Args:
        f: callable(x) → array.
        x: evaluation point.
        bandwidth: p such that J[i,j] = 0 if |i-j| > p.
    """
    n = len(x)
    # Build sparsity pattern
    sparsity = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            sparsity[i, j] = True

    return sparse_jacobian(f, x, sparsity, h)


def detect_sparsity(
    f,
    x: np.ndarray,
    h: float = 1e-7,
    threshold: float = 1e-10,
) -> np.ndarray:
    """Detect sparsity pattern by probing with finite differences.

    Perturbs each variable and checks which outputs change.
    Cost: n+1 evaluations (same as dense Jacobian).

    Args:
        f: callable(x) → array.
        x: evaluation point.
        threshold: below this, entry is considered zero.

    Returns:
        (m, n) boolean sparsity pattern.
    """
    f0 = f(x)
    m = len(f0)
    n = len(x)
    sparsity = np.zeros((m, n), dtype=bool)

    for j in range(n):
        x_pert = x.copy()
        x_pert[j] += h
        f_pert = f(x_pert)
        df = np.abs(f_pert - f0)
        sparsity[:, j] = df > threshold

    return sparsity
