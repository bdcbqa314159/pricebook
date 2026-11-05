"""The pseudo-square-root of a covariance matrix (L0). JAECKEL-2002 eqs 12.4, 12.6, 12.7.

Factorises a covariance `C` as `C ≈ A·Aᵀ`, optionally truncated to `m` factors, then rescales so

    Σ_j a_ij² = c_ii                                                                   (12.6)

holds **exactly**. Jäckel's reason, in his words: *"in order to retain the calibration of the options
on the FRAs, i.e. the caplets"*. The diagonal of `C` is each forward rate's own total variance, so if
factor truncation disturbed it, every caplet would reprice wrong — and a market model that cannot
reprice the caplets it was calibrated to is not calibrated. Eq 12.7 is the rescaling:

    a_ij = ã_ij · √( c_ii / Σ_k ã_ik² )                                                (12.7)

**Spectral decomposition, not Cholesky, and the reason is truncation.** Cholesky gives a factor per
dimension in a fixed order and has no notion of "the most important `m` factors"; the eigenvalue
decomposition orders factors by the variance they explain, so truncating to `m` keeps the largest.
That is what makes 12.7 a *rescaling of a good approximation* rather than a repair of an arbitrary
one. (Jäckel's 12.4 is written as a general pseudo-square-root; the spectral route is the standard
realisation of it and the one that makes `m`-factor truncation meaningful.)

**Jäckel also states the trade-off, so the oracle asserts it rather than hoping:** *"the individual
variances of each of the rates are still correct, even if we have reduced the number of driving
factors to one, but the effective covariances will differ. For instance, for a single factor model,
all of the correlation coefficients will be unity."* Both halves are oracled — the diagonal exact at
any factor count, and the one-factor case driving every implied correlation to 1.

**Finance-free, which is why it is L0.** The finance is *which* matrix to factorise; a decomposition
is arithmetic. The matrix comes from L3's model.

Provenance:
  stock:  — the quarry decomposed per-site; built fresh as one adapter
  source: JAECKEL-2002 (docs/references.md) §12.2 eqs 12.4/12.6/12.7, read in `-raw` and `-layout`,
          which agree; numpy's `linalg.eigh` as the symmetric-eigenvalue provider (pinned, since the
          factorisation feeds a reproducibility contract)
  oracle: tests/L0/test_linalg.py — full-factor reconstruction `A·Aᵀ = C`; **eq 12.6 exactly at every
          factor count** (the caplet guarantee); the one-factor case driving all correlations to
          unity AND preserving the diagonal; truncation shown to really change the off-diagonals, so
          the preservation is not vacuous; a diagonal matrix factorising to its standard deviations;
          non-symmetric, non-positive-definite, non-positive-diagonal and bad factor counts refused
  crosscheck: none — a symmetric-eigenvalue factorisation, no finance convention
  slice:  tranche 16 — the LIBOR market model by Monte Carlo
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Matrix = Sequence[Sequence[float]]

#: Eigenvalues below this are treated as numerically zero rather than negative. Symmetric
#: decomposition of a positive-semidefinite matrix can return a tiny negative eigenvalue purely from
#: rounding, and clipping those is not the same as accepting a genuinely indefinite matrix — the
#: threshold is scaled by the matrix's own magnitude so it means the same thing at any units.
_EIGENVALUE_FLOOR_SCALE = 1e-12


def pseudo_square_root(covariance: Matrix, factors: int) -> tuple[tuple[float, ...], ...]:
    """`A` such that `A·Aᵀ ≈ covariance`, truncated to `factors` columns and rescaled by eq 12.7.

    The rescaling makes `Σ_j a_ij² = c_ii` hold **exactly** whatever `factors` is, which is the
    caplet-calibration guarantee; the off-diagonal covariances are what truncation sacrifices."""
    n = len(covariance)
    if n == 0 or any(len(row) != n for row in covariance):
        raise ValueError("a covariance matrix must be square and non-empty")
    if not 1 <= factors <= n:
        raise ValueError(f"factors must lie between 1 and {n}, got {factors}")

    matrix = np.array([[float(x) for x in row] for row in covariance], dtype=float)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-14):
        raise ValueError("a covariance matrix must be symmetric")
    for i in range(n):
        if matrix[i, i] <= 0.0:
            raise ValueError(
                f"a covariance diagonal must be positive — eq 12.7 rescales by sqrt(c_ii) — "
                f"got {matrix[i, i]} at {i}"
            )

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = -_EIGENVALUE_FLOOR_SCALE * float(np.max(np.abs(eigenvalues)))
    if float(np.min(eigenvalues)) < floor:
        raise ValueError(
            f"a covariance matrix must be positive semidefinite; its smallest eigenvalue is "
            f"{float(np.min(eigenvalues))}, which is not rounding"
        )
    clipped = np.clip(eigenvalues, 0.0, None)

    # Largest factors first, so truncating to `m` keeps the most explanatory ones -- which is what
    # makes eq 12.7 a rescaling of a good approximation rather than a repair of an arbitrary one.
    order = np.argsort(clipped)[::-1][:factors]
    unscaled = eigenvectors[:, order] * np.sqrt(clipped[order])

    rows: list[tuple[float, ...]] = []
    for i in range(n):
        row = unscaled[i, :]
        norm_squared = float(np.dot(row, row))
        if norm_squared <= 0.0:
            raise ValueError(
                f"row {i} has no weight in the retained factors, so eq 12.7's rescaling is "
                f"undefined; use more factors"
            )
        scale = math.sqrt(float(matrix[i, i]) / norm_squared)
        rows.append(tuple(float(x) * scale for x in row))
    return tuple(rows)
