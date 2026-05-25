"""2D interpolation and RBF: bilinear, bicubic, radial basis functions.

    from pricebook.numerical import bilinear, bicubic, rbf_interpolate
    from pricebook.numerical import InterpMethod2D, RBFKernel

Extends pricebook.interpolation (1D) with multi-dimensional methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class InterpMethod2D(Enum):
    """2D interpolation methods."""
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    RBF = "rbf"


class RBFKernel(Enum):
    """Radial basis function kernels."""
    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"
    GAUSSIAN = "gaussian"
    THIN_PLATE = "thin_plate"
    LINEAR = "linear"


def bilinear(
    x: float,
    y: float,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> float:
    """Bilinear interpolation on a regular 2D grid.

    Args:
        x, y: query point.
        xs: sorted x-coordinates (n_x,).
        ys: sorted y-coordinates (n_y,).
        zs: values on grid (n_x, n_y).

    Returns:
        Interpolated value at (x, y).
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    zs = np.asarray(zs)

    # Clamp to grid
    x = np.clip(x, xs[0], xs[-1])
    y = np.clip(y, ys[0], ys[-1])

    # Find cell
    ix = min(int(np.searchsorted(xs, x, side="right")) - 1, len(xs) - 2)
    iy = min(int(np.searchsorted(ys, y, side="right")) - 1, len(ys) - 2)
    ix = max(ix, 0)
    iy = max(iy, 0)

    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]

    dx = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
    dy = (y - y0) / (y1 - y0) if y1 != y0 else 0.0

    z00 = zs[ix, iy]
    z10 = zs[ix + 1, iy]
    z01 = zs[ix, iy + 1]
    z11 = zs[ix + 1, iy + 1]

    return float(
        z00 * (1 - dx) * (1 - dy) +
        z10 * dx * (1 - dy) +
        z01 * (1 - dx) * dy +
        z11 * dx * dy
    )


def bicubic(
    x: float,
    y: float,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> float:
    """Bicubic interpolation on a regular 2D grid.

    Uses scipy.interpolate.RectBivariateSpline for smooth C1 interpolation.
    """
    from scipy.interpolate import RectBivariateSpline

    spline = RectBivariateSpline(xs, ys, zs, kx=3, ky=3)
    return float(spline(x, y)[0, 0])


def interpolate_2d(
    x: float,
    y: float,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    method: InterpMethod2D | str = InterpMethod2D.BILINEAR,
) -> float:
    """Unified 2D interpolation dispatcher.

    Args:
        x, y: query point.
        xs, ys: grid coordinates.
        zs: grid values.
        method: InterpMethod2D enum or string name.
    """
    if isinstance(method, str):
        method = InterpMethod2D(method.lower())

    if method == InterpMethod2D.BILINEAR:
        return bilinear(x, y, xs, ys, zs)
    if method == InterpMethod2D.BICUBIC:
        return bicubic(x, y, xs, ys, zs)
    raise ValueError(f"Use rbf_interpolate() for RBF; method={method!r}")


def _rbf_kernel_matrix(D: np.ndarray, kernel: RBFKernel) -> np.ndarray:
    """Compute RBF kernel matrix from distance matrix."""
    if kernel == RBFKernel.MULTIQUADRIC:
        return np.sqrt(1 + D ** 2)
    if kernel == RBFKernel.INVERSE_MULTIQUADRIC:
        return 1.0 / np.sqrt(1 + D ** 2)
    if kernel == RBFKernel.GAUSSIAN:
        return np.exp(-D ** 2)
    if kernel == RBFKernel.THIN_PLATE:
        return np.where(D > 0, D ** 2 * np.log(D + 1e-300), 0.0)
    # LINEAR
    return D


@dataclass
class RBFResult:
    """RBF interpolation result."""
    weights: np.ndarray
    centers: np.ndarray
    kernel: str

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate interpolant at new points."""
        x = np.atleast_2d(x)
        n = len(self.centers)
        m = len(x)
        D = np.zeros((m, n))
        for j in range(n):
            D[:, j] = np.sqrt(np.sum((x - self.centers[j]) ** 2, axis=1))
        kernel_enum = RBFKernel(self.kernel)
        Phi = _rbf_kernel_matrix(D, kernel_enum)
        return Phi @ self.weights

    def to_dict(self) -> dict:
        return {"kernel": self.kernel, "n_centers": len(self.centers)}


def rbf_interpolate(
    centers: np.ndarray,
    values: np.ndarray,
    kernel: RBFKernel | str = RBFKernel.MULTIQUADRIC,
) -> RBFResult:
    """Radial Basis Function interpolation for scattered data.

    Args:
        centers: (n, d) data point locations.
        values: (n,) values at centers.
        kernel: RBFKernel enum or string name.

    Returns:
        RBFResult with weights and evaluate() method.
    """
    if isinstance(kernel, str):
        kernel = RBFKernel(kernel.lower())

    centers = np.atleast_2d(centers)
    values = np.asarray(values)
    n = len(centers)

    # Build distance matrix
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

    Phi = _rbf_kernel_matrix(D, kernel)

    # Add small regularization for numerical stability
    Phi += 1e-10 * np.eye(n)
    weights = np.linalg.solve(Phi, values)

    return RBFResult(weights, centers, kernel.value)
