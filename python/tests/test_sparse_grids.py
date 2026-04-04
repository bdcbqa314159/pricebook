"""Tests for sparse grids (Smolyak quadrature)."""

import math
import pytest
import numpy as np

from pricebook.sparse_grids import (
    clenshaw_curtis_nodes,
    smolyak_grid,
    sparse_grid_count,
    sparse_grid_integrate,
)


class TestClenshawCurtis:
    def test_level_0(self):
        nodes, weights = clenshaw_curtis_nodes(0)
        assert len(nodes) == 1
        assert nodes[0] == pytest.approx(0.0)
        assert weights[0] == pytest.approx(2.0)

    def test_level_1(self):
        nodes, weights = clenshaw_curtis_nodes(1)
        assert len(nodes) == 3

    def test_weights_sum(self):
        """Weights should sum to 2 (length of [-1,1])."""
        for level in range(5):
            _, w = clenshaw_curtis_nodes(level)
            assert sum(w) == pytest.approx(2.0, abs=1e-10)

    def test_integrates_polynomial(self):
        """Level k should integrate degree 2k+1 polynomial exactly (CC property)."""
        nodes, weights = clenshaw_curtis_nodes(3)
        # Integrate x^4 over [-1, 1] = 2/5
        result = sum(w * x**4 for x, w in zip(nodes, weights))
        assert result == pytest.approx(2.0 / 5.0, abs=1e-10)

    def test_nested(self):
        """Level k+1 nodes should contain level k nodes."""
        n1, _ = clenshaw_curtis_nodes(1)
        n2, _ = clenshaw_curtis_nodes(2)
        for x in n1:
            assert any(abs(x - y) < 1e-14 for y in n2)


class TestSmolyakGrid:
    def test_1d_matches_cc(self):
        """In 1D, Smolyak = Clenshaw-Curtis."""
        nodes, weights = smolyak_grid(1, 3)
        cc_nodes, cc_weights = clenshaw_curtis_nodes(3)
        assert len(nodes) == len(cc_nodes)

    def test_2d_fewer_than_tensor(self):
        """Sparse grid should have fewer points than full tensor product."""
        sg_count = sparse_grid_count(2, 3)
        cc_count = len(clenshaw_curtis_nodes(3)[0])
        tensor_count = cc_count ** 2
        assert sg_count < tensor_count

    def test_3d_much_fewer(self):
        sg_count = sparse_grid_count(3, 2)
        cc_count = len(clenshaw_curtis_nodes(2)[0])
        tensor_count = cc_count ** 3
        assert sg_count < tensor_count

    def test_weights_sum_2d(self):
        """Weights should sum to volume of [-1,1]^d = 2^d."""
        nodes, weights = smolyak_grid(2, 3)
        assert sum(weights) == pytest.approx(4.0, abs=0.01)

    def test_weights_sum_3d(self):
        nodes, weights = smolyak_grid(3, 2)
        assert sum(weights) == pytest.approx(8.0, abs=0.01)


class TestSparseGridIntegrate:
    def test_constant(self):
        """Integral of 1 over [-1,1]^2 = 4."""
        result = sparse_grid_integrate(lambda x: 1.0, dim=2, level=1)
        assert result == pytest.approx(4.0, abs=0.01)

    def test_polynomial_2d(self):
        """Integrate x^2 + y^2 over [-1,1]^2 = 2*(2/3)*2 = 8/3."""
        def f(x):
            return x[0]**2 + x[1]**2
        result = sparse_grid_integrate(f, dim=2, level=3)
        assert result == pytest.approx(8.0 / 3.0, abs=0.01)

    def test_product_polynomial_2d(self):
        """Integrate x^2 * y^2 over [0,1]^2 = (1/3)^2 = 1/9."""
        def f(x):
            return x[0]**2 * x[1]**2
        result = sparse_grid_integrate(f, dim=2, level=3, bounds=[(0, 1)] * 2)
        assert result == pytest.approx(1.0 / 9.0, abs=0.001)

    def test_3d(self):
        """Integrate x*y*z over [0,1]^3 = (1/2)^3 = 1/8."""
        def f(x):
            return x[0] * x[1] * x[2]
        result = sparse_grid_integrate(f, dim=3, level=3, bounds=[(0, 1)] * 3)
        assert result == pytest.approx(0.125, abs=0.01)

    def test_custom_bounds(self):
        """Integrate 1 over [0,2]^2 = 4."""
        result = sparse_grid_integrate(lambda x: 1.0, dim=2, level=1,
                                       bounds=[(0, 2), (0, 2)])
        assert result == pytest.approx(4.0, abs=0.01)

    def test_sparse_vs_full_tensor(self):
        """Sparse grid uses fewer points than full tensor for similar accuracy."""
        def f(x):
            return x[0]**2 * x[1]**2

        # Sparse grid
        sg_result = sparse_grid_integrate(f, dim=2, level=3, bounds=[(0, 1)] * 2)
        n_sg = sparse_grid_count(2, 3)

        # Exact: (1/3)^2 = 1/9
        expected = 1.0 / 9.0
        assert sg_result == pytest.approx(expected, abs=0.001)
        # Sparse grid should use fewer than 9^2=81 points (full 9-point tensor)
        assert n_sg < 81
