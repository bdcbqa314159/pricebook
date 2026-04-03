"""Tests for AAD-aware interpolation."""

import math
import pytest

from pricebook.aad import Number, Tape
from pricebook.aad_interp import aad_linear_interp, aad_log_linear_interp


class TestAADLinearInterp:
    def test_exact_knot(self):
        with Tape() as tape:
            ys = [Number(1.0), Number(2.0), Number(3.0)]
            xs = [0.0, 1.0, 2.0]
            v = aad_linear_interp(1.0, xs, ys)
            assert v.value == pytest.approx(2.0)

    def test_midpoint(self):
        with Tape() as tape:
            ys = [Number(1.0), Number(3.0)]
            xs = [0.0, 1.0]
            v = aad_linear_interp(0.5, xs, ys)
            assert v.value == pytest.approx(2.0)

    def test_derivatives_two_points(self):
        """d/dy[i] should be the interpolation weights."""
        with Tape() as tape:
            y0 = Number(1.0)
            y1 = Number(3.0)
            v = aad_linear_interp(0.25, [0.0, 1.0], [y0, y1])
            v.propagate_to_start()
            # At x=0.25: w0 = 0.75, w1 = 0.25
            assert y0.adjoint == pytest.approx(0.75)
            assert y1.adjoint == pytest.approx(0.25)

    def test_derivatives_three_points(self):
        """Only the two bracketing knots get non-zero adjoints."""
        with Tape() as tape:
            ys = [Number(1.0), Number(2.0), Number(4.0)]
            xs = [0.0, 1.0, 2.0]
            v = aad_linear_interp(1.5, xs, ys)
            v.propagate_to_start()
            assert ys[0].adjoint == pytest.approx(0.0)
            assert ys[1].adjoint == pytest.approx(0.5)
            assert ys[2].adjoint == pytest.approx(0.5)

    def test_flat_extrapolation_left(self):
        with Tape() as tape:
            ys = [Number(10.0), Number(20.0)]
            v = aad_linear_interp(-1.0, [0.0, 1.0], ys)
            assert v.value == pytest.approx(10.0)

    def test_flat_extrapolation_right(self):
        with Tape() as tape:
            ys = [Number(10.0), Number(20.0)]
            v = aad_linear_interp(5.0, [0.0, 1.0], ys)
            assert v.value == pytest.approx(20.0)

    def test_matches_float_interp(self):
        """AAD values match plain float interpolation."""
        from pricebook.interpolation import LinearInterpolator
        import numpy as np

        xs = [0.0, 1.0, 2.0, 5.0]
        y_vals = [1.0, 0.98, 0.95, 0.85]
        float_interp = LinearInterpolator(np.array(xs), np.array(y_vals))

        for x in [0.5, 1.5, 3.0, 4.5]:
            with Tape() as tape:
                ys = [Number(v) for v in y_vals]
                aad_val = aad_linear_interp(x, xs, ys).value
                float_val = float_interp(x)
                assert aad_val == pytest.approx(float_val, rel=1e-12)

    def test_fd_check(self):
        """AAD derivatives match finite differences."""
        xs = [0.0, 1.0, 2.0, 5.0]
        y_vals = [1.0, 0.98, 0.95, 0.85]
        query_x = 1.5
        eps = 1e-7

        # AAD
        with Tape() as tape:
            ys = [Number(v) for v in y_vals]
            v = aad_linear_interp(query_x, xs, ys)
            v.propagate_to_start()
            aad_derivs = [y.adjoint for y in ys]

        # Finite difference
        for idx in range(len(y_vals)):
            y_up = list(y_vals)
            y_up[idx] += eps
            y_dn = list(y_vals)
            y_dn[idx] -= eps

            with Tape() as tape:
                v_up = aad_linear_interp(query_x, xs, [Number(v) for v in y_up]).value
            with Tape() as tape:
                v_dn = aad_linear_interp(query_x, xs, [Number(v) for v in y_dn]).value

            fd = (v_up - v_dn) / (2 * eps)
            assert aad_derivs[idx] == pytest.approx(fd, abs=1e-5)


class TestAADLogLinearInterp:
    def test_exact_knot(self):
        with Tape() as tape:
            ys = [Number(1.0), Number(0.5), Number(0.25)]
            xs = [0.0, 1.0, 2.0]
            v = aad_log_linear_interp(1.0, xs, ys)
            assert v.value == pytest.approx(0.5)

    def test_midpoint(self):
        with Tape() as tape:
            y0 = Number(1.0)
            y1 = Number(math.exp(-0.1))  # df at t=1
            v = aad_log_linear_interp(0.5, [0.0, 1.0], [y0, y1])
            # log-linear midpoint: exp(0.5 * ln(1) + 0.5 * ln(e^{-0.1})) = exp(-0.05)
            assert v.value == pytest.approx(math.exp(-0.05))

    def test_derivatives_flow(self):
        """Derivatives through log-linear interp are non-zero for bracketing knots."""
        with Tape() as tape:
            ys = [Number(1.0), Number(0.9), Number(0.8)]
            xs = [0.0, 1.0, 2.0]
            v = aad_log_linear_interp(0.5, xs, ys)
            v.propagate_to_start()
            assert ys[0].adjoint != 0.0
            assert ys[1].adjoint != 0.0
            assert ys[2].adjoint == pytest.approx(0.0, abs=1e-14)

    def test_matches_float_interp(self):
        """AAD values match plain float log-linear interpolation."""
        from pricebook.interpolation import LogLinearInterpolator
        import numpy as np

        xs = [0.0, 1.0, 2.0, 5.0]
        y_vals = [1.0, 0.98, 0.95, 0.85]
        float_interp = LogLinearInterpolator(np.array(xs), np.array(y_vals))

        for x in [0.5, 1.5, 3.0, 4.5]:
            with Tape() as tape:
                ys = [Number(v) for v in y_vals]
                aad_val = aad_log_linear_interp(x, xs, ys).value
                float_val = float_interp(x)
                assert aad_val == pytest.approx(float_val, rel=1e-10)

    def test_fd_check(self):
        """AAD derivatives match finite differences for log-linear."""
        xs = [0.0, 1.0, 2.0, 5.0]
        y_vals = [1.0, 0.98, 0.95, 0.85]
        query_x = 1.5
        eps = 1e-7

        with Tape() as tape:
            ys = [Number(v) for v in y_vals]
            v = aad_log_linear_interp(query_x, xs, ys)
            v.propagate_to_start()
            aad_derivs = [y.adjoint for y in ys]

        for idx in range(len(y_vals)):
            y_up = list(y_vals)
            y_up[idx] += eps
            y_dn = list(y_vals)
            y_dn[idx] -= eps

            with Tape() as tape:
                v_up = aad_log_linear_interp(query_x, xs, [Number(v) for v in y_up]).value
            with Tape() as tape:
                v_dn = aad_log_linear_interp(query_x, xs, [Number(v) for v in y_dn]).value

            fd = (v_up - v_dn) / (2 * eps)
            assert aad_derivs[idx] == pytest.approx(fd, abs=1e-5)

    def test_flat_extrapolation(self):
        with Tape() as tape:
            ys = [Number(1.0), Number(0.9)]
            v_left = aad_log_linear_interp(-1.0, [0.0, 1.0], ys)
            v_right = aad_log_linear_interp(5.0, [0.0, 1.0], ys)
            assert v_left.value == pytest.approx(1.0)
            assert v_right.value == pytest.approx(0.9)
