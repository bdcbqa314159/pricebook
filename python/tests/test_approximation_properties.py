"""Property-based tests for the approximation / Chebyshev cluster (plan 0d, P1).

Every property here deliberately breaks symmetry on *all* axes at once —
asymmetric interval ``[a, b]`` with ``a != -b``, a non-even / non-odd target
function, and off-centre, off-node query points. The four historical Chebyshev
bugs (sign-flipped diff matrix, reversed-interval evaluate, operator-specific
BVP, silent Padé truncation) all survived because the tests were symmetric on
some axis (symmetric ``f``, midpoint-only evaluation, sign-invariant checks).
Hypothesis removes the human from input selection, so that whole class of
"test-shaped-wrong" miss cannot recur.
"""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from pricebook.core.approximation import chebyshev_interpolate, chebyshev_nodes
from pricebook.numerical._spectral import chebyshev_expand

# Asymmetric interval: a in [-3, 1], width in [1, 4] -> b = a + width.
# Generically a != -b, so any reversed/negated domain map is exposed.
_intervals = st.tuples(
    st.floats(min_value=-3.0, max_value=1.0),
    st.floats(min_value=1.0, max_value=4.0),
).map(lambda aw: (aw[0], aw[0] + aw[1]))


def _poly_coeffs(max_deg):
    """Random polynomial coefficients (low→high), degree exactly max_deg."""
    return st.lists(
        st.floats(min_value=-2.0, max_value=2.0),
        min_size=max_deg + 1,
        max_size=max_deg + 1,
    )


class TestChebyshevReproducesPolynomials:
    """Degree-≤n interpolation must reproduce polynomials *everywhere*, not just
    at the symmetric/midpoint points the old tests happened to check."""

    @given(interval=_intervals, coeffs=_poly_coeffs(5))
    def test_reproduces_polynomial_at_nodes_and_off_node(self, interval, coeffs):
        a, b = interval
        f = np.polynomial.Polynomial(coeffs)
        deg = len(coeffs) - 1
        n = deg + 2  # n >= deg => exact in exact arithmetic

        interp = chebyshev_interpolate(f, a, b, n)
        nodes = chebyshev_nodes(n, a, b)
        scale = max(1.0, float(np.max(np.abs(f(nodes)))))

        # Exact at the collocation nodes (interpolation property).
        assert np.max(np.abs(interp.evaluate(nodes) - f(nodes))) <= 1e-9 * scale

        # Exact at off-node, off-centre interior points (polynomial exactness).
        q = a + (b - a) * np.array([0.17, 0.34, 0.58, 0.81])
        assert np.max(np.abs(interp.evaluate(q) - f(q))) <= 1e-7 * scale


class TestSpectralEvaluateNotMirrored:
    """Reproduces the reversed-interval mirror bug class: a non-symmetric f
    evaluated off-centre must give f(q), never f(a+b-q)."""

    @given(interval=_intervals, t=st.floats(min_value=0.05, max_value=0.95))
    def test_off_centre_value_is_not_mirrored(self, interval, t):
        a, b = interval
        # Non-even, non-odd, smooth: a mirror about the midpoint changes the value.

        def f(x):
            return np.exp(0.3 * x) + 0.5 * x

        res = chebyshev_expand(f, 24, a, b)
        q = a + t * (b - a)  # off-centre whenever t != 0.5
        val = float(np.atleast_1d(res.evaluate(q))[0])
        assert abs(val - f(q)) <= 1e-7 * max(1.0, abs(f(q)))
