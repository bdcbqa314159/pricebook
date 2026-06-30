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
from hypothesis import assume, given
from hypothesis import strategies as st

from pricebook.core.approximation import (
    barycentric_interpolate,
    chebyshev_interpolate,
    chebyshev_nodes,
    pade_approximant,
)
from pricebook.numerical._spectral import chebyshev_expand


def _maclaurin_of_ratio(num, den, order):
    """Maclaurin series s_0..s_order of the rational num(x)/den(x).

    Both coefficient arrays are low→high. Computes the reciprocal series of
    `den` (q_0 ≠ 0) then convolves with `num`. Used to verify that a Padé
    [L/M] approximant actually matches its Taylor input to order L+M.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    recip = np.zeros(order + 1)
    recip[0] = 1.0 / den[0]
    for k in range(1, order + 1):
        acc = sum(den[j] * recip[k - j] for j in range(1, min(k, len(den) - 1) + 1))
        recip[k] = -acc / den[0]
    s = np.zeros(order + 1)
    for k in range(order + 1):
        s[k] = sum(num[i] * recip[k - i] for i in range(min(k, len(num) - 1) + 1))
    return s


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


class TestPadeOrderMatching:
    """The *defining* Padé contract: P/Q must match the Taylor input to order
    L+M. Verifying only a value (as the old tests did) lets a silent truncation
    pass; re-expanding P/Q as a power series and comparing every coefficient
    does not."""

    @given(data=st.data())
    def test_pade_matches_taylor_to_order_L_plus_M(self, data):
        L = data.draw(st.integers(min_value=1, max_value=3))
        M = data.draw(st.integers(min_value=1, max_value=3))
        n = L + M
        c = [data.draw(st.floats(min_value=0.5, max_value=2.0))]  # c0 != 0
        c += data.draw(
            st.lists(
                st.floats(min_value=-1.0, max_value=1.0),
                min_size=n,
                max_size=n,
            )
        )

        try:
            pade = pade_approximant(c, L, M)
        except ValueError:
            assume(False)  # singular / ill-conditioned draw — not a valid [L/M]

        # Restrict to genuinely well-conditioned approximants: the cond<1e12
        # guard is generous, and near it the recovered q blows up and the
        # re-expansion amplifies roundoff. Large |q| is a cheap conditioning
        # proxy; the matching contract is what we test, on the regime where it
        # is numerically meaningful.
        qmax = float(np.max(np.abs(pade.denominator)))
        assume(qmax < 1e3)

        s = _maclaurin_of_ratio(pade.numerator, pade.denominator, n)
        scale = max(1.0, float(np.max(np.abs(c))))
        # Matching error scales with conditioning (∝ |q|·eps); a flat 1e-7 is too
        # tight for moderately-conditioned [3/3]. This stays ≤ 1e-5 (qmax<1e3) —
        # still ~5 orders below the O(1) error a silent truncation would produce.
        tol = 1e-8 * scale * max(1.0, qmax)
        assert np.max(np.abs(s - np.asarray(c))) <= tol


class TestBarycentricReproducesPolynomials:
    """Barycentric interpolation through arbitrary nodes must reproduce a
    degree-≤n polynomial everywhere — at the nodes and off them. Hypothesis
    supplies asymmetric, irregular node sets the hand-written tests wouldn't."""

    @given(
        anchors=st.lists(
            st.floats(min_value=-4.0, max_value=4.0), min_size=4, max_size=8, unique=True
        ),
        coeffs=st.lists(st.floats(min_value=-2.0, max_value=2.0), min_size=3, max_size=4),
    )
    def test_reproduces_polynomial(self, anchors, coeffs):
        nodes = np.sort(np.asarray(anchors))
        # Well-separated nodes only: float-`unique` can still place nodes ~1e-7
        # apart, which makes the barycentric weights blow up (ill-conditioned).
        # The interpolation contract is what we test, on the meaningful regime.
        assume(np.min(np.diff(nodes)) > 0.1)
        f = np.polynomial.Polynomial(coeffs)  # degree ≤ len(nodes)-1
        interp = barycentric_interpolate(nodes, f(nodes))
        scale = max(1.0, float(np.max(np.abs(f(nodes)))))
        # at the nodes
        assert np.max(np.abs(interp.evaluate(nodes) - f(nodes))) <= 1e-9 * scale
        # off-node interior points
        q = np.linspace(nodes[0], nodes[-1], 11)[1:-1]
        assert np.max(np.abs(interp.evaluate(q) - f(q))) <= 1e-7 * scale
