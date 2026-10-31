"""The normal CDF adapter -- testing OUR wrapper, not scipy's numerics.

The distinction matters: scipy is a pinned dependency whose correctness is not this library's
claim. What is this library's claim is that the adapter exposes it faithfully, and in particular
that the symmetry the cap/floor's put-call-parity oracle leans on holds exactly.
"""

import pytest

from pricebook.foundation.distributions import norm_cdf


def test_phi_of_zero_is_one_half() -> None:
    assert norm_cdf(0.0) == 0.5


def test_the_symmetry_the_parity_ORACLE_depends_on() -> None:
    """`Phi(x) + Phi(-x) = 1`. The cap/floor parity `cap - floor = N*psi*Pn*(m - K)` follows from
    exactly this, so it is asserted here rather than assumed there."""
    for x in (0.001, 0.1, 0.5, 1.0, 2.5, 6.0):
        assert norm_cdf(x) + norm_cdf(-x) == pytest.approx(1.0, abs=1e-15)


def test_it_is_monotone_and_its_tails_saturate() -> None:
    xs = [-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0]
    values = [norm_cdf(x) for x in xs]
    assert values == sorted(values)
    assert norm_cdf(-40.0) == pytest.approx(0.0, abs=1e-15)
    assert norm_cdf(40.0) == pytest.approx(1.0, abs=1e-15)


def test_the_familiar_quantiles() -> None:
    """Textbook values, as a sanity check that the adapter is not silently shifted or scaled."""
    assert norm_cdf(1.0) == pytest.approx(0.8413447461, abs=1e-9)
    assert norm_cdf(1.959963985) == pytest.approx(0.975, abs=1e-9)
    assert norm_cdf(-2.326347874) == pytest.approx(0.01, abs=1e-9)
