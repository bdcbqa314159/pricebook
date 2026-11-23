"""The `SmileLike` protocol (L1). Extracted at tranche 76, ruling 0401(2).

Its home is here because the protocol lives at L1 beside `ImpliedVolSkew`, so the L4 replication
strip depends **down** on it (§2) and the model-smile presenters at L3/L4 implement a lower-layer
type. These are the protocol's own oracles; the strip's behaviour through it is tested at L4.

**A protocol with no members is satisfied by everything** (tranche 33b's input-population shape), so
the only thing worth asserting about a `@runtime_checkable` protocol is that it **discriminates** —
and that is what a mutation removing its single member destroys while every price stays green.
"""

import pytest

from pricebook.market.equity_skew import ImpliedVolSkew, SmileLike


class _Smile:
    """The minimum that satisfies the protocol — one method, nothing inherited."""

    def volatility(self, strike: float, forward: float, years: float = 0.0) -> float:
        return 0.2


class _NotASmile:
    """Everything except the member that matters."""

    def variance(self, strike: float, forward: float, years: float = 0.0) -> float:
        return 0.04


def test_the_concrete_SKEW_satisfies_the_protocol_it_was_abstracted_from():
    assert isinstance(ImpliedVolSkew(0.28, -0.15), SmileLike)
    assert isinstance(ImpliedVolSkew(0.2, 0.0, 0.01), SmileLike)


def test_a_MINIMAL_implementation_satisfies_it_so_nothing_is_inherited():
    # The point of the extraction: a presenter needs the one method and no base class. Tranche 75's
    # Heston smile was a *subclass* of `ImpliedVolSkew` before this existed.
    assert isinstance(_Smile(), SmileLike)


@pytest.mark.parametrize("candidate", [object(), _NotASmile(), 3.14, "not a smile", None])
def test_the_protocol_REJECTS_what_is_not_a_smile_which_an_empty_one_could_not(candidate):
    # The non-degeneracy premise (T33b): remove `volatility` from the protocol and every one of
    # these starts passing, while no price changes — which is exactly how an empty protocol hides.
    assert not isinstance(candidate, SmileLike)


def test_the_protocol_has_EXACTLY_the_member_the_strip_calls():
    # One real member, deliberately: `volatility` is the only thing the replication strip ever asks
    # of a smile, so the interface is the minimum rather than a copy of `ImpliedVolSkew`'s surface.
    assert hasattr(SmileLike, "volatility")
    assert not hasattr(SmileLike, "atm_volatility")
    assert not hasattr(SmileLike, "is_floored")
