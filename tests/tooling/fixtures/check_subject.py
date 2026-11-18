"""The subject's oracle — deliberately weak in one place.

Named `check_` rather than `test_` so pytest does not collect it with the real suite: it exists to
be *run by the sweep*, and a planted hole that the main suite reported as a failure would be a
self-inflicted red. pytest still collects it when the path is passed explicitly, which is how the
sweep runs it.
"""

from fixtures.subject import guarded, twin_a, twin_b, unguarded


def test_guarded_is_pinned() -> None:
    assert guarded(0.0) == 1.0
    assert guarded(2.0) == 5.0


def test_unguarded_is_merely_CALLED() -> None:
    """The planted hole: the value is never asserted, so any change to it survives."""
    assert isinstance(unguarded(1.0), float)


def test_both_twins_are_pinned() -> None:
    assert twin_a(2.0) == 4.0
    assert twin_b(2.0) == 5.0
