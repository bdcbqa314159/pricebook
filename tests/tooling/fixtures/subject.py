"""A tiny subject for the mutation sweep's own tests — deliberately not part of the library.

Two functions with a **known** oracle situation, so the tool's verdict is checked against a known
answer rather than trusted:

* `guarded` is pinned by `check_subject.py` at two inputs, so breaking it must be **killed**;
* `unguarded` is called but never asserted about, so breaking it must **survive** — a planted hole;
* `twin_a` and `twin_b` contain **textually identical** load-bearing lines, which is the shape a
  string-replacement sweep refuses as non-unique and silently leaves uncounted.
"""


def guarded(x: float) -> float:
    return 2.0 * x + 1.0


def unguarded(x: float) -> float:
    return 3.0 * x - 7.0


def twin_a(x: float) -> float:
    scaled = x * 1.5
    return scaled + 1.0


def twin_b(x: float) -> float:
    scaled = x * 1.5
    return scaled + 2.0
