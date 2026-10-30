"""The scaffold's own check: the package imports and states its version.

Trivial on purpose — it exists so the test gates are *exercised* rather than merely configured.
A green CI over zero collected tests is the floor-rule case this project learned the hard way
(CI once gated 177 of 366 tests while every report read "green"), so `verify.py tests` holds a
collection floor and this is the first thing on it.
"""

import pricebook


def test_the_package_imports_and_reports_its_version() -> None:
    assert pricebook.__version__ == "0.0.0"
