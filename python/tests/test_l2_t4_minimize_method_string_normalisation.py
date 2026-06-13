"""Regression for L2 Wave-2 audit — `minimize(method=...)` rejected the
natural hyphenated method string form.

Pre-fix the string-to-enum dispatch did only ``OptimMethod(method.lower())``,
so passing the form scipy itself uses (``"Nelder-Mead"``, ``"L-BFGS-B"``)
raised:

    ValueError: 'nelder-mead' is not a valid OptimMethod

The enum values use underscores (``"nelder_mead"``).  This is an
ergonomic trap: users copy-paste a method name from scipy docs and get
a hard error, with no hint that the fix is to swap ``-`` for ``_``.

Post-fix the dispatch normalises hyphens to underscores in addition to
lower-casing, so both forms work.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.numerical._optimize import OptimMethod, minimize


def _quad(x):
    return float(np.sum(np.asarray(x) ** 2))


class TestHyphenatedNamesAccepted:
    def test_Nelder_Mead_with_hyphen(self):
        r = minimize(_quad, x0=np.array([3.0, 4.0]), method="Nelder-Mead")
        np.testing.assert_allclose(r.x, [0.0, 0.0], atol=1e-4)

    def test_L_BFGS_B_with_hyphens(self):
        r = minimize(_quad, x0=np.array([3.0, 4.0]), method="L-BFGS-B")
        np.testing.assert_allclose(r.x, [0.0, 0.0], atol=1e-4)


class TestUnderscoreNamesStillAccepted:
    def test_nelder_mead_underscore(self):
        r = minimize(_quad, x0=np.array([3.0, 4.0]), method="nelder_mead")
        np.testing.assert_allclose(r.x, [0.0, 0.0], atol=1e-4)

    def test_l_bfgs_b_underscore(self):
        r = minimize(_quad, x0=np.array([3.0, 4.0]), method="l_bfgs_b")
        np.testing.assert_allclose(r.x, [0.0, 0.0], atol=1e-4)


class TestEnumStillAccepted:
    def test_enum_method(self):
        r = minimize(_quad, x0=np.array([3.0, 4.0]),
                     method=OptimMethod.NELDER_MEAD)
        np.testing.assert_allclose(r.x, [0.0, 0.0], atol=1e-4)


class TestUnknownStillRaises:
    def test_unknown_method_string_raises(self):
        with pytest.raises(ValueError):
            minimize(_quad, x0=np.array([3.0, 4.0]), method="cosmic_descent")
