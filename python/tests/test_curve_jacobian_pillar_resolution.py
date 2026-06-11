"""Regression for L1 C.2 B1 — curve_jacobian resolves pillar_tenors to indices.

Pre-fix, `bumped_at(j, ...)` was called with the enumeration index `j` of the
user-supplied `pillar_tenors` list. If `pillar_tenors` was shorter than (or
not aligned with) the curve's actual pillars, the Jacobian columns were
silently mislabeled — saying "1y pillar bumped" when actually the 0.25y pillar
was bumped.

Post-fix, each requested tenor is resolved to its actual curve pillar index
before bumping; unresolvable tenors raise `ValueError`.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.curves.curve_risk import curve_jacobian


@pytest.fixture
def curve_with_standard_pillars():
    ref = date(2024, 1, 1)
    return DiscountCurve.flat(ref, 0.05)   # pillars at 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20


class TestPillarResolution:
    def test_default_pillars_unchanged(self, curve_with_standard_pillars):
        """When pillar_tenors is None, the curve's own grid is used."""
        J = curve_jacobian(
            curve_with_standard_pillars,
            query_tenors=[1.0],
        )
        # J has 1 row, n_curve_pillars columns. For a log-linear curve,
        # bumping the 1y pillar produces non-zero sensitivity at 1y.
        assert J.shape == (1, 10)   # 10 non-zero curve pillars

    def test_custom_pillar_tenors_get_correct_columns(self, curve_with_standard_pillars):
        """Custom pillar_tenors=[1, 2, 5] → bump the 1y, 2y, 5y pillars
        (NOT the curve's first three pillars 0.25, 0.5, 1)."""
        J = curve_jacobian(
            curve_with_standard_pillars,
            query_tenors=[1.0],
            pillar_tenors=[1.0, 2.0, 5.0],
        )
        # Bumping the 1y pillar should give the strongest sensitivity for
        # zero_rate at 1y. Columns: 1y, 2y, 5y. Column 0 should dominate.
        assert J[0, 0] > J[0, 1]   # 1y bump > 2y bump for 1y zero
        assert J[0, 0] > J[0, 2]   # 1y bump > 5y bump for 1y zero
        # With log-linear interp, bumping a pillar that's NOT the query
        # tenor itself gives no first-order effect — so column 1 and 2
        # should be ~0 while column 0 ≈ 1.
        assert J[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert abs(J[0, 1]) < 1e-6
        assert abs(J[0, 2]) < 1e-6

    def test_unrecognised_tenor_raises(self, curve_with_standard_pillars):
        """A tenor that doesn't match any curve pillar must raise — silent
        misalignment was the C.2 B1 bug."""
        with pytest.raises(ValueError, match="does not match any curve pillar"):
            curve_jacobian(
                curve_with_standard_pillars,
                query_tenors=[1.0],
                pillar_tenors=[4.0],   # 4y not in [0.25, 0.5, 1, 2, 3, 5, ...]
            )

    def test_tolerance_allows_close_match(self, curve_with_standard_pillars):
        """Sub-tolerance tenor differences match the nearest pillar (the
        curve uses date_from_year_fraction with 365.25 so 1.0y → 1.00274y)."""
        # Curve's actual 1y pillar lands at year_fraction ≈ 1.0 — but with
        # day-count rounding it's ~1.0. Confirm tol=1e-2 succeeds.
        J = curve_jacobian(
            curve_with_standard_pillars,
            query_tenors=[1.0],
            pillar_tenors=[1.001],   # within default tol on curve's pillar
            pillar_tol=1e-2,
        )
        assert J.shape == (1, 1)

    def test_columns_attribute_to_user_supplied_order(self, curve_with_standard_pillars):
        """If user passes pillar_tenors=[5, 2, 1] (out of order), columns of J
        correspond to that order."""
        J = curve_jacobian(
            curve_with_standard_pillars,
            query_tenors=[5.0],
            pillar_tenors=[5.0, 2.0, 1.0],
        )
        # Column 0 (5y bump) should dominate for 5y query; columns 1 and 2 ≈ 0.
        assert J[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert abs(J[0, 1]) < 1e-6
        assert abs(J[0, 2]) < 1e-6
