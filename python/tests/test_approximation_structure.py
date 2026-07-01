"""Structural guards for the approximation / Chebyshev cluster (plan 0d, P10).

These tests check *structure*, not numerics. They make the consolidation and the
layering impossible to silently undo: the Chebyshev kernel was duplicated across
three modules and diverged (one differentiation matrix was sign-flipped for
years precisely because two copies existed). The guards below fail the moment a
second kernel definition reappears, or `core.approximation` grows an upward
import (it is the lowest layer — the spectral/model layers depend on it).
"""

import ast
from pathlib import Path

import pricebook

PKG = Path(pricebook.__file__).parent  # scans the package only, not tests/


def _module_level_defs(name: str) -> list[str]:
    """Package files that define a module-level ``def name`` (imports excluded)."""
    hits = []
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                hits.append(path.relative_to(PKG).as_posix())
    return sorted(hits)


class TestNoChebyshevKernelDivergence:
    def test_kernel_defined_once_in_core(self):
        for name in (
            "chebyshev_nodes",
            "chebyshev_coefficients",
            "chebyshev_evaluate",
            "chebyshev_interpolate",
            "barycentric_interpolate",
            "remez",
            "hermite_interpolate",
        ):
            assert _module_level_defs(name) == ["core/approximation.py"], (
                f"{name} must be defined once in core/approximation.py"
            )

    def test_diff_matrix_defined_once_in_spectral(self):
        assert _module_level_defs("chebyshev_diff_matrix") == ["numerical/_spectral.py"]

    def test_no_private_diff_matrix_copy_reappears(self):
        # The pde_advanced copy was deleted in v1.191; it must not return.
        assert _module_level_defs("_chebyshev_diff_matrix") == []


class TestApproximationLayering:
    def test_core_approximation_imports_nothing_upward(self):
        """core is the lowest layer: it may import stdlib/numpy and (in principle)
        other pricebook.core modules, but NOTHING from numerical/models/etc."""
        tree = ast.parse((PKG / "core" / "approximation.py").read_text(encoding="utf-8"))
        upward = []
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                if m.startswith("pricebook.") and not m.startswith("pricebook.core"):
                    upward.append(m)
        assert upward == [], f"core.approximation has upward imports: {upward}"
