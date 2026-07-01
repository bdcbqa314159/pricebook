"""Structural guard against Gauss-Legendre quadrature re-divergence (from v1.207).

The Gauss-Legendre *integration logic* (map `leggauss` nodes to [a, b] + weighted
sum) was duplicated across `_integrate` and `_spectral`; v1.207 consolidated it so
`numerical/_integrate.py` is the single home and `spectral_integrate` delegates.
These guards fail if a new integrator re-implements it, or if `spectral_integrate`
re-inlines it — the quadrature analogue of the Chebyshev P10 guard.
"""

import ast
from pathlib import Path

import pricebook

PKG = Path(pricebook.__file__).parent

# The only legitimate `leggauss` users, each for a distinct reason. A new file
# calling leggauss is almost certainly a re-implemented integrator — trip the guard.
_LEGGAUSS_ALLOWED = {
    "numerical/_integrate.py",  # canonical Gauss-Legendre integrator (+ complex contour)
    "numerical/_spectral.py",  # legendre_nodes_weights accessor
    "numerical/_qmc.py",  # sparse-grid node/weight construction (not integration)
}


def _module_level_defs(name: str) -> list[str]:
    hits = []
    for path in PKG.rglob("*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                hits.append(path.relative_to(PKG).as_posix())
    return sorted(hits)


def _files_containing(text: str) -> set[str]:
    return {
        p.relative_to(PKG).as_posix()
        for p in PKG.rglob("*.py")
        if text in p.read_text(encoding="utf-8")
    }


class TestQuadratureSingleSourced:
    def test_gauss_legendre_backend_defined_once(self):
        """The canonical Gauss-Legendre integrator lives once, in _integrate."""
        assert _module_level_defs("_gauss_legendre") == ["numerical/_integrate.py"]

    def test_integrate_dispatcher_defined_once(self):
        assert _module_level_defs("integrate") == ["numerical/_integrate.py"]

    def test_no_new_leggauss_home(self):
        extra = _files_containing("leggauss") - _LEGGAUSS_ALLOWED
        assert extra == set(), (
            f"new Gauss-Legendre call site(s): {extra} — reuse _integrate.integrate "
            f"instead of re-deriving quadrature"
        )

    def test_spectral_integrate_delegates(self):
        """spectral_integrate must call the canonical engine, not re-inline it."""
        src = (PKG / "numerical" / "_spectral.py").read_text(encoding="utf-8")
        fn = next(
            n
            for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name == "spectral_integrate"
        )
        body = ast.get_source_segment(src, fn)
        assert "integrate(" in body, "spectral_integrate must delegate to _integrate.integrate"
        assert "leggauss" not in body, "spectral_integrate must not re-inline Gauss-Legendre"
