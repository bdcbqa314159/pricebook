"""L5 risk — generic bump-and-reprice greeks on the `Priceable` protocol (CLAUDE.md §1).

Risk depends DOWN on the engine + snapshot; it never inspects an instrument's type (the engine
registry dispatches). Greeks are generic over the snapshot's open keys — one key-blind FD core,
one `Bump` strategy per shape. Nothing at L≤4 imports this layer (enforced by `verify.py acyclic`).
"""

from pricebook_ng.risk.bump import Bump, CurveBasisBump, CurveIdentityBump, SurfaceBump
from pricebook_ng.risk.greeks import central_diff, ir_basis_delta, ir_delta, vol_vega
from pricebook_ng.risk.priceable import Priceable, priceable

__all__ = [
    "Priceable",
    "priceable",
    "Bump",
    "CurveIdentityBump",
    "CurveBasisBump",
    "SurfaceBump",
    "central_diff",
    "ir_delta",
    "ir_basis_delta",
    "vol_vega",
]
