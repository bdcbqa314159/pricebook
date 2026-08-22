"""L6 shell — the imperative shell: frozen Trade/Book, marking, portfolio risk (CLAUDE.md §1).

The stateful shell that holds positions and CALLS the stateless core, never computing. Depends DOWN
on the engine (L4), risk (L5), and products (L2); imported by NOTHING (enforced by `verify.py acyclic`).
This is the LAST spine layer — with it, all seven layers (L0–L6) are live.
"""

from pricebook_ng.shell.benefit import benefit, register_benefit
from pricebook_ng.shell.booking import BookedTrade, realized, total
from pricebook_ng.shell.portfolio import (
    Book,
    Product,
    Trade,
    book_dv01,
    book_priceable,
    mark,
    mark_book,
)

__all__ = [
    "Trade",
    "Book",
    "Product",
    "mark",
    "mark_book",
    "book_priceable",
    "book_dv01",
    "BookedTrade",
    "realized",
    "total",
    "benefit",
    "register_benefit",
]
