"""L4 engine — the stateless heart: product→pricer via the registry (CLAUDE.md §1).

Importing the package registers every pricer (their `@register` decorators run at import),
so `price` dispatches over all product types. New pricer modules are imported here to
populate the registry.
"""

from pricebook_ng.engine.linear import price, price_swap
from pricebook_ng.engine.registry import register

__all__ = ["price", "price_swap", "register"]
