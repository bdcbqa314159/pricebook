"""L4 — the stateless engine. `price(product, model) -> PricingResult | PricingFailure`.

Importing this package registers every pricer (their `@register` decorators run at import), so
`price` dispatches over all product types. A new pricer module is imported here to be reachable.
"""

from pricebook.engine.cash import price_fixed_cashflow
from pricebook.engine.inflation import price_inflation_linked_bond, price_zc_inflation_swap
from pricebook.engine.registry import dispatch as price
from pricebook.engine.registry import register

__all__ = ["price", "price_fixed_cashflow", "price_inflation_linked_bond",
           "price_zc_inflation_swap", "register"]
