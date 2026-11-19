"""L4 — the stateless engine. `price(product, model) -> PricingResult | PricingFailure`.

Importing this package registers every pricer (their `@register` decorators run at import), so
`price` dispatches over all product types. A new pricer module is imported here to be reachable.
"""

from pricebook.engine.capfloor import price_inflation_capfloor
from pricebook.engine.cash import price_fixed_cashflow
from pricebook.engine.equity_vanilla import price_european_equity_option
from pricebook.engine.commodity_vanilla import price_commodity_option
from pricebook.engine.commodity_spread import (
    kirk_weight,
    price_commodity_spread_option,
    spread_total_variance,
)
from pricebook.engine.commodity_asian import (
    geometric_average_forward,
    geometric_average_variance,
    arithmetic_average_moments,
    price_commodity_arithmetic_asian,
    price_commodity_geometric_asian,
)
from pricebook.engine.variance_replication import fair_variance, price_variance_swap
from pricebook.engine.heston_fourier import FourierGrid, heston_call_price, heston_probabilities
from pricebook.engine.equity_rate_hybrid import price_european_equity_option_hull_white
from pricebook.engine.defaultable_equity import price_european_equity_option_defaultable
from pricebook.engine.credit_default_swap import price_credit_default_swap
from pricebook.engine.heston_mc import (
    heston_call_estimate,
    heston_forward_estimate,
    simulate_heston_spots,
)
from pricebook.engine.nominal_cap import price_nominal_cap
from pricebook.engine.inflation import price_inflation_linked_bond, price_zc_inflation_swap
from pricebook.engine import interest_rate_swap as _interest_rate_swap

assert _interest_rate_swap is not None
from pricebook.engine.registry import dispatch as price
from pricebook.engine.yoy import price_yoy_inflation_swap
from pricebook.engine.mc_path_engine import PathGrid, mc_caplet_price, simulate_terminal_forwards
from pricebook.engine.forward_cds import (
    price_credit_default_swaption,
    price_forward_cds,
)
from pricebook.engine.first_to_default_basket import price_first_to_default_basket
from pricebook.engine.synthetic_cdo_tranche import price_synthetic_cdo_tranche
from pricebook.engine.loss_recursion import price_recursion_tranche
from pricebook.engine.registry import register

__all__ = [
    "FourierGrid",
    "heston_call_price",
    "heston_probabilities",
    "simulate_heston_spots",
    "fair_variance",
    "price_variance_swap",
    "price_european_equity_option",
    "price_commodity_option",
    "price_commodity_geometric_asian",
    "price_commodity_arithmetic_asian",
    "price_commodity_spread_option",
    "kirk_weight",
    "spread_total_variance",
    "arithmetic_average_moments",
    "geometric_average_forward",
    "geometric_average_variance","price", "PathGrid", "mc_caplet_price", "simulate_terminal_forwards", "price_nominal_cap", "price_inflation_capfloor", "price_fixed_cashflow", "price_inflation_linked_bond",
           "price_yoy_inflation_swap",
           "price_zc_inflation_swap", "register"]
