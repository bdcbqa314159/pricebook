"""L4 — the stateless engine. `price(product, model) -> PricingResult | PricingFailure`.

Importing this package registers every pricer (their `@register` decorators run at import), so
`price` dispatches over all product types. A new pricer module is imported here to be reachable.
"""

from pricebook.engine.capfloor import price_inflation_capfloor
from pricebook.engine.cash import price_fixed_cashflow
from pricebook.engine.equity_vanilla import price_european_equity_option
from pricebook.engine.commodity_vanilla import price_commodity_option
from pricebook.engine.schwartz import (
    price_commodity_option_schwartz,
    schwartz_futures,
    schwartz_total_variance,
)
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
from pricebook.engine.variance_replication import (
    fair_variance,
    generalized_fair_variance,
    price_corridor_variance_swap,
    price_gamma_swap,
    price_variance_swap,
)
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
from pricebook.engine.cev_caplet import (
    cev_absorption_probability,
    cev_call,
    cev_caplet,
    price_cev_cap,
)
from pricebook.engine.smile_caplet import (
    bachelier_caplet,
    displaced_caplet,
    price_bachelier_cap,
    price_displaced_cap,
)
from pricebook.engine.inflation import price_inflation_linked_bond, price_zc_inflation_swap
from pricebook.engine.interest_rate_swap import price_interest_rate_swap
from pricebook.engine.realized_variance_option import (
    heston_fair_variance,
    integrated_variance_moments,
    price_realized_variance_option,
    realized_variance_option_value,
)
from pricebook.engine.vix import (
    price_vix_future,
    price_vix_option,
    vix_affine_coefficients,
    vix_future_value,
    vix_option_value,
    vix_squared,
)
from pricebook.engine.three_halves import (
    expected_variance,
    price_three_halves_variance_swap,
    reciprocal_cir_law,
    three_halves_fair_variance,
)
from pricebook.engine.time_changed_levy import (
    clock_increment_variance,
    discrete_monitoring_adjustment,
    levy_cumulant,
    levy_fair_variance,
    levy_mean,
    levy_multiplier,
    levy_variance,
    price_discrete_levy_variance_swap,
    price_levy_variance_swap,
)
from pricebook.engine.timer_option import (
    price_timer_option,
    simulate_timer_first_passage,
    timer_option_estimate,
)
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
    "price_european_equity_option_hull_white",
    "price_european_equity_option_defaultable",
    "price_credit_default_swap",
    "heston_call_estimate",
    "heston_forward_estimate",
    "price_interest_rate_swap",
    "price_credit_default_swaption",
    "price_forward_cds",
    "price_first_to_default_basket",
    "price_synthetic_cdo_tranche",
    "price_recursion_tranche",
    "FourierGrid",
    "heston_call_price",
    "heston_probabilities",
    "simulate_heston_spots",
    "fair_variance", "generalized_fair_variance",
    "heston_fair_variance", "integrated_variance_moments",
    "realized_variance_option_value", "price_realized_variance_option",
    "vix_squared", "vix_affine_coefficients", "vix_future_value", "vix_option_value",
    "price_vix_future", "price_vix_option",
    "reciprocal_cir_law", "expected_variance", "three_halves_fair_variance",
    "price_three_halves_variance_swap",
    "levy_cumulant", "levy_variance", "levy_mean", "levy_multiplier",
    "clock_increment_variance", "levy_fair_variance", "discrete_monitoring_adjustment",
    "price_levy_variance_swap", "price_discrete_levy_variance_swap",
    "simulate_timer_first_passage", "timer_option_estimate", "price_timer_option",
    "price_gamma_swap", "price_corridor_variance_swap",
    "price_variance_swap",
    "price_european_equity_option",
    "price_commodity_option",
    "price_commodity_geometric_asian",
    "price_commodity_arithmetic_asian",
    "price_commodity_spread_option",
    "price_commodity_option_schwartz",
    "schwartz_futures",
    "schwartz_total_variance",
    "kirk_weight",
    "spread_total_variance",
    "arithmetic_average_moments",
    "geometric_average_forward",
    "geometric_average_variance","price", "PathGrid", "mc_caplet_price", "simulate_terminal_forwards", "price_nominal_cap", "price_bachelier_cap", "price_displaced_cap",
    "bachelier_caplet", "displaced_caplet",
    "price_cev_cap", "cev_call", "cev_caplet", "cev_absorption_probability", "price_inflation_capfloor", "price_fixed_cashflow", "price_inflation_linked_bond",
           "price_yoy_inflation_swap",
           "price_zc_inflation_swap", "register"]
