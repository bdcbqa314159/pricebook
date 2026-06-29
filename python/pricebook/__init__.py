__version__ = "1.194.0"


def __getattr__(name: str):
    """Lazy top-level re-exports — avoids circular import chains."""

    _map = {
        # core
        "PricingContext": ("pricebook.core.pricing_context", "PricingContext"),
        "DiscountCurve": ("pricebook.core.discount_curve", "DiscountCurve"),
        "SurvivalCurve": ("pricebook.core.survival_curve", "SurvivalCurve"),
        "Trade": ("pricebook.core.trade", "Trade"),
        "Portfolio": ("pricebook.core.trade", "Portfolio"),
        # serialization
        "to_json": ("pricebook.core.serialization", "to_json"),
        "from_json": ("pricebook.core.serialization", "from_json"),
        "load_trade": ("pricebook.core.serialization", "load_trade"),
        "load_portfolio": ("pricebook.core.serialization", "load_portfolio"),
        # instruments
        "InterestRateSwap": ("pricebook.fixed_income.swap", "InterestRateSwap"),
        "FixedRateBond": ("pricebook.fixed_income.bond", "FixedRateBond"),
        "FRA": ("pricebook.fixed_income.fra", "FRA"),
        "CDS": ("pricebook.credit.cds", "CDS"),
        "Swaption": ("pricebook.options.swaption", "Swaption"),
        # registry
        "get_solver": ("pricebook.registry", "get_solver"),
        "get_tree_european": ("pricebook.registry", "get_tree_european"),
    }

    if name in _map:
        module_path, attr = _map[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)

    raise AttributeError(f"module 'pricebook' has no attribute {name!r}")
