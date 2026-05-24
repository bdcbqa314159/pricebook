"""Binomial tree — delegates to numerical._trees.solve_tree."""

from pricebook.numerical._trees import solve_tree, TreeMethod, ExerciseType

# Re-export OptionType if used by callers
try:
    from pricebook.options.option_types import OptionType
except ImportError:
    class OptionType:
        CALL = "call"
        PUT = "put"


def binomial_european(spot, strike, rate, vol, T, n_steps=200,
                       option_type=None, div_yield=0.0):
    is_call = option_type is None or str(getattr(option_type, 'value', option_type)).lower() != "put"
    return solve_tree(spot, strike, rate, vol, T, TreeMethod.CRR, n_steps,
                       ExerciseType.EUROPEAN, is_call=is_call, div_yield=div_yield).price


def binomial_american(spot, strike, rate, vol, T, n_steps=200,
                       option_type=None, div_yield=0.0):
    is_call = option_type is None or str(getattr(option_type, 'value', option_type)).lower() != "put"
    return solve_tree(spot, strike, rate, vol, T, TreeMethod.CRR, n_steps,
                       ExerciseType.AMERICAN, is_call=is_call, div_yield=div_yield).price
