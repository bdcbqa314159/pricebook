"""Trinomial tree — delegates to numerical._trees.solve_tree."""

from pricebook.numerical._trees import solve_tree, TreeMethod, ExerciseType


def trinomial_european(spot, strike, rate, vol, T, n_steps=200,
                        option_type=None, div_yield=0.0):
    is_call = option_type is None or str(getattr(option_type, 'value', option_type)).lower() != "put"
    return solve_tree(spot, strike, rate, vol, T, TreeMethod.TRINOMIAL, n_steps,
                       ExerciseType.EUROPEAN, is_call=is_call, div_yield=div_yield).price


def trinomial_american(spot, strike, rate, vol, T, n_steps=200,
                        option_type=None, div_yield=0.0):
    is_call = option_type is None or str(getattr(option_type, 'value', option_type)).lower() != "put"
    return solve_tree(spot, strike, rate, vol, T, TreeMethod.TRINOMIAL, n_steps,
                       ExerciseType.AMERICAN, is_call=is_call, div_yield=div_yield).price
