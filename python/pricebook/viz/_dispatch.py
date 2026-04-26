"""Registry-based dispatch for plot()."""

from __future__ import annotations

from typing import Callable

_INSTRUMENT_REGISTRY: dict[type, Callable] = {}
_RESULT_REGISTRY: dict[type, Callable] = {}
_PANEL_HANDLERS: dict[type, dict[str, Callable]] = {}


def register_instrument(cls):
    """Decorator: register a default dashboard for an instrument class."""
    def decorator(fn):
        _INSTRUMENT_REGISTRY[cls] = fn
        return fn
    return decorator


def register_result(cls):
    """Decorator: register a default dashboard for a result class."""
    def decorator(fn):
        _RESULT_REGISTRY[cls] = fn
        return fn
    return decorator


def register_panels(cls, panels: dict[str, Callable]):
    """Register named panel handlers for an instrument class."""
    _PANEL_HANDLERS[cls] = panels


def get_panel_handler(instrument_type, panel_name: str):
    """Look up a panel handler by instrument type and panel name."""
    handlers = _PANEL_HANDLERS.get(instrument_type, {})
    return handlers.get(panel_name)


def plot(target, curve=None, *, figsize=None, dark=None, **kwargs):
    """Auto-detect instrument/result type and show default dashboard.

    Returns a matplotlib Figure (never calls plt.show()).

        from pricebook.viz import plot
        fig = plot(tlock, curve)        # instrument + curve
        fig = plot(result)              # result only
    """
    from pricebook.viz._backend import apply_theme, create_figure
    from pricebook.viz._generic import plot_summary_table
    from pricebook.viz._theme import get_theme

    theme = get_theme(dark)

    # Dispatch on type
    target_type = type(target)

    # Check instrument registry
    if target_type in _INSTRUMENT_REGISTRY and curve is not None:
        with apply_theme(theme):
            return _INSTRUMENT_REGISTRY[target_type](target, curve, figsize=figsize,
                                                      theme=theme, **kwargs)

    # Check result registry
    if target_type in _RESULT_REGISTRY:
        with apply_theme(theme):
            return _RESULT_REGISTRY[target_type](target, figsize=figsize,
                                                  theme=theme, **kwargs)

    # Fallback: if it has .to_dict(), show summary table
    if hasattr(target, 'to_dict'):
        with apply_theme(theme):
            fig, [ax] = create_figure(1, figsize)
            plot_summary_table(ax, target, curve)
            fig.tight_layout()
            return fig

    supported = list(_INSTRUMENT_REGISTRY.keys()) + list(_RESULT_REGISTRY.keys())
    raise TypeError(
        f"Cannot plot {target_type.__name__}. "
        f"Supported types: {[t.__name__ for t in supported]}"
    )
