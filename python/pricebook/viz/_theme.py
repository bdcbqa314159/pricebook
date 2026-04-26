"""Pricebook visual theme: colors, fonts, light/dark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PricebookTheme:
    """Visual theme for pricebook charts."""
    background: str
    foreground: str
    grid_color: str
    grid_alpha: float
    colors: tuple[str, ...]
    font_family: str
    font_size: int
    title_size: int
    line_width: float


LIGHT = PricebookTheme(
    background="#ffffff",
    foreground="#1a1a2e",
    grid_color="#cccccc",
    grid_alpha=0.3,
    colors=("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#be185d"),
    font_family="sans-serif",
    font_size=11,
    title_size=13,
    line_width=1.8,
)

DARK = PricebookTheme(
    background="#1a1a2e",
    foreground="#e0e0e0",
    grid_color="#444444",
    grid_alpha=0.3,
    colors=("#60a5fa", "#f87171", "#34d399", "#fbbf24", "#a78bfa", "#f472b6"),
    font_family="sans-serif",
    font_size=11,
    title_size=13,
    line_width=1.8,
)

_current_theme: PricebookTheme = LIGHT


def configure_theme(theme: PricebookTheme | None = None, dark: bool = False) -> None:
    """Set the global pricebook theme."""
    global _current_theme
    if theme is not None:
        _current_theme = theme
    elif dark:
        _current_theme = DARK
    else:
        _current_theme = LIGHT


def get_theme(dark: bool | None = None) -> PricebookTheme:
    """Get the active theme, optionally overriding dark mode."""
    if dark is None:
        return _current_theme
    return DARK if dark else LIGHT
