"""Convention data loading — JSON files ↔ convention dataclasses.

Provides :func:`load_conventions` and :func:`save_conventions` for
reading/writing convention registries to JSON files in ``pricebook/data/``.

    from pricebook.core.data_registry import load_conventions, save_conventions, DATA_DIR

    # Load sovereign conventions from JSON (or fall back to defaults)
    conventions = load_conventions(
        "sovereign_conventions.json",
        SovereignConventions,
    )

    # Save updated registry back to JSON
    save_conventions("sovereign_conventions.json", conventions)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

# Data directory: python/pricebook/data/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_conventions(
    filename: str,
    item_type: type[T],
    key_fn=None,
) -> list[T]:
    """Load convention objects from a JSON file.

    Args:
        filename: JSON file name (relative to DATA_DIR).
        item_type: dataclass type with from_dict() classmethod.
        key_fn: optional function to extract key (for dedup).

    Returns:
        List of convention objects. Empty list if file not found.
    """
    path = DATA_DIR / filename
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    items = []
    for entry in data:
        try:
            obj = item_type.from_dict(entry)
            items.append(obj)
        except Exception as e:
            import warnings
            warnings.warn(
                f"Skipping invalid entry in {filename}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    return items


def save_conventions(
    filename: str,
    items: list,
) -> Path:
    """Save convention objects to a JSON file.

    Args:
        filename: JSON file name (relative to DATA_DIR).
        items: list of convention objects with to_dict() method.

    Returns:
        Path to the written file.
    """
    path = DATA_DIR / filename
    data = [item.to_dict() for item in items]

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")  # trailing newline

    return path


def load_or_default(
    filename: str,
    item_type: type[T],
    defaults: list[T],
    key_fn=None,
) -> list[T]:
    """Load from JSON if present, otherwise return defaults.

    This is the standard pattern for convention modules:
    1. Try to load from JSON
    2. If file missing or empty, use hardcoded defaults
    3. Return the list either way

    Args:
        filename: JSON file name.
        item_type: dataclass type with from_dict().
        defaults: hardcoded default conventions.
        key_fn: optional key function for dedup.
    """
    loaded = load_conventions(filename, item_type, key_fn)
    if loaded:
        return loaded
    return defaults
