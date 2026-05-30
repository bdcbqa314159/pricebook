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


def _validate_filename(filename: str) -> None:
    """Guard against path traversal and invalid filenames."""
    if not filename or ".." in filename or filename.startswith("/"):
        raise ValueError(f"Invalid filename: {filename!r}. Must be a simple name relative to DATA_DIR.")


def load_conventions(
    filename: str,
    item_type: type[T],
    key_fn=None,
) -> list[T]:
    """Load convention objects from a JSON file.

    Args:
        filename: JSON file name (relative to DATA_DIR). Must not contain '..'.
        item_type: dataclass type with from_dict() classmethod.
        key_fn: optional function to extract key (for dedup).

    Returns:
        List of convention objects. Empty list if file not found.
    """
    _validate_filename(filename)
    path = DATA_DIR / filename
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        import warnings
        warnings.warn(f"{filename}: expected JSON array, got {type(data).__name__}", RuntimeWarning, stacklevel=2)
        return []

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
        filename: JSON file name (relative to DATA_DIR). Must not contain '..'.
        items: list of convention objects with to_dict() method.

    Returns:
        Path to the written file.
    """
    _validate_filename(filename)
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


def load_registry(
    filename: str,
    item_type: type[T],
    key_fn,
    defaults: dict[str, T],
) -> dict[str, T]:
    """Load a keyed registry from JSON, falling back to hardcoded defaults.

    Args:
        filename: JSON file name (relative to DATA_DIR).
        item_type: dataclass type with from_dict().
        key_fn: function to extract dict key from each item (e.g. lambda c: c.market_code).
            Must not be None.
        defaults: hardcoded default registry dict.

    Returns:
        Dict mapping key → convention object.

    Raises:
        ValueError: if key_fn is None.
    """
    if key_fn is None:
        raise ValueError("key_fn must not be None for load_registry")
    loaded = load_conventions(filename, item_type)
    if loaded:
        return {key_fn(item): item for item in loaded}
    return defaults
