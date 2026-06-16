from typing import Any


def cast_to_bool(value: Any) -> bool:
    """Convert common setup-file values to a boolean."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "f", "no", "n", "off"}:
            return False
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        raise ValueError(f"Cannot cast string value {value!r} to bool")

    return bool(value)
