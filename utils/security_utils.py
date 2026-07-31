import os
import re
from typing import Any


def get_env_value(name: str, default: Any = None, *, cast=str):
    """Safely read an environment value with optional casting."""
    value = os.getenv(name, default)
    if value is None:
        return default

    if cast is bool:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if cast is int:
        try:
            return int(str(value).strip())
        except ValueError:
            return default
    if cast is float:
        try:
            return float(str(value).strip())
        except ValueError:
            return default
    return value


def normalize_text(value: Any, default: str = "N/A") -> str:
    """Return a cleaned display string for user-facing content."""
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def validate_password_strength(password: str):
    """Validate password strength and return a list of violations."""
    errors = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter")
    if not re.search(r"[0-9]", password):
        errors.append("Password must contain at least one digit")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        errors.append("Password must contain at least one special character")
    return errors
