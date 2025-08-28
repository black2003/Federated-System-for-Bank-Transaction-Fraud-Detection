# config.py
"""
Simple configuration values for the fraud dashboard.
Place this file in the project root (same dir as run.py).

You can override these values with environment variables:
  - ADMIN_USERNAME
  - ADMIN_PASSWORD
  - SECRET_KEY
  - STORAGE_DIR
  - MODEL_DIR
"""

import os

# Admin credentials (default). **Change these for any deployment.**
# Can be overridden via environment variables (recommended).
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

# Flask secret key (used for sessions). Override in production.
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-VERY-IMPORTANT")

# Storage locations (relative to project root by default)
STORAGE_DIR = os.environ.get("STORAGE_DIR", "storage")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(STORAGE_DIR, "models"))

# Convenience helper (optional) to check credentials in one place.
def check_admin(username: str, password: str) -> bool:
    """
    Simple check for admin credentials using the config values.
    Replace/extend with hashed password checks for production.
    """
    return (str(username) == str(ADMIN_USERNAME)) and (str(password) == str(ADMIN_PASSWORD))
