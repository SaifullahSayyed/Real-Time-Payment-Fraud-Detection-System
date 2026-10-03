"""
tests/conftest.py
──────────────────
Pytest configuration and global test fixtures.

Provides default environment variables (e.g. ANALYST_TOKEN) so that test runs
in fresh environments or CI have consistent test tokens set.
Tests that test missing or invalid tokens can explicitly modify or pop the env var.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Ensure standard test environment variables are set during test runs."""
    old_token = os.environ.get("ANALYST_TOKEN")
    if not old_token:
        os.environ["ANALYST_TOKEN"] = "test-analyst-token"
    yield
    # Restore prior state if it existed, or leave clean
    if old_token is None:
        os.environ.pop("ANALYST_TOKEN", None)
    else:
        os.environ["ANALYST_TOKEN"] = old_token
