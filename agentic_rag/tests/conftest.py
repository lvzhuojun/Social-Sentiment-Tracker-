"""
agentic_rag/tests/conftest.py — Pytest session setup for agentic_rag tests.

Sets a dummy OPENAI_API_KEY before any test module is collected so that
agentic_rag/config.py can be imported without a real .env file.
Tests that exercise the LLM API directly must monkeypatch the client.
"""

import os

# Must run before any agentic_rag module is imported.
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-pytest")
