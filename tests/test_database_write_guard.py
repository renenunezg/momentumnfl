"""The production write gate blocks writes unless CI or a human opts in."""

from __future__ import annotations

from backend.db import writes_allowed


def test_writes_allowed_logic(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("MOMENTUMNFL_DB_WRITES", raising=False)
    assert writes_allowed() is False
    monkeypatch.setenv("MOMENTUMNFL_DB_WRITES", "1")
    assert writes_allowed() is True
    monkeypatch.delenv("MOMENTUMNFL_DB_WRITES", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert writes_allowed() is True
