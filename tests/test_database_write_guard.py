"""The production write gate blocks writes unless CI or a human opts in."""

from __future__ import annotations

from backend.db import _is_write_statement, writes_allowed


def test_writes_allowed_logic(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("MOMENTUMNFL_DB_WRITES", raising=False)
    assert writes_allowed() is False
    monkeypatch.setenv("MOMENTUMNFL_DB_WRITES", "1")
    assert writes_allowed() is True
    monkeypatch.delenv("MOMENTUMNFL_DB_WRITES", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert writes_allowed() is True


def test_is_write_statement():
    for w in (
        "INSERT INTO x VALUES (1)",
        "  \n  UPDATE x SET a=1",
        "DELETE FROM x",
        "TRUNCATE TABLE x",
        "drop table x",
        "ALTER TABLE x ADD c INT",
    ):
        assert _is_write_statement(w) is True, w
    for r in (
        "SELECT * FROM x",
        "  WITH c AS (SELECT 1) SELECT * FROM c",
        "BEGIN",
        "COMMIT",
    ):
        assert _is_write_statement(r) is False, r
