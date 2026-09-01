import os

from sqlalchemy import create_engine, event

from backend import config  # noqa: F401  (importing loads .env)

_WRITE_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "truncate",
    "create",
    "drop",
    "alter",
    "replace",
    "merge",
)


def _is_write_statement(statement: str) -> bool:
    return statement.lstrip().lower().startswith(_WRITE_KEYWORDS)


def writes_allowed() -> bool:
    """DATABASE_URL is the live production Supabase, not a dev copy, so a bare
    local run is denied by default. CI opts in automatically (GitHub Actions
    sets GITHUB_ACTIONS=true); a human sets MOMENTUMNFL_DB_WRITES=1 to mutate
    production intentionally."""
    return (
        os.getenv("GITHUB_ACTIONS") == "true"
        or os.getenv("MOMENTUMNFL_DB_WRITES") == "1"
    )


def _block_unauthorized_writes(
    conn, cursor, statement, parameters, context, executemany
):
    # Fails fast with a clear message; the read-only session below is the
    # real guard.
    if _is_write_statement(statement) and not writes_allowed():
        raise RuntimeError(
            "Refusing to write to the production database (DATABASE_URL is "
            "the live Supabase). Re-run with MOMENTUMNFL_DB_WRITES=1 to "
            "mutate production intentionally. CI runs are allowed "
            "automatically."
        )


def _configure_session(dbapi_connection, connection_record):
    # publish.py qualifies every statement; the search_path only serves
    # ad-hoc queries. A session SET beats the shared role's own setting.
    with dbapi_connection.cursor() as cursor:
        cursor.execute("SET search_path TO nfl, public")
        if not writes_allowed():
            cursor.execute("SET default_transaction_read_only = on")


_engine = None


def __getattr__(name):
    # The engine is created on first use so the module (and the write-guard
    # helpers above) import cleanly without DATABASE_URL configured.
    if name == "engine":
        global _engine
        if _engine is None:
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise ValueError("DATABASE_URL not set in .env")
            _engine = create_engine(database_url)
            event.listens_for(_engine, "connect")(_configure_session)
            event.listens_for(_engine, "before_cursor_execute")(
                _block_unauthorized_writes
            )
        return _engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
