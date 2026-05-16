"""Sync psycopg3 connection pool for /v1/ingest.

Uses psycopg_pool.ConnectionPool instead of opening a bare connection on every
call.  The key parameters:

  min_size=0   — no connections are opened at cold start; the first caller
                 blocks briefly while the pool creates one.
  max_size=2   — typical single-request concurrency plus a small buffer for
                 an ingest thread overlapping with a health-check DB touch.
  open=False   — pool is constructed in closed state; init_pool() opens it
                 explicitly from main.py so the timing is predictable.

All callers keep the existing `with get_conn() as conn:` pattern unchanged.
pool.connection() is already a context manager that commits on clean exit and
rolls back on exception, so the explicit conn.commit() calls in ingest.py are
redundant-but-harmless.

init_pool() / close_pool() are called by main.py's CLI entry point so the
pool drains cleanly when the process exits.
"""
import logging

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from core.config import settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def _configure_conn(conn: psycopg.Connection) -> None:
    """Called by the pool whenever it creates a new physical connection."""
    try:
        register_vector(conn)
    except Exception as exc:
        # Fresh DBs don't have `CREATE EXTENSION vector` yet — scripts/
        # migrate.py applies 001_init_extensions.sql on the first connection.
        # Without this guard the migrate one-shot container exits before
        # any SQL runs.
        logger.debug("db: pgvector not registered on this connection: %s", exc)


def init_pool() -> None:
    """Open the module-level connection pool.

    No-op when DATABASE_URL is unset (extract-only deployments that never
    call /v1/ingest don't need a DB connection at all).
    """
    global _pool
    if not settings.database_url:
        logger.info("db: DATABASE_URL not set — pool not initialised (extract-only mode)")
        return
    _pool = ConnectionPool(
        settings.database_url,
        min_size=0,
        max_size=2,
        open=False,
        configure=_configure_conn,
    )
    # open(wait=False) starts the connection handshake in the background so
    # startup doesn't block; the first get_conn() call will wait for a slot.
    _pool.open(wait=False)
    logger.info("db: connection pool opened (min=0, max=2)")


def close_pool() -> None:
    """Drain open connections and close the pool.

    Called from main.py's `finally` block so PostgreSQL Terminate messages
    are sent cleanly before the process exits.
    """
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("db: connection pool closed")


def get_conn():
    """Return a pooled connection context manager with pgvector registered.

    Usage (unchanged from the old bare-connection pattern):

        with get_conn() as conn:
            conn.execute(...)
            conn.commit()

    pool.connection() commits on clean exit and rolls back on exception, then
    returns the connection to the pool rather than closing the underlying TCP
    socket — eliminating per-call open/close churn.
    """
    if _pool is None:
        raise RuntimeError(
            "DB pool is not initialised — either DATABASE_URL is unset or "
            "init_pool() was not called before the first DB access."
        )
    return _pool.connection()
