"""PostgreSQL connection pool with pgvector type registration."""

from __future__ import annotations

import time
from functools import lru_cache

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .settings import get_settings

# Transient connect failures (DNS, timeout, host unreachable) are retried.
_CONNECT_RETRY_ATTEMPTS = 5
_CONNECT_RETRY_BASE_SECONDS = 1.0


def _configure(conn: psycopg.Connection) -> None:
    # Requires the `vector` extension to already exist in the database.
    register_vector(conn)


@lru_cache
def get_pool() -> ConnectionPool:
    settings = get_settings()
    pool = ConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=10,
        configure=_configure,
        open=True,
        kwargs={"autocommit": False},
    )
    return pool


def raw_connect() -> psycopg.Connection:
    """A plain connection without vector registration (used for schema init)."""
    settings = get_settings()
    return psycopg.connect(settings.database_url, autocommit=True)


def check_connection(timeout: int = 5) -> tuple[bool, str]:
    """Verify the database is reachable and the schema exists.

    Returns (ok, message). Uses a short-lived direct connection with a connect
    timeout. Transient connect failures (``OperationalError``, including DNS
    ``getaddrinfo failed``) are retried up to 5 times with exponential backoff
    (1s, 2s, 4s, 8s). Schema-missing results are not retried.
    """
    settings = get_settings()
    last_exc: BaseException | None = None
    for attempt in range(1, _CONNECT_RETRY_ATTEMPTS + 1):
        try:
            with psycopg.connect(
                settings.database_url, connect_timeout=timeout, autocommit=True
            ) as conn:
                conn.execute("SELECT 1")
                reg = conn.execute("SELECT to_regclass('public.documents')").fetchone()[0]
                fn = conn.execute(
                    "SELECT to_regprocedure('search_hybrid(text,vector,int,int,int)')"
                ).fetchone()[0]
            if reg is None:
                return (
                    False,
                    "Connected, but the 'documents' table is missing. Run sql/schema.sql.",
                )
            if fn is None:
                return (
                    False,
                    "Connected, but the search_hybrid() function is missing. Run sql/schema.sql.",
                )
            return (True, "Connection OK.")
        except psycopg.OperationalError as exc:
            last_exc = exc
            if attempt < _CONNECT_RETRY_ATTEMPTS:
                time.sleep(_CONNECT_RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
                continue
        except Exception as exc:  # noqa: BLE001 - surface any other driver error immediately
            return (False, f"{type(exc).__name__}: {exc}")

    return (
        False,
        f"{type(last_exc).__name__}: {last_exc} "
        f"(failed after {_CONNECT_RETRY_ATTEMPTS} attempts with exponential backoff)",
    )
