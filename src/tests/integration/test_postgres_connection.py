import psycopg
import pytest

from models.config import settings


@pytest.mark.integration
def test_postgres_is_reachable():
    conn = psycopg.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=settings.pg_database,
        connect_timeout=5,
    )
    try:
        result = conn.execute("SELECT 1").fetchone()
        assert result == (1,)
    finally:
        conn.close()


@pytest.mark.integration
def test_pgvector_extension_is_available():
    conn = psycopg.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=settings.pg_database,
        connect_timeout=5,
    )
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        row = conn.execute(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        assert row == ("vector",)
    finally:
        conn.close()
