"""Standalone check that the local Postgres + pgvector container is reachable.

Run from src/: python scripts/check_postgres.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

from models.config import settings


def main() -> None:
    print(f"Connecting to postgres://{settings.pg_user}@{settings.pg_host}:{settings.pg_port}/{settings.pg_database} ...")

    try:
        conn = psycopg.connect(
            host=settings.pg_host,
            port=settings.pg_port,
            user=settings.pg_user,
            password=settings.pg_password,
            dbname=settings.pg_database,
            connect_timeout=5,
        )
    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {exc}")
        sys.exit(1)

    with conn:
        version = conn.execute("SELECT version()").fetchone()[0]
        print(f"Connected. {version}")

        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        row = conn.execute(
            "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        if row:
            print(f"pgvector extension available (version {row[1]})")
        else:
            print("pgvector extension is NOT available")
            sys.exit(1)


if __name__ == "__main__":
    main()
