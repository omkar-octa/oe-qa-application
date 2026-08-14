"""Standalone browser for what's currently in Postgres.

Prints the 10 most recently ingested documents and 10 sample index_entries
rows so you can eyeball what's in the database without a separate DB client.

Run from src/: python scripts/peek_postgres.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

from models.config import settings


def main() -> None:
    conn = psycopg.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=settings.pg_database,
        connect_timeout=5,
    )

    with conn:
        print("-- Top 10 documents (most recently ingested) --")
        rows = conn.execute(
            "SELECT doc_id, file_name, file_type, page_count, ingested_at, "
            "left(doc_summary, 200) AS summary_preview "
            "FROM documents ORDER BY ingested_at DESC LIMIT 10"
        ).fetchall()
        if not rows:
            print("(no rows)")
        for row in rows:
            print(row)

        print("\n-- Top 10 index_entries --")
        rows = conn.execute(
            "SELECT chunk_id, file_name, granularity, page_start, page_end, "
            "left(display_text, 80) AS preview "
            "FROM index_entries LIMIT 10"
        ).fetchall()
        if not rows:
            print("(no rows)")
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
