import os
import sys
from dotenv import load_dotenv
import psycopg2


def main():
    load_dotenv()
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    db = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    if not all([db, user, password]):
        print("Missing DB env vars: POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD", file=sys.stderr)
        sys.exit(2)

    conn = psycopg2.connect(host=host, port=port, dbname=db, user=user, password=password)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("truncate table public.signals;")
        print("Truncated public.signals")
    finally:
        conn.close()


if __name__ == "__main__":
    main()