import os
import sys
from dotenv import load_dotenv
import psycopg2


def main():
    load_dotenv()
    host = os.getenv("DB_HOST")
    port = int(os.getenv("DB_PORT"))
    db = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    if not all([host, port, db, user, password]):
        print("Missing DB env vars: DB_HOST,DB_PORT, DB_NAME, DB_USER, DB_PASSWORD", file=sys.stderr)
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