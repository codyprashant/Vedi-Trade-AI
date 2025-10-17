from datetime import datetime, timedelta
from app.db import get_connection  # use existing pool/conn factory
import json
import logging

logger = logging.getLogger(__name__)


def ensure_signal_trace_table():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_traces (
                    id BIGSERIAL PRIMARY KEY,
                    ts_utc TIMESTAMPTZ DEFAULT NOW(),
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    event TEXT NOT NULL,
                    data JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_signal_traces_symbol_time
                ON signal_traces(symbol, ts_utc DESC);
                """
            )
            conn.commit()
    finally:
        conn.close()


def insert_signal_trace(symbol: str, timeframe: str, data: dict):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO signal_traces (symbol, timeframe, event, data)
                VALUES (%s, %s, %s, %s);
                """,
                (symbol, timeframe, "symbol_trace", json.dumps(data)),
            )
            conn.commit()
    finally:
        conn.close()


def cleanup_signal_traces(days: int = 7):
    conn = get_connection()
    cutoff = datetime.utcnow() - timedelta(days=days)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM signal_traces WHERE ts_utc < %s;", (cutoff,))
            deleted = cur.rowcount
            conn.commit()
    finally:
        conn.close()
    logger.info(f"Signal trace cleanup done ({deleted} rows removed).")
