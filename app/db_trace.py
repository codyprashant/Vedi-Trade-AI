from __future__ import annotations

from datetime import datetime, timedelta
from app.db import get_connection  # use existing pool/conn factory
import json
import logging
from typing import Any, List, Dict, Optional

from psycopg2.extras import RealDictCursor

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
                    data JSONB NOT NULL,
                    signal_id BIGINT
                );
                CREATE INDEX IF NOT EXISTS idx_signal_traces_symbol_time
                ON signal_traces(symbol, ts_utc DESC);
                CREATE INDEX IF NOT EXISTS idx_signal_traces_signal_id
                ON signal_traces(signal_id);
                """
            )
            cur.execute(
                "ALTER TABLE signal_traces ADD COLUMN IF NOT EXISTS signal_id BIGINT"
            )
            conn.commit()
    finally:
        conn.close()


def insert_signal_trace(
    symbol: str,
    timeframe: str,
    data: dict,
    *,
    signal_id: Optional[int] = None,
    event: str = "symbol_trace",
):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO signal_traces (symbol, timeframe, event, data, signal_id)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (
                    symbol,
                    timeframe,
                    event,
                    json.dumps(data),
                    signal_id,
                ),
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


def fetch_signal_traces(signal_id: int) -> List[Dict[str, Any]]:
    """Return trace rows associated with a signal id ordered by timestamp."""

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    ts_utc AS timestamp,
                    ts_utc,
                    symbol,
                    timeframe,
                    event,
                    data,
                    signal_id
                FROM signal_traces
                WHERE signal_id = %s
                ORDER BY ts_utc ASC
                """,
                (signal_id,),
            )
            rows = cur.fetchall() or []
            return [dict(row) for row in rows]
    finally:
        conn.close()
