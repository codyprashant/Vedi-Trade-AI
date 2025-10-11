from __future__ import annotations

import os
from typing import Dict, Any

from supabase import create_client, Client


def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
    return create_client(url, key)


def save_signal(client: Client, table: str, record: Dict[str, Any]) -> None:
    client.table(table).insert(record).execute()