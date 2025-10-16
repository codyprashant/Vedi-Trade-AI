"""Utilities for tracking per-indicator reliability statistics across sessions."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict


class IndicatorStats:
    """Track win/loss counts for indicators and persist them to disk."""

    def __init__(self, path: str = "data/indicator_stats.json") -> None:
        self.path = path
        self.stats: defaultdict[str, Dict[str, float]] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "total": 0}
        )
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    for key, value in data.items():
                        bucket = self.stats[key]
                        if isinstance(value, dict):
                            bucket.update({
                                "wins": int(value.get("wins", bucket["wins"])),
                                "losses": int(value.get("losses", bucket["losses"])),
                                "total": int(value.get("total", bucket["total"]))
                            })
            except Exception:
                # Corrupted or unreadable file; start fresh without raising.
                pass

    def record(self, indicator: str, success: bool) -> None:
        bucket = self.stats[indicator]
        if success:
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1
        bucket["total"] += 1

    def win_rate(self, indicator: str) -> float:
        bucket = self.stats[indicator]
        total = bucket.get("total", 0)
        if not total:
            return 0.0
        return round(100.0 * float(bucket.get("wins", 0)) / float(total), 2)

    def save(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        serializable = {
            key: {
                "wins": int(value.get("wins", 0)),
                "losses": int(value.get("losses", 0)),
                "total": int(value.get("total", 0)),
            }
            for key, value in self.stats.items()
        }
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)
