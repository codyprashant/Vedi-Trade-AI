"""EWMA-based auto-threshold calibrator for adaptive signal gating."""

import json
import os
from collections import defaultdict
from typing import Dict


class AutoThresholdCalibrator:
    """Maintains EWMA win-rate per (symbol, regime) and maps to thresholds."""

    def __init__(
        self,
        path: str = "data/auto_threshold.json",
        alpha: float = 0.2,
        base: float = 60.0,
        thr_min: float = 50.0,
        thr_max: float = 80.0,
    ) -> None:
        self.path = path
        self.alpha = float(alpha)
        self.base = float(base)
        self.thr_min = float(thr_min)
        self.thr_max = float(thr_max)
        self._ewma: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"wr": 0.55, "n": 0}
        )
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            self._ewma[key].update(value)
            except Exception:
                # Corrupt state is ignored and rebuilt progressively.
                pass

    def _key(self, symbol: str, regime: str) -> str:
        return f"{symbol}|{regime}"

    def update(self, symbol: str, regime: str, success: bool) -> None:
        key = self._key(symbol, regime)
        record = self._ewma[key]
        win_rate = float(record.get("wr", 0.55))
        count = int(record.get("n", 0))
        outcome = 1.0 if success else 0.0
        new_wr = (1 - self.alpha) * win_rate + self.alpha * outcome
        self._ewma[key] = {"wr": float(new_wr), "n": count + 1}

    def threshold(self, symbol: str, regime: str) -> float:
        record = self._ewma[self._key(symbol, regime)]
        win_rate = float(record.get("wr", 0.55))
        offset = -10.0 * (win_rate - 0.55)
        threshold_value = self.base + offset
        threshold_value = max(self.thr_min, min(self.thr_max, threshold_value))
        return float(threshold_value)

    def save(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self._ewma, handle, indent=2)
