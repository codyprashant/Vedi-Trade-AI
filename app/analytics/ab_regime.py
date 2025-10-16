import hashlib
import json
import os
from collections import defaultdict, deque


class RegimeAB:
    """Per-symbol, per-regime A/B tracker with deterministic arm assignment."""

    def __init__(self, path: str = "data/ab_regime.json", window: int = 300, bucket: int = 7):
        self.path = path
        self.window = window
        self.bucket = bucket
        self.history = defaultdict(lambda: {"A": deque(maxlen=window), "B": deque(maxlen=window)})
        if os.path.exists(path):
            try:
                raw = json.load(open(path))
            except Exception:
                raw = {}
            for key, values in raw.items():
                self.history[key]["A"] = deque(values.get("A", []), maxlen=window)
                self.history[key]["B"] = deque(values.get("B", []), maxlen=window)

    def _key(self, symbol: str, regime: str) -> str:
        return f"{symbol}|{regime}"

    def assign(self, symbol: str, regime: str, seed: str = "") -> str:
        payload = f"{symbol}|{regime}|{seed}"
        hashed = int(hashlib.md5(payload.encode()).hexdigest(), 16)
        return "A" if (hashed % 100) < (50 + self.bucket) else "B"

    def record(self, symbol: str, regime: str, arm: str, success: bool) -> None:
        arm_key = "A" if arm == "A" else "B"
        self.history[self._key(symbol, regime)][arm_key].append(1 if success else 0)

    def wr(self, symbol: str, regime: str, arm: str) -> float:
        arm_key = "A" if arm == "A" else "B"
        window = self.history[self._key(symbol, regime)][arm_key]
        return (100.0 * sum(window) / len(window)) if window else 0.0

    def save(self) -> None:
        payload = {}
        for key, record in self.history.items():
            payload[key] = {"A": list(record["A"]), "B": list(record["B"])}
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w") as handle:
            json.dump(payload, handle, indent=2)
