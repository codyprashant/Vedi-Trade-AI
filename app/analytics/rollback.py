import json
import os
from collections import deque


class RollbackGuard:
    """Tracks rolling win-rate deltas between current and baseline strategies."""

    def __init__(
        self,
        path: str = "data/rollback.json",
        window: int = 300,
        min_trades: int = 100,
        budget: float = -5.0,
    ):
        self.path = path
        self.window = window
        self.min_trades = min_trades
        self.budget = budget
        self.current = deque(maxlen=window)
        self.baseline = deque(maxlen=window)
        if os.path.exists(path):
            try:
                payload = json.load(open(path))
            except Exception:
                payload = {}
            self.current = deque(payload.get("current", []), maxlen=window)
            self.baseline = deque(payload.get("baseline", []), maxlen=window)

    def record(self, success: bool, baseline_success: bool) -> None:
        self.current.append(1 if success else 0)
        self.baseline.append(1 if baseline_success else 0)

    def delta_wr(self) -> float:
        if not self.current or not self.baseline:
            return 0.0
        current_wr = 100.0 * sum(self.current) / len(self.current)
        baseline_wr = 100.0 * sum(self.baseline) / len(self.baseline)
        return current_wr - baseline_wr

    def should_rollback(self) -> bool:
        return len(self.current) >= self.min_trades and self.delta_wr() <= self.budget

    def save(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w") as handle:
            json.dump({"current": list(self.current), "baseline": list(self.baseline)}, handle, indent=2)
