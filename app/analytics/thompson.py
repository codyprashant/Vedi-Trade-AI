import json
import os
import random


class BetaBandit:
    """Two-armed Thompson Sampling bandit with persisted Beta posteriors."""

    def __init__(self, path: str = "data/ts_filters.json"):
        self.path = path
        self.state = {"A": {"a": 1, "b": 1}, "B": {"a": 1, "b": 1}}
        if os.path.exists(path):
            try:
                saved = json.load(open(path))
                self.state.update(saved)
            except Exception:
                pass

    def sample_arm(self) -> str:
        def draw(alpha: float, beta: float) -> float:
            try:
                return random.betavariate(alpha, beta)
            except Exception:
                return alpha / (alpha + beta)

        value_a = draw(self.state["A"]["a"], self.state["A"]["b"])
        value_b = draw(self.state["B"]["a"], self.state["B"]["b"])
        return "A" if value_a >= value_b else "B"

    def update(self, arm: str, success: bool) -> None:
        key = "A" if arm == "A" else "B"
        if success:
            self.state[key]["a"] += 1
        else:
            self.state[key]["b"] += 1

    def save(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w") as handle:
            json.dump(self.state, handle, indent=2)
