"""Multiplicative weight learner for indicator contributions."""

import json
import os
from typing import Dict, Optional

import numpy as np


class WeightLearner:
    """Multiplicative weights update on indicator weights based on outcomes."""

    def __init__(
        self,
        path: str = "data/weights_learned.json",
        base_weights: Optional[Dict[str, float]] = None,
        lr: float = 0.05,
        w_min: float = 2.0,
        w_max: float = 30.0,
        seed: int = 7,
    ) -> None:
        self.path = path
        self.lr = float(lr)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.rng = np.random.default_rng(seed)
        self.weights: Dict[str, float] = dict(base_weights or {})
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    self.weights.update({k: float(v) for k, v in data.items()})
            except Exception:
                # Ignore corrupt state and rebuild using base weights.
                pass
        if not self.weights:
            # Ensure at least one weight exists to avoid division by zero.
            self.weights = {"fallback": 100.0}
        self._normalize()

    def _normalize(self) -> None:
        total = float(sum(self.weights.values())) or 1.0
        for key in list(self.weights.keys()):
            self.weights[key] = 100.0 * float(self.weights[key]) / total

    def propose(self) -> Dict[str, float]:
        return dict(self.weights)

    def update(self, contributions: Dict[str, float], reward: float) -> None:
        if not contributions:
            return
        reward_sign = float(np.sign(reward) or 0.0)
        for indicator, contribution in contributions.items():
            if indicator not in self.weights:
                continue
            contrib_value = float(contribution)
            delta = self.lr * reward_sign * contrib_value
            self.weights[indicator] *= float(np.exp(delta))
        for key in list(self.weights.keys()):
            self.weights[key] = float(
                np.clip(self.weights[key], self.w_min, self.w_max)
            )
        self._normalize()

    def save(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.weights, handle, indent=2)

    def snapshot(self, snap_path: str = "data/weights_learned.snap.json") -> None:
        directory = os.path.dirname(snap_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(snap_path, "w", encoding="utf-8") as handle:
            json.dump(self.weights, handle, indent=2)

    def restore(self, snap_path: str = "data/weights_learned.snap.json") -> None:
        if not os.path.exists(snap_path):
            return
        try:
            with open(snap_path, "r", encoding="utf-8") as handle:
                restored = json.load(handle)
            if isinstance(restored, dict):
                self.weights.update({k: float(v) for k, v in restored.items()})
                self._normalize()
                self.save()
        except Exception:
            # Ignore corrupt snapshot files.
            pass
