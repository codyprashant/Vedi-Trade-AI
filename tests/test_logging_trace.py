import asyncio
import ast
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")
import pandas as pd

import app.signal_engine as signal_module
from app.signal_engine import SignalEngine


class StubIndicator:
    def __init__(self, direction: str, value: dict | None = None, strength: float = 70.0):
        self.direction = direction
        self.value = value or {}
        self.vote = 1
        self.strength = strength
        self.label = "stub"


def _make_dataframe(count: int, freq: str) -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp.utcnow(), periods=count, freq=freq)
    data = {
        "time": index,
        "open": [1800 + i * 0.1 for i in range(count)],
        "high": [1800 + i * 0.1 + 0.2 for i in range(count)],
        "low": [1800 + i * 0.1 - 0.2 for i in range(count)],
        "close": [1800 + i * 0.1 for i in range(count)],
        "volume": [100 + i for i in range(count)],
    }
    return pd.DataFrame(data)


def _fake_fetch(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    freq_map = {"15min": "15T", "1h": "1H", "4h": "4H"}
    freq = freq_map.get(timeframe, "15T")
    return _make_dataframe(max(count, 60), freq)


@pytest.mark.asyncio
async def test_logging_trace_events(caplog, monkeypatch):
    caplog.set_level("DEBUG")

    engine = SignalEngine(_fake_fetch)

    # Strategy configuration stub
    strategy_config = {
        "indicator_params": {},
        "weights": {
            "RSI": 15,
            "MACD": 20,
            "STOCH": 10,
            "BBANDS": 10,
            "SMA": 10,
            "EMA": 10,
            "MTF": 10,
            "ATR_STABILITY": 5,
            "PRICE_ACTION": 10,
        },
        "symbols": ["TEST"],
        "primary_timeframe": "15min",
        "confirmation_timeframe": "1h",
        "trend_timeframe": "4h",
        "run_interval_seconds": 0,
    }
    monkeypatch.setattr(signal_module, "get_active_strategy_config", lambda: strategy_config)

    stub_res = {
        "RSI": StubIndicator("buy", {"rsi": 55.0}),
        "MACD": StubIndicator("buy", {"histogram": 0.02}),
        "STOCH": StubIndicator("buy", {"k": 70.0, "d": 65.0}),
        "BBANDS": StubIndicator("buy", {"upper": 1.0, "lower": -1.0}),
        "SMA": StubIndicator("buy", {"sma": 0.0}),
        "EMA": StubIndicator("buy", {"ema": 0.0}),
        "MTF": StubIndicator("buy", {}),
        "ATR": StubIndicator("buy", {"atr": 1.0}),
        "PRICE_ACTION": StubIndicator("buy", {"direction": "buy"}),
    }

    monkeypatch.setattr(signal_module, "compute_indicators", lambda df, params: {})
    monkeypatch.setattr(signal_module, "evaluate_signals", lambda df, ind, params: stub_res)
    monkeypatch.setattr(signal_module, "compute_strategy_strength", lambda res, weights: {})
    monkeypatch.setattr(signal_module, "best_signal", lambda strat: {
        "direction": "buy",
        "strength": 80.0,
        "strategy": "stub",
        "contributions": {},
        "components": {"core": {"strength": 80.0}},
    })

    def fake_vote(_res, _weights, threshold):
        return {
            "final_direction": "buy",
            "confidence": 0.75,
            "strong_signals": 3,
            "threshold_used": threshold,
        }

    monkeypatch.setattr(signal_module, "compute_weighted_vote_aggregation", fake_vote)
    monkeypatch.setattr(signal_module, "price_action_direction", lambda df, lookback, params: "buy")
    monkeypatch.setattr(signal_module, "atr_last", lambda df, params=None: 1.0)
    monkeypatch.setattr(signal_module, "atr_last_and_mean", lambda df, length=14, mean_window=50, params=None: (1.0, 1.0))
    monkeypatch.setattr(signal_module, "compute_bounded_alignment_boost", lambda **kwargs: (1.1, {}))
    monkeypatch.setattr(signal_module, "ema_trend_direction", lambda df, short_len=50, long_len=200: "Bullish")
    monkeypatch.setattr(signal_module, "insert_indicator_snapshot", lambda payload: None)
    monkeypatch.setattr(signal_module, "insert_signal", lambda payload: None)

    # Sanity filter stub to guarantee pass
    engine.sanity_filter.validate_signal = lambda **kwargs: (True, "pass", {"volatility": 0.5})

    # Threshold manager stubs
    engine.threshold_manager.compute_adaptive_threshold = lambda **kwargs: (55.0, {})
    engine.threshold_manager.compute_dynamic_threshold_with_votes = lambda **kwargs: (
        60.0,
        {"vote_adjustments": {"total_adjustment": -5.0}},
    )

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(signal_module.asyncio, "sleep", fake_sleep)

    def extract_events(records):
        events = []
        for record in records:
            msg = record.getMessage()
            if isinstance(msg, str) and msg.startswith("{") and msg.endswith("}"):
                try:
                    data = ast.literal_eval(msg)
                except Exception:
                    continue
                if isinstance(data, dict) and "event" in data:
                    events.append(data["event"])
        return events

    async def run_once(mtf_ok: bool):
        caplog.clear()

        engine.mtf_confirmation.confirm_signal = lambda **kwargs: SimpleNamespace(
            confirmed=mtf_ok,
            confidence_adjustment=1.0,
            reason="aligned" if mtf_ok else "misaligned",
            signal_history=[],
        )

        run_task = asyncio.create_task(engine.run())
        # Yield control to allow evaluation to execute
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        engine.stop()
        await run_task
        return extract_events(caplog.records)

    events = await run_once(mtf_ok=True)
    for required in {"threshold_analysis", "sanity_check", "confidence", "final_signal"}:
        assert required in events

    events = await run_once(mtf_ok=False)
    assert "mtf_gate" in events
