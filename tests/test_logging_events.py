import pytest

pytest.importorskip("pandas")
import pandas as pd

from app.config import THRESHOLD_MANAGER_CONFIG, SANITY_FILTER_CONFIG
from app.signal_engine import SignalEngine
from app.threshold_manager import ThresholdManagerFactory
from app.sanity_filter import SignalSanityFilterFactory
from app.mtf_confirmation import MultiTimeframeConfirmation


def _fake_history(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp.utcnow(), periods=max(count, 120), freq="15T")
    data = {
        "time": index,
        "open": [100 + i * 0.1 for i in range(len(index))],
        "high": [100 + i * 0.1 + 0.2 for i in range(len(index))],
        "low": [100 + i * 0.1 - 0.2 for i in range(len(index))],
        "close": [100 + i * 0.1 for i in range(len(index))],
        "volume": [1000 + i for i in range(len(index))],
    }
    return pd.DataFrame(data)


@pytest.fixture
def engine():
    tm = ThresholdManagerFactory.create_from_config(THRESHOLD_MANAGER_CONFIG)
    sf = SignalSanityFilterFactory.create_from_config(SANITY_FILTER_CONFIG["strict"])
    mtf = MultiTimeframeConfirmation()
    return SignalEngine(
        fetch_history_func=_fake_history,
        threshold_manager=tm,
        sanity_filter=sf,
        mtf_confirmation=mtf,
    )


def test_structured_logs(caplog, engine):
    caplog.set_level("DEBUG")
    engine.evaluate_symbol("TEST")
    events = [record.getMessage() for record in caplog.records]
    for key in ["threshold_analysis", "sanity_check", "confidence", "final_signal"]:
        assert any(key in message for message in events)
    assert any("\"event\": \"perf\"" in message or "'event': 'perf'" in message for message in events)
    assert any("\"event\": \"profiling\"" in message or "'event': 'profiling'" in message for message in events)
