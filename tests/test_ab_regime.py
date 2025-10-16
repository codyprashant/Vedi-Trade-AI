from app.analytics.ab_regime import RegimeAB


def test_deterministic_assignment():
    tracker = RegimeAB(window=10, bucket=7)
    arm1 = tracker.assign("EURUSD", "trend", "M15")
    arm2 = tracker.assign("EURUSD", "trend", "M15")
    assert arm1 == arm2
    tracker.record("EURUSD", "trend", arm1, True)
    tracker.save()
    assert tracker.wr("EURUSD", "trend", arm1) >= 0.0
