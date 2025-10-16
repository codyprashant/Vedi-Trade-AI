from app.analytics.rollback import RollbackGuard


def test_rollback_trigger():
    guard = RollbackGuard(window=50, min_trades=20, budget=-5.0)
    for _ in range(20):
        guard.record(False, True)
    assert guard.should_rollback()
