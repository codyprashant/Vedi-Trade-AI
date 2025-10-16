from app.analytics.auto_threshold import AutoThresholdCalibrator


def test_auto_threshold_moves_safely(tmp_path):
    path = tmp_path / "auto.json"
    calibrator = AutoThresholdCalibrator(
        path=str(path),
        alpha=0.5,
        base=60.0,
        thr_min=50.0,
        thr_max=80.0,
    )

    initial = calibrator.threshold("EURUSD", "range")
    calibrator.update("EURUSD", "range", False)
    tightened = calibrator.threshold("EURUSD", "range")
    assert tightened >= initial

    calibrator.update("EURUSD", "range", True)
    relaxed = calibrator.threshold("EURUSD", "range")
    assert relaxed <= tightened
