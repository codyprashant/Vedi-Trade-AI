from app.analytics.weight_learner import WeightLearner


def test_weight_update_clamped(tmp_path):
    learner = WeightLearner(
        path=str(tmp_path / "weights.json"),
        base_weights={"MACD": 20.0, "RSI": 15.0, "STOCH": 10.0},
        lr=0.1,
        w_min=5.0,
        w_max=30.0,
    )

    before = learner.propose()
    learner.update({"MACD": 1.0, "RSI": -0.5, "STOCH": 0.0}, reward=1.0)
    after = learner.propose()

    assert abs(sum(after.values()) - 100.0) < 1e-6
    assert after["MACD"] >= before["MACD"]
    assert after["RSI"] <= before["RSI"]
