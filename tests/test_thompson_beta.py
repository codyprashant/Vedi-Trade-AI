from app.analytics.thompson import BetaBandit


def test_ts_updates_bias_selection(tmp_path):
    path = tmp_path / "ts.json"
    bandit = BetaBandit(path=str(path))
    chosen = bandit.sample_arm()
    bandit.update(chosen, True)
    bandit.save()

    wins_arm = chosen
    for _ in range(50):
        bandit.update(wins_arm, True)
    preferences = [bandit.sample_arm() for _ in range(200)]
    assert preferences.count(wins_arm) > 100
