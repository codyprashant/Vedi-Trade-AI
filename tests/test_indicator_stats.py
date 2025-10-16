from app.analytics.indicator_stats import IndicatorStats


def test_indicator_stats_record(tmp_path):
    path = tmp_path / "stats.json"
    stats = IndicatorStats(str(path))
    stats.record("RSI", True)
    stats.record("RSI", False)
    assert stats.stats["RSI"]["total"] == 2
    rate = stats.win_rate("RSI")
    assert 0.0 <= rate <= 100.0
    stats.save()
    reloaded = IndicatorStats(str(path))
    assert reloaded.stats["RSI"]["total"] == 2
