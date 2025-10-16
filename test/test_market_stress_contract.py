import pytest
from app.threshold_manager import ThresholdManager

class TestMarketStressContract:
    def test_market_stress_returns_dict_and_positive_adjustment_on_high_stress(self):
        tm = ThresholdManager(base_threshold=60.0)
        res = tm.detect_market_stress({
            'atr_ratio': 2.1,
            'rsi': 82,
            'macd_histogram': -1.2,
            'price_ma_deviation': 0.07
        })
        assert isinstance(res, dict)
        assert 'is_stressed' in res and 'threshold_recommendation' in res and 'recommended_adjustment' in res
        assert res['stress_level'] in ('low','medium','high','extreme','none')
        if res['stress_level'] in ('high','extreme'):
            assert res['recommended_adjustment'] > 0

    def test_market_stress_contract_all_required_keys(self):
        """Test that all required keys are present in the return dict"""
        tm = ThresholdManager(base_threshold=60.0)
        res = tm.detect_market_stress({
            'atr_ratio': 1.8,
            'rsi': 75,
            'macd_histogram': 0.9,
            'price_ma_deviation': 0.04
        })
        
        # Check all required top-level keys
        required_keys = [
            'is_stressed', 'stress_level', 'stress_score', 'indicators',
            'threshold_recommendation', 'recommended_adjustment', 'details'
        ]
        for key in required_keys:
            assert key in res, f"Missing required key: {key}"
        
        # Check types
        assert isinstance(res['is_stressed'], bool)
        assert isinstance(res['stress_level'], str)
        assert isinstance(res['stress_score'], (int, float))
        assert isinstance(res['indicators'], dict)
        assert isinstance(res['threshold_recommendation'], str)
        assert isinstance(res['recommended_adjustment'], (int, float))
        assert isinstance(res['details'], dict)

    def test_market_stress_disabled_returns_proper_dict(self):
        """Test that when stress detection is disabled, proper dict is returned"""
        tm = ThresholdManager(base_threshold=60.0, stress_detection_enabled=False)
        res = tm.detect_market_stress({
            'atr_ratio': 3.0,
            'rsi': 90,
            'macd_histogram': 2.5,
            'price_ma_deviation': 0.15
        })
        
        assert isinstance(res, dict)
        assert res['is_stressed'] is False
        assert res['stress_level'] == 'none'
        assert res['stress_score'] == 0.0
        assert res['threshold_recommendation'] == 'none'
        assert res['recommended_adjustment'] == 0.0
        assert 'stress_detection_disabled' in res['details']

    def test_market_stress_extreme_conditions(self):
        """Test extreme market conditions produce expected stress levels"""
        tm = ThresholdManager(base_threshold=60.0)
        res = tm.detect_market_stress({
            'atr_ratio': 2.5,  # extreme
            'rsi': 15,         # extreme
            'macd_histogram': 2.2,  # extreme
            'price_ma_deviation': 0.12  # extreme
        })
        
        assert res['stress_level'] == 'extreme'
        assert res['is_stressed'] is True
        assert res['threshold_recommendation'] == 'increase'
        assert res['recommended_adjustment'] == 8.0
        assert res['stress_score'] >= 3.0

    def test_market_stress_normal_conditions(self):
        """Test normal market conditions produce no stress"""
        tm = ThresholdManager(base_threshold=60.0)
        res = tm.detect_market_stress({
            'atr_ratio': 1.0,
            'rsi': 50,
            'macd_histogram': 0.1,
            'price_ma_deviation': 0.01
        })
        
        assert res['stress_level'] == 'none'
        assert res['is_stressed'] is False
        assert res['threshold_recommendation'] == 'none'
        assert res['recommended_adjustment'] == 0.0
        assert res['stress_score'] == 0.0

    def test_market_stress_details_contain_input_values(self):
        """Test that details section contains the input market condition values"""
        tm = ThresholdManager(base_threshold=60.0)
        input_conditions = {
            'atr_ratio': 1.7,
            'rsi': 65,
            'macd_histogram': 0.5,
            'price_ma_deviation': 0.03
        }
        res = tm.detect_market_stress(input_conditions)
        
        details = res['details']
        assert details['atr_ratio'] == 1.7
        assert details['rsi'] == 65
        assert details['macd_histogram'] == 0.5
        assert details['price_ma_deviation'] == 0.03

    def test_market_stress_stress_level_progression(self):
        """Test that stress levels progress correctly with increasing stress"""
        tm = ThresholdManager(base_threshold=60.0)
        
        # Low stress
        res_low = tm.detect_market_stress({
            'atr_ratio': 1.6,  # atr_high = 1.0 point
            'rsi': 50,
            'macd_histogram': 0.0,
            'price_ma_deviation': 0.0
        })
        assert res_low['stress_level'] == 'medium'  # 1.0 point = medium
        
        # High stress
        res_high = tm.detect_market_stress({
            'atr_ratio': 2.1,  # atr_extreme = 2.0 points
            'rsi': 50,
            'macd_histogram': 0.0,
            'price_ma_deviation': 0.0
        })
        assert res_high['stress_level'] == 'high'  # 2.0 points = high
        
        # Extreme stress
        res_extreme = tm.detect_market_stress({
            'atr_ratio': 2.1,  # atr_extreme = 2.0 points
            'rsi': 85,         # rsi_extreme = 1.0 point
            'macd_histogram': 0.0,
            'price_ma_deviation': 0.0
        })
        assert res_extreme['stress_level'] == 'extreme'  # 3.0 points = extreme