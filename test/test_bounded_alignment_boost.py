import pytest
from app.utils_time import compute_bounded_alignment_boost


class TestBoundedAlignmentBoost:
    """Test bounded alignment boost functionality."""
    
    def test_no_alignment_no_boost(self):
        """Test that no alignment results in no boost."""
        multiplier, details = compute_bounded_alignment_boost(
            base_strength=60.0,
            h1_aligned=False,
            h4_aligned=False,
            atr_ratio=1.0
        )
        
        assert multiplier == 1.0
        assert details["final_boost"] == 0.0
        assert details["h1_aligned"] == False
        assert details["h4_aligned"] == False
    
    def test_h1_alignment_only(self):
        """Test H1 alignment only gives expected boost."""
        multiplier, details = compute_bounded_alignment_boost(
            base_strength=60.0,
            h1_aligned=True,
            h4_aligned=False,
            atr_ratio=1.0
        )
        
        # Should get 10% boost (H1 only)
        expected_multiplier = 1.10
        assert abs(multiplier - expected_multiplier) < 0.001
        assert details["final_boost"] == 10.0
        assert details["h1_aligned"] == True
        assert details["h4_aligned"] == False
    
    def test_full_alignment_normal_conditions(self):
        """Test full alignment under normal conditions."""
        multiplier, details = compute_bounded_alignment_boost(
            base_strength=60.0,
            h1_aligned=True,
            h4_aligned=True,
            atr_ratio=1.0
        )
        
        # Should get 15% boost (H1 + H4)
        expected_multiplier = 1.15
        assert abs(multiplier - expected_multiplier) < 0.001
        assert details["final_boost"] == 15.0
        assert details["total_raw_boost"] == 15.0
    
    def test_high_volatility_scaling(self):
        """Test that high volatility reduces boost."""
        multiplier, details = compute_bounded_alignment_boost(
            base_strength=60.0,
            h1_aligned=True,
            h4_aligned=True,
            atr_ratio=2.5  # High volatility
        )
        
        # Should get reduced boost due to high volatility
        assert multiplier < 1.15  # Less than normal full alignment
        assert details["volatility_factor"] == 0.7  # High volatility cap
        assert details["final_boost"] < 15.0
    
    def test_extreme_volatility_scaling(self):
        """Test that extreme volatility heavily reduces boost."""
        multiplier, details = compute_bounded_alignment_boost(
            base_strength=60.0,
            h1_aligned=True,
            h4_aligned=True,
            atr_ratio=3.5  # Extreme volatility
        )
        
        # Should get heavily reduced boost due to extreme volatility
        assert multiplier < 1.10  # Much less than normal full alignment
        assert details["volatility_factor"] == 0.5  # Extreme volatility cap
        assert details["final_boost"] < 10.0
    
    def test_high_strength_diminishing_returns(self):
        """Test that high base strength applies diminishing returns."""
        multiplier, details = compute_bounded_alignment_boost(
            base_strength=85.0,  # High strength
            h1_aligned=True,
            h4_aligned=True,
            atr_ratio=1.0
        )
        
        # Should get reduced boost due to high base strength
        assert multiplier < 1.15  # Less than normal full alignment
        assert details["strength_factor"] == 0.6  # Diminishing factor
        assert details["final_boost"] < 15.0
    
    def test_total_boost_cap(self):
        """Test that total boost is capped at maximum."""
        # Use custom config with lower cap for testing
        custom_config = {
            "max_total_boost": 12.0,  # Lower than normal 15%
            "max_individual_boost": 15.0,
            "volatility_scaling": False,
            "strength_based_scaling": False
        }
        
        multiplier, details = compute_bounded_alignment_boost(
            base_strength=60.0,
            h1_aligned=True,
            h4_aligned=True,
            atr_ratio=1.0,
            boost_config=custom_config
        )
        
        # Should be capped at 12%
        expected_multiplier = 1.12
        assert abs(multiplier - expected_multiplier) < 0.001
        assert details["final_boost"] == 12.0
        assert details["capped_boost"] == 12.0
    
    def test_combined_scaling_factors(self):
        """Test that multiple scaling factors combine correctly."""
        multiplier, details = compute_bounded_alignment_boost(
            base_strength=85.0,  # High strength (0.6 factor)
            h1_aligned=True,
            h4_aligned=True,
            atr_ratio=2.5,  # High volatility (0.7 factor)
        )
        
        # Should apply both volatility and strength scaling
        # 15% * 0.7 * 0.6 = 6.3%
        expected_boost = 15.0 * 0.7 * 0.6
        expected_multiplier = 1.0 + (expected_boost / 100.0)
        
        assert abs(multiplier - expected_multiplier) < 0.001
        assert abs(details["final_boost"] - expected_boost) < 0.001
        assert details["volatility_factor"] == 0.7
        assert details["strength_factor"] == 0.6