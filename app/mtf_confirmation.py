"""
Multi-Timeframe Confirmation Module

Provides temporal filtering and signal consistency validation across different timeframes
to prevent rapid flip-flops in choppy markets and enhance signal quality.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TimeframeHierarchy(Enum):
    """Timeframe hierarchy for multi-timeframe analysis"""
    M1 = 1
    M5 = 5
    M15 = 15
    M30 = 30
    H1 = 60
    H4 = 240
    D1 = 1440


@dataclass
class SignalHistory:
    """Stores historical signal information for consistency checking"""
    timestamp: datetime
    direction: str  # 'BUY' or 'SELL'
    confidence: float
    timeframe: str
    symbol: str


@dataclass
class MTFConfirmationResult:
    """Result of multi-timeframe confirmation analysis"""
    confirmed: bool
    confidence_adjustment: float
    consistency_score: float
    higher_tf_alignment: bool
    consecutive_signals: int
    reason: str
    metadata: Dict[str, Any]


class MultiTimeframeConfirmation:
    """
    Multi-timeframe confirmation system for signal validation
    
    Features:
    - Checks alignment between current timeframe and higher timeframes
    - Validates consecutive signal consistency
    - Applies confidence adjustments based on temporal patterns
    - Prevents rapid signal flip-flops in choppy markets
    """
    
    def __init__(self, 
                 consistency_window_minutes: int = 60,
                 min_consecutive_signals: int = 2,
                 alignment_boost: float = 1.1,
                 flip_penalty: float = 0.8,
                 higher_tf_weight: float = 1.2):
        """
        Initialize multi-timeframe confirmation system
        
        Args:
            consistency_window_minutes: Time window to check for signal consistency
            min_consecutive_signals: Minimum consecutive signals required for confirmation
            alignment_boost: Confidence multiplier for aligned signals
            flip_penalty: Confidence multiplier for direction flips
            higher_tf_weight: Weight for higher timeframe alignment
        """
        self.consistency_window = timedelta(minutes=consistency_window_minutes)
        self.min_consecutive_signals = min_consecutive_signals
        self.alignment_boost = alignment_boost
        self.flip_penalty = flip_penalty
        self.higher_tf_weight = higher_tf_weight
        
        # Signal history storage (in production, this would be database-backed)
        self.signal_history: Dict[str, List[SignalHistory]] = {}
        
    def add_signal_to_history(self, symbol: str, timeframe: str, direction: str, 
                            confidence: float, timestamp: datetime = None):
        """Add a signal to the history for future consistency checks"""
        if timestamp is None:
            timestamp = datetime.now()
            
        signal = SignalHistory(
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            symbol=symbol
        )
        
        key = f"{symbol}_{timeframe}"
        if key not in self.signal_history:
            self.signal_history[key] = []
            
        self.signal_history[key].append(signal)
        
        # Keep only recent signals within the consistency window
        cutoff_time = timestamp - self.consistency_window
        self.signal_history[key] = [
            s for s in self.signal_history[key] 
            if s.timestamp >= cutoff_time
        ]
        
    def get_timeframe_hierarchy_level(self, timeframe: str) -> int:
        """Get the hierarchy level for a timeframe"""
        tf_map = {
            'M1': TimeframeHierarchy.M1.value,
            'M5': TimeframeHierarchy.M5.value,
            'M15': TimeframeHierarchy.M15.value,
            'M30': TimeframeHierarchy.M30.value,
            'H1': TimeframeHierarchy.H1.value,
            'H4': TimeframeHierarchy.H4.value,
            'D1': TimeframeHierarchy.D1.value
        }
        return tf_map.get(timeframe, TimeframeHierarchy.M15.value)
        
    def get_higher_timeframes(self, current_tf: str) -> List[str]:
        """Get list of higher timeframes for alignment checking"""
        current_level = self.get_timeframe_hierarchy_level(current_tf)
        
        higher_tfs = []
        for tf_name, tf_enum in [
            ('M30', TimeframeHierarchy.M30),
            ('H1', TimeframeHierarchy.H1),
            ('H4', TimeframeHierarchy.H4),
            ('D1', TimeframeHierarchy.D1)
        ]:
            if tf_enum.value > current_level:
                higher_tfs.append(tf_name)
                
        return higher_tfs
        
    def check_consecutive_consistency(self, symbol: str, timeframe: str, 
                                    current_direction: str) -> Tuple[int, float]:
        """
        Check for consecutive signal consistency
        
        Returns:
            Tuple of (consecutive_count, consistency_score)
        """
        key = f"{symbol}_{timeframe}"
        if key not in self.signal_history:
            return 0, 0.0
            
        signals = sorted(self.signal_history[key], key=lambda x: x.timestamp, reverse=True)
        
        consecutive_count = 0
        total_signals = len(signals)
        same_direction_count = 0
        
        for signal in signals:
            if signal.direction == current_direction:
                same_direction_count += 1
                if consecutive_count == same_direction_count - 1:
                    consecutive_count += 1
            else:
                break
                
        consistency_score = same_direction_count / max(total_signals, 1)
        
        return consecutive_count, consistency_score
        
    def check_higher_timeframe_alignment(self, symbol: str, current_tf: str, 
                                       current_direction: str) -> Tuple[bool, float]:
        """
        Check alignment with higher timeframes
        
        Returns:
            Tuple of (is_aligned, alignment_strength)
        """
        higher_tfs = self.get_higher_timeframes(current_tf)
        
        if not higher_tfs:
            return True, 1.0  # No higher timeframes to check
            
        aligned_count = 0
        total_checked = 0
        
        for higher_tf in higher_tfs:
            key = f"{symbol}_{higher_tf}"
            if key in self.signal_history and self.signal_history[key]:
                # Get the most recent signal from higher timeframe
                recent_signal = max(self.signal_history[key], key=lambda x: x.timestamp)
                
                # Check if it's within a reasonable time window
                time_diff = datetime.now() - recent_signal.timestamp
                if time_diff <= self.consistency_window * 2:  # Extended window for higher TFs
                    total_checked += 1
                    if recent_signal.direction == current_direction:
                        aligned_count += 1
                        
        if total_checked == 0:
            return True, 1.0  # No recent higher TF signals to compare
            
        alignment_strength = aligned_count / max(total_checked, 1)  # Prevent division by zero
        is_aligned = alignment_strength >= 0.5  # At least 50% alignment
        
        return is_aligned, alignment_strength
        
    def confirm_signal(self, symbol: str, timeframe: str, direction: str, 
                      confidence: float, timestamp: datetime = None) -> MTFConfirmationResult:
        """
        Perform multi-timeframe confirmation analysis
        
        Args:
            symbol: Trading symbol
            timeframe: Current timeframe
            direction: Signal direction ('BUY' or 'SELL')
            confidence: Initial signal confidence
            timestamp: Signal timestamp (defaults to now)
            
        Returns:
            MTFConfirmationResult with confirmation details
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Check consecutive consistency
        consecutive_count, consistency_score = self.check_consecutive_consistency(
            symbol, timeframe, direction
        )
        
        # Check higher timeframe alignment
        higher_tf_aligned, alignment_strength = self.check_higher_timeframe_alignment(
            symbol, timeframe, direction
        )
        
        # Calculate confidence adjustment
        confidence_adjustment = 1.0
        
        # Boost for consecutive signals
        if consecutive_count >= self.min_consecutive_signals:
            confidence_adjustment *= self.alignment_boost
            
        # Penalty for direction flips (low consecutive count)
        elif consecutive_count == 0:
            confidence_adjustment *= self.flip_penalty
            
        # Boost for higher timeframe alignment
        if higher_tf_aligned:
            confidence_adjustment *= (1.0 + (alignment_strength - 0.5) * self.higher_tf_weight)
        else:
            confidence_adjustment *= (0.9 - (0.5 - alignment_strength) * 0.2)
            
        # Determine confirmation
        confirmed = (
            consecutive_count >= 1 and  # At least some consistency
            consistency_score >= 0.3 and  # Reasonable consistency ratio
            confidence_adjustment >= 0.9  # Not too heavily penalized
        )
        
        # Generate reason
        reasons = []
        if consecutive_count >= self.min_consecutive_signals:
            reasons.append(f"consecutive_signals_{consecutive_count}")
        if higher_tf_aligned:
            reasons.append(f"higher_tf_aligned_{alignment_strength:.2f}")
        if consistency_score >= 0.7:
            reasons.append(f"high_consistency_{consistency_score:.2f}")
        if not confirmed:
            reasons.append("insufficient_confirmation")
            
        reason = "|".join(reasons) if reasons else "no_data"
        
        # Add signal to history
        self.add_signal_to_history(symbol, timeframe, direction, confidence, timestamp)
        
        return MTFConfirmationResult(
            confirmed=confirmed,
            confidence_adjustment=confidence_adjustment,
            consistency_score=consistency_score,
            higher_tf_alignment=higher_tf_aligned,
            consecutive_signals=consecutive_count,
            reason=reason,
            metadata={
                'alignment_strength': alignment_strength,
                'original_confidence': confidence,
                'adjusted_confidence': confidence * confidence_adjustment,
                'timeframe_hierarchy_level': self.get_timeframe_hierarchy_level(timeframe),
                'higher_timeframes_checked': self.get_higher_timeframes(timeframe)
            }
        )
        
    def clear_old_history(self, cutoff_hours: int = 24):
        """Clear signal history older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=cutoff_hours)
        
        for key in self.signal_history:
            self.signal_history[key] = [
                signal for signal in self.signal_history[key]
                if signal.timestamp >= cutoff_time
            ]
            
    def get_signal_statistics(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get statistics about recent signals for a symbol/timeframe"""
        key = f"{symbol}_{timeframe}"
        if key not in self.signal_history:
            return {'total_signals': 0}
            
        signals = self.signal_history[key]
        if not signals:
            return {'total_signals': 0}
            
        buy_signals = [s for s in signals if s.direction == 'BUY']
        sell_signals = [s for s in signals if s.direction == 'SELL']
        
        return {
            'total_signals': len(signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_confidence': sum(s.confidence for s in signals) / max(len(signals), 1),
            'latest_signal': {
                'direction': signals[-1].direction,
                'confidence': signals[-1].confidence,
                'timestamp': signals[-1].timestamp.isoformat()
            } if signals else None,
            'time_span_minutes': (
                (signals[-1].timestamp - signals[0].timestamp).total_seconds() / 60
                if len(signals) > 1 else 0
            )
        }


class MTFConfirmationFactory:
    """Factory for creating MTF confirmation instances with different configurations"""
    
    @staticmethod
    def create_strict() -> MultiTimeframeConfirmation:
        """Create strict MTF confirmation (conservative)"""
        return MultiTimeframeConfirmation(
            consistency_window_minutes=90,
            min_consecutive_signals=3,
            alignment_boost=1.15,
            flip_penalty=0.7,
            higher_tf_weight=1.3
        )
        
    @staticmethod
    def create_permissive() -> MultiTimeframeConfirmation:
        """Create permissive MTF confirmation (aggressive)"""
        return MultiTimeframeConfirmation(
            consistency_window_minutes=30,
            min_consecutive_signals=1,
            alignment_boost=1.05,
            flip_penalty=0.9,
            higher_tf_weight=1.1
        )
        
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> MultiTimeframeConfirmation:
        """Create MTF confirmation from configuration dictionary"""
        return MultiTimeframeConfirmation(
            consistency_window_minutes=config.get('consistency_window_minutes', 60),
            min_consecutive_signals=config.get('min_consecutive_signals', 2),
            alignment_boost=config.get('alignment_boost', 1.1),
            flip_penalty=config.get('flip_penalty', 0.8),
            higher_tf_weight=config.get('higher_tf_weight', 1.2)
        )