"""
Unified Backtesting Engine for VediTrading AI

This module provides a comprehensive backtesting framework that:
- Reuses live signal generation logic for realistic simulations
- Simulates trade execution with realistic P/L calculations
- Persists full backtest metadata and results in the database
- Provides performance analytics and ROI projections
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from app.signal_engine import SignalEngine
from app.db import (
    get_pool, _get_conn, _put_conn,
    insert_signal, fetch_signals_by_date
)
from app.yahoo_server import fetch_range_df
from app.analytics.indicator_stats import IndicatorStats
from app.analytics.auto_threshold import AutoThresholdCalibrator
from app.analytics.weight_learner import WeightLearner
from app.analytics.ab_regime import RegimeAB
from app.analytics.thompson import BetaBandit
from app.analytics.rollback import RollbackGuard
from app import config, WEIGHTS as BASE_WEIGHTS


logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Unified backtesting engine that generates realistic performance analytics
    using live signal generation logic and simulated trade execution.
    """
    
    def __init__(self, strategy_id: int, symbol: str, start_date: str, end_date: str,
                 investment: float = 10000, timeframe: str = "15m", log_level: str = None,
                 stats_path: str = "data/indicator_stats.json"):
        """
        Initialize the backtesting engine.
        
        Args:
            strategy_id: Strategy configuration ID from database
            symbol: Trading symbol (e.g., 'XAUUSD')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            investment: Initial capital amount
            timeframe: Data timeframe (15m, 1h, etc.)
        """
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.start_date = datetime.fromisoformat(start_date)
        self.end_date = datetime.fromisoformat(end_date)
        self.investment = investment
        self.timeframe = timeframe
        self.logger = logger
        self.log_level = (log_level or config.LOG_LEVEL).lower()
        
        # Results storage
        self.signals = []
        self.results = []
        self.backtest_id = None
        
        # Initialize signal engine for live logic reuse
        self.signal_engine = SignalEngine(fetch_range_df)
        self.stats = IndicatorStats(stats_path)
        base_threshold = getattr(self.signal_engine.threshold_manager, "base_threshold", 60.0)
        self.calibrator = (
            AutoThresholdCalibrator(
                path="data/auto_threshold.json",
                alpha=config.THRESH_EWMA_ALPHA,
                base=base_threshold,
                thr_min=config.THRESH_MIN,
                thr_max=config.THRESH_MAX,
            )
            if config.AUTO_THRESHOLD_ENABLED
            else None
        )
        self.weight_learner = (
            WeightLearner(
                base_weights=BASE_WEIGHTS,
                lr=config.WEIGHT_LR,
                w_min=config.WEIGHT_MIN,
                w_max=config.WEIGHT_MAX,
            )
            if config.WEIGHT_LEARNING_ENABLED
            else None
        )
        if self.weight_learner is not None:
            self.weight_learner.snapshot()

        self.ab_tracker = RegimeAB(bucket=config.AB_BUCKET) if config.AB_ENABLED else None
        self.ts_bandit = BetaBandit() if config.TS_ENABLED else None
        self.rollback_guard = (
            RollbackGuard(
                window=config.ROLLBACK_WINDOW,
                min_trades=config.ROLLBACK_MIN_TRADES,
                budget=config.ROLLBACK_WINRATE_BUDGET,
            )
            if config.ROLLBACK_ENABLED
            else None
        )

        self.logger.info(f"BacktestEngine initialized: {symbol} from {start_date} to {end_date}")
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Execute the complete backtesting process.
        
        Returns:
            Dictionary containing backtest summary and results
        """
        try:
            self.logger.info(f"Starting backtest for {self.symbol} ({self.start_date} to {self.end_date})")
            
            # Step 1: Fetch historical market data
            market_data = self._fetch_market_data()
            if market_data.empty:
                raise ValueError("No market data available for the specified period")
            
            # Step 2: Generate signals using live logic
            self._generate_signals(market_data)
            
            # Step 3: Simulate trades for each signal
            self._simulate_trades(market_data)
            
            # Step 4: Calculate performance metrics and store results
            summary = self.finalize()
            
            self.logger.info(f"Backtest completed: {len(self.signals)} signals generated, "
                             f"{summary['efficiency_pct']:.1f}% efficiency")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
    
    def _fetch_market_data(self) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for the backtest period.
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert timeframe to Yahoo Finance format
            tf_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
            yahoo_tf = tf_map.get(self.timeframe, "15m")
            
            # Fetch data with some buffer for indicators
            buffer_start = self.start_date - timedelta(days=30)
            
            data = fetch_range_df(
                symbol=self.symbol,
                start_date=buffer_start.strftime("%Y-%m-%d"),
                end_date=self.end_date.strftime("%Y-%m-%d"),
                timeframe=yahoo_tf
            )
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Filter to actual backtest period
            df = df[self.start_date:self.end_date]
            
            self.logger.info(f"Fetched {len(df)} candles for {self.symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {e}")
            return pd.DataFrame()
    
    def _generate_signals(self, market_data: pd.DataFrame) -> None:
        """
        Generate trading signals using the live signal engine logic.
        
        Args:
            market_data: Historical OHLCV data
        """
        try:
            # Process data in chunks to simulate real-time signal generation
            # Stop one bar before the end to ensure we have next bar for entry price
            for i in range(100, len(market_data) - 1):  # Start after enough data for indicators
                # Get data slice up to current point
                current_data = market_data.iloc[:i+1]
                
                # Generate signal using simplified approach for backtesting
                signal_result = self._evaluate_signal_for_backtest(
                    historical_data=current_data,
                    current_timestamp=current_data.index[-1]
                )
                
                if signal_result and signal_result.get('direction') in ['buy', 'sell']:
                    # Use next bar's open price for realistic entry timing
                    next_bar = market_data.iloc[i+1]
                    realistic_entry_price = next_bar['open']
                    
                    # Recalculate stop loss and take profit based on realistic entry price
                    original_entry = signal_result['entry_price']
                    side = signal_result['direction']
                    
                    # Calculate the adjustment needed for SL/TP levels
                    entry_adjustment = realistic_entry_price - original_entry
                    
                    # Adjust SL and TP to maintain the same risk/reward distances
                    adjusted_sl = signal_result.get('stop_loss_price', 0) + entry_adjustment
                    adjusted_tp = signal_result.get('take_profit_price', 0) + entry_adjustment
                    
                    # Add timestamp and create signal record
                    signal = {
                        'id': f"backtest_{len(self.signals)}",
                        'timestamp': current_data.index[-1],  # Signal generated at end of current bar
                        'symbol': self.symbol,
                        'side': side,
                        'entry_price': realistic_entry_price,  # Execute at next bar open
                        'stop_loss_price': adjusted_sl,
                        'take_profit_price': adjusted_tp,
                        'strength': signal_result.get('strength', 0),
                        'confidence': signal_result.get('confidence', 'medium'),
                        'reason': signal_result.get('reason', ''),
                        'indicators': signal_result.get('indicators', {}),
                        'indicator_contributions': signal_result.get('indicator_contributions', {}),
                        'metadata': signal_result.get('metadata', {}),
                        'risk_reward_ratio': signal_result.get('risk_reward_ratio', 1.0),
                        'original_entry_price': original_entry,  # Keep for reference
                        'entry_adjustment': entry_adjustment  # Track the adjustment made
                    }
                    
                    self.signals.append(signal)
                    if self.log_level == "debug":
                        self.logger.debug(
                            f"Generated signal: {signal['side']} at {realistic_entry_price:.2f} "
                            f"(next bar open, adjusted from {original_entry:.2f})"
                        )
            
            self.logger.info(f"Generated {len(self.signals)} signals with realistic next-bar-open timing")
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            raise
    
    def _evaluate_signal_for_backtest(self, historical_data: pd.DataFrame, current_timestamp) -> Optional[Dict[str, Any]]:
        """
        Comprehensive signal evaluation using the same gate logic as live SignalEngine.
        This ensures backtesting results accurately reflect live trading performance.
        """
        try:
            from app.utils_time import as_utc_index
            
            # Ensure proper time indexing
            historical_data = as_utc_index(historical_data)
            
            # Use the SignalEngine's comprehensive evaluation logic
            # This includes all gates: Tier1, Tier2 (ThresholdManager), Sanity, and MTF
            signal_result = self.signal_engine.evaluate_signal(
                symbol=self.symbol,
                historical_data=historical_data,
                current_timestamp=current_timestamp
            )
            
            # Check if signal passed all gates
            if signal_result and signal_result.get('had_signal') and signal_result.get('confidence_passed'):
                # Extract signal information
                direction = signal_result.get('direction')
                current_close = historical_data.iloc[-1]['close']
                
                # Use SignalEngine's calculated levels if available, otherwise fallback to ATR-based
                if 'stop_loss_price' in signal_result and 'take_profit_price' in signal_result:
                    entry_price = current_close
                    stop_loss_price = signal_result['stop_loss_price']
                    take_profit_price = signal_result['take_profit_price']
                else:
                    # Fallback to ATR-based calculation
                    atr_period = 14
                    if len(historical_data) >= atr_period:
                        high_low = historical_data['high'] - historical_data['low']
                        high_close = abs(historical_data['high'] - historical_data['close'].shift(1))
                        low_close = abs(historical_data['low'] - historical_data['close'].shift(1))
                        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                        atr = true_range.rolling(window=atr_period).mean().iloc[-1]
                    else:
                        atr = current_close * 0.02  # 2% fallback
                    
                    if direction == 'buy':
                        entry_price = current_close
                        stop_loss_price = current_close - (2 * atr)
                        take_profit_price = current_close + (3 * atr)
                    elif direction == 'sell':
                        entry_price = current_close
                        stop_loss_price = current_close + (2 * atr)
                        take_profit_price = current_close - (3 * atr)
                    else:
                        return None
                
                metadata = signal_result.get('metadata') or {
                    "market_regime": signal_result.get('market_regime', "unknown"),
                    "indicator_contributions": signal_result.get('indicator_contributions', {}),
                    "market_conditions": signal_result.get('market_conditions', {}),
                    "threshold_factors": signal_result.get('threshold_factors', {}),
                    "ab_arm": signal_result.get('ab_arm', 'A'),
                    "ts_arm": signal_result.get('ts_arm', 'A'),
                }

                return {
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'confidence': signal_result.get('confidence', 'medium'),
                    'strength': signal_result.get('final_strength', signal_result.get('strength', 0)),
                    'reason': signal_result.get('decision_summary', ''),
                    'indicators': signal_result.get('indicators', {}),
                    'threshold_factors': signal_result.get('threshold_factors', {}),
                    'indicator_contributions': signal_result.get('indicator_contributions', {}),
                    'metadata': metadata,
                    'validation_results': {
                        'tier1_passed': signal_result.get('tier1_passed', False),
                        'tier2_passed': signal_result.get('tier2_passed', False),
                        'sanity_passed': signal_result.get('sanity_passed', False),
                        'mtf_confirmed': signal_result.get('mtf_confirmed', False)
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in signal evaluation: {e}")
            return None
    
    def simulate_trade(self, signal: Dict[str, Any], data) -> Tuple[str, float]:
        """
        Simulate trade execution for a given signal.
        
        Args:
            signal: Signal dictionary with entry/exit prices
            data: Market data (DataFrame or list of dicts)
            
        Returns:
            Tuple of (result, exit_price) where result is 'profit', 'loss', or 'open'
        """
        try:
            entry_price = signal['entry_price']
            side = signal.get('side', signal.get('direction'))
            
            # Calculate stop loss and take profit levels
            if signal.get('stop_loss_price') and signal.get('take_profit_price'):
                sl = signal['stop_loss_price']
                tp = signal['take_profit_price']
            elif 'take_profit_percent' in signal and 'stop_loss_percent' in signal:
                # Use percentage-based levels from signal
                tp_pct = signal['take_profit_percent'] / 100
                sl_pct = signal['stop_loss_percent'] / 100
                
                if side == 'buy':
                    tp = entry_price * (1 + tp_pct)
                    sl = entry_price * (1 - sl_pct)
                else:  # sell
                    tp = entry_price * (1 - tp_pct)
                    sl = entry_price * (1 + sl_pct)
            else:
                # Default 2% stop loss, 4% take profit
                if side == 'buy':
                    sl = entry_price * 0.98
                    tp = entry_price * 1.04
                else:
                    sl = entry_price * 1.02
                    tp = entry_price * 0.96
            
            # Handle both DataFrame and list formats
            if isinstance(data, list):
                # For test data (list of dicts)
                for candle in data:
                    if side == "buy":
                        # Check take profit first (high of candle)
                        if candle['high'] >= tp:
                            return "profit", tp
                        # Then check stop loss (low of candle)
                        elif candle['low'] <= sl:
                            return "loss", sl
                    else:  # sell
                        # Check take profit first (low of candle)
                        if candle['low'] <= tp:
                            return "profit", tp
                        # Then check stop loss (high of candle)
                        elif candle['high'] >= sl:
                            return "loss", sl
                
                # If no exit condition met, trade remains open
                return "open", data[-1]['close'] if data else entry_price
            
            else:
                # For DataFrame data (production)
                signal_time = signal['timestamp']
                future_data = data[data.index > signal_time]
                
                if future_data.empty:
                    return "open", data.iloc[-1]['close']
                
                # Simulate trade execution
                for timestamp, candle in future_data.iterrows():
                    if side == "buy":
                        # Check take profit first (high of candle)
                        if candle['high'] >= tp:
                            return "profit", tp
                        # Then check stop loss (low of candle)
                        elif candle['low'] <= sl:
                            return "loss", sl
                    else:  # sell
                        # Check take profit first (low of candle)
                        if candle['low'] <= tp:
                            return "profit", tp
                        # Then check stop loss (high of candle)
                        elif candle['high'] >= sl:
                            return "loss", sl
                
                # If no exit condition met, trade remains open
                return "open", future_data.iloc[-1]['close']
            
        except Exception as e:
            self.logger.error(f"Trade simulation failed: {e}")
            return "error", entry_price
    
    def _simulate_trades(self, market_data: pd.DataFrame) -> None:
        """
        Simulate trades for all generated signals.
        
        Args:
            market_data: Complete market data for the backtest period
        """
        try:
            for signal in self.signals:
                result, exit_price = self.simulate_trade(signal, market_data)
                
                # Calculate profit percentage
                entry_price = signal['entry_price']
                side = signal['side']
                
                if result in ['profit', 'loss']:
                    if side == 'buy':
                        profit_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        profit_pct = ((entry_price - exit_price) / entry_price) * 100
                else:
                    profit_pct = 0.0
                
                # Store result
                trade_result = {
                    "signal_id": signal['id'],
                    "result": result,
                    "profit_pct": profit_pct,
                    "pnl": profit_pct,
                    "exit_price": exit_price,
                    "entry_price": entry_price,
                    "side": side,
                    "timestamp": signal['timestamp'],
                    "symbol": signal.get('symbol', self.symbol),
                    "metadata": signal.get('metadata', {}),
                }

                self.results.append(trade_result)

                self._record_trade_outcome(signal, trade_result)
                
                if self.log_level == "debug":
                    self.logger.debug(f"Trade result: {result} ({profit_pct:.2f}%)")
            
            self.logger.info(f"Simulated {len(self.results)} trades")

        except Exception as e:
            self.logger.error(f"Trade simulation failed: {e}")
            raise

    def _record_trade_outcome(self, signal: Dict[str, Any], trade: Dict[str, Any]) -> None:
        try:
            success = trade.get("result") == "profit"
            indicators = signal.get("indicators", {}) if isinstance(signal, dict) else {}
            if isinstance(indicators, dict):
                for name, info in indicators.items():
                    if not isinstance(name, str):
                        continue
                    active = True
                    if isinstance(info, dict):
                        active = bool(info.get("active", True))
                    elif hasattr(info, "get"):
                        try:
                            active = bool(info.get("active", True))  # type: ignore[call-arg]
                        except Exception:
                            active = True
                    if active:
                        self.stats.record(name, success)
            self.stats.save()

            metadata: Dict[str, Any] = {}
            if isinstance(signal, dict):
                metadata.update(signal.get("metadata", {}) or {})
                metadata.setdefault(
                    "indicator_contributions",
                    signal.get("indicator_contributions") or {},
                )
            if isinstance(trade, dict):
                trade_meta = trade.get("metadata", {}) or {}
                metadata.update(trade_meta)
                metadata.setdefault(
                    "indicator_contributions",
                    trade_meta.get("indicator_contributions", {}),
                )

            symbol = trade.get("symbol") or (signal.get("symbol") if isinstance(signal, dict) else None)
            regime = metadata.get("market_regime", "unknown")
            ab_arm = metadata.get("ab_arm", "A")
            ts_arm = metadata.get("ts_arm", "A")
            if self.calibrator is not None and config.AUTO_THRESHOLD_ENABLED and symbol:
                self.calibrator.update(symbol, regime, success)
                self.calibrator.save()

            contributions = metadata.get("indicator_contributions") or {}
            if not contributions and isinstance(signal, dict):
                contributions = signal.get("indicator_contributions", {}) or {}
            if self.weight_learner is not None and config.WEIGHT_LEARNING_ENABLED and contributions:
                pnl = float(trade.get("pnl", trade.get("profit_pct", 0.0)))
                reward = 1.0 if pnl > 0 else -1.0 if pnl < 0 else 0.0
                # Ensure contributions are floats for stable updates
                numeric_contrib = {k: float(v) for k, v in contributions.items()}
                self.weight_learner.update(numeric_contrib, reward)
                self.weight_learner.save()

            if self.ab_tracker is not None and config.AB_ENABLED and symbol:
                self.ab_tracker.record(symbol, regime, ab_arm, success)
                self.ab_tracker.save()
                self.logger.info(
                    {
                        "event": "ab_outcome",
                        "symbol": symbol,
                        "regime": regime,
                        "arm": ab_arm,
                        "success": success,
                        "wr_A": self.ab_tracker.wr(symbol, regime, "A"),
                        "wr_B": self.ab_tracker.wr(symbol, regime, "B"),
                    }
                )

            if self.ts_bandit is not None and config.TS_ENABLED:
                self.ts_bandit.update(ts_arm, success)
                self.ts_bandit.save()

            baseline_success = metadata.get("baseline_success")
            if baseline_success is None:
                baseline_success = success if ab_arm == "A" else metadata.get("baseline_result", success)
            if isinstance(baseline_success, str):
                baseline_flag = baseline_success.lower() in {"true", "1", "profit", "win", "success"}
            else:
                baseline_flag = bool(baseline_success)

            if self.rollback_guard is not None:
                self.rollback_guard.record(success, baseline_flag)
                triggered = False
                if (
                    config.ROLLBACK_ENABLED
                    and self.weight_learner is not None
                    and self.rollback_guard.should_rollback()
                ):
                    self.weight_learner.restore("data/weights_learned.snap.json")
                    triggered = True
                self.rollback_guard.save()
                self.logger.info(
                    {
                        "event": "rollback",
                        "delta_wr": self.rollback_guard.delta_wr(),
                        "activated": triggered,
                    }
                )
        except Exception:
            self.logger.debug("Failed to record indicator stats", exc_info=True)
    
    def finalize(self) -> Dict[str, Any]:
        """
        Calculate final performance metrics and store backtest results.
        
        Returns:
            Dictionary containing backtest summary
        """
        try:
            if not self.results:
                return {
                    "backtest_id": None,
                    "total_signals": 0,
                    "total_return_pct": 0.0,
                    "efficiency_pct": 0.0,
                    "win_count": 0,
                    "loss_count": 0,
                    "open_count": 0
                }
            
            # Calculate metrics
            win_count = sum(1 for r in self.results if r["result"] == "profit")
            loss_count = sum(1 for r in self.results if r["result"] == "loss")
            open_count = sum(1 for r in self.results if r["result"] == "open")
            
            closed_trades = [r for r in self.results if r["result"] in ["profit", "loss"]]
            avg_return = np.mean([r["profit_pct"] for r in closed_trades]) if closed_trades else 0.0
            efficiency = (win_count / max(1, win_count + loss_count)) * 100
            
            # Calculate total return (simplified cumulative)
            total_return = avg_return * len(closed_trades) / 100 if closed_trades else 0.0
            
            # Store in database
            self.backtest_id = self._store_backtest_results(
                total_return_pct=total_return * 100,
                efficiency_pct=efficiency,
                win_count=win_count,
                loss_count=loss_count,
                open_count=open_count
            )
            
            summary = {
                "backtest_id": self.backtest_id,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "investment": self.investment,
                "total_signals": len(self.signals),
                "total_return_pct": total_return * 100,
                "efficiency_pct": efficiency,
                "win_count": win_count,
                "loss_count": loss_count,
                "open_count": open_count,
                "avg_return_per_trade": avg_return
            }
            
            self.logger.info(f"Backtest finalized: {efficiency:.1f}% efficiency, "
                             f"{total_return*100:.2f}% total return")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Backtest finalization failed: {e}")
            raise
    
    def _store_backtest_results(self, total_return_pct: float, efficiency_pct: float,
                               win_count: int, loss_count: int, open_count: int) -> int:
        """
        Store backtest results in the database.
        
        Returns:
            The backtest ID from the database
        """
        conn = _get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    # Insert backtest summary
                    cur.execute("""
                        INSERT INTO public.backtests 
                        (strategy_id, symbol, timeframe, start_date, end_date, investment,
                         total_return_pct, efficiency_pct, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        RETURNING id
                    """, (
                        int(self.strategy_id), str(self.symbol), str(self.timeframe),
                        self.start_date, self.end_date, float(self.investment),
                        float(total_return_pct), float(efficiency_pct)
                    ))
                    
                    backtest_id = cur.fetchone()[0]
                    
                    # Insert individual signal results
                    for i, (signal, result) in enumerate(zip(self.signals, self.results)):
                        cur.execute("""
                            INSERT INTO public.backtest_signals
                            (backtest_id, signal_time, direction, entry_price, exit_price,
                             profit_pct, result, confidence, reason, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        """, (
                            int(backtest_id), signal['timestamp'], str(signal.get('side', signal.get('direction'))),
                            float(signal['entry_price']), float(result['exit_price']),
                            float(result['profit_pct']), str(result['result']),
                            str(signal.get('confidence', 'medium')),
                            str(signal.get('reason', ''))
                        ))
                    
                    self.logger.info(f"Stored backtest results with ID: {backtest_id}")
                    return backtest_id
                    
        except Exception as e:
            self.logger.error(f"Failed to store backtest results: {e}")
            raise
        finally:
            _put_conn(conn)