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


logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Unified backtesting engine that generates realistic performance analytics
    using live signal generation logic and simulated trade execution.
    """
    
    def __init__(self, strategy_id: int, symbol: str, start_date: str, end_date: str, 
                 investment: float = 10000, timeframe: str = "15m"):
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
        
        # Results storage
        self.signals = []
        self.results = []
        self.backtest_id = None
        
        # Initialize signal engine for live logic reuse
        self.signal_engine = SignalEngine(fetch_range_df)
        
        logger.info(f"BacktestEngine initialized: {symbol} from {start_date} to {end_date}")
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Execute the complete backtesting process.
        
        Returns:
            Dictionary containing backtest summary and results
        """
        try:
            logger.info(f"Starting backtest for {self.symbol} ({self.start_date} to {self.end_date})")
            
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
            
            logger.info(f"Backtest completed: {len(self.signals)} signals generated, "
                       f"{summary['efficiency_pct']:.1f}% efficiency")
            
            return summary
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
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
            
            logger.info(f"Fetched {len(df)} candles for {self.symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return pd.DataFrame()
    
    def _generate_signals(self, market_data: pd.DataFrame) -> None:
        """
        Generate trading signals using the live signal engine logic.
        
        Args:
            market_data: Historical OHLCV data
        """
        try:
            # Process data in chunks to simulate real-time signal generation
            for i in range(100, len(market_data)):  # Start after enough data for indicators
                # Get data slice up to current point
                current_data = market_data.iloc[:i+1]
                
                # Generate signal using live engine
                signal_result = self.signal_engine.generate_signal(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    ohlcv_data=current_data.to_dict('records')
                )
                
                if signal_result and signal_result.get('is_valid', False):
                    # Add timestamp and create signal record
                    signal = {
                        'id': f"backtest_{len(self.signals)}",
                        'timestamp': current_data.index[-1],
                        'symbol': self.symbol,
                        'side': signal_result['side'],
                        'entry_price': signal_result['entry_price'],
                        'stop_loss_price': signal_result.get('stop_loss_price'),
                        'take_profit_price': signal_result.get('take_profit_price'),
                        'strength': signal_result.get('final_signal_strength', 0),
                        'confidence': signal_result.get('direction_confidence', 'medium'),
                        'reason': signal_result.get('direction_reason', ''),
                        'indicators': signal_result.get('indicators', {}),
                        'risk_reward_ratio': signal_result.get('risk_reward_ratio', 1.0)
                    }
                    
                    self.signals.append(signal)
                    logger.debug(f"Generated signal: {signal['side']} at {signal['entry_price']}")
            
            logger.info(f"Generated {len(self.signals)} signals")
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            raise
    
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
            logger.error(f"Trade simulation failed: {e}")
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
                    "exit_price": exit_price,
                    "entry_price": entry_price,
                    "side": side,
                    "timestamp": signal['timestamp']
                }
                
                self.results.append(trade_result)
                
                logger.debug(f"Trade result: {result} ({profit_pct:.2f}%)")
            
            logger.info(f"Simulated {len(self.results)} trades")
            
        except Exception as e:
            logger.error(f"Trade simulation failed: {e}")
            raise
    
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
            
            logger.info(f"Backtest finalized: {efficiency:.1f}% efficiency, "
                       f"{total_return*100:.2f}% total return")
            
            return summary
            
        except Exception as e:
            logger.error(f"Backtest finalization failed: {e}")
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
                    
                    logger.info(f"Stored backtest results with ID: {backtest_id}")
                    return backtest_id
                    
        except Exception as e:
            logger.error(f"Failed to store backtest results: {e}")
            raise
        finally:
            _put_conn(conn)