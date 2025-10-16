from __future__ import annotations

import asyncio
import time
from typing import Dict, Any, Optional, List

import pandas as pd
import math

from .config import (
    DEFAULT_SYMBOL,
    ALLOWED_SYMBOLS,
    DEFAULT_TIMEFRAME,
    DEFAULT_HISTORY_COUNT,
    WEIGHTS,
    SUPABASE_TABLE,
    PRIMARY_TIMEFRAME,
    CONFIRMATION_TIMEFRAME,
    TREND_TIMEFRAME,
    ALIGNMENT_BOOST_H1,
    ALIGNMENT_BOOST_H4,
    ALIGNMENT_BOOST_CONFIG,
    ATR_STABILITY_BONUS,
    PRICE_ACTION_BONUS,
    CONFIDENCE_ZONES,
    DEBUG_SIGNALS,
)
from .logging_config import get_logger, log_signal_event, log_performance, safe_log_value
from .utils_time import as_utc_index, last_closed, safe_float, compute_bounded_alignment_boost
from .indicators import (
    compute_indicators,
    evaluate_signals,
    compute_strategy_strength,
    best_signal,
    get_signal_confidence_zone,
    ema_trend_direction,
    atr_last,
    atr_last_and_mean,
    price_action_direction,
    compute_weighted_vote_aggregation,
)
from .db import insert_signal
from .db import insert_indicator_snapshot
from .db import get_active_strategy_config
from .threshold_manager import ThresholdManagerFactory, ThresholdManager
from .sanity_filter import SignalSanityFilterFactory
from .mtf_confirmation import MultiTimeframeConfirmation


def _calibrate_conf(strength: float, strong_count: int) -> float:
    """Calibrate raw signal strength into a normalized confidence score."""
    strength = float(strength)
    strong_count = float(strong_count)
    s = max(0.0, min(1.0, strength / 100.0))
    c = 0.6 * s + 0.4 * max(0.0, min(1.0, strong_count / 4.0))
    return max(0.0, min(1.0, c))


def _ts_changed(prev, curr):
    """
    Helper function to compare timestamps safely.
    Returns True if timestamps are different or if either is None/invalid.
    """
    import pandas as pd
    if prev is None or curr is None:
        return True
    try:
        return pd.to_datetime(prev, utc=True) != pd.to_datetime(curr, utc=True)
    except Exception:
        return True


class SignalEngine:
    def __init__(self, fetch_history_func):
        """
        fetch_history_func(symbol: str, timeframe: str, count: int) -> pd.DataFrame
        Expected df columns: time (ns or iso), open, high, low, close, volume
        """
        self.fetch_history = fetch_history_func
        self.running = False
        self.h1_cache_by_symbol: Dict[str, Dict[str, Any]] = {}
        self.h4_cache_by_symbol: Dict[str, Dict[str, Any]] = {}
        self.last_snapshot_ts_by_symbol: Dict[str, pd.Timestamp] = {}
        # Metrics tracking per symbol
        self.attempt_counts_by_symbol: Dict[str, int] = {}
        self.signal_counts_by_symbol: Dict[str, int] = {}
        
        # Initialize adaptive threshold and filtering components with configuration
        from .config import THRESHOLD_MANAGER_CONFIG, SANITY_FILTER_CONFIG
        self.threshold_manager = ThresholdManagerFactory.create_from_config(THRESHOLD_MANAGER_CONFIG)
        self.sanity_filter = SignalSanityFilterFactory.create_from_config(SANITY_FILTER_CONFIG["strict"])
        self.mtf_confirmation = MultiTimeframeConfirmation()
        self.debug = DEBUG_SIGNALS

        # Initialize logger
        self.logger = get_logger('signals')

    def _confidence_zone(self, confidence: float, final_strength: float) -> str:
        """Map confidence to qualitative zones using configured strength bands."""
        try:
            return get_signal_confidence_zone(float(final_strength))
        except Exception:
            if confidence >= 0.7:
                return "strong"
            if confidence >= 0.5:
                return "weak"
            return "neutral"

    async def run(self):
        self.running = True
        # DB-based persistence; will attempt inserts, ignore failures

        while self.running:
            try:
                # Load active strategy config from DB
                strategy = get_active_strategy_config()
                indicator_params = strategy.get("indicator_params", {})
                weights = strategy.get("weights", {})
                primary_tf = strategy.get("primary_timeframe", DEFAULT_TIMEFRAME)
                confirmation_tf = strategy.get("confirmation_timeframe", "1h")
                trend_tf = strategy.get("trend_timeframe", "4h")
                # Enforce 10-minute monitoring interval (600s)
                run_interval_seconds = max(600, int(strategy.get("run_interval_seconds", 600)))
                # Use ThresholdManager for all threshold decisions - no direct signal_threshold usage
                # The threshold will be computed dynamically by ThresholdManager based on market conditions

                # Iterate over configured symbols and evaluate
                symbols: List[str] = strategy.get("symbols", ALLOWED_SYMBOLS)
                for sym in symbols:
                    # Track attempts
                    self.attempt_counts_by_symbol[sym] = int(self.attempt_counts_by_symbol.get(sym, 0)) + 1
                    
                    # Initialize variables that will be used in logging regardless of signal outcome
                    confidence_status = "Not calculated"
                    decision_summary = "No signal generated"
                    # Fetch M15/H1/H4 concurrently for each symbol, capturing per-timeframe statuses
                    fetch_tasks = [
                        asyncio.to_thread(self.fetch_history, sym, primary_tf, DEFAULT_HISTORY_COUNT),
                        asyncio.to_thread(self.fetch_history, sym, confirmation_tf, DEFAULT_HISTORY_COUNT),
                        asyncio.to_thread(self.fetch_history, sym, trend_tf, DEFAULT_HISTORY_COUNT),
                    ]
                    m15_res, h1_res, h4_res = await asyncio.gather(*fetch_tasks, return_exceptions=True)

                    # Build fetch status report
                    def _df_state(res, tf):
                        if isinstance(res, Exception):
                            return {"tf": tf, "status": f"error:{res}"}
                        if res is None:
                            return {"tf": tf, "status": "neutral", "len": 0}
                        try:
                            last_ts = str(pd.to_datetime(res.iloc[-1]["time"]))
                        except Exception:
                            last_ts = "unknown"
                        return {"tf": tf, "status": "ok", "len": len(res), "last": last_ts}

                    fetch_report = [
                        _df_state(m15_res, primary_tf),
                        _df_state(h1_res, confirmation_tf),
                        _df_state(h4_res, trend_tf),
                    ]

                    # Convert results to DataFrames or None
                    m15_df = m15_res if isinstance(m15_res, pd.DataFrame) else None
                    h1_df = h1_res if isinstance(h1_res, pd.DataFrame) else None
                    h4_df = h4_res if isinstance(h4_res, pd.DataFrame) else None

                    # Basic validation
                    if any(df is None or len(df) < 50 for df in (m15_df, h1_df, h4_df)):
                        self.logger.warning(
                            f"Signal Flow | {sym} {primary_tf} | gain N/A | "
                            f"Fetch {fetch_report} | Result: insufficient_data",
                            extra={'symbol': sym, 'timeframe': primary_tf, 'fetch_report': fetch_report}
                        )
                        self.logger.warning(f"Insufficient data for {sym}; skipping.", 
                                          extra={'symbol': sym, 'reason': 'insufficient_data'})
                        continue

                    # Normalize all DataFrames to UTC timezone
                    m15_df = as_utc_index(m15_df)
                    h1_df = as_utc_index(h1_df)
                    h4_df = as_utc_index(h4_df)

                    # Closed-bar alignment: align all timeframes to the last common closed bar
                    try:
                        # Check if DataFrames have datetime indices before attempting alignment
                        has_datetime_indices = all(
                            isinstance(df.index, pd.DatetimeIndex) for df in (m15_df, h1_df, h4_df)
                        )
                        
                        if has_datetime_indices:
                            anchor = min(
                                last_closed(m15_df.index[-1], "15min"),
                                last_closed(h1_df.index[-1], "1h"),
                                last_closed(h4_df.index[-1], "4h")
                            )
                            m15_df = m15_df.loc[:anchor]
                            h1_df = h1_df.loc[:anchor]
                            h4_df = h4_df.loc[:anchor]
                        else:
                            # If DataFrames don't have datetime indices, use time column for alignment
                            if all('time' in df.columns for df in (m15_df, h1_df, h4_df)):
                                anchor = min(
                                    last_closed(m15_df.iloc[-1]['time'], "15min"),
                                    last_closed(h1_df.iloc[-1]['time'], "1h"),
                                    last_closed(h4_df.iloc[-1]['time'], "4h")
                                )
                                # Filter by time column instead of index
                                m15_df = m15_df[m15_df['time'] <= anchor]
                                h1_df = h1_df[h1_df['time'] <= anchor]
                                h4_df = h4_df[h4_df['time'] <= anchor]
                            else:
                                self.logger.warning(f"Cannot align bars for {sym}: no datetime index or time column", 
                                                  extra={'symbol': sym, 'reason': 'no_datetime_reference'})
                        
                        # Validate we still have enough data after alignment
                        if any(len(df) < 50 for df in (m15_df, h1_df, h4_df)):
                            self.logger.warning(f"Insufficient data for {sym} after alignment; skipping.", 
                                              extra={'symbol': sym, 'reason': 'insufficient_data_after_alignment'})
                            continue
                    except Exception as e:
                        self.logger.warning(f"Error aligning bars for {sym}: {e}; skipping.", 
                                          extra={'symbol': sym, 'error': str(e), 'reason': 'alignment_error'})
                        continue

                    # Cache and compute H1 trend (per symbol) - MOVED BEFORE WEIGHTED VOTING
                    h1_ts = h1_df.iloc[-1]["time"]
                    cache_h1 = self.h1_cache_by_symbol.get(sym)
                    if cache_h1 is None or _ts_changed(cache_h1.get("ts"), h1_ts):
                        h1_dir = ema_trend_direction(h1_df, short_len=50, long_len=200)
                        h1_atr_last, h1_atr_mean50 = atr_last_and_mean(
                            h1_df,
                            length=indicator_params.get("ATR", {}).get("length", 14),
                            mean_window=50,
                            params=indicator_params,
                        )
                        self.h1_cache_by_symbol[sym] = {
                            "ts": h1_ts,
                            "dir": h1_dir,
                            "atr_last": h1_atr_last,
                            "atr_mean50": h1_atr_mean50,
                        }
                    else:
                        h1_dir = cache_h1["dir"]

                    # Cache and compute H4 trend (per symbol) - MOVED BEFORE WEIGHTED VOTING
                    h4_ts = h4_df.iloc[-1]["time"]
                    cache_h4 = self.h4_cache_by_symbol.get(sym)
                    if cache_h4 is None or _ts_changed(cache_h4.get("ts"), h4_ts):
                        h4_dir = ema_trend_direction(h4_df, short_len=50, long_len=200)
                        self.h4_cache_by_symbol[sym] = {"ts": h4_ts, "dir": h4_dir, "atr": atr_last(h4_df, params=indicator_params)}
                    else:
                        h4_dir = cache_h4["dir"]

                    # Compute primary indicators (M15)
                    ind = compute_indicators(m15_df, indicator_params)
                    res = evaluate_signals(m15_df, ind, indicator_params)
                    
                    # Debug: Log individual indicator results
                    if self.debug:
                        indicator_details = []
                        for indicator_name, indicator_result in res.items():
                            values = {}
                            if hasattr(indicator_result, "value") and isinstance(indicator_result.value, dict):
                                values = {
                                    key: (float(val) if isinstance(val, (int, float)) else safe_log_value(val))
                                    for key, val in indicator_result.value.items()
                                }
                            indicator_details.append({
                                "indicator": indicator_name,
                                "direction": getattr(indicator_result, "direction", None),
                                "vote": getattr(indicator_result, "vote", None),
                                "strength": float(getattr(indicator_result, "strength", 0.0) or 0.0),
                                "label": getattr(indicator_result, "label", None),
                                "values": values,
                            })
                        self.logger.debug({
                            "event": "debug_indicator_results",
                            "symbol": sym,
                            "details": indicator_details,
                        })
                    
                    # ENHANCED WEIGHTED VOTING SYSTEM WITH DYNAMIC THRESHOLD
                    # Step 1: Calculate preliminary dynamic threshold for weighted voting decision mapping
                    h1_atr_last = float(self.h1_cache_by_symbol[sym]["atr_last"])
                    h1_atr_mean50 = float(self.h1_cache_by_symbol[sym]["atr_mean50"])
                    atr_ratio = h1_atr_last / h1_atr_mean50 if h1_atr_mean50 > 0 else 1.0
                    
                    # RSI deviation from neutral (50)
                    rsi_value = res["RSI"].value.get("rsi", 50.0)
                    rsi_deviation = abs(rsi_value - 50.0) / 50.0  # Normalized 0-1
                    
                    # MACD histogram magnitude (normalized)
                    macd_hist = res["MACD"].value.get("histogram", 0.0)
                    macd_deviation = min(abs(macd_hist) * 1000, 1.0)  # Scale and cap at 1.0
                    
                    # Calculate preliminary adaptive threshold
                    preliminary_threshold, threshold_metadata = self.threshold_manager.compute_adaptive_threshold(
                        atr_ratio=atr_ratio,
                        rsi_deviation=rsi_deviation,
                        macd_histogram=macd_deviation,
                        price_ma_deviation=0.0,
                        symbol=sym,
                        timeframe="M15"
                    )
                    
                    # Step 2: Enhanced weighted vote aggregation with dynamic threshold
                    vote_result = compute_weighted_vote_aggregation(res, weights, threshold=preliminary_threshold)
                    
                    # Compute weighted vote aggregation (using base threshold for this simplified path)
                    vote_result = compute_weighted_vote_aggregation(res, weights, threshold=self.threshold_manager.base_threshold)
                    
                    # PATCH 1: Block neutral→trade upgrades
                    base_dir = vote_result.get("final_direction", "neutral")
                    decision = base_dir
                    if base_dir == "neutral":
                        decision = "none"
                        vote_result["blocked_reason"] = "neutral_base_vote"
                        # bonuses may still be logged but MUST NOT flip to trade
                    
                    strat = compute_strategy_strength(res, weights)
                    best = best_signal(strat)
                    ts = pd.to_datetime(m15_df.iloc[-1]["time"]).isoformat()

                    # Compute indicator validity and direction distribution
                    try:
                        total_inds = len(res)
                        valid_inds = 0
                        buy_cnt = 0
                        sell_cnt = 0
                        neutral_cnt = 0
                        for name, r in res.items():
                            vals = list(r.value.values())
                            is_valid = all(
                                (v is not None) and (not math.isnan(float(v)))
                                for v in vals
                            )
                            if is_valid:
                                valid_inds += 1
                            if r.direction == "buy":
                                buy_cnt += 1
                            elif r.direction == "sell":
                                sell_cnt += 1
                            else:
                                neutral_cnt += 1
                        valid_pct = round(100.0 * valid_inds / max(1, total_inds), 1)
                    except Exception:
                        total_inds, valid_inds, buy_cnt, sell_cnt, neutral_cnt, valid_pct = 0, 0, 0, 0, 0, 0.0

                    # Persist indicator snapshot opportunistically (do not enforce 10-minute spacing here)
                    try:
                        insert_indicator_snapshot({
                            "timestamp": ts,
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "indicators": {k: v.value for k, v in res.items()},
                            "evaluation": {k: v.direction for k, v in res.items()},
                            "strategy": best.get("strategy") if best else None,
                        })
                        snapshot_status = "ok"
                    except Exception as snap_err:
                        snapshot_status = f"error: {snap_err}"

                    if not best or best["direction"] not in ("buy", "sell"):
                        # No actionable signal for this symbol — summarize and continue
                        freq_attempts = int(self.attempt_counts_by_symbol.get(sym, 0))
                        freq_signals = int(self.signal_counts_by_symbol.get(sym, 0))
                        freq_pct = (100.0 * freq_signals / freq_attempts) if freq_attempts > 0 else 0.0
                        
                        # Enhanced base_strength calculation with weighted vote integration and fallbacks
                        base_strength = None
                        
                        # Primary: Use weighted vote confidence if available
                        if vote_result and vote_result.get('confidence') is not None:
                            base_strength = vote_result['confidence'] * 100.0  # Convert to percentage
                        
                        # Secondary: Use best signal strength
                        elif best and "strength" in best and best["strength"] is not None:
                            base_strength = best["strength"]
                        
                        # Tertiary: Calculate from available strategy data
                        elif best:
                            base_strength = 0.0
                            for strategy_name, strategy_data in best.get("components", {}).items():
                                if isinstance(strategy_data, dict) and "strength" in strategy_data and strategy_data["strength"] is not None:
                                    base_strength = max(base_strength, strategy_data["strength"])
                        
                        # Ultimate fallback: return neutral strength instead of None
                        if base_strength is None:
                            base_strength = 0.0
                        self.logger.debug({
                            "event": "signal_metrics",
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "attempts": int(freq_attempts),
                            "signals": int(freq_signals),
                            "hit_rate_pct": float(round(freq_pct, 1)),
                            "valid_indicators": int(valid_inds),
                            "total_indicators": int(total_inds),
                            "valid_pct": float(round(valid_pct, 1)),
                            "dir_counts": {
                                "buy": int(buy_cnt),
                                "sell": int(sell_cnt),
                                "neutral": int(neutral_cnt),
                            },
                            "signal_generated": False,
                        })
                        self.logger.debug({
                            "event": "final_signal",
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "decision": "none",
                            "final_strength": float(base_strength),
                            "threshold": None,
                            "confidence": 0.0,
                            "zone": "neutral",
                            "blocked_reason": "best_direction_missing",
                        })
                        continue

                    # Cache population moved earlier in the flow before weighted voting system

                    # Validate M15 signal with H1 direction
                    m15_side = best["direction"]
                    h1_is_bull = h1_dir == "Bullish"
                    aligns_h1 = (m15_side == "buy" and h1_is_bull) or (m15_side == "sell" and not h1_is_bull)
                    
                    # Debug: Log multi-timeframe analysis
                    if self.debug:
                        self.logger.debug({
                            "event": "debug_mtf_analysis",
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "m15_direction": m15_side,
                            "h1_trend": h1_dir,
                            "h4_trend": h4_dir,
                            "h1_alignment": bool(aligns_h1),
                        })

                    if not aligns_h1:
                        self.logger.debug({
                            "event": "alignment_gate",
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "decision": "skipped",
                            "reason": "h1_misaligned",
                            "m15_direction": m15_side,
                            "h1_trend": h1_dir,
                        })
                        continue

                    # Volatility classification using H1 ATR vs 50-period avg
                    h1_atr_last = float(self.h1_cache_by_symbol[sym]["atr_last"])
                    h1_atr_mean50 = float(self.h1_cache_by_symbol[sym]["atr_mean50"])
                    extreme_vol = h1_atr_last > (3.0 * h1_atr_mean50)

                    # Debug: Log volatility analysis
                    if self.debug:
                        self.logger.debug({
                            "event": "debug_volatility",
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "h1_atr_last": float(h1_atr_last),
                            "h1_atr_mean50": float(h1_atr_mean50),
                            "atr_ratio": float(h1_atr_last / h1_atr_mean50) if h1_atr_mean50 else None,
                            "extreme": bool(extreme_vol),
                        })

                    if extreme_vol:
                        self.logger.debug({
                            "event": "volatility_gate",
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "decision": "skipped",
                            "reason": "atr_extreme",
                            "h1_atr_last": float(h1_atr_last),
                            "h1_atr_mean50": float(h1_atr_mean50),
                        })
                        continue

                    h1_atr_last = float(self.h1_cache_by_symbol[sym]["atr_last"])
                    h1_atr_mean50 = float(self.h1_cache_by_symbol[sym]["atr_mean50"]) 
                    if h1_atr_last > 1.2 * h1_atr_mean50:
                        volatility_state = "High"
                    elif h1_atr_last < 0.8 * h1_atr_mean50:
                        volatility_state = "Low"
                    else:
                        volatility_state = "Normal"

                    # Entry/SL/TP computation per spec (use H1 ATR)
                    close_m15 = float(m15_df.iloc[-1]["close"])  # last close on M15
                    entry_offset = 0.1 * h1_atr_last
                    entry_price = close_m15 - entry_offset if m15_side == "buy" else close_m15 + entry_offset

                    # Stop-loss distance multiplier by volatility and RR
                    if volatility_state == "High":
                        sl_mult = 2.0
                        rr = 1.2
                    elif volatility_state == "Low":
                        sl_mult = 1.0
                        rr = 1.8
                    else:
                        sl_mult = 1.5
                        rr = 1.5

                    sl_distance = sl_mult * h1_atr_last
                    # Sanity bounds for SL distance: 0.25%–1.2% of price
                    min_sl = 0.0025 * close_m15
                    max_sl = 0.0120 * close_m15
                    if sl_distance < min_sl:
                        sl_distance = min_sl
                    elif sl_distance > max_sl:
                        sl_distance = max_sl

                    # Take-profit distance by R:R and bounds 0.4%–2.0%
                    tp_distance = rr * sl_distance
                    min_tp = 0.0040 * close_m15
                    max_tp = 0.0200 * close_m15
                    if tp_distance < min_tp:
                        tp_distance = min_tp
                    elif tp_distance > max_tp:
                        tp_distance = max_tp

                    stop_loss_price = entry_price - sl_distance if m15_side == "buy" else entry_price + sl_distance
                    take_profit_price = entry_price + tp_distance if m15_side == "buy" else entry_price - tp_distance

                    # Pips conversion assumption: XAUUSD pip = 0.01
                    pip_value = 0.01
                    sl_pips = sl_distance / pip_value
                    tp_pips = tp_distance / pip_value

                    # Bounded alignment boost: prevents excessive signal amplification
                    atr_ratio = atr_last(m15_df) / atr_last_and_mean(m15_df)[1] if len(m15_df) > 20 else 1.0
                    h4_aligned = aligns_h1 and h4_dir == h1_dir
                    alignment_multiplier, boost_details = compute_bounded_alignment_boost(
                        base_strength=best["strength"],
                        h1_aligned=aligns_h1,
                        h4_aligned=h4_aligned,
                        atr_ratio=atr_ratio,
                        boost_config=ALIGNMENT_BOOST_CONFIG
                    )

                    # Price action heuristic over last 5 candles
                    pa_dir = price_action_direction(m15_df, lookback=5, params=indicator_params)

                    # Confidence contributions per spec
                    contrib_new = {
                        "RSI": (weights.get("RSI", 0) if res["RSI"].direction == m15_side else 0),
                        "MACD": (weights.get("MACD", 0) if res["MACD"].direction == m15_side else 0),
                        "STOCH": (weights.get("STOCH", 0) if res["STOCH"].direction == m15_side else 0),
                        "BBANDS": (weights.get("BBANDS", 0) if res["BBANDS"].direction == m15_side else 0),
                        "SMA_EMA": (
                            weights.get("SMA", 0) + weights.get("EMA", 0)
                            if (res["SMA"].direction == m15_side and res["EMA"].direction == m15_side)
                            else 0
                        ),
                        "MTF": (weights.get("MTF", 0) if aligns_h1 else 0),
                        "ATR_STABILITY": (weights.get("ATR_STABILITY", 0) if volatility_state == "Normal" else 0),
                        "PRICE_ACTION": (weights.get("PRICE_ACTION", 0) if pa_dir == m15_side else 0),
                    }
                    base_strength = float(sum(contrib_new.values()))

                    # Apply bonuses for ATR stability and price action
                    atr_bonus = ATR_STABILITY_BONUS if volatility_state == "Normal" else 0.0
                    pa_bonus = PRICE_ACTION_BONUS if pa_dir == m15_side else 0.0
                    
                    # Calculate final strength with multiplicative alignment and additive bonuses
                    strength_with_alignment = base_strength * alignment_multiplier
                    final_strength = min(100.0, strength_with_alignment + atr_bonus + pa_bonus)
                    alignment_boost = final_strength - base_strength  # For logging compatibility
                    
                    strong_count = int(vote_result.get("strong_signals", 0))
                    confidence = _calibrate_conf(final_strength, strong_count)
                    confidence_zone = self._confidence_zone(confidence, final_strength)
                    vote_result["confidence"] = float(confidence)
                    self.logger.debug({
                        "event": "confidence",
                        "symbol": sym,
                        "value": float(confidence),
                        "zone": confidence_zone,
                    })
                    
                    # Prepare market conditions for threshold calculation
                    market_conditions = {
                        'atr_ratio': atr_ratio,
                        'rsi_deviation': rsi_deviation,
                        'macd_histogram': macd_deviation,
                        'price_ma_deviation': 0.0
                    }
                    
                    # REFINED DYNAMIC THRESHOLD WITH ENHANCED WEIGHTED VOTES
                    dynamic_threshold, threshold_factors = self.threshold_manager.compute_dynamic_threshold_with_votes(
                        vote_result=vote_result,
                        market_conditions=market_conditions,
                        symbol=sym,
                        timeframe="M15"
                    )
                    
                    # Enhanced threshold logging for transparency
                    threshold_change = dynamic_threshold - self.threshold_manager.base_threshold
                    vote_adjustments = threshold_factors.get("vote_adjustments", {})
                    self.logger.debug({
                        "event": "threshold_analysis",
                        "symbol": sym,
                        "base": float(self.threshold_manager.base_threshold),
                        "final": float(dynamic_threshold),
                        "delta": float(threshold_change),
                        "atr_ratio": float(atr_ratio) if atr_ratio is not None else None,
                        "rsi_deviation": float(rsi_deviation) if rsi_deviation is not None else None,
                        "vote_adj_total": float(vote_adjustments.get('total_adjustment', 0.0)),
                        "market_regime": market_conditions.get('regime', 'unknown'),
                    })
                    
                    # SANITY FILTER CHECK
                    # Get last candle data for sanity filtering
                    last_candle = {
                        "open": float(m15_df.iloc[-1]["open"]),
                        "high": float(m15_df.iloc[-1]["high"]),
                        "low": float(m15_df.iloc[-1]["low"]),
                        "close": float(m15_df.iloc[-1]["close"])
                    }
                    
                    # Apply sanity filter with enhanced validation
                    sanity_passed, filter_reason, validation_metadata = self.sanity_filter.validate_signal(
                        candle=last_candle,
                        atr=h1_atr_last,
                        direction_confidence=final_strength,
                        signal_strength=final_strength,
                        symbol=sym,
                        additional_data={
                            'signal_direction': m15_side,
                            'h1_alignment': aligns_h1,
                            'volatility_regime': 'extreme' if extreme_vol else 'normal',
                            'atr_ratio': h1_atr_last / h1_atr_mean50 if h1_atr_mean50 > 0 else 0
                        }
                    )

                    sanity_payload = {
                        key: (float(value) if isinstance(value, (int, float)) else value)
                        for key, value in (validation_metadata or {}).items()
                    }
                    self.logger.debug({
                        "event": "sanity_check",
                        "symbol": sym,
                        "passed": bool(sanity_passed),
                        "reasons": filter_reason,
                        **sanity_payload,
                    })

                    # Two-tiered confidence filtering - initialize variables for all paths
                    directional_bias_threshold = 60.0  # 60% of total weighted bias
                    directional_bias_confidence = 0.0
                    tier1_passed = False
                    tier2_passed = False
                    confidence_passed = False
                    
                    # Debug logging
                    if DEBUG_SIGNALS:
                        self.logger.debug({
                            "event": "debug_signal_processing",
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "base_strength": float(base_strength),
                            "alignment_multiplier": float(alignment_multiplier),
                            "atr_bonus": float(atr_bonus),
                            "pa_bonus": float(pa_bonus),
                            "final_strength": float(final_strength),
                            "confidence_zone": confidence_zone,
                            "confidence_status": confidence_status,
                            "contributions": {
                                key: (float(val) if isinstance(val, (int, float)) else val)
                                for key, val in contrib_new.items()
                            },
                            "strategy": best.get('strategy', 'N/A'),
                            "direction": m15_side,
                        })

                if best and best["direction"] in ("buy", "sell"):
                    # Tier 1: Directional Bias calculation
                    if best and "components" in best:
                        # Calculate total possible weight and actual directional weight
                        total_possible_weight = 0.0
                        actual_directional_weight = 0.0
                        
                        for strategy_name, strategy_data in best["components"].items():
                            if isinstance(strategy_data, dict):
                                strategy_strength = strategy_data.get("strength", 0.0)
                                total_possible_weight += 100.0  # Each strategy can contribute up to 100%
                                actual_directional_weight += strategy_strength
                        
                        if total_possible_weight > 0:
                            directional_bias_confidence = (actual_directional_weight / total_possible_weight) * 100.0
                    
                    # Tier 2: Signal Strength - must exceed dynamic adaptive threshold
                    signal_strength_confidence = final_strength
                    
                    # Multi-timeframe confirmation
                    mtf_result = self.mtf_confirmation.confirm_signal(
                        symbol=sym,
                        timeframe=primary_tf,
                        direction=m15_side.upper(),
                        confidence=final_strength
                    )
                    
                    # Apply confidence adjustment from MTF confirmation
                    mtf_adjusted_confidence = final_strength * mtf_result.confidence_adjustment
                    
                    # All filters must be satisfied for signal generation
                    tier1_passed = directional_bias_confidence >= directional_bias_threshold
                    tier2_passed = signal_strength_confidence >= dynamic_threshold  # Use dynamic threshold
                    sanity_passed_check = sanity_passed
                    mtf_confirmed = mtf_result.confirmed

                    if m15_side in ("buy", "sell") and not mtf_confirmed:
                        self.logger.debug({
                            "event": "mtf_gate",
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "ok": False,
                        })
                        vote_result["blocked_reason"] = "mtf_gate"

                    confidence_passed = tier1_passed and tier2_passed and sanity_passed_check and mtf_confirmed
                    
                    # Enhanced logging for confidence filtering
                    confidence_status = (
                        f"Tier1: {directional_bias_confidence:.1f}% ({'PASS' if tier1_passed else 'FAIL'}), "
                        f"Tier2: {signal_strength_confidence:.1f}% vs {dynamic_threshold:.1f}% ({'PASS' if tier2_passed else 'FAIL'}), "
                        f"Sanity: {'PASS' if sanity_passed_check else 'FAIL'} ({filter_reason}), "
                        f"MTF: {'PASS' if mtf_confirmed else 'FAIL'} (adj: {mtf_result.confidence_adjustment:.2f}x)"
                    )
                    
                    # Decision reasoning summary
                    decision_reasons = []
                    if not tier1_passed:
                        decision_reasons.append(f"Directional bias too weak ({directional_bias_confidence:.1f}% < {directional_bias_threshold}%)")
                    if not tier2_passed:
                        decision_reasons.append(f"Signal strength insufficient ({signal_strength_confidence:.1f}% < {dynamic_threshold:.1f}%)")
                    if not sanity_passed_check:
                        decision_reasons.append(f"Sanity filter failed ({filter_reason})")
                    if not mtf_confirmed:
                        decision_reasons.append(f"MTF confirmation failed ({mtf_result.reason})")
                    if aligns_h1:
                        decision_reasons.append(f"H1 alignment confirmed ({h1_dir})")
                    else:
                        decision_reasons.append(f"H1 alignment failed ({h1_dir} vs {m15_side})")
                    if not extreme_vol:
                        decision_reasons.append(f"Volatility normal ({h1_atr_last/h1_atr_mean50:.1f}x)")
                    else:
                        decision_reasons.append(f"Extreme volatility ({h1_atr_last/h1_atr_mean50:.1f}x > 3.0x)")
                    
                    decision_summary = f"{'ACCEPT' if confidence_passed else 'REJECT'}: {'; '.join(decision_reasons)}"
                    
                    # Debug: Log decision reasoning
                    if self.debug:
                        self.logger.debug({
                            "event": "debug_decision_reasoning",
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "summary": decision_summary,
                            "reasons": decision_reasons,
                        })
                    
                    insert_status = "skipped"
                    signal_generated = False
                    decision_for_log = "none"

                    if confidence_passed:
                        record: Dict[str, Any] = {
                            "timestamp": ts,
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "side": m15_side,
                            "signal_type": m15_side.upper(),
                            "strength": base_strength,
                            "strategy": best["strategy"],
                            "indicators": {k: v.value for k, v in res.items()},
                            "contributions": best["contributions"],
                            "indicator_contributions": contrib_new,
                            # MTF fields
                            "primary_timeframe": primary_tf,
                            "confirmation_timeframe": confirmation_tf,
                            "trend_timeframe": trend_tf,
                            "h1_trend_direction": h1_dir,
                            "h4_trend_direction": h4_dir,
                            "alignment_boost": alignment_boost,
                            "final_signal_strength": final_strength,
                            # Trade plan fields
                            "entry_price": entry_price,
                            "stop_loss_price": stop_loss_price,
                            "take_profit_price": take_profit_price,
                            "stop_loss_distance_pips": sl_pips,
                            "take_profit_distance_pips": tp_pips,
                            "risk_reward_ratio": rr,
                            "volatility_state": volatility_state,
                            # Enhanced signal system fields
                            "direction_confidence": confidence_status,
                            "direction_reason": decision_summary,
                            # Adaptive threshold and filter metadata
                            "dynamic_threshold": dynamic_threshold,
                            "threshold_factors": threshold_factors,
                            "filter_reason": filter_reason,
                            "sanity_check_passed": sanity_passed,
                            "mtf_confirmation": {
                                "confirmed": mtf_result.confirmed,
                                "confidence_adjustment": mtf_result.confidence_adjustment,
                                "reason": mtf_result.reason,
                                "signal_history_count": len(mtf_result.signal_history)
                            },
                            # Validation
                            "is_valid": True,
                        }
                        try:
                            insert_signal(record)
                            self.signal_counts_by_symbol[sym] = int(self.signal_counts_by_symbol.get(sym, 0)) + 1
                            insert_status = "ok"
                            signal_generated = True
                            decision_for_log = m15_side
                            self.logger.debug({
                                "event": "signal_persist",
                                "symbol": sym,
                                "timeframe": primary_tf,
                                "side": m15_side,
                                "entry_price": float(entry_price),
                                "stop_loss": float(stop_loss_price),
                                "take_profit": float(take_profit_price),
                                "risk_reward": float(rr),
                                "volatility_state": volatility_state,
                                "alignment_boost": float(alignment_boost),
                                "dynamic_threshold": float(dynamic_threshold),
                                "sanity_reason": filter_reason,
                                "mtf_reason": mtf_result.reason,
                                "contributions": {
                                    key: (float(val) if isinstance(val, (int, float)) else val)
                                    for key, val in contrib_new.items()
                                },
                            })
                        except Exception as e:
                            insert_status = f"error: {e}"
                            self.logger.error({
                                "event": "signal_persist_error",
                                "symbol": sym,
                                "timeframe": primary_tf,
                                "error": str(e),
                            })
                    else:
                        if "blocked_reason" not in vote_result:
                            vote_result["blocked_reason"] = "confidence_gate"
                        threshold_gap = float(dynamic_threshold - final_strength)
                        self.logger.debug({
                            "event": "signal_rejected",
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "direction": m15_side,
                            "final_strength": float(final_strength),
                            "threshold": float(dynamic_threshold),
                            "decision_summary": decision_summary,
                            "blocked_reason": vote_result.get("blocked_reason"),
                        })
                        self.logger.debug({
                            "event": "rejection_details",
                            "symbol": sym,
                            "timeframe": primary_tf,
                            "tier1_pct": float(directional_bias_confidence),
                            "tier1_threshold": float(directional_bias_threshold),
                            "tier1_passed": bool(tier1_passed),
                            "tier2_pct": float(signal_strength_confidence),
                            "tier2_threshold": float(dynamic_threshold),
                            "tier2_passed": bool(tier2_passed),
                            "sanity_passed": bool(sanity_passed_check),
                            "sanity_reason": filter_reason,
                            "mtf_confirmed": bool(mtf_confirmed),
                            "mtf_reason": mtf_result.reason,
                        })
                        self.logger.debug({
                            "event": "block_analysis",
                            "symbol": sym,
                            "threshold_gap": threshold_gap,
                            "base_threshold": float(self.threshold_manager.base_threshold),
                            "threshold_adjustment": float(threshold_change),
                            "market_regime": market_conditions.get('regime', 'unknown'),
                            "volatility_state": volatility_state,
                            "atr_ratio": float(atr_ratio) if atr_ratio is not None else None,
                        })

                    freq_attempts = int(self.attempt_counts_by_symbol.get(sym, 0))
                    freq_signals = int(self.signal_counts_by_symbol.get(sym, 0))
                    freq_pct = (100.0 * freq_signals / freq_attempts) if freq_attempts > 0 else 0.0
                    base_strength_metric = float(best["strength"]) if best and "strength" in best and best["strength"] is not None else 0.0

                    self.logger.debug({
                        "event": "signal_metrics",
                        "symbol": sym,
                        "timeframe": primary_tf,
                        "attempts": int(freq_attempts),
                        "signals": int(freq_signals),
                        "hit_rate_pct": float(round(freq_pct, 1)),
                        "valid_indicators": int(valid_inds),
                        "total_indicators": int(total_inds),
                        "valid_pct": float(round(valid_pct, 1)),
                        "dir_counts": {
                            "buy": int(buy_cnt),
                            "sell": int(sell_cnt),
                            "neutral": int(neutral_cnt),
                        },
                        "signal_generated": bool(signal_generated),
                        "insert_status": insert_status,
                        "base_strength": base_strength_metric,
                    })

                    self.logger.debug({
                        "event": "final_signal",
                        "symbol": sym,
                        "timeframe": primary_tf,
                        "decision": decision_for_log,
                        "final_strength": float(final_strength),
                        "threshold": float(dynamic_threshold),
                        "confidence": float(confidence),
                        "zone": confidence_zone,
                        "blocked_reason": vote_result.get("blocked_reason"),
                    })


                # Yield control briefly between symbols
                await asyncio.sleep(0)
            except Exception as e:
                err_type = type(e).__name__
                self.logger.exception({
                    "event": "engine_loop_error",
                    "error_type": err_type,
                    "message": str(e),
                })
                await asyncio.sleep(5)

            # Respect configured run interval between iterations
            try:
                await asyncio.sleep(run_interval_seconds)
            except Exception:
                await asyncio.sleep(5)

    def stop(self):
        self.running = False

    async def compute_once(self, symbols_override: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Run a single compute iteration for all (or provided) symbols and return a summary."""
        results: List[Dict[str, Any]] = []
        try:
            # Load active strategy config from DB
            strategy = get_active_strategy_config()
            indicator_params = strategy.get("indicator_params", {})
            weights = strategy.get("weights", {})
            primary_tf = strategy.get("primary_timeframe", DEFAULT_TIMEFRAME)
            confirmation_tf = strategy.get("confirmation_timeframe", "1h")
            trend_tf = strategy.get("trend_timeframe", "4h")
            # Use ThresholdManager for all threshold decisions - no direct signal_threshold usage
            # The threshold will be computed dynamically by ThresholdManager based on market conditions

            # Iterate over configured or overridden symbols
            symbols: List[str] = symbols_override or strategy.get("symbols", ALLOWED_SYMBOLS)
            for sym in symbols:
                # Track attempts
                self.attempt_counts_by_symbol[sym] = int(self.attempt_counts_by_symbol.get(sym, 0)) + 1
                
                # Initialize variables that will be used in logging regardless of signal outcome
                confidence_status = "Not calculated"
                decision_summary = "No signal generated"
                confidence_passed = False
                alignment_boost = 0.0
                volatility_state = "Normal"
                rr = 1.5
                # Fetch M15/H1/H4 concurrently for each symbol, capturing per-timeframe statuses
                fetch_tasks = [
                    asyncio.to_thread(self.fetch_history, sym, primary_tf, DEFAULT_HISTORY_COUNT),
                    asyncio.to_thread(self.fetch_history, sym, confirmation_tf, DEFAULT_HISTORY_COUNT),
                    asyncio.to_thread(self.fetch_history, sym, trend_tf, DEFAULT_HISTORY_COUNT),
                ]
                m15_res, h1_res, h4_res = await asyncio.gather(*fetch_tasks, return_exceptions=True)

                def _df_state(res, tf):
                    if isinstance(res, Exception):
                        return {"tf": tf, "status": f"error:{res}"}
                    if res is None:
                        return {"tf": tf, "status": "neutral", "len": 0}
                    try:
                        last_ts = str(pd.to_datetime(res.iloc[-1]["time"]))
                    except Exception:
                        last_ts = "unknown"
                    return {"tf": tf, "status": "ok", "len": len(res), "last": last_ts}

                fetch_report = [
                    _df_state(m15_res, primary_tf),
                    _df_state(h1_res, confirmation_tf),
                    _df_state(h4_res, trend_tf),
                ]

                m15_df = m15_res if isinstance(m15_res, pd.DataFrame) else None
                h1_df = h1_res if isinstance(h1_res, pd.DataFrame) else None
                h4_df = h4_res if isinstance(h4_res, pd.DataFrame) else None

                # Basic validation
                if any(df is None or len(df) < 50 for df in (m15_df, h1_df, h4_df)):
                    self.logger.debug({
                        "event": "compute_skip",
                        "symbol": sym,
                        "timeframe": primary_tf,
                        "reason": "insufficient_data",
                        "fetch": fetch_report,
                    })
                    results.append({
                        "symbol": sym,
                        "status": "insufficient_data",
                        "fetch": fetch_report,
                    })
                    continue

                # Normalize all DataFrames to UTC timezone
                m15_df = as_utc_index(m15_df)
                h1_df = as_utc_index(h1_df)
                h4_df = as_utc_index(h4_df)

                # Closed-bar alignment: align all timeframes to the last common closed bar
                try:
                    anchor = min(
                        last_closed(m15_df.index[-1], "15min"),
                        last_closed(h1_df.index[-1], "1h"),
                        last_closed(h4_df.index[-1], "4h")
                    )
                    m15_df = m15_df.loc[:anchor]
                    h1_df = h1_df.loc[:anchor]
                    h4_df = h4_df.loc[:anchor]
                    
                    # Validate we still have enough data after alignment
                    if any(len(df) < 50 for df in (m15_df, h1_df, h4_df)):
                        results.append({
                            "symbol": sym,
                            "status": "insufficient_data_after_alignment",
                            "fetch": fetch_report,
                        })
                        continue
                except Exception as e:
                    results.append({
                        "symbol": sym,
                        "status": f"alignment_error: {e}",
                        "fetch": fetch_report,
                    })
                    continue

                # Compute primary indicators (M15)
                ind = compute_indicators(m15_df, indicator_params)
                res = evaluate_signals(m15_df, ind, indicator_params)
                
                # Compute weighted vote aggregation (using base threshold for this simplified path)
                vote_result = compute_weighted_vote_aggregation(res, weights, threshold=self.threshold_manager.base_threshold)
                
                # PATCH 1: Block neutral→trade upgrades
                base_dir = vote_result.get("final_direction", "neutral")
                decision = base_dir
                if base_dir == "neutral":
                    decision = "none"
                    vote_result["blocked_reason"] = "neutral_base_vote"
                    # bonuses may still be logged but MUST NOT flip to trade
                
                strat = compute_strategy_strength(res, weights)
                best = best_signal(strat)
                ts = pd.to_datetime(m15_df.iloc[-1]["time"]).isoformat()

                # Compute indicator validity and direction distribution
                try:
                    total_inds = len(res)
                    valid_inds = 0
                    buy_cnt = 0
                    sell_cnt = 0
                    neutral_cnt = 0
                    for name, r in res.items():
                        vals = list(r.value.values())
                        is_valid = all(
                            (v is not None) and (not math.isnan(float(v)))
                            for v in vals
                        )
                        if is_valid:
                            valid_inds += 1
                        if r.direction == "buy":
                            buy_cnt += 1
                        elif r.direction == "sell":
                            sell_cnt += 1
                        else:
                            neutral_cnt += 1
                    valid_pct = round(100.0 * valid_inds / max(1, total_inds), 1)
                except Exception:
                    total_inds, valid_inds, buy_cnt, sell_cnt, neutral_cnt, valid_pct = 0, 0, 0, 0, 0, 0.0

                # Persist indicator snapshot opportunistically (do not enforce 10-minute spacing here)
                try:
                    insert_indicator_snapshot({
                        "timestamp": ts,
                        "symbol": sym,
                        "timeframe": primary_tf,
                        "indicators": {k: v.value for k, v in res.items()},
                        "evaluation": {k: v.direction for k, v in res.items()},
                        "strategy": best.get("strategy") if best else None,
                    })
                    snapshot_status = "ok"
                except Exception as snap_err:
                    snapshot_status = f"error: {snap_err}"

                if not best or best["direction"] not in ("buy", "sell"):
                    # No actionable signal for this symbol — summarize and continue
                    freq_attempts = int(self.attempt_counts_by_symbol.get(sym, 0))
                    freq_signals = int(self.signal_counts_by_symbol.get(sym, 0))
                    freq_pct = (100.0 * freq_signals / freq_attempts) if freq_attempts > 0 else 0.0
                    
                    # Enhanced base_strength calculation with weighted vote integration and fallbacks
                    base_strength = None
                    
                    # Primary: Use weighted vote confidence if available
                    if vote_result and vote_result.get('confidence') is not None:
                        base_strength = vote_result['confidence'] * 100.0  # Convert to percentage
                    
                    # Secondary: Use best signal strength
                    elif best and "strength" in best and best["strength"] is not None:
                        base_strength = best["strength"]
                    
                    # Tertiary: Calculate from available strategy data
                    elif best:
                        base_strength = 0.0
                        for strategy_name, strategy_data in best.get("components", {}).items():
                            if isinstance(strategy_data, dict) and "strength" in strategy_data and strategy_data["strength"] is not None:
                                base_strength = max(base_strength, strategy_data["strength"])
                    
                    # Ultimate fallback: return neutral strength instead of None
                    if base_strength is None:
                        base_strength = 0.0
                    self.logger.debug({
                        "event": "final_signal",
                        "symbol": sym,
                        "timeframe": primary_tf,
                        "decision": "none",
                        "final_strength": float(base_strength),
                        "threshold": None,
                        "confidence": 0.0,
                        "zone": "neutral",
                        "blocked_reason": "best_direction_missing",
                    })
                    results.append({
                        "symbol": sym,
                        "timeframe": primary_tf,
                        "indicator_valid": f"{valid_inds}/{total_inds}",
                        "indicator_valid_pct": valid_pct,
                        "dirs": {"buy": buy_cnt, "sell": sell_cnt, "neutral": neutral_cnt},
                        "base_strength": base_strength,
                        "had_signal": False,
                        "insert_status": "skipped",
                        "snapshot_status": snapshot_status,
                        "signal_frequency": {
                            "signals": freq_signals,
                            "attempts": freq_attempts,
                            "percent": round(freq_pct, 1),
                        },
                        "fetch": fetch_report,
                    })
                    continue

                # Cache and compute H1 trend (per symbol)
                h1_ts = h1_df.iloc[-1]["time"]
                cache_h1 = self.h1_cache_by_symbol.get(sym)
                if cache_h1 is None or _ts_changed(cache_h1.get("ts"), h1_ts):
                    h1_dir = ema_trend_direction(h1_df, short_len=50, long_len=200)
                    h1_atr_last, h1_atr_mean50 = atr_last_and_mean(
                        h1_df,
                        length=indicator_params.get("ATR", {}).get("length", 14),
                        mean_window=50,
                        params=indicator_params,
                    )
                    self.h1_cache_by_symbol[sym] = {
                        "ts": h1_ts,
                        "dir": h1_dir,
                        "atr_last": h1_atr_last,
                        "atr_mean50": h1_atr_mean50,
                    }
                else:
                    h1_dir = cache_h1["dir"]

                # Compute H4 trend (per symbol)
                h4_ts = h4_df.iloc[-1]["time"]
                cache_h4 = self.h4_cache_by_symbol.get(sym)
                if cache_h4 is None or _ts_changed(cache_h4.get("ts"), h4_ts):
                    h4_dir = ema_trend_direction(h4_df, short_len=50, long_len=200)
                    self.h4_cache_by_symbol[sym] = {"ts": h4_ts, "dir": h4_dir, "atr": atr_last(h4_df, params=indicator_params)}
                else:
                    h4_dir = cache_h4["dir"]

                # Validate alignment and compute final strength
                m15_side = best["direction"]
                h1_is_bull = h1_dir == "Bullish"
                aligns_h1 = (m15_side == "buy" and h1_is_bull) or (m15_side == "sell" and not h1_is_bull)
                # Bounded alignment boost: prevents excessive signal amplification
                atr_ratio = atr_last(m15_df) / atr_last_and_mean(m15_df)[1] if len(m15_df) > 20 else 1.0
                h4_aligned = aligns_h1 and h4_dir == h1_dir
                alignment_multiplier, boost_details = compute_bounded_alignment_boost(
                    base_strength=best["strength"],
                    h1_aligned=aligns_h1,
                    h4_aligned=h4_aligned,
                    atr_ratio=atr_ratio,
                    boost_config=ALIGNMENT_BOOST_CONFIG
                )
                volatility_state = "Normal"  # Simplified for manual compute summary
                entry_price = float(m15_df.iloc[-1]["close"]) if not math.isnan(float(m15_df.iloc[-1]["close"])) else None
                stop_loss_price = None
                take_profit_price = None
                rr = 1.5  # Default RR ratio for logging

                # Save only when final strength >= adaptive threshold (computed by ThresholdManager)
                base_strength_val = best["strength"]
                final_strength = min(100.0, base_strength_val * alignment_multiplier)
                alignment_boost = final_strength - base_strength_val  # For logging compatibility
                
                # Use ThresholdManager for adaptive threshold computation
                atr_ratio = atr_last(m15_df) / atr_last_and_mean(m15_df)[1] if len(m15_df) > 20 else 1.0
                rsi_deviation = abs(res["RSI"].value - 50.0) if "RSI" in res else 0.0
                macd_histogram = abs(res["MACD"].value) if "MACD" in res else 0.0
                adaptive_threshold, threshold_details = self.threshold_manager.compute_adaptive_threshold(
                    atr_ratio=atr_ratio,
                    rsi_deviation=rsi_deviation,
                    macd_histogram=macd_histogram,
                    symbol=sym,
                    timeframe=primary_tf
                )
                
                had_signal = False
                insert_status = "skipped"
                if final_strength >= adaptive_threshold:
                    record: Dict[str, Any] = {
                        "timestamp": ts,
                        "symbol": sym,
                        "timeframe": primary_tf,
                        "side": m15_side,
                        "strength": base_strength_val,
                        "strategy": best["strategy"],
                        "indicators": {k: v.value for k, v in res.items()},
                        "contributions": best["contributions"],
                        "indicator_contributions": {
                            "SMA_EMA": (
                                weights.get("SMA", 0) + weights.get("EMA", 0)
                                if (res["SMA"].direction == m15_side and res["EMA"].direction == m15_side)
                                else 0
                            ),
                            "MACD": (weights.get("MACD", 0) if res["MACD"].direction == m15_side else 0),
                            "ATR_STABILITY": (weights.get("ATR_STABILITY", 0)),
                            "PRICE_ACTION": (weights.get("PRICE_ACTION", 0)),
                        },
                        "signal_type": m15_side.upper(),
                        "primary_timeframe": primary_tf,
                        "confirmation_timeframe": confirmation_tf,
                        "trend_timeframe": trend_tf,
                        "h1_trend_direction": h1_dir,
                        "h4_trend_direction": h4_dir,
                        "alignment_boost": alignment_boost,
                        "final_signal_strength": final_strength,
                        # Trade plan fields
                        "entry_price": entry_price,
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price,
                        "risk_reward_ratio": rr,
                        "volatility_state": volatility_state,
                        # Enhanced signal system fields
                        "direction_confidence": confidence_status,
                        "direction_reason": decision_summary,
                        # Validation
                        "is_valid": True,
                    }
                    try:
                        insert_signal(record)
                        # Track signal frequency
                        self.signal_counts_by_symbol[sym] = int(self.signal_counts_by_symbol.get(sym, 0)) + 1
                        insert_status = "ok"
                        had_signal = True
                    except Exception as e:
                        insert_status = f"error: {e}"

                # Summarize
                freq_attempts = int(self.attempt_counts_by_symbol.get(sym, 0))
                freq_signals = int(self.signal_counts_by_symbol.get(sym, 0))
                freq_pct = (100.0 * freq_signals / freq_attempts) if freq_attempts > 0 else 0.0
                results.append({
                    "symbol": sym,
                    "timeframe": primary_tf,
                    "indicator_valid": f"{valid_inds}/{total_inds}",
                    "indicator_valid_pct": valid_pct,
                    "dirs": {"buy": buy_cnt, "sell": sell_cnt, "neutral": neutral_cnt},
                    "base_strength": base_strength_val,
                    "final_strength": final_strength,
                    "had_signal": had_signal,
                    "insert_status": insert_status,
                    "snapshot_status": snapshot_status,
                    "signal_frequency": {
                        "signals": freq_signals,
                        "attempts": freq_attempts,
                        "percent": round(freq_pct, 1),
                    },
                    "fetch": fetch_report,
                })
                self.logger.debug({
                    "event": "final_signal",
                    "symbol": sym,
                    "timeframe": primary_tf,
                    "decision": "buy" if had_signal and confidence_passed else "none",
                    "final_strength": float(final_strength),
                    "threshold": None,
                    "confidence": float(vote_result.get("confidence", 0.0)),
                    "zone": self._confidence_zone(vote_result.get("confidence", 0.0), final_strength) if vote_result.get("confidence") is not None else "neutral",
                    "blocked_reason": None if had_signal and confidence_passed else "confidence_gate",
                    "details": {
                        "confidence_status": confidence_status,
                        "decision_summary": decision_summary,
                        "alignment_boost": float(alignment_boost),
                        "volatility_state": volatility_state,
                        "risk_reward": float(rr),
                        "insert_status": insert_status,
                    },
                })
        except Exception as e:
            results.append({"status": f"compute_once_error: {e}"})
        return results
