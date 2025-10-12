from __future__ import annotations

import asyncio
import time
import traceback
from typing import Dict, Any, Optional

import pandas as pd

from .config import (
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    DEFAULT_HISTORY_COUNT,
    SIGNAL_THRESHOLD,
    WEIGHTS,
    SUPABASE_TABLE,
    PRIMARY_TIMEFRAME,
    CONFIRMATION_TIMEFRAME,
    TREND_TIMEFRAME,
)
from .indicators import (
    compute_indicators,
    evaluate_signals,
    compute_strategy_strength,
    best_signal,
    ema_trend_direction,
    atr_last,
    atr_last_and_mean,
    price_action_direction,
)
from .db import insert_signal
from .db import get_active_strategy_config


class SignalEngine:
    def __init__(self, fetch_history_func):
        """
        fetch_history_func(symbol: str, timeframe: str, count: int) -> pd.DataFrame
        Expected df columns: time (ns or iso), open, high, low, close, volume
        """
        self.fetch_history = fetch_history_func
        self.running = False
        self.h1_cache: Optional[Dict[str, Any]] = None
        self.h4_cache: Optional[Dict[str, Any]] = None

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
                run_interval_seconds = int(strategy.get("run_interval_seconds", 5))
                signal_threshold = float(strategy.get("signal_threshold", SIGNAL_THRESHOLD))

                # Fetch M15/H1/H4 concurrently
                m15_df, h1_df, h4_df = await asyncio.gather(
                    asyncio.to_thread(self.fetch_history, DEFAULT_SYMBOL, primary_tf, DEFAULT_HISTORY_COUNT),
                    asyncio.to_thread(self.fetch_history, DEFAULT_SYMBOL, confirmation_tf, DEFAULT_HISTORY_COUNT),
                    asyncio.to_thread(self.fetch_history, DEFAULT_SYMBOL, trend_tf, DEFAULT_HISTORY_COUNT),
                )

                # Basic validation
                if any(df is None or len(df) < 50 for df in (m15_df, h1_df, h4_df)):
                    await asyncio.sleep(5)
                    continue

                # Compute primary indicators (M15)
                ind = compute_indicators(m15_df, indicator_params)
                res = evaluate_signals(m15_df, ind, indicator_params)
                strat = compute_strategy_strength(res, weights)
                best = best_signal(strat)
                ts = pd.to_datetime(m15_df.iloc[-1]["time"]).isoformat()

                if not best or best["direction"] not in ("buy", "sell"):
                    await asyncio.sleep(5)
                    continue

                # Cache and compute H1 trend
                h1_ts = h1_df.iloc[-1]["time"]
                if self.h1_cache is None or str(self.h1_cache.get("ts")) != str(h1_ts):
                    h1_dir = ema_trend_direction(h1_df, short_len=50, long_len=200)
                    h1_atr_last, h1_atr_mean50 = atr_last_and_mean(h1_df, length=indicator_params.get("ATR", {}).get("length", 14), mean_window=50, params=indicator_params)
                    self.h1_cache = {
                        "ts": h1_ts,
                        "dir": h1_dir,
                        "atr_last": h1_atr_last,
                        "atr_mean50": h1_atr_mean50,
                    }
                else:
                    h1_dir = self.h1_cache["dir"]

                # Cache and compute H4 trend
                h4_ts = h4_df.iloc[-1]["time"]
                if self.h4_cache is None or str(self.h4_cache.get("ts")) != str(h4_ts):
                    h4_dir = ema_trend_direction(h4_df, short_len=50, long_len=200)
                    self.h4_cache = {"ts": h4_ts, "dir": h4_dir, "atr": atr_last(h4_df, params=indicator_params)}
                else:
                    h4_dir = self.h4_cache["dir"]

                # Validate M15 signal with H1 direction
                m15_side = best["direction"]
                h1_is_bull = h1_dir == "Bullish"
                aligns_h1 = (m15_side == "buy" and h1_is_bull) or (m15_side == "sell" and not h1_is_bull)
                if not aligns_h1:
                    # Low-confidence or ignore
                    print(f"Ignored low-confidence M15 {m15_side.upper()} — H1 trend {h1_dir} misaligned.")
                    await asyncio.sleep(5)
                    continue

                # Volatility classification using cached H1 ATR vs 50-period avg
                if self.h1_cache and "atr_last" in self.h1_cache and "atr_mean50" in self.h1_cache:
                    h1_atr_last = float(self.h1_cache["atr_last"])
                    h1_atr_mean50 = float(self.h1_cache["atr_mean50"])
                else:
                    h1_atr_last, h1_atr_mean50 = atr_last_and_mean(h1_df, length=indicator_params.get("ATR", {}).get("length", 14), mean_window=50, params=indicator_params)
                    if self.h1_cache is not None:
                        self.h1_cache["atr_last"] = h1_atr_last
                        self.h1_cache["atr_mean50"] = h1_atr_mean50
                extreme_vol = h1_atr_last > (3.0 * h1_atr_mean50)
                if extreme_vol:
                    print("Skipped trade due to extreme volatility (ATR > 3x mean).")
                    await asyncio.sleep(5)
                    continue

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

                # Stop-loss distance multiplier by volatility
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

                # Compute alignment boost: only H4 vs H1 per spec (+10 / -10)
                alignment_boost = 10.0 if h4_dir == h1_dir else -10.0

                # Price action heuristic over last 5 candles
                pa_dir = price_action_direction(m15_df, lookback=5, params=indicator_params)

                # Confidence contributions (new spec)
                contrib_new = {
                    "RSI": (weights.get("RSI", 0) if res["RSI"].direction == m15_side else 0),
                    "MACD": (weights.get("MACD", 0) if res["MACD"].direction == m15_side else 0),
                    "STOCH": (weights.get("STOCH", 0) if res["STOCH"].direction == m15_side else 0),
                    "BBANDS": (weights.get("BBANDS", 0) if res["BBANDS"].direction == m15_side else 0),
                    "SMA_EMA": (
                        weights.get("SMA_EMA", 0)
                        if (res["SMA"].direction == m15_side and res["EMA"].direction == m15_side)
                        else 0
                    ),
                    "MTF": (weights.get("MTF", 0) if aligns_h1 else 0),
                    "ATR_STABILITY": (weights.get("ATR_STABILITY", 0) if volatility_state == "Normal" else 0),
                    "PRICE_ACTION": (weights.get("PRICE_ACTION", 0) if pa_dir == m15_side else 0),
                }
                base_strength = float(sum(contrib_new.values()))

                final_strength = min(100.0, base_strength + alignment_boost)

                # Save only when final strength >= threshold
                if final_strength >= signal_threshold:
                    record: Dict[str, Any] = {
                        "timestamp": ts,
                        "symbol": DEFAULT_SYMBOL,
                        "timeframe": primary_tf,
                        "side": m15_side,
                        "signal_type": m15_side.upper(),
                        "strength": base_strength,  # base strength per new spec
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
                        # Validation
                        "is_valid": True,
                    }
                    try:
                        insert_signal(record)
                        # Log summary
                        print(
                            f"Signal: {m15_side.upper()} — Entry {entry_price:.2f}, SL {stop_loss_price:.2f}, TP {take_profit_price:.2f} | "
                            f"Volatility {volatility_state}, RR {rr:.2f}, Final {final_strength:.1f}% | "
                            f"H1 {h1_dir}, H4 {h4_dir} (boost {alignment_boost:+.0f}%) | "
                            f"Contrib {contrib_new}"
                        )
                    except Exception as e:
                        print(f"Save failed: {e}")

                await asyncio.sleep(5)
            except Exception as e:
                err_type = type(e).__name__
                print(f"Engine loop error [{err_type}]: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)

            # Respect configured run interval between iterations
            try:
                await asyncio.sleep(run_interval_seconds)
            except Exception:
                await asyncio.sleep(5)

    def stop(self):
        self.running = False