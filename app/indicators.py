from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from .config import (
    INDICATOR_PARAMS,
    WEIGHTS,
    MACD_HIST_MIN,
    NEUTRAL_WEIGHT_FACTOR,
    TREND_WEIGHT_RATIO,
    MOMENTUM_WEIGHT_RATIO,
    CONFIDENCE_ZONES,
    ATR_STABILITY_BONUS,
    PRICE_ACTION_BONUS,
)


Direction = Literal["buy", "sell", "neutral"]


@dataclass
class IndicatorResult:
    direction: Direction
    value: Dict[str, float]
    contribution: float  # weight if aligned, else 0


def compute_indicators(df: pd.DataFrame, params: Dict[str, Dict] | None = None) -> Dict[str, pd.Series]:
    params = params or INDICATOR_PARAMS

    def _nan_series(name: str) -> pd.Series:
        return pd.Series([np.nan] * len(df), index=df.index, name=name)

    ind: Dict[str, pd.Series] = {}

    # RSI - Multiple periods
    try:
        rsi_periods = params["RSI"].get("periods", [14])
        for period in rsi_periods:
            rsi = ta.rsi(df["close"], length=period)
            key = f"rsi_{period}"
            ind[key] = rsi.rename(key) if isinstance(rsi, pd.Series) else _nan_series(key)
        # Keep backward compatibility with single RSI
        if 14 in rsi_periods:
            ind["rsi"] = ind["rsi_14"]
    except Exception:
        for period in params["RSI"].get("periods", [14]):
            ind[f"rsi_{period}"] = _nan_series(f"rsi_{period}")
        ind["rsi"] = _nan_series("rsi")

    # MACD
    try:
        macd = ta.macd(
            df["close"],
            fast=params["MACD"]["fast"],
            slow=params["MACD"]["slow"],
            signal=params["MACD"]["signal"],
        )
        if isinstance(macd, pd.DataFrame) and macd.shape[1] >= 2:
            ind["macd"] = macd[macd.columns[0]].rename("macd")
            ind["macd_signal"] = macd[macd.columns[1]].rename("macd_signal")
        else:
            ind["macd"] = _nan_series("macd")
            ind["macd_signal"] = _nan_series("macd_signal")
    except Exception:
        ind["macd"] = _nan_series("macd")
        ind["macd_signal"] = _nan_series("macd_signal")

    # SMA - Multiple periods
    try:
        sma_periods = params["SMA"].get("periods", [50, 200])
        for period in sma_periods:
            sma = ta.sma(df["close"], length=period)
            key = f"sma_{period}"
            ind[key] = sma.rename(key) if isinstance(sma, pd.Series) else _nan_series(key)
        # Keep backward compatibility
        if 50 in sma_periods:
            ind["sma_short"] = ind["sma_50"]
        if 200 in sma_periods:
            ind["sma_long"] = ind["sma_200"]
    except Exception:
        for period in params["SMA"].get("periods", [50, 200]):
            ind[f"sma_{period}"] = _nan_series(f"sma_{period}")
        ind["sma_short"] = _nan_series("sma_short")
        ind["sma_long"] = _nan_series("sma_long")
    # EMA - Multiple periods
    try:
        ema_periods = params["EMA"].get("periods", [20, 50])
        for period in ema_periods:
            ema = ta.ema(df["close"], length=period)
            key = f"ema_{period}"
            ind[key] = ema.rename(key) if isinstance(ema, pd.Series) else _nan_series(key)
        # Keep backward compatibility
        if 20 in ema_periods:
            ind["ema_short"] = ind["ema_20"]
        elif 9 in ema_periods:
            ind["ema_short"] = ind["ema_9"]
        if 50 in ema_periods:
            ind["ema_long"] = ind["ema_50"]
        elif 55 in ema_periods:
            ind["ema_long"] = ind["ema_55"]
    except Exception:
        for period in params["EMA"].get("periods", [20, 50]):
            ind[f"ema_{period}"] = _nan_series(f"ema_{period}")
        ind["ema_short"] = _nan_series("ema_short")
        ind["ema_long"] = _nan_series("ema_long")

    # Bollinger Bands
    try:
        bb = ta.bbands(df["close"], length=params["BBANDS"]["length"], std=params["BBANDS"]["std"])
        if isinstance(bb, pd.DataFrame) and bb.shape[1] >= 3:
            ind["bb_low"] = bb[bb.columns[0]].rename("bb_low")
            ind["bb_mid"] = bb[bb.columns[1]].rename("bb_mid")
            ind["bb_high"] = bb[bb.columns[2]].rename("bb_high")
        else:
            ind["bb_low"] = _nan_series("bb_low")
            ind["bb_mid"] = _nan_series("bb_mid")
            ind["bb_high"] = _nan_series("bb_high")
    except Exception:
        ind["bb_low"] = _nan_series("bb_low")
        ind["bb_mid"] = _nan_series("bb_mid")
        ind["bb_high"] = _nan_series("bb_high")

    # Stochastic
    try:
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=params["STOCH"]["k"], d=params["STOCH"]["d"])
        if isinstance(stoch, pd.DataFrame) and stoch.shape[1] >= 2:
            ind["stoch_k"] = stoch[stoch.columns[0]].rename("stoch_k")
            ind["stoch_d"] = stoch[stoch.columns[1]].rename("stoch_d")
        else:
            ind["stoch_k"] = _nan_series("stoch_k")
            ind["stoch_d"] = _nan_series("stoch_d")
    except Exception:
        ind["stoch_k"] = _nan_series("stoch_k")
        ind["stoch_d"] = _nan_series("stoch_d")

    # ATR
    try:
        atr = ta.atr(df["high"], df["low"], df["close"], length=params["ATR"]["length"])
        ind["atr"] = atr.rename("atr") if isinstance(atr, pd.Series) else _nan_series("atr")
    except Exception:
        ind["atr"] = _nan_series("atr")

    return ind


def _cross_over(prev_a: float, prev_b: float, a: float, b: float) -> bool:
    return prev_a <= prev_b and a > b


def _cross_under(prev_a: float, prev_b: float, a: float, b: float) -> bool:
    return prev_a >= prev_b and a < b


def evaluate_signals(df: pd.DataFrame, ind: Dict[str, pd.Series], params: Dict[str, Dict] | None = None) -> Dict[str, IndicatorResult]:
    params = params or INDICATOR_PARAMS

    last = df.iloc[-1]
    prev = df.iloc[-2]

    results: Dict[str, IndicatorResult] = {}

    # Helper function to safely convert to float and check for NaN
    def safe_float(val):
        try:
            f_val = float(val)
            return f_val if not np.isnan(f_val) else None
        except (ValueError, TypeError):
            return None

    # RSI
    rsi_val = safe_float(ind["rsi"].iloc[-1])
    if rsi_val is not None:
        rsi_dir: Direction = "buy" if rsi_val < params["RSI"]["oversold"] else (
            "sell" if rsi_val > params["RSI"]["overbought"] else "neutral"
        )
    else:
        rsi_dir = "neutral"
        rsi_val = np.nan
    results["RSI"] = IndicatorResult(rsi_dir, {"rsi": rsi_val}, 0)

    # MACD cross and histogram
    macd_val = safe_float(ind["macd"].iloc[-1])
    macd_sig = safe_float(ind["macd_signal"].iloc[-1])
    macd_prev = safe_float(ind["macd"].iloc[-2])
    macd_sig_prev = safe_float(ind["macd_signal"].iloc[-2])
    macd_hist = safe_float(ind["macd_histogram"].iloc[-1]) if "macd_histogram" in ind else None
    macd_dir: Direction = "neutral"
    
    if all(v is not None for v in [macd_val, macd_sig, macd_prev, macd_sig_prev]):
        # Traditional crossover signals
        if _cross_over(macd_prev, macd_sig_prev, macd_val, macd_sig):
            macd_dir = "buy"
        elif _cross_under(macd_prev, macd_sig_prev, macd_val, macd_sig):
            macd_dir = "sell"
        # Enhanced: histogram magnitude check
        elif macd_hist is not None and abs(macd_hist) >= params.get("MACD_HIST_MIN", MACD_HIST_MIN):
            if macd_hist > 0:
                macd_dir = "buy"
            else:
                macd_dir = "sell"
    
    # Use original values for output (including NaN if present)
    macd_val_out = float(ind["macd"].iloc[-1]) if macd_val is not None else np.nan
    macd_sig_out = float(ind["macd_signal"].iloc[-1]) if macd_sig is not None else np.nan
    results["MACD"] = IndicatorResult(macd_dir, {"macd": macd_val_out, "signal": macd_sig_out}, 0)

    # SMA cross
    sma_s = safe_float(ind["sma_short"].iloc[-1])
    sma_l = safe_float(ind["sma_long"].iloc[-1])
    sma_s_prev = safe_float(ind["sma_short"].iloc[-2])
    sma_l_prev = safe_float(ind["sma_long"].iloc[-2])
    sma_dir: Direction = "neutral"
    
    if all(v is not None for v in [sma_s, sma_l, sma_s_prev, sma_l_prev]):
        if _cross_over(sma_s_prev, sma_l_prev, sma_s, sma_l):
            sma_dir = "buy"
        elif _cross_under(sma_s_prev, sma_l_prev, sma_s, sma_l):
            sma_dir = "sell"
    
    # Use original values for output (including NaN if present)
    sma_s_out = float(ind["sma_short"].iloc[-1]) if sma_s is not None else np.nan
    sma_l_out = float(ind["sma_long"].iloc[-1]) if sma_l is not None else np.nan
    results["SMA"] = IndicatorResult(sma_dir, {"sma50": sma_s_out, "sma200": sma_l_out}, 0)

    # EMA cross
    ema_s = safe_float(ind["ema_short"].iloc[-1])
    ema_l = safe_float(ind["ema_long"].iloc[-1])
    ema_s_prev = safe_float(ind["ema_short"].iloc[-2])
    ema_l_prev = safe_float(ind["ema_long"].iloc[-2])
    ema_dir: Direction = "neutral"
    
    if all(v is not None for v in [ema_s, ema_l, ema_s_prev, ema_l_prev]):
        if _cross_over(ema_s_prev, ema_l_prev, ema_s, ema_l):
            ema_dir = "buy"
        elif _cross_under(ema_s_prev, ema_l_prev, ema_s, ema_l):
            ema_dir = "sell"
    
    # Use original values for output (including NaN if present)
    ema_s_out = float(ind["ema_short"].iloc[-1]) if ema_s is not None else np.nan
    ema_l_out = float(ind["ema_long"].iloc[-1]) if ema_l is not None else np.nan
    results["EMA"] = IndicatorResult(ema_dir, {"ema20": ema_s_out, "ema50": ema_l_out}, 0)

    # Bollinger + RSI condition
    bb_low = safe_float(ind["bb_low"].iloc[-1])
    bb_high = safe_float(ind["bb_high"].iloc[-1])
    close = safe_float(last["close"])
    bb_dir: Direction = "neutral"
    
    if all(v is not None for v in [bb_low, bb_high, close, rsi_val]):
        if close <= bb_low and rsi_val < params["RSI"]["oversold"]:
            bb_dir = "buy"
        elif close >= bb_high and rsi_val > params["RSI"]["overbought"]:
            bb_dir = "sell"
    
    # Use original values for output (including NaN if present)
    bb_low_out = float(ind["bb_low"].iloc[-1]) if bb_low is not None else np.nan
    bb_high_out = float(ind["bb_high"].iloc[-1]) if bb_high is not None else np.nan
    close_out = float(last["close"]) if close is not None else np.nan
    results["BBANDS"] = IndicatorResult(bb_dir, {"bb_low": bb_low_out, "bb_high": bb_high_out, "close": close_out}, 0)

    # Stochastic cross
    k = safe_float(ind["stoch_k"].iloc[-1])
    d = safe_float(ind["stoch_d"].iloc[-1])
    k_prev = safe_float(ind["stoch_k"].iloc[-2])
    d_prev = safe_float(ind["stoch_d"].iloc[-2])
    st_dir: Direction = "neutral"
    
    if all(v is not None for v in [k, d, k_prev, d_prev]):
        if _cross_over(k_prev, d_prev, k, d) and k < params["STOCH"]["oversold"]:
            st_dir = "buy"
        elif _cross_under(k_prev, d_prev, k, d) and k > params["STOCH"]["overbought"]:
            st_dir = "sell"
    
    # Use original values for output (including NaN if present)
    k_out = float(ind["stoch_k"].iloc[-1]) if k is not None else np.nan
    d_out = float(ind["stoch_d"].iloc[-1]) if d is not None else np.nan
    results["STOCH"] = IndicatorResult(st_dir, {"k": k_out, "d": d_out}, 0)

    # ATR filter
    atr = safe_float(ind["atr"].iloc[-1])
    atr_dir: Direction = "neutral"
    atr_ratio = 0.0
    
    if atr is not None and close is not None and close > 0:
        atr_ratio = atr / close
        atr_ok = atr_ratio >= params["ATR"]["min_ratio"]
        atr_dir = "buy" if atr_ok else "neutral"
    
    # Use original values for output (including NaN if present)
    atr_out = float(ind["atr"].iloc[-1]) if atr is not None else np.nan
    results["ATR"] = IndicatorResult(atr_dir, {"atr": atr_out, "atr_ratio": atr_ratio}, 0)

    return results


def _get_indicator_contribution(weight: float, is_aligned: bool, is_neutral: bool) -> float:
    """Calculate indicator contribution with partial credit for neutral indicators."""
    if is_aligned:
        return weight
    elif is_neutral:
        return weight * NEUTRAL_WEIGHT_FACTOR
    else:
        return 0.0


def _strength(direction: Direction, contributions: Dict[str, float]) -> float:
    return sum(v for k, v in contributions.items()) if direction in ("buy", "sell") else 0.0


def compute_strategy_strength(results: Dict[str, IndicatorResult], weights: Dict[str, float] | None = None) -> Dict[str, Dict]:
    weights = weights or WEIGHTS

    # Build contributions for each strategy
    # Trend: SMA/EMA cross + MACD + ATR
    trend_dir: Direction = "neutral"
    # Decide direction via majority among SMA, EMA, MACD (ignore ATR for direction)
    dirs = [results["SMA"].direction, results["EMA"].direction, results["MACD"].direction]
    buys = dirs.count("buy")
    sells = dirs.count("sell")
    if buys > sells and buys > 0:
        trend_dir = "buy"
    elif sells > buys and sells > 0:
        trend_dir = "sell"

    trend_contrib = {
        "SMA_EMA": _get_indicator_contribution(
            weights["SMA_EMA"], 
            results["SMA"].direction == trend_dir and results["EMA"].direction == trend_dir,
            results["SMA"].direction == "neutral" or results["EMA"].direction == "neutral"
        ),
        "MACD": _get_indicator_contribution(
            weights["MACD"], 
            results["MACD"].direction == trend_dir,
            results["MACD"].direction == "neutral"
        ),
        "ATR_STABILITY": (weights.get("ATR_STABILITY", weights.get("ATR", 0)) if results["ATR"].direction == "buy" else 0),  # ATR acts as filter; contributes when OK
    }
    trend_strength = _strength(trend_dir, trend_contrib)

    # Momentum: RSI + Stochastic + Bollinger
    momentum_dir: Direction = "neutral"
    dirs_m = [results["RSI"].direction, results["STOCH"].direction, results["BBANDS"].direction]
    buys_m = dirs_m.count("buy")
    sells_m = dirs_m.count("sell")
    if buys_m > sells_m and buys_m > 0:
        momentum_dir = "buy"
    elif sells_m > buys_m and sells_m > 0:
        momentum_dir = "sell"

    momentum_contrib = {
        "RSI": _get_indicator_contribution(
            weights["RSI"], 
            results["RSI"].direction == momentum_dir,
            results["RSI"].direction == "neutral"
        ),
        "STOCH": _get_indicator_contribution(
            weights["STOCH"], 
            results["STOCH"].direction == momentum_dir,
            results["STOCH"].direction == "neutral"
        ),
        "BBANDS": _get_indicator_contribution(
            weights["BBANDS"], 
            results["BBANDS"].direction == momentum_dir,
            results["BBANDS"].direction == "neutral"
        ),
    }
    momentum_strength = _strength(momentum_dir, momentum_contrib)

    # Combined: all indicators together
    # Direction by majority among all (excluding ATR)
    all_dirs = [
        results["RSI"].direction,
        results["MACD"].direction,
        results["SMA"].direction,
        results["EMA"].direction,
        results["BBANDS"].direction,
        results["STOCH"].direction,
    ]
    buys_c = all_dirs.count("buy")
    sells_c = all_dirs.count("sell")
    combined_dir: Direction = "neutral"
    if buys_c > sells_c and buys_c > 0:
        combined_dir = "buy"
    elif sells_c > buys_c and sells_c > 0:
        combined_dir = "sell"

    combined_contrib = {
        "RSI": _get_indicator_contribution(
            weights["RSI"], 
            results["RSI"].direction == combined_dir,
            results["RSI"].direction == "neutral"
        ),
        "MACD": _get_indicator_contribution(
            weights["MACD"], 
            results["MACD"].direction == combined_dir,
            results["MACD"].direction == "neutral"
        ),
        "SMA_EMA": _get_indicator_contribution(
            weights["SMA_EMA"], 
            results["SMA"].direction == combined_dir and results["EMA"].direction == combined_dir,
            results["SMA"].direction == "neutral" or results["EMA"].direction == "neutral"
        ),
        "BBANDS": _get_indicator_contribution(
            weights["BBANDS"], 
            results["BBANDS"].direction == combined_dir,
            results["BBANDS"].direction == "neutral"
        ),
        "STOCH": _get_indicator_contribution(
            weights["STOCH"], 
            results["STOCH"].direction == combined_dir,
            results["STOCH"].direction == "neutral"
        ),
        "ATR_STABILITY": (weights.get("ATR_STABILITY", weights.get("ATR", 0)) if results["ATR"].direction == "buy" else 0),
    }
    combined_strength = _strength(combined_dir, combined_contrib)

    return {
        "trend": {"direction": trend_dir, "strength": trend_strength, "contributions": trend_contrib},
        "momentum": {"direction": momentum_dir, "strength": momentum_strength, "contributions": momentum_contrib},
        "combined": {"direction": combined_dir, "strength": combined_strength, "contributions": combined_contrib},
    }


def get_signal_confidence_zone(strength: float) -> str:
    """Determine signal confidence zone based on strength."""
    # Sort zones by minimum threshold in descending order
    sorted_zones = sorted(CONFIDENCE_ZONES.items(), key=lambda x: x[1]["min"], reverse=True)
    
    for zone_name, zone_config in sorted_zones:
        if strength >= zone_config["min"]:
            return zone_name
    return "neutral"


def best_signal(strategies: Dict[str, Dict]) -> Optional[Dict]:
    """Calculate weighted blend of strategies instead of single best."""
    if not strategies:
        return None
    
    # Calculate weighted blend
    trend_data = strategies.get("trend", {})
    momentum_data = strategies.get("momentum", {})
    combined_data = strategies.get("combined", {})
    
    trend_strength = trend_data.get("strength", 0.0)
    momentum_strength = momentum_data.get("strength", 0.0)
    combined_strength = combined_data.get("strength", 0.0)
    
    # Apply strategy weight ratios
    weighted_trend = trend_strength * TREND_WEIGHT_RATIO
    weighted_momentum = momentum_strength * MOMENTUM_WEIGHT_RATIO
    weighted_combined = combined_strength * (1.0 - TREND_WEIGHT_RATIO - MOMENTUM_WEIGHT_RATIO)
    
    # Calculate final blended strength
    final_strength = weighted_trend + weighted_momentum + weighted_combined
    
    # Determine direction based on strongest individual strategy
    best_individual = None
    max_strength = 0.0
    for name, data in strategies.items():
        if data.get("strength", 0.0) > max_strength:
            max_strength = data.get("strength", 0.0)
            best_individual = {"strategy": name, **data}
    
    if best_individual:
        return {
            "strategy": "weighted_blend",
            "direction": best_individual["direction"],
            "strength": final_strength,
            "components": {
                "trend": {"strength": trend_strength, "weighted": weighted_trend},
                "momentum": {"strength": momentum_strength, "weighted": weighted_momentum},
                "combined": {"strength": combined_strength, "weighted": weighted_combined},
            },
            "best_individual": best_individual["strategy"]
        }
    
    return None


# --- Multi-timeframe helpers ---
def _safe_ema_series(close: pd.Series, length: int) -> pd.Series:
    """Return a pd.Series for EMA(length); if pandas_ta returns None or errors, return NaN series."""
    try:
        s = ta.ema(close, length=length)
        if isinstance(s, pd.Series):
            return s
        # Fallback: same index, NaNs
        return pd.Series([np.nan] * len(close), index=close.index)
    except Exception:
        return pd.Series([np.nan] * len(close), index=close.index)
def ema_trend_direction(df: pd.DataFrame, short_len: int = 50, long_len: int = 200) -> str:
    """Return 'Bullish' if EMA(short_len) > EMA(long_len), else 'Bearish'."""
    ema_s_series = _safe_ema_series(df["close"], short_len)
    ema_l_series = _safe_ema_series(df["close"], long_len)
    # Use last values; NaNs will default to Bearish due to comparison semantics
    ema_s_last = float(ema_s_series.iloc[-1]) if len(ema_s_series) else float("nan")
    ema_l_last = float(ema_l_series.iloc[-1]) if len(ema_l_series) else float("nan")
    return "Bullish" if (ema_s_last > ema_l_last) else "Bearish"


def atr_last(df: pd.DataFrame, length: int = None, params: Dict[str, Dict] | None = None) -> float:
    """Return last ATR value using configured length by default."""
    p = params or INDICATOR_PARAMS
    l = length or p["ATR"]["length"]
    atr_series = ta.atr(df["high"], df["low"], df["close"], length=l)
    return float(atr_series.iloc[-1])


def atr_last_and_mean(df: pd.DataFrame, length: int = None, mean_window: int = 50, params: Dict[str, Dict] | None = None) -> (float, float):
    """Return (last ATR, rolling mean of ATR over mean_window)."""
    p = params or INDICATOR_PARAMS
    l = length or p["ATR"]["length"]
    atr_series = ta.atr(df["high"], df["low"], df["close"], length=l)
    last = float(atr_series.iloc[-1])
    mean = float(atr_series.rolling(mean_window).mean().iloc[-1])
    return last, mean


def price_action_direction(df: pd.DataFrame, lookback: int = 5, params: Dict[str, Dict] | None = None) -> Direction:
    """
    Enriched price action heuristic using common patterns:
    - Engulfing (bullish/bearish) using last two candles
    - Pin bar (bullish/bearish) using wick/body ratios on last candle
    - Trend continuation using EMA20 slope and close position
    Fallback: majority-of-closes with EMA20 filter over `lookback` candles
    """
    p = params or INDICATOR_PARAMS
    if len(df) < max(lookback, p["EMA"]["short"]):
        return "neutral"

    # Prepare last candles
    last = df.iloc[-1]
    prev = df.iloc[-2]
    open_last, close_last, high_last, low_last = float(last["open"]), float(last["close"]), float(last["high"]), float(last["low"])
    open_prev, close_prev = float(prev["open"]), float(prev["close"])

    body_last = abs(close_last - open_last)
    lower_wick_last = min(open_last, close_last) - low_last
    upper_wick_last = high_last - max(open_last, close_last)

    ema20_series = _safe_ema_series(df["close"], length=p["EMA"]["short"])  # EMA short
    ema20_last = float(ema20_series.iloc[-1]) if len(ema20_series) else float("nan")
    ema20_prev3 = float(ema20_series.iloc[-3]) if len(ema20_series) >= 3 else ema20_last

    buy_score = 0
    sell_score = 0

    # Engulfing detection (last two candles)
    prev_bear = close_prev < open_prev
    prev_bull = close_prev > open_prev
    curr_bull = close_last > open_last
    curr_bear = close_last < open_last
    prev_body = abs(close_prev - open_prev)
    engulf_bullish = prev_bear and curr_bull and (open_last <= close_prev) and (close_last >= open_prev) and (body_last >= 0.8 * prev_body)
    engulf_bearish = prev_bull and curr_bear and (open_last >= close_prev) and (close_last <= open_prev) and (body_last >= 0.8 * prev_body)
    if engulf_bullish:
        buy_score += 2
    if engulf_bearish:
        sell_score += 2

    # Pin bar detection on last candle
    body_min = max(body_last, 1e-6)
    lower_ratio = lower_wick_last / body_min
    upper_ratio = upper_wick_last / body_min
    bullish_pin = lower_ratio >= 2.0 and upper_ratio <= 0.7 and close_last >= open_last
    bearish_pin = upper_ratio >= 2.0 and lower_ratio <= 0.7 and close_last <= open_last
    if bullish_pin:
        buy_score += 1
    if bearish_pin:
        sell_score += 1

    # Trend continuation using EMA20 slope + close position
    ema_slope_up = ema20_last > ema20_prev3
    ema_slope_down = ema20_last < ema20_prev3
    above_ema = close_last > ema20_last and float(df["close"].iloc[-2]) > ema20_last
    below_ema = close_last < ema20_last and float(df["close"].iloc[-2]) < ema20_last
    if ema_slope_up and above_ema:
        buy_score += 1
    if ema_slope_down and below_ema:
        sell_score += 1

    if buy_score > sell_score:
        return "buy"
    if sell_score > buy_score:
        return "sell"

    # Fallback: majority of closes with EMA20 filter over lookback
    closes = df["close"].iloc[-lookback:]
    deltas = closes.diff().fillna(0)
    ups = int((deltas > 0).sum())
    downs = int((deltas < 0).sum())
    if ups >= (lookback // 2 + 1) and close_last > ema20_last:
        return "buy"
    if downs >= (lookback // 2 + 1) and close_last < ema20_last:
        return "sell"
    return "neutral"