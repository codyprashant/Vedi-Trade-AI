from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from .config import INDICATOR_PARAMS, WEIGHTS


Direction = Literal["buy", "sell", "none"]


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

    # RSI
    try:
        rsi = ta.rsi(df["close"], length=params["RSI"]["length"])
        ind["rsi"] = rsi.rename("rsi") if isinstance(rsi, pd.Series) else _nan_series("rsi")
    except Exception:
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

    # SMA/EMA
    try:
        sma_s = ta.sma(df["close"], length=params["SMA"]["short"])
        ind["sma_short"] = sma_s.rename("sma_short") if isinstance(sma_s, pd.Series) else _nan_series("sma_short")
    except Exception:
        ind["sma_short"] = _nan_series("sma_short")
    try:
        sma_l = ta.sma(df["close"], length=params["SMA"]["long"])
        ind["sma_long"] = sma_l.rename("sma_long") if isinstance(sma_l, pd.Series) else _nan_series("sma_long")
    except Exception:
        ind["sma_long"] = _nan_series("sma_long")
    try:
        ema_s = ta.ema(df["close"], length=params["EMA"]["short"])
        ind["ema_short"] = ema_s.rename("ema_short") if isinstance(ema_s, pd.Series) else _nan_series("ema_short")
    except Exception:
        ind["ema_short"] = _nan_series("ema_short")
    try:
        ema_l = ta.ema(df["close"], length=params["EMA"]["long"])
        ind["ema_long"] = ema_l.rename("ema_long") if isinstance(ema_l, pd.Series) else _nan_series("ema_long")
    except Exception:
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

    # RSI
    rsi_val = float(ind["rsi"].iloc[-1])
    rsi_dir: Direction = "buy" if rsi_val < params["RSI"]["oversold"] else (
        "sell" if rsi_val > params["RSI"]["overbought"] else "none"
    )
    results["RSI"] = IndicatorResult(rsi_dir, {"rsi": rsi_val}, 0)

    # MACD cross
    macd_val = float(ind["macd"].iloc[-1])
    macd_sig = float(ind["macd_signal"].iloc[-1])
    macd_prev = float(ind["macd"].iloc[-2])
    macd_sig_prev = float(ind["macd_signal"].iloc[-2])
    macd_dir: Direction = "none"
    if _cross_over(macd_prev, macd_sig_prev, macd_val, macd_sig):
        macd_dir = "buy"
    elif _cross_under(macd_prev, macd_sig_prev, macd_val, macd_sig):
        macd_dir = "sell"
    results["MACD"] = IndicatorResult(macd_dir, {"macd": macd_val, "signal": macd_sig}, 0)

    # SMA cross
    sma_s = float(ind["sma_short"].iloc[-1])
    sma_l = float(ind["sma_long"].iloc[-1])
    sma_s_prev = float(ind["sma_short"].iloc[-2])
    sma_l_prev = float(ind["sma_long"].iloc[-2])
    sma_dir: Direction = "none"
    if _cross_over(sma_s_prev, sma_l_prev, sma_s, sma_l):
        sma_dir = "buy"
    elif _cross_under(sma_s_prev, sma_l_prev, sma_s, sma_l):
        sma_dir = "sell"
    results["SMA"] = IndicatorResult(sma_dir, {"sma50": sma_s, "sma200": sma_l}, 0)

    # EMA cross
    ema_s = float(ind["ema_short"].iloc[-1])
    ema_l = float(ind["ema_long"].iloc[-1])
    ema_s_prev = float(ind["ema_short"].iloc[-2])
    ema_l_prev = float(ind["ema_long"].iloc[-2])
    ema_dir: Direction = "none"
    if _cross_over(ema_s_prev, ema_l_prev, ema_s, ema_l):
        ema_dir = "buy"
    elif _cross_under(ema_s_prev, ema_l_prev, ema_s, ema_l):
        ema_dir = "sell"
    results["EMA"] = IndicatorResult(ema_dir, {"ema20": ema_s, "ema50": ema_l}, 0)

    # Bollinger + RSI condition
    bb_low = float(ind["bb_low"].iloc[-1])
    bb_high = float(ind["bb_high"].iloc[-1])
    close = float(last["close"])
    bb_dir: Direction = "none"
    if close <= bb_low and rsi_val < params["RSI"]["oversold"]:
        bb_dir = "buy"
    elif close >= bb_high and rsi_val > params["RSI"]["overbought"]:
        bb_dir = "sell"
    results["BBANDS"] = IndicatorResult(bb_dir, {"bb_low": bb_low, "bb_high": bb_high, "close": close}, 0)

    # Stochastic cross
    k = float(ind["stoch_k"].iloc[-1])
    d = float(ind["stoch_d"].iloc[-1])
    k_prev = float(ind["stoch_k"].iloc[-2])
    d_prev = float(ind["stoch_d"].iloc[-2])
    st_dir: Direction = "none"
    if _cross_over(k_prev, d_prev, k, d) and k < params["STOCH"]["oversold"]:
        st_dir = "buy"
    elif _cross_under(k_prev, d_prev, k, d) and k > params["STOCH"]["overbought"]:
        st_dir = "sell"
    results["STOCH"] = IndicatorResult(st_dir, {"k": k, "d": d}, 0)

    # ATR filter
    atr = float(ind["atr"].iloc[-1])
    atr_ratio = atr / close if close else 0.0
    atr_ok = atr_ratio >= params["ATR"]["min_ratio"]
    results["ATR"] = IndicatorResult("buy" if atr_ok else "none", {"atr": atr, "atr_ratio": atr_ratio}, 0)

    return results


def _strength(direction: Direction, contributions: Dict[str, float]) -> float:
    return sum(v for k, v in contributions.items()) if direction in ("buy", "sell") else 0.0


def compute_strategy_strength(results: Dict[str, IndicatorResult], weights: Dict[str, float] | None = None) -> Dict[str, Dict]:
    weights = weights or WEIGHTS

    # Build contributions for each strategy
    # Trend: SMA/EMA cross + MACD + ATR
    trend_dir: Direction = "none"
    # Decide direction via majority among SMA, EMA, MACD (ignore ATR for direction)
    dirs = [results["SMA"].direction, results["EMA"].direction, results["MACD"].direction]
    buys = dirs.count("buy")
    sells = dirs.count("sell")
    if buys > sells and buys > 0:
        trend_dir = "buy"
    elif sells > buys and sells > 0:
        trend_dir = "sell"

    trend_contrib = {
        "SMA_EMA": (weights["SMA_EMA"] if results["SMA"].direction == trend_dir and results["EMA"].direction == trend_dir else 0),
        "MACD": (weights["MACD"] if results["MACD"].direction == trend_dir else 0),
        "ATR": (weights.get("ATR", 0) if results["ATR"].direction == "buy" else 0),  # ATR acts as filter; contributes when OK
    }
    trend_strength = _strength(trend_dir, trend_contrib)

    # Momentum: RSI + Stochastic + Bollinger
    momentum_dir: Direction = "none"
    dirs_m = [results["RSI"].direction, results["STOCH"].direction, results["BBANDS"].direction]
    buys_m = dirs_m.count("buy")
    sells_m = dirs_m.count("sell")
    if buys_m > sells_m and buys_m > 0:
        momentum_dir = "buy"
    elif sells_m > buys_m and sells_m > 0:
        momentum_dir = "sell"

    momentum_contrib = {
        "RSI": (weights["RSI"] if results["RSI"].direction == momentum_dir else 0),
        "STOCH": (weights["STOCH"] if results["STOCH"].direction == momentum_dir else 0),
        "BBANDS": (weights["BBANDS"] if results["BBANDS"].direction == momentum_dir else 0),
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
    combined_dir: Direction = "none"
    if buys_c > sells_c and buys_c > 0:
        combined_dir = "buy"
    elif sells_c > buys_c and sells_c > 0:
        combined_dir = "sell"

    combined_contrib = {
        "RSI": (WEIGHTS["RSI"] if results["RSI"].direction == combined_dir else 0),
        "MACD": (WEIGHTS["MACD"] if results["MACD"].direction == combined_dir else 0),
        "SMA_EMA": (
            WEIGHTS["SMA_EMA"] if (results["SMA"].direction == combined_dir and results["EMA"].direction == combined_dir) else 0
        ),
        "BBANDS": (WEIGHTS["BBANDS"] if results["BBANDS"].direction == combined_dir else 0),
        "STOCH": (WEIGHTS["STOCH"] if results["STOCH"].direction == combined_dir else 0),
        "ATR": (WEIGHTS.get("ATR", 0) if results["ATR"].direction == "buy" else 0),
    }
    combined_strength = _strength(combined_dir, combined_contrib)

    return {
        "trend": {"direction": trend_dir, "strength": trend_strength, "contributions": trend_contrib},
        "momentum": {"direction": momentum_dir, "strength": momentum_strength, "contributions": momentum_contrib},
        "combined": {"direction": combined_dir, "strength": combined_strength, "contributions": combined_contrib},
    }


def best_signal(strategies: Dict[str, Dict]) -> Optional[Dict]:
    best = None
    for name, data in strategies.items():
        if best is None or data["strength"] > best["strength"]:
            best = {"strategy": name, **data}
    return best


# --- Multi-timeframe helpers ---
def ema_trend_direction(df: pd.DataFrame, short_len: int = 50, long_len: int = 200) -> str:
    """Return 'Bullish' if EMA(short_len) > EMA(long_len), else 'Bearish'."""
    ema_s = ta.ema(df["close"], length=short_len).iloc[-1]
    ema_l = ta.ema(df["close"], length=long_len).iloc[-1]
    return "Bullish" if float(ema_s) > float(ema_l) else "Bearish"


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
        return "none"

    # Prepare last candles
    last = df.iloc[-1]
    prev = df.iloc[-2]
    open_last, close_last, high_last, low_last = float(last["open"]), float(last["close"]), float(last["high"]), float(last["low"])
    open_prev, close_prev = float(prev["open"]), float(prev["close"])

    body_last = abs(close_last - open_last)
    lower_wick_last = min(open_last, close_last) - low_last
    upper_wick_last = high_last - max(open_last, close_last)

    ema20_series = ta.ema(df["close"], length=p["EMA"]["short"])  # EMA short
    ema20_last = float(ema20_series.iloc[-1])
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
    return "none"