from __future__ import annotations

import time
import uuid
from typing import Dict, Any, List, Tuple

import pandas as pd

from .config import (
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    SIGNAL_THRESHOLD,
    WEIGHTS,
    PRIMARY_TIMEFRAME,
    CONFIRMATION_TIMEFRAME,
    TREND_TIMEFRAME,
)
from .indicators import (
    compute_indicators,
    evaluate_signals,
    best_signal,
    ema_trend_direction,
    atr_last_and_mean,
    price_action_direction,
)
from .db import (
    insert_backtesting_signals_batch,
    insert_backtesting_run,
    fetch_backtesting_signals_by_run,
)


def _alignment_and_volatility(m15_side: str, h1_df: pd.DataFrame, h4_df: pd.DataFrame) -> Tuple[bool, str, str, str, float, float, bool]:
    h1_dir = ema_trend_direction(h1_df, short_len=50, long_len=200)
    h4_dir = ema_trend_direction(h4_df, short_len=50, long_len=200)
    h1_is_bull = h1_dir == "Bullish"
    aligns_h1 = (m15_side == "buy" and h1_is_bull) or (m15_side == "sell" and not h1_is_bull)

    # Volatility classification
    h1_atr_last, h1_atr_mean50 = atr_last_and_mean(h1_df, length=14, mean_window=50)
    extreme_vol = h1_atr_last > (3.0 * h1_atr_mean50)
    if h1_atr_last > 1.2 * h1_atr_mean50:
        volatility_state = "High"
    elif h1_atr_last < 0.8 * h1_atr_mean50:
        volatility_state = "Low"
    else:
        volatility_state = "Normal"

    return aligns_h1, h1_dir, h4_dir, volatility_state, float(h1_atr_last), float(h1_atr_mean50), extreme_vol


def _trade_plan(m15_side: str, close_price: float, h1_atr_last: float, volatility_state: str) -> Tuple[float, float, float, float, float, float]:
    entry_offset = 0.1 * h1_atr_last
    entry_price = close_price - entry_offset if m15_side == "buy" else close_price + entry_offset

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
    min_sl = 0.0025 * close_price
    max_sl = 0.0120 * close_price
    if sl_distance < min_sl:
        sl_distance = min_sl
    elif sl_distance > max_sl:
        sl_distance = max_sl

    tp_distance = rr * sl_distance
    min_tp = 0.0040 * close_price
    max_tp = 0.0200 * close_price
    if tp_distance < min_tp:
        tp_distance = min_tp
    elif tp_distance > max_tp:
        tp_distance = max_tp

    stop_loss_price = entry_price - sl_distance if m15_side == "buy" else entry_price + sl_distance
    take_profit_price = entry_price + tp_distance if m15_side == "buy" else entry_price - tp_distance

    pip_value = 0.01
    sl_pips = sl_distance / pip_value
    tp_pips = tp_distance / pip_value
    return entry_price, stop_loss_price, take_profit_price, sl_pips, tp_pips, rr


def run_manual_backtest(
    fetch_range_func,
    start_date: str,
    end_date: str,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    initial_balance: float | None = None,
    commission_per_trade: float | None = None,
    spread_adjustment: float | None = None,
) -> Dict[str, Any]:
    ts_start = pd.to_datetime(start_date)
    ts_end = pd.to_datetime(end_date)
    manual_run_id = f"manual_run_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
    t0 = time.time()

    # Fetch data ranges with buffer for EMA/ATR computations
    buffer_days = 10
    buf_start = ts_start - pd.Timedelta(days=buffer_days)

    m15_df = fetch_range_func(symbol, timeframe, buf_start, ts_end)
    h1_df = fetch_range_func(symbol, "1h", buf_start, ts_end)
    h4_df = fetch_range_func(symbol, "4h", buf_start, ts_end)

    records: List[Dict[str, Any]] = []

    for i in range(len(m15_df)):
        ts = pd.to_datetime(m15_df.iloc[i]["time"])
        if ts < ts_start or ts > ts_end:
            continue
        # Need enough history for indicators
        if i < 50:
            continue

        m15_slice = m15_df.iloc[: i + 1]
        h1_slice = h1_df[h1_df["time"] <= ts]
        h4_slice = h4_df[h4_df["time"] <= ts]
        if len(h1_slice) < 50 or len(h4_slice) < 50:
            continue

        ind = compute_indicators(m15_slice)
        res = evaluate_signals(m15_slice, ind)
        best = best_signal({"combined": {"direction": res["EMA"].direction, "strength": 0, "contributions": {}}})

        # Determine strategy via existing approach: compute best among strategies
        # Reuse indicators to evaluate combined/trend/momentum strengths
        from .indicators import compute_strategy_strength
        strat = compute_strategy_strength(res)
        best = best_signal(strat)

        if not best or best["direction"] not in ("buy", "sell"):
            continue
        m15_side = best["direction"]

        aligns_h1, h1_dir, h4_dir, vol_state, h1_atr_last, h1_atr_mean50, extreme = _alignment_and_volatility(
            m15_side, h1_slice, h4_slice
        )
        if not aligns_h1 or extreme:
            continue

        close_m15 = float(m15_slice.iloc[-1]["close"])
        entry_price, sl_price, tp_price, sl_pips, tp_pips, rr = _trade_plan(m15_side, close_m15, h1_atr_last, vol_state)

        alignment_boost = 10.0 if h4_dir == h1_dir else -10.0
        pa_dir = price_action_direction(m15_slice, lookback=5)

        contrib = {
            "RSI": (WEIGHTS["RSI"] if res["RSI"].direction == m15_side else 0),
            "MACD": (WEIGHTS["MACD"] if res["MACD"].direction == m15_side else 0),
            "STOCH": (WEIGHTS["STOCH"] if res["STOCH"].direction == m15_side else 0),
            "BBANDS": (WEIGHTS["BBANDS"] if res["BBANDS"].direction == m15_side else 0),
            "SMA_EMA": (
                WEIGHTS["SMA_EMA"] if (res["SMA"].direction == m15_side and res["EMA"].direction == m15_side) else 0
            ),
            "MTF": (WEIGHTS["MTF"] if aligns_h1 else 0),
            "ATR_STABILITY": (WEIGHTS["ATR_STABILITY"] if vol_state == "Normal" else 0),
            "PRICE_ACTION": (WEIGHTS["PRICE_ACTION"] if pa_dir == m15_side else 0),
        }
        base_strength = float(sum(contrib.values()))
        final_strength = min(100.0, base_strength + alignment_boost)

        if final_strength >= float(SIGNAL_THRESHOLD):
            records.append(
                {
                    "manual_run_id": manual_run_id,
                    "timestamp": ts.isoformat(),
                    "symbol": symbol,
                    "signal_type": m15_side.upper(),
                    "entry_price": entry_price,
                    "stop_loss_price": sl_price,
                    "take_profit_price": tp_price,
                    "final_signal_strength": final_strength,
                    "volatility_state": vol_state,
                    "risk_reward_ratio": rr,
                    "indicator_contributions": contrib,
                    "created_at": pd.Timestamp.utcnow().isoformat(),
                    "source_mode": "manual_backtest",
                }
            )

    signals_generated = len(records)
    avg_conf = float(pd.Series([r["final_signal_strength"] for r in records]).mean()) if records else 0.0
    avg_rr = float(pd.Series([r["risk_reward_ratio"] for r in records]).mean()) if records else 0.0
    run_duration = time.time() - t0

    # Persist
    if records:
        insert_backtesting_signals_batch(records)

    insert_backtesting_run(
        {
            "manual_run_id": manual_run_id,
            "start_date": ts_start.isoformat(),
            "end_date": ts_end.isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "signals_generated": signals_generated,
            "average_confidence": avg_conf,
            "average_rr_ratio": avg_rr,
            "run_duration_seconds": run_duration,
            "status": "completed",
            "created_at": pd.Timestamp.utcnow().isoformat(),
        }
    )

    return {
        "manual_run_id": manual_run_id,
        "signals_generated": signals_generated,
        "average_confidence": avg_conf,
        "status": "completed",
    }


def execute_manual_run(
    fetch_range_func,
    manual_run_id: str,
    initial_balance: float,
    risk_per_trade_percent: float,
    commission_per_trade: float,
    slippage_percent: float,
) -> Dict[str, Any]:
    signals = fetch_backtesting_signals_by_run(manual_run_id)
    balance = float(initial_balance)
    wins = 0
    losses = 0
    equity_curve: List[float] = [balance]
    profits: List[float] = []
    losses_list: List[float] = []

    # Use M1 data to evaluate SL/TP hit after signal timestamp
    for s in sorted(signals, key=lambda r: r["timestamp"]):
        side = s["signal_type"].lower()
        ts = pd.to_datetime(s["timestamp"])  # entry time
        entry = float(s["entry_price"])
        sl = float(s["stop_loss_price"])
        tp = float(s["take_profit_price"])

        risk_amt = balance * (risk_per_trade_percent / 100.0)
        sl_distance = abs(entry - sl)
        tp_distance = abs(tp - entry)
        if sl_distance <= 0:
            continue
        position_size = risk_amt / sl_distance

        # Fetch 2 days of M1 after entry to determine outcome
        end = ts + pd.Timedelta(days=2)
        m1 = fetch_range_func(s["symbol"], "1m", ts, end)
        result = None
        for _, row in m1.iterrows():
            high = float(row["high"])  # next minute high
            low = float(row["low"])   # next minute low
            if side == "buy":
                if low <= sl:
                    result = "sl"
                    break
                if high >= tp:
                    result = "tp"
                    break
            else:
                if high >= sl:
                    result = "sl"
                    break
                if low <= tp:
                    result = "tp"
                    break

        if result == "tp":
            gross = position_size * tp_distance
            cost = commission_per_trade + (slippage_percent / 100.0) * risk_amt
            pnl = gross - cost
            balance += pnl
            wins += 1
            profits.append(pnl)
        else:
            # Assume SL if tp not reached within window or SL hit first
            gross_loss = position_size * sl_distance
            cost = commission_per_trade + (slippage_percent / 100.0) * risk_amt
            pnl = -(gross_loss + cost)
            balance += pnl
            losses += 1
            losses_list.append(-pnl)

        equity_curve.append(balance)
        if balance <= 0:
            break

    total_trades = wins + losses
    final_balance = balance
    net_profit_percent = ((final_balance - initial_balance) / initial_balance) * 100.0
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    max_drawdown = 0.0
    peak = equity_curve[0]
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100.0
        if dd > max_drawdown:
            max_drawdown = dd

    profit_factor = (sum(profits) / sum(losses_list)) if losses_list else 0.0
    avg_rr = float(pd.Series([abs(tp - entry) / abs(entry - sl) for entry, sl, tp in [
        (float(s["entry_price"]), float(s["stop_loss_price"]), float(s["take_profit_price"])) for s in signals
    ]]).mean()) if signals else 0.0

    account_blown = final_balance <= 0
    critical_event = None
    if account_blown:
        critical_event = "Account liquidated due to losses during simulation."

    return {
        "manual_run_id": manual_run_id,
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "net_profit_percent": net_profit_percent,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_percent": win_rate,
        "max_drawdown_percent": max_drawdown,
        "account_blown": account_blown,
        "profit_factor": profit_factor,
        "average_rr_ratio": avg_rr,
        "result_summary": (
            "Profitable run with moderate drawdown and stable performance." if final_balance > initial_balance else
            "Unprofitable run; consider revising parameters."
        ),
        "critical_event": critical_event,
    }