import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, Set, Optional, Any

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from .env_loader import load_dotenv_file
from .signal_engine import SignalEngine
from .indicators import (
    compute_indicators,
    evaluate_signals,
    compute_strategy_strength,
    best_signal,
    ema_trend_direction,
    atr_last_and_mean,
    price_action_direction,
)
from .db import (
    ensure_signals_table,
    ensure_backtesting_tables,
    ensure_strategy_tables,
    create_default_gold_strategy_if_missing,
    fetch_recent_signals,
    fetch_recent_signals_by_symbol,
    fetch_latest_signal_by_symbol,
    ensure_indicator_snapshots_table,
    fetch_latest_indicator_snapshots,
    insert_backtesting_signals_batch,
    insert_backtesting_run,
    fetch_backtesting_runs,
    fetch_backtesting_signals_by_run,
    get_strategies,
    get_strategy_details,
    update_indicator_params,
    update_strategy_weights,
    update_strategy_schedule,
    update_strategy_threshold,
    set_active_strategy,
)
from .config import (
    ALLOWED_SYMBOLS,
    INDICATOR_PARAMS,
    WEIGHTS,
    SIGNAL_THRESHOLD,
    PRIMARY_TIMEFRAME,
    CONFIRMATION_TIMEFRAME,
    TREND_TIMEFRAME,
)


# Load env early so CORS can read .env values
try:
    load_dotenv_file()
except Exception:
    pass
load_dotenv()


app = FastAPI(
    title="PriceTracker Yahoo API",
    description="Stream prices via WebSocket and fetch OHLCV using Yahoo Finance.",
    version="0.1.0",
)


# --- CORS configuration (allow_origins from .env via CORS_ALLOW_ORIGINS) ---
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "https://vedi-trade-ai.base44.app"
]

def _parse_cors_origins(raw: Optional[str]) -> list[str]:
    if not raw:
        return DEFAULT_CORS_ORIGINS
    raw = raw.strip()
    if raw == "*":
        return ["*"]
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]

allow_origins_cfg = _parse_cors_origins(os.getenv("CORS_ALLOW_ORIGINS"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_cfg,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Tracks subscribers per symbol
subscriptions: Dict[str, Set[WebSocket]] = {}

# Background polling task
poll_task: Optional[asyncio.Task] = None

# Inactivity shutdown watcher
inactivity_task: Optional[asyncio.Task] = None

# Signal engine task
signal_task: Optional[asyncio.Task] = None
signal_engine_instance: Optional[SignalEngine] = None


def total_subscribers() -> int:
    return sum(len(s) for s in subscriptions.values())


async def inactivity_watch():
    """If no subscribers for 60s, stop polling task."""
    global inactivity_task, poll_task
    try:
        await asyncio.sleep(60)
        if total_subscribers() == 0 and poll_task is not None:
            try:
                poll_task.cancel()
            except Exception:
                pass
            poll_task = None
    finally:
        inactivity_task = None


def ensure_inactivity_watch():
    global inactivity_task
    if total_subscribers() == 0 and inactivity_task is None:
        inactivity_task = asyncio.create_task(inactivity_watch())


# --- Symbol normalization and resolution (canonical -> Yahoo ticker) ---
# Canonicalization: map common aliases to one canonical symbol used everywhere
CANONICAL_SYMBOL_MAP = {
    # Gold aliases
    "XAUUSD=X": "XAUUSD",
    "XAU=X": "XAUUSD",
    "XAU": "XAUUSD",
    "GOLD": "XAUUSD",
    "GC=F": "XAUUSD",
}

# Yahoo tickers for our canonical symbols
YAHOO_SYMBOL_FOR_CANONICAL = {
    "XAUUSD": "GC=F",
    "USDCAD": "USDCAD=X",
    "USDJPY": "USDJPY=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "AUS200": "^AXJO",
    "UK100": "^FTSE",
    "DJ30": "^DJI",
    "SPX": "^GSPC",
    "NAS100": "^NDX",
    "GER40": "^GDAXI",
    "FRA40": "^FCHI",
}

def canonical_symbol(symbol: str) -> str:
    sym = symbol.upper().strip()
    return CANONICAL_SYMBOL_MAP.get(sym, sym)

def resolve_symbol(symbol: str) -> str:
    canon = canonical_symbol(symbol)
    if canon not in ALLOWED_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Symbol not allowed: {symbol}. Allowed: {', '.join(ALLOWED_SYMBOLS)}")
    return YAHOO_SYMBOL_FOR_CANONICAL.get(canon, canon)


# --- Timeframe parsing ---
TF_MAP = {
    # Map app timeframe strings to Yahoo intervals
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "h1": "60m",
    "4h": "4h",
    "h4": "4h",
    "1d": "1d",
    "1w": "1wk",
    "1mo": "1mo",
    "1mon": "1mo",
    "1month": "1mo",
    "5min": "5m",
    "15min": "15m",
    "1hour": "60m",
    "4hour": "240m",
    "1day": "1d",
    # MT5-style uppercase shorthands
    "m1": "1m",
    "m5": "5m",
    "m15": "15m",
    "m30": "30m",
    "h1": "60m",
    # h4 handled via synthetic 4h resampling
    "d1": "1d",
}


def parse_timeframe(tf_str: str) -> Optional[str]:
    key = tf_str.strip().lower()
    return TF_MAP.get(key)


def _approx_period_for_count(interval: str, count: int) -> str:
    """Return a Yahoo period that likely covers `count` candles for an interval.

    This is heuristic to avoid throttling while fetching sufficient data.
    """
    count = max(1, int(count))
    if interval.endswith("m"):
        minutes = int(interval.replace("m", ""))
        total_minutes = minutes * count
        # Cap to 7d for 1m (Yahoo limit); otherwise stretch appropriately
        if minutes == 1:
            days = min(7, max(1, (total_minutes // (60 * 24)) + 1))
        else:
            days = max(1, (total_minutes // (60 * 24)) + 1)
        return f"{days}d"
    if interval.endswith("h"):
        hours = int(interval.replace("h", ""))
        total_hours = hours * count
        days = max(1, (total_hours // 24) + 1)
        return f"{days}d"
    if interval == "1d":
        weeks = max(1, (count // 5) + 1)
        return f"{weeks}wk"
    if interval == "1wk":
        months = max(1, (count // 4) + 1)
        return f"{months}mo"
    if interval == "1mo":
        years = max(1, (count // 12) + 1)
        return f"{years}y"
    return "1mo"


async def poll_loop():
    """Poll Yahoo Finance data for subscribed symbols and broadcast every 120s."""
    global poll_task
    try:
        while True:
            if total_subscribers() == 0:
                ensure_inactivity_watch()
                await asyncio.sleep(1)
                continue

            for symbol, sockets in list(subscriptions.items()):
                if not sockets:
                    continue

                y_symbol = resolve_symbol(symbol)
                ticker = yf.Ticker(y_symbol)

                bid = None
                previous_close = None
                market_state = None
                regular_market_price = None
                ts_iso = datetime.now(timezone.utc).isoformat()

                # Try fast_info first (lightweight)
                try:
                    fi = getattr(ticker, "fast_info", None)
                    if fi:
                        # Use lastPrice as bid value
                        lastPrice_raw = fi.get("lastPrice", None)
                        bid = float(lastPrice_raw) if lastPrice_raw is not None else None
                        
                        # Extract additional fields
                        previousClose_raw = fi.get("previousClose", None)
                        previous_close = float(previousClose_raw) if previousClose_raw is not None else None
                        
                        market_state = fi.get("marketState", None)
                        
                        regularMarketPrice_raw = fi.get("regularMarketPrice", None)
                        regular_market_price = float(regularMarketPrice_raw) if regularMarketPrice_raw is not None else None
                except Exception as e:
                    print(f"Error getting fast_info for {symbol}: {e}")

                # Fallback to latest 1m candle close
                if bid is None:
                    try:
                        hist = ticker.history(period="1d", interval="1m")
                        if hist is not None and not hist.empty:
                            row = hist.tail(1)
                            bid = float(row["Close"].iloc[0])
                            idx = row.index[-1]
                            
                            try:
                                ts_iso = pd.Timestamp(idx).tz_convert("UTC").isoformat()
                            except Exception:
                                ts_iso = pd.Timestamp(idx).tz_localize("UTC").isoformat()
                    except Exception as e:
                        print(f"Error getting historical data for {symbol}: {e}")

                # Compute indicators for the symbol (using 15m timeframe by default)
                indicators_data = None
                evaluation_data = None
                try:
                    timeframe = "15m"  # Default timeframe for indicators
                    count = 200  # Default count for indicators calculation
                    
                    # Fetch recent history and compute indicators
                    df = fetch_history_df(symbol, timeframe, max(100, count))
                    if df is not None and len(df) >= 3:
                        ind = compute_indicators(df, INDICATOR_PARAMS)
                        res = evaluate_signals(df, ind, INDICATOR_PARAMS)
                        
                        # Extract last values for each indicator series
                        last_vals: dict[str, Any] = {}
                        for k, series in ind.items():
                            try:
                                last_vals[k] = float(series.iloc[-1])
                            except Exception:
                                last_vals[k] = None
                        
                        directions = {name: r.direction for name, r in res.items()}
                        indicators_data = last_vals
                        evaluation_data = directions
                except Exception as e:
                    print(f"Error computing indicators for {symbol}: {e}")

                payload = {
                    "symbol": symbol,
                    "time": ts_iso,
                    "bid": bid,
                    "previousClose": previous_close,
                    "marketState": market_state,
                    "regularMarketPrice": regular_market_price,
                    "indicators": indicators_data,
                    "evaluation": evaluation_data,
                }

                dead: Set[WebSocket] = set()
                for ws in list(sockets):
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        dead.add(ws)

                for ws in dead:
                    sockets.discard(ws)

            # Yahoo API throttle: poll every 120 seconds
            await asyncio.sleep(120)
    finally:
        poll_task = None


def ensure_polling():
    global poll_task
    if poll_task is None:
        poll_task = asyncio.create_task(poll_loop())


@app.get("/health")
async def health():
    return {
        "subscribers": total_subscribers(),
        "symbols": {k: len(v) for k, v in subscriptions.items()},
        "data_source": "yahoo",
        "signal_engine_running": signal_task is not None,
    }


@app.get("/history")
async def history(symbol: str = "XAUUSD", timeframe: str = "15m", count: int = 500):
    interval = parse_timeframe(timeframe)
    if not interval:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {timeframe}")
    # Validate and normalize symbol
    y_symbol = resolve_symbol(symbol)
    ticker = yf.Ticker(y_symbol)

    period = _approx_period_for_count(interval, count)
    try:
        df = ticker.history(period=period, interval=interval)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yahoo history fetch failed: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {y_symbol} {timeframe}")

    # Keep only last `count` rows
    df = df.tail(count)

    candles = []
    for ts, row in df.iterrows():
        # Ensure UTC ISO timestamp
        try:
            ts_iso = pd.Timestamp(ts).tz_convert("UTC").isoformat()
        except Exception:
            ts_iso = pd.Timestamp(ts).tz_localize("UTC").isoformat()
        candles.append(
            {
                "t": ts_iso,
                "o": float(row["Open"]),
                "h": float(row["High"]),
                "l": float(row["Low"]),
                "c": float(row["Close"]),
                "v": int(row.get("Volume", 0) or 0),
            }
        )

    return {"symbol": symbol, "symbol_yahoo": y_symbol, "timeframe": timeframe, "count": len(candles), "candles": candles}


def fetch_history_df(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    interval = parse_timeframe(timeframe)
    if not interval:
        raise RuntimeError(f"Unsupported timeframe: {timeframe}")
    # Validate and normalize symbol
    y_symbol = resolve_symbol(symbol)
    ticker = yf.Ticker(y_symbol)
    period = _approx_period_for_count(interval, count)
    df = ticker.history(period=period, interval=interval)
    # Generic fallbacks: resample from smaller intervals if native interval empty
    def try_resample(from_interval: str, rule: str, mult: int) -> Optional[pd.DataFrame]:
        try:
            alt_count = max(count * mult, 200)
            alt_period = _approx_period_for_count(from_interval, alt_count)
            df1 = ticker.history(period=alt_period, interval=from_interval)
            if df1 is None or df1.empty:
                return None
            df1 = df1.reset_index().rename(columns={"Datetime": "time"})
            df1["time"] = pd.to_datetime(df1["time"])  # preserve tz
            df1 = df1.set_index("time")
            ohlc = df1[["Open", "High", "Low", "Close"]].resample(rule, label="right", closed="right").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last"
            })
            vol = df1[["Volume"]].resample(rule, label="right", closed="right").sum()
            out = pd.concat([ohlc, vol], axis=1).dropna()
            return out
        except Exception:
            return None

    if df is None or df.empty:
        if interval == "4h":
            df = try_resample("1h", "4H", 6) or try_resample("30m", "4H", 12) or try_resample("15m", "4H", 24)
        elif interval in ("60m", "1h"):
            df = try_resample("30m", "1H", 4) or try_resample("15m", "1H", 8) or try_resample("5m", "1H", 24)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {y_symbol} {timeframe}")
    df = df.tail(count)
    df = df.reset_index()
    df = df.rename(columns={"Datetime": "time"})
    df["time"] = pd.to_datetime(df["time"])  # keep timezone if present
    return df[["time", "Open", "High", "Low", "Close", "Volume"]].rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    })


def fetch_range_df(symbol: str, timeframe: str, start_ts, end_ts) -> pd.DataFrame:
    interval = parse_timeframe(timeframe)
    if not interval:
        raise RuntimeError(f"Unsupported timeframe: {timeframe}")
    # Validate and normalize symbol
    y_symbol = resolve_symbol(symbol)
    start_dt = pd.to_datetime(start_ts)
    end_dt = pd.to_datetime(end_ts)
    ticker = yf.Ticker(y_symbol)
    df = ticker.history(start=start_dt.to_pydatetime(), end=end_dt.to_pydatetime(), interval=interval)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {y_symbol} {timeframe} in range")
    df = df.reset_index()
    df = df.rename(columns={"Datetime": "time"})
    df["time"] = pd.to_datetime(df["time"])  # keep timezone if present
    return df[["time", "Open", "High", "Low", "Close", "Volume"]].rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    })


def _normalize_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lower-case columns
    cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols)
    # Map aliases
    alias_map = {
        "datetime": "time",
        "date": "time",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    for src, dst in alias_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    # Ensure required columns exist
    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing required columns: {missing}")
    # Parse time column
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    return df[["time", "open", "high", "low", "close", "volume"]]


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    idx = df.set_index("time").index
    df = df.set_index("time")
    ohlc = df[["open", "high", "low", "close"]].resample(rule, label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    })
    vol = df[["volume"]].resample(rule, label="right", closed="right").sum()
    out = pd.concat([ohlc, vol], axis=1).dropna()
    out = out.reset_index().rename(columns={"index": "time"})
    return out[["time", "open", "high", "low", "close", "volume"]]


def fetch_range_df_csv(symbol: str, timeframe: str, start_ts, end_ts) -> pd.DataFrame:
    """Read OHLCV from CSV at data/backtesting/<SYMBOL>_<TIMEFRAME>.csv and filter by range."""
    tf = timeframe.strip()
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "backtesting", f"{symbol}_{tf}.csv")
    if not os.path.exists(path):
        raise RuntimeError(f"CSV not found: {path}")
    raw = pd.read_csv(path)
    df = _normalize_csv_columns(raw)
    start_dt = pd.to_datetime(start_ts)
    end_dt = pd.to_datetime(end_ts)
    df = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]
    if df.empty:
        raise RuntimeError("CSV has no rows within the requested range")
    return df


# --- Strategy configuration models ---
class UpdateIndicatorRequest(BaseModel):
    params: Dict[str, Any]

class UpdateWeightsRequest(BaseModel):
    weights: Dict[str, Any]

class UpdateScheduleRequest(BaseModel):
    run_interval_seconds: int

class UpdateThresholdRequest(BaseModel):
    signal_threshold: float

# --- Backtesting models ---
class ManualBacktestRequest(BaseModel):
    symbol: str
    timeframe: str | None = None
    start_date: str
    end_date: str
    manual_run_id: str | None = None
    source_mode: str | None = None  # 'csv' or 'yahoo'


@app.post("/api/backtest/manual/generate")
async def backtest_manual_generate(req: ManualBacktestRequest):
    """
    Generate backtesting signals using Yahoo Finance OHLCV and the current indicator/strategy config.

    - Computes signals on the requested `timeframe` (defaults to configured primary timeframe).
    - Aligns with H1 confirmation and H4 trend to compute final confidence and trade plan.
    - Persists signals and run metadata to Postgres backtesting tables.
    """
    try:
        primary_tf = req.timeframe or PRIMARY_TIMEFRAME
        confirmation_tf = CONFIRMATION_TIMEFRAME
        trend_tf = TREND_TIMEFRAME

        use_csv = (req.source_mode or "").lower() == "csv"
        if use_csv:
            # Load primary timeframe from CSV; derive H1/H4 by resampling if needed
            df = fetch_range_df_csv(req.symbol, primary_tf, req.start_date, req.end_date)
            # Derive H1/H4
            # Map timeframe to pandas resample rule
            tf_rules = {
                "1m": "1T", "5m": "5T", "15m": "15T", "30m": "30T",
                "1h": "1H", "h1": "1H", "4h": "4H", "h4": "4H", "1d": "1D"
            }
            # Confirmation: 1H
            if primary_tf.lower() in ("1h", "h1"):
                h1_df = df.copy()
            else:
                h1_df = resample_ohlcv(df, "1H")
            # Trend: 4H
            if primary_tf.lower() in ("4h", "h4"):
                h4_df = df.copy()
            else:
                # If we already have h1_df, prefer resampling from it
                base_for_h4 = h1_df if len(h1_df) > 0 else df
                h4_df = resample_ohlcv(base_for_h4, "4H")
        else:
            # Fetch primary + higher timeframes from Yahoo
            df = fetch_range_df(req.symbol, primary_tf, req.start_date, req.end_date)
            h1_df = fetch_range_df(req.symbol, confirmation_tf, req.start_date, req.end_date)
            h4_df = fetch_range_df(req.symbol, trend_tf, req.start_date, req.end_date)

        if len(df) < 3:
            raise HTTPException(status_code=400, detail="Insufficient data for backtesting range")

        records: list[dict] = []
        strengths: list[float] = []
        rrs: list[float] = []

        # Iterate candles; compute indicators per step to ensure correct previous values
        for i in range(1, len(df)):
            df_i = df.iloc[: i + 1]
            # Compute indicators for the current slice
            ind = compute_indicators(df_i, INDICATOR_PARAMS)
            # Validate last two values exist for all needed series
            needed = [
                "rsi",
                "macd",
                "macd_signal",
                "sma_short",
                "sma_long",
                "ema_short",
                "ema_long",
                "bb_low",
                "bb_high",
                "stoch_k",
                "stoch_d",
                "atr",
            ]
            if any(len(ind[n]) < 2 or pd.isna(ind[n].iloc[-1]) or pd.isna(ind[n].iloc[-2]) for n in needed):
                continue

            # Evaluate and compute strategy strength
            res = evaluate_signals(df_i, ind, INDICATOR_PARAMS)
            strat = compute_strategy_strength(res, WEIGHTS)
            best = best_signal(strat)
            if not best or best["direction"] not in ("buy", "sell"):
                continue

            ts = pd.to_datetime(df_i.iloc[-1]["time"]).isoformat()

            # Find corresponding higher timeframe slices up to ts
            h1_slice = h1_df[h1_df["time"] <= df_i.iloc[-1]["time"]]
            h4_slice = h4_df[h4_df["time"] <= df_i.iloc[-1]["time"]]
            if len(h1_slice) < 5 or len(h4_slice) < 5:
                # Need at least a few points to compute EMA/ATR reliably
                continue

            h1_dir = ema_trend_direction(h1_slice, short_len=50, long_len=200)
            h1_atr_last, h1_atr_mean50 = atr_last_and_mean(
                h1_slice,
                length=INDICATOR_PARAMS.get("ATR", {}).get("length", 14),
                mean_window=50,
                params=INDICATOR_PARAMS,
            )
            h4_dir = ema_trend_direction(h4_slice, short_len=50, long_len=200)

            # Alignment validation and volatility checks
            m15_side = best["direction"]
            h1_is_bull = h1_dir == "Bullish"
            aligns_h1 = (m15_side == "buy" and h1_is_bull) or (m15_side == "sell" and not h1_is_bull)
            if not aligns_h1:
                continue

            extreme_vol = float(h1_atr_last) > (3.0 * float(h1_atr_mean50))
            if extreme_vol:
                continue

            if float(h1_atr_last) > 1.2 * float(h1_atr_mean50):
                volatility_state = "High"
                rr = 1.2
                sl_mult = 2.0
            elif float(h1_atr_last) < 0.8 * float(h1_atr_mean50):
                volatility_state = "Low"
                rr = 1.8
                sl_mult = 1.0
            else:
                volatility_state = "Normal"
                rr = 1.5
                sl_mult = 1.5

            # Trade plan
            close_m15 = float(df_i.iloc[-1]["close"])  # last close on primary TF
            entry_offset = 0.1 * float(h1_atr_last)
            entry_price = close_m15 - entry_offset if m15_side == "buy" else close_m15 + entry_offset

            sl_distance = sl_mult * float(h1_atr_last)
            min_sl = 0.0025 * close_m15
            max_sl = 0.0120 * close_m15
            if sl_distance < min_sl:
                sl_distance = min_sl
            elif sl_distance > max_sl:
                sl_distance = max_sl

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

            alignment_boost = 10.0 if h4_dir == h1_dir else -10.0
            pa_dir = price_action_direction(df_i, lookback=5, params=INDICATOR_PARAMS)

            contrib_new = {
                "RSI": (WEIGHTS.get("RSI", 0) if res["RSI"].direction == m15_side else 0),
                "MACD": (WEIGHTS.get("MACD", 0) if res["MACD"].direction == m15_side else 0),
                "STOCH": (WEIGHTS.get("STOCH", 0) if res["STOCH"].direction == m15_side else 0),
                "BBANDS": (WEIGHTS.get("BBANDS", 0) if res["BBANDS"].direction == m15_side else 0),
                "SMA_EMA": (
                    WEIGHTS.get("SMA_EMA", 0)
                    if (res["SMA"].direction == m15_side and res["EMA"].direction == m15_side)
                    else 0
                ),
                "MTF": (WEIGHTS.get("MTF", 0) if aligns_h1 else 0),
                "ATR_STABILITY": (WEIGHTS.get("ATR_STABILITY", 0) if volatility_state == "Normal" else 0),
                "PRICE_ACTION": (WEIGHTS.get("PRICE_ACTION", 0) if pa_dir == m15_side else 0),
            }
            base_strength = float(sum(contrib_new.values()))
            final_strength = min(100.0, base_strength + alignment_boost)

            if final_strength >= float(SIGNAL_THRESHOLD):
                records.append(
                    {
                        "manual_run_id": req.manual_run_id or f"run_{req.symbol}_{primary_tf}_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}",
                        "timestamp": ts,
                        "symbol": req.symbol,
                        "signal_type": m15_side.upper(),
                        "entry_price": entry_price,
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price,
                        "final_signal_strength": final_strength,
                        "volatility_state": volatility_state,
                        "risk_reward_ratio": rr,
                        "indicator_contributions": contrib_new,
                        "source_mode": "yahoo_backtest",
                    }
                )
                strengths.append(final_strength)
                rrs.append(rr)

        if not records:
            # Still insert a run with zero signals for traceability
            run_id = req.manual_run_id or f"run_{req.symbol}_{primary_tf}_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}"
            insert_backtesting_run(
                {
                    "manual_run_id": run_id,
                    "start_date": pd.to_datetime(req.start_date).isoformat(),
                    "end_date": pd.to_datetime(req.end_date).isoformat(),
                    "symbol": req.symbol,
                    "timeframe": primary_tf,
                    "signals_generated": 0,
                    "average_confidence": None,
                    "average_rr_ratio": None,
                    "run_duration_seconds": 0.0,
                    "status": "completed",
                    "created_at": pd.Timestamp.utcnow().isoformat(),
                }
            )
            return {"manual_run_id": run_id, "signals_generated": 0}

        # Persist signals and run meta
        run_id = records[0]["manual_run_id"]
        insert_backtesting_signals_batch(records)
        insert_backtesting_run(
            {
                "manual_run_id": run_id,
                "start_date": pd.to_datetime(req.start_date).isoformat(),
                "end_date": pd.to_datetime(req.end_date).isoformat(),
                "symbol": req.symbol,
                "timeframe": primary_tf,
                "signals_generated": len(records),
                "average_confidence": float(sum(strengths) / len(strengths)) if strengths else None,
                "average_rr_ratio": float(sum(rrs) / len(rrs)) if rrs else None,
                "run_duration_seconds": 0.0,
                "status": "completed",
                "created_at": pd.Timestamp.utcnow().isoformat(),
            }
        )
        return {
            "manual_run_id": run_id,
            "signals_generated": len(records),
            "average_confidence": float(sum(strengths) / len(strengths)) if strengths else None,
            "average_rr_ratio": float(sum(rrs) / len(rrs)) if rrs else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtesting failed: {e}")


@app.get("/api/backtest/manual/runs")
async def backtest_runs(from_date: str | None = None, to_date: str | None = None, symbol: str | None = None, min_signal_strength: float | None = None):
    try:
        rows = fetch_backtesting_runs(from_date=from_date, to_date=to_date, symbol=symbol, min_signal_strength=min_signal_strength)
        return {"count": len(rows), "runs": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch backtesting runs: {e}")


@app.get("/api/backtest/manual/signals")
async def backtest_signals(manual_run_id: str):
    try:
        rows = fetch_backtesting_signals_by_run(manual_run_id)
        return {"count": len(rows), "signals": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch backtesting signals: {e}")


@app.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket, symbol: str = "XAUUSD"):
    await websocket.accept()
    # Validate and normalize symbol up-front; store canonical in subscriptions
    canon = canonical_symbol(symbol)
    if canon not in ALLOWED_SYMBOLS:
        try:
            await websocket.send_json({"type": "error", "error": "symbol_not_allowed", "allowed": ALLOWED_SYMBOLS})
        except Exception:
            pass
        await websocket.close()
        return

    symbol = canon
    sockets = subscriptions.setdefault(symbol, set())
    sockets.add(websocket)

    global inactivity_task
    if inactivity_task is not None:
        try:
            inactivity_task.cancel()
        except Exception:
            pass
        inactivity_task = None

    ensure_polling()

    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({"type": "heartbeat", "ts": datetime.now(timezone.utc).isoformat()})
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        sockets.discard(websocket)
        if len(sockets) == 0 and symbol in subscriptions and subscriptions[symbol] == set():
            subscriptions.pop(symbol, None)
        ensure_inactivity_watch()


@app.get("/signals/recent")
async def recent_signals(limit: int = 20, min_strength: float | None = None, symbol: str | None = None):
    try:
        if symbol:
            canon = canonical_symbol(symbol)
            if canon not in ALLOWED_SYMBOLS:
                raise HTTPException(status_code=400, detail=f"symbol_not_allowed: {symbol}")
            rows = fetch_recent_signals_by_symbol(canon, limit)
        else:
            rows = fetch_recent_signals(limit)
        if min_strength is not None:
            rows = [r for r in rows if float(r.get("final_signal_strength") or 0) >= float(min_strength)]
        return {"count": len(rows), "signals": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Postgres query failed: {e}")


@app.get("/signals/latest")
async def latest_signal(symbol: str):
    """Return the most recent signal for a specific symbol, including all indicator values and contributions."""
    try:
        canon = canonical_symbol(symbol)
        if canon not in ALLOWED_SYMBOLS:
            raise HTTPException(status_code=400, detail=f"symbol_not_allowed: {symbol}")
        row = fetch_latest_signal_by_symbol(canon)
        if not row:
            raise HTTPException(status_code=404, detail=f"No signal found for {canon}")
        return {"symbol": canon, "signal": row}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch latest signal: {e}")


@app.get("/indicators/latest")
async def indicators_latest(symbols: str | None = None):
    """Return latest indicator analysis snapshot per symbol.

    Query param `symbols` can be a comma-separated list; defaults to configured symbols.
    """
    try:
        if symbols:
            sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
        else:
            sym_list = ALLOWED_SYMBOLS
        rows = fetch_latest_indicator_snapshots(sym_list)
        return {"count": len(rows), "snapshots": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Snapshot query failed: {e}")


@app.on_event("startup")
async def init_env_and_start_signal_engine():
    # Ensure DB tables exist and start signal engine using Yahoo history fetch
    try:
        ensure_signals_table()
    except Exception as e:
        print(f"Postgres init failed (signals): {e}")
    try:
        ensure_backtesting_tables()
    except Exception as e:
        print(f"Postgres init failed (backtesting): {e}")
    try:
        ensure_strategy_tables()
        create_default_gold_strategy_if_missing()
    except Exception as e:
        print(f"Postgres init failed (strategies): {e}")
    try:
        ensure_indicator_snapshots_table()
    except Exception as e:
        print(f"Postgres init failed (indicator_snapshots): {e}")

    # Start signal engine (fetches Yahoo data)
    global signal_task, signal_engine_instance
    try:
        signal_engine_instance = SignalEngine(fetch_history_df)
        signal_task = asyncio.create_task(signal_engine_instance.run())
    except Exception as e:
        print(f"Failed to start signal engine: {e}")


@app.on_event("shutdown")
async def stop_signal_engine():
    global signal_task, signal_engine_instance
    if signal_engine_instance:
        signal_engine_instance.stop()
    if signal_task:
        try:
            await asyncio.wait_for(signal_task, timeout=2)
        except Exception:
            pass


@app.post("/api/signals/compute")
async def signals_compute(symbols: str | None = None):
    """Manually trigger a one-off signal computation for all or specified symbols.

    Query param `symbols` can be a comma-separated list; defaults to strategy-configured symbols.
    Returns a per-symbol summary of indicator validity, directions, strengths, and DB insert status.
    """
    try:
        # Ensure engine instance exists
        global signal_engine_instance
        if signal_engine_instance is None:
            signal_engine_instance = SignalEngine(fetch_history_df)

        sym_list = None
        if symbols:
            sym_list = [s.strip() for s in symbols.split(",") if s.strip()]

        results = await signal_engine_instance.compute_once(sym_list)
        return {"count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual compute failed: {e}")


@app.get("/indicators/live")
async def indicators_live(symbol: str, timeframe: str = "15m", count: int = 200):
    """Compute live technical indicators for the latest candle set from Yahoo, respecting the latest price.

    Returns last indicator values and evaluation directions for the most recent candle.
    """
    try:
        interval = parse_timeframe(timeframe)
        if not interval:
            raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {timeframe}")
        canon = canonical_symbol(symbol)
        if canon not in ALLOWED_SYMBOLS:
            raise HTTPException(status_code=400, detail=f"symbol_not_allowed: {symbol}")
        # Fetch recent history and compute indicators on it
        df = fetch_history_df(canon, timeframe, max(100, count))
        if df is None or len(df) < 3:
            raise HTTPException(status_code=400, detail="Insufficient data to compute indicators")
        ind = compute_indicators(df, INDICATOR_PARAMS)
        res = evaluate_signals(df, ind, INDICATOR_PARAMS)
        ts = pd.to_datetime(df.iloc[-1]["time"]).isoformat()
        # Extract last values for each indicator series
        last_vals: dict[str, Any] = {}
        for k, series in ind.items():
            try:
                last_vals[k] = float(series.iloc[-1])
            except Exception:
                last_vals[k] = None
        directions = {name: r.direction for name, r in res.items()}
        return {
            "symbol": canon,
            "timeframe": timeframe,
            "timestamp": ts,
            "indicators": last_vals,
            "evaluation": directions,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute live indicators: {e}")


# --- Strategy Configuration API Endpoints ---

@app.get("/api/config/strategies")
async def get_all_strategies():
    """Get all strategies with their basic information."""
    try:
        strategies = get_strategies()
        return {"strategies": strategies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch strategies: {e}")


@app.get("/api/config/strategies/{strategy_id}")
async def get_strategy_by_id(strategy_id: int):
    """Get detailed information about a specific strategy including indicators and weights."""
    try:
        strategy = get_strategy_details(strategy_id)
        return strategy
    except RuntimeError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch strategy: {e}")


@app.patch("/api/config/strategies/{strategy_id}/indicator/{indicator_name}")
async def update_strategy_indicator(strategy_id: int, indicator_name: str, req: UpdateIndicatorRequest):
    """Update indicator parameters for a specific strategy."""
    try:
        update_indicator_params(strategy_id, indicator_name, req.params)
        return {"message": f"Updated {indicator_name} parameters for strategy {strategy_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update indicator parameters: {e}")


@app.patch("/api/config/strategies/{strategy_id}/weights")
async def update_strategy_weights(strategy_id: int, req: UpdateWeightsRequest):
    """Update weights for a specific strategy."""
    try:
        update_strategy_weights(strategy_id, req.weights)
        return {"message": f"Updated weights for strategy {strategy_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update weights: {e}")


@app.patch("/api/config/strategies/{strategy_id}/schedule")
async def update_strategy_schedule(strategy_id: int, req: UpdateScheduleRequest):
    """Update run interval schedule for a specific strategy."""
    try:
        update_strategy_schedule(strategy_id, req.run_interval_seconds)
        return {"message": f"Updated schedule for strategy {strategy_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update schedule: {e}")


@app.patch("/api/config/strategies/{strategy_id}/threshold")
async def update_strategy_threshold(strategy_id: int, req: UpdateThresholdRequest):
    """Update signal threshold for a specific strategy."""
    try:
        update_strategy_threshold(strategy_id, req.signal_threshold)
        return {"message": f"Updated threshold for strategy {strategy_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update threshold: {e}")


@app.post("/api/config/strategies/{strategy_id}/activate")
async def activate_strategy(strategy_id: int):
    """Activate a specific strategy (deactivates all others)."""
    try:
        set_active_strategy(strategy_id)
        return {"message": f"Activated strategy {strategy_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate strategy: {e}")