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
from .utils_time import retry, last_closed
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
    ensure_new_backtesting_tables,
    ensure_strategy_tables,
    create_default_gold_strategy_if_missing,
    fetch_recent_signals,
    fetch_recent_signals_by_symbol,
    fetch_latest_signal_by_symbol,
    ensure_indicator_snapshots_table,
    fetch_latest_indicator_snapshots,
    fetch_backtest_summary,
    fetch_backtest_signals,
    fetch_all_backtests,
    get_strategies,
    get_strategy_details,
    update_indicator_params,
    update_strategy_weights,
    update_strategy_schedule,
    update_strategy_threshold,
    set_active_strategy,
    check_database_health,
    fetch_indicator_contributions,
)
from .db_trace import ensure_signal_trace_table, fetch_signal_traces
from .config import (
    ALLOWED_SYMBOLS,
    INDICATOR_PARAMS,
    WEIGHTS,
    SIGNAL_THRESHOLD,
    PRIMARY_TIMEFRAME,
    CONFIRMATION_TIMEFRAME,
    TREND_TIMEFRAME,
    DEBUG_WEBSOCKET,
    DEBUG_SIGNALS,
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


# --- Helper: map app timeframe to pandas frequency string for flooring ---
def _freq_for_timeframe(tf: str) -> str:
    tf = tf.strip().lower()
    if tf in ("15m", "15min", "m15"):
        return "15min"
    if tf in ("1h", "60m", "h1", "1hour"):
        return "1h"
    if tf in ("4h", "h4", "240m", "4hour"):
        return "4h"
    if tf in ("1d", "d1", "1day"):
        return "1d"
    # Default to minutes if unspecified
    return "15min"


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
                    if DEBUG_WEBSOCKET:
                        print(f"WebSocket Debug - {symbol}: fast_info available: {fi is not None}")
                    if fi:
                        if DEBUG_WEBSOCKET:
                            print(f"WebSocket Debug - {symbol}: fast_info keys: {list(fi.keys()) if hasattr(fi, 'keys') else 'not dict-like'}")
                        
                        # Use lastPrice as bid value
                        lastPrice_raw = fi.get("lastPrice", None)
                        bid = float(lastPrice_raw) if lastPrice_raw is not None else None
                        if DEBUG_WEBSOCKET:
                            print(f"WebSocket Debug - {symbol}: lastPrice_raw={lastPrice_raw}, bid={bid}")
                        
                        # Extract additional fields
                        previousClose_raw = fi.get("previousClose", None)
                        previous_close = float(previousClose_raw) if previousClose_raw is not None else None
                        if DEBUG_WEBSOCKET:
                            print(f"WebSocket Debug - {symbol}: previousClose_raw={previousClose_raw}, previous_close={previous_close}")
                        
                        market_state = fi.get("marketState", None)
                        if DEBUG_WEBSOCKET:
                            print(f"WebSocket Debug - {symbol}: market_state={market_state}")
                        
                        regularMarketPrice_raw = fi.get("regularMarketPrice", None)
                        regular_market_price = float(regularMarketPrice_raw) if regularMarketPrice_raw is not None else None
                        if DEBUG_WEBSOCKET:
                            print(f"WebSocket Debug - {symbol}: regularMarketPrice_raw={regularMarketPrice_raw}, regular_market_price={regular_market_price}")
                    else:
                        if DEBUG_WEBSOCKET:
                            print(f"WebSocket Debug - {symbol}: fast_info is None or not available")
                except Exception as e:
                    print(f"Error getting fast_info for {symbol}: {e}")
                    import traceback
                    traceback.print_exc()

                # Fallback to latest 1m candle close
                if bid is None:
                    try:
                        hist = retry(lambda: ticker.history(period="1d", interval="1m"))
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
                    count = 300  # Increased to 300 to ensure enough data for SMA 200 + buffer
                    
                    if DEBUG_WEBSOCKET:
                        print(f"WebSocket Debug - {symbol}: Starting indicator computation with timeframe={timeframe}, count={count}")
                    
                    # Fetch recent history and compute indicators
                    df = fetch_history_df(symbol, timeframe, max(100, count))
                    # Crop to last closed bar to avoid partial candle effects
                    try:
                        if df is not None and len(df) >= 1:
                            freq = _freq_for_timeframe(timeframe)
                            anchor = last_closed(pd.to_datetime(df.iloc[-1]["time"]), freq)
                            before_len = len(df)
                            df = df[df["time"] <= anchor]
                            after_len = len(df)
                            if DEBUG_WEBSOCKET:
                                last_ts = pd.to_datetime(df.iloc[-1]["time"]).isoformat() if after_len else "<empty>"
                                print(
                                    f"WebSocket Debug - {symbol}: Cropped partial bars ({before_len} -> {after_len}); anchor={anchor}, last_ts={last_ts}"
                                )
                    except Exception as crop_err:
                        if DEBUG_WEBSOCKET:
                            print(f"WebSocket Debug - {symbol}: Cropping error: {crop_err}")
                    if DEBUG_WEBSOCKET:
                        print(f"WebSocket Debug - {symbol}: Fetched df: {df is not None}, len={len(df) if df is not None else 0}")
                    
                    if df is not None and len(df) >= 3:
                        if DEBUG_WEBSOCKET:
                            print(f"WebSocket Debug - {symbol}: Computing indicators with INDICATOR_PARAMS keys: {list(INDICATOR_PARAMS.keys())}")
                        ind = compute_indicators(df, INDICATOR_PARAMS)
                        if DEBUG_WEBSOCKET:
                            print(f"WebSocket Debug - {symbol}: Computed indicators keys: {list(ind.keys())}")
                        
                        res = evaluate_signals(df, ind, INDICATOR_PARAMS)
                        if DEBUG_WEBSOCKET:
                            print(f"WebSocket Debug - {symbol}: Evaluated signals keys: {list(res.keys())}")
                        
                        # Extract last values for each indicator series (limit to 2 decimal places)
                        last_vals: dict[str, Any] = {}
                        for k, series in ind.items():
                            try:
                                last_val = float(series.iloc[-1])
                                # Limit to 2 decimal places
                                last_vals[k] = round(last_val, 2)
                                if DEBUG_WEBSOCKET:
                                    print(f"WebSocket Debug - {symbol}: {k} = {last_vals[k]}")
                            except Exception as ex:
                                last_vals[k] = None
                                if DEBUG_WEBSOCKET:
                                    print(f"WebSocket Debug - {symbol}: {k} = None (error: {ex})")
                        
                        # Convert 'none' to 'neutral' in evaluations
                        directions = {}
                        for name, r in res.items():
                            direction = r.direction
                            # Direction is already standardized to "neutral" in the codebase
                            directions[name] = direction
                        if DEBUG_WEBSOCKET:
                            print(f"WebSocket Debug - {symbol}: Directions: {directions}")
                        
                        indicators_data = last_vals
                        evaluation_data = directions
                    else:
                        if DEBUG_WEBSOCKET:
                            print(f"WebSocket Debug - {symbol}: Insufficient data for indicators (df={df is not None}, len={len(df) if df is not None else 0})")
                except Exception as e:
                    print(f"Error computing indicators for {symbol}: {e}")
                    import traceback
                    traceback.print_exc()

                # Fetch latest signal and signal history for this symbol
                latest_signal = None
                signal_history = []
                try:
                    latest_signal = fetch_latest_signal_by_symbol(symbol)
                    signal_history = fetch_recent_signals_by_symbol(symbol, 10)
                except Exception as e:
                    if DEBUG_WEBSOCKET:
                        print(f"WebSocket Debug - {symbol}: Error fetching signals: {e}")

                payload = {
                    "symbol": symbol,
                    "time": ts_iso,
                    "bid": bid,
                    "previousClose": previous_close,
                    "indicators": indicators_data,
                    "evaluation": evaluation_data,
                    "latestSignal": latest_signal,
                    "signalHistory": signal_history,
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
        df = retry(lambda: ticker.history(period=period, interval=interval))
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
    df = retry(lambda: ticker.history(period=period, interval=interval))
    # Generic fallbacks: resample from smaller intervals if native interval empty
    def try_resample(from_interval: str, rule: str, mult: int) -> Optional[pd.DataFrame]:
        try:
            alt_count = max(count * mult, 200)
            alt_period = _approx_period_for_count(from_interval, alt_count)
            df1 = retry(lambda: ticker.history(period=alt_period, interval=from_interval))
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
    df = retry(lambda: ticker.history(start=start_dt.to_pydatetime(), end=end_dt.to_pydatetime(), interval=interval))
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

# Legacy backtesting endpoints removed - use unified BacktestEngine via /api/backtest/* endpoints


# --- New Unified Backtesting Endpoints ---

@app.post("/api/backtest/run")
async def run_backtest(
    strategy_id: int,
    symbol: str,
    start_date: str,
    end_date: str,
    investment: float = 10000,
    timeframe: str = "15m"
):
    """Run a new backtest using the unified BacktestEngine."""
    try:
        from backtest.backtest_engine import BacktestEngine
        
        engine = BacktestEngine(
            strategy_id=strategy_id,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            investment=investment,
            timeframe=timeframe
        )
        
        summary = engine.run_backtest()
        return {"success": True, "backtest": summary}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest execution failed: {e}")


@app.get("/api/backtest/roi")
async def calculate_roi(backtest_id: int, amount: float):
    """Calculate ROI projection for a specific backtest and investment amount."""
    try:
        # Fetch backtest summary
        backtest = fetch_backtest_summary(backtest_id)
        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        # Calculate ROI
        total_return_pct = backtest['total_return_pct']
        roi_amount = amount * (1 + total_return_pct / 100)
        
        return {
            "backtest_id": backtest_id,
            "initial": amount,
            "final": roi_amount,
            "return_pct": total_return_pct,
            "profit": roi_amount - amount,
            "symbol": backtest['symbol'],
            "timeframe": backtest['timeframe'],
            "efficiency_pct": backtest['efficiency_pct']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ROI calculation failed: {e}")


@app.get("/api/backtest/{backtest_id}/results")
async def get_backtest_results(backtest_id: int):
    """Get detailed backtest results including summary and all signals."""
    try:
        # Fetch backtest summary
        backtest = fetch_backtest_summary(backtest_id)
        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        # Fetch all signals for this backtest
        signals = fetch_backtest_signals(backtest_id)
        
        # Calculate additional metrics
        win_signals = [s for s in signals if s['result'] == 'profit']
        loss_signals = [s for s in signals if s['result'] == 'loss']
        open_signals = [s for s in signals if s['result'] == 'open']
        
        # Format signals for response
        formatted_signals = []
        for signal in signals:
            formatted_signals.append({
                "time": signal['signal_time'].isoformat() if signal['signal_time'] else None,
                "direction": signal['direction'],
                "entry_price": signal['entry_price'],
                "exit_price": signal['exit_price'],
                "profit_pct": signal['profit_pct'],
                "result": signal['result'],
                "confidence": signal['confidence'],
                "reason": signal['reason']
            })
        
        return {
            "summary": {
                "backtest_id": backtest_id,
                "symbol": backtest['symbol'],
                "timeframe": backtest['timeframe'],
                "start_date": backtest['start_date'].isoformat() if backtest['start_date'] else None,
                "end_date": backtest['end_date'].isoformat() if backtest['end_date'] else None,
                "investment": backtest['investment'],
                "total_return_pct": backtest['total_return_pct'],
                "efficiency_pct": backtest['efficiency_pct'],
                "total_signals": len(signals),
                "win_count": len(win_signals),
                "loss_count": len(loss_signals),
                "open_count": len(open_signals),
                "created_at": backtest['created_at'].isoformat() if backtest['created_at'] else None
            },
            "signals": formatted_signals
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch backtest results: {e}")


@app.get("/api/backtest/list")
async def list_backtests(symbol: str = None, limit: int = 50):
    """List all backtests with optional filtering."""
    try:
        backtests = fetch_all_backtests(symbol=symbol, limit=limit)
        
        # Format response
        formatted_backtests = []
        for bt in backtests:
            formatted_backtests.append({
                "id": bt['id'],
                "symbol": bt['symbol'],
                "timeframe": bt['timeframe'],
                "start_date": bt['start_date'].isoformat() if bt['start_date'] else None,
                "end_date": bt['end_date'].isoformat() if bt['end_date'] else None,
                "investment": bt['investment'],
                "total_return_pct": bt['total_return_pct'],
                "efficiency_pct": bt['efficiency_pct'],
                "total_signals": bt.get('total_signals', 0),
                "win_count": bt.get('win_count', 0),
                "loss_count": bt.get('loss_count', 0),
                "open_count": bt.get('open_count', 0),
                "created_at": bt['created_at'].isoformat() if bt['created_at'] else None
            })
        
        return {
            "count": len(formatted_backtests),
            "backtests": formatted_backtests
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch backtests: {e}")


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
        ensure_new_backtesting_tables()
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
    try:
        ensure_signal_trace_table()
    except Exception as e:
        print(f"Postgres init failed (signal_traces): {e}")

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


@app.get("/api/trace/{signal_id}")
async def get_trace(signal_id: int):
    """Return stored trace log rows for a specific signal."""

    try:
        traces = fetch_signal_traces(signal_id)
        return {"signal_id": signal_id, "traces": traces}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch trace logs: {e}")


@app.get("/api/indicators/contributions/{signal_id}")
async def get_indicator_contributions(signal_id: int):
    """Return the indicator contribution payload stored with a signal."""

    try:
        contributions = fetch_indicator_contributions(signal_id)
        if contributions is None:
            raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")
        return {
            "signal_id": signal_id,
            "indicator_contributions": contributions,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch indicator contributions: {e}")


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


# --- Health Check Endpoint ---

@app.get("/health")
async def health_check():
    """Lightweight health check endpoint for monitoring."""
    import time
    
    start_time = time.time()
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": None,
        "database": None,
        "signal_engine": None,
        "websocket_subscribers": total_subscribers(),
        "response_time_ms": None
    }
    
    # Check database health
    try:
        db_health = check_database_health()
        health_data["database"] = db_health
        if db_health["status"] != "healthy":
            health_data["status"] = "degraded"
    except Exception as e:
        health_data["database"] = {"status": "unhealthy", "error": str(e)}
        health_data["status"] = "degraded"
    
    # Check signal engine status
    try:
        global signal_engine_instance, signal_task
        if signal_engine_instance and signal_task and not signal_task.done():
            health_data["signal_engine"] = {"status": "running"}
        else:
            health_data["signal_engine"] = {"status": "stopped"}
            if health_data["status"] == "healthy":
                health_data["status"] = "degraded"
    except Exception as e:
        health_data["signal_engine"] = {"status": "error", "error": str(e)}
        health_data["status"] = "degraded"
    
    # Calculate response time
    response_time = (time.time() - start_time) * 1000
    health_data["response_time_ms"] = round(response_time, 2)
    
    return health_data