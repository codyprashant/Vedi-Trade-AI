import asyncio
import os
from datetime import datetime
from typing import Dict, Set, Optional, Any
import pandas as pd

import MetaTrader5 as mt5
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi import status

from .signal_engine import SignalEngine
from .config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, DEFAULT_HISTORY_COUNT
from .env_loader import load_dotenv_file
from .db import (
    ensure_signals_table,
    ensure_backtesting_tables,
    ensure_strategy_tables,
    create_default_gold_strategy_if_missing,
    fetch_recent_signals,
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
from .backtesting import run_manual_backtest, execute_manual_run
from dotenv import load_dotenv


app = FastAPI(
    title="PriceTracker API",
    description="Stream live ticks via WebSocket and fetch OHLCV history via REST.",
    version="0.1.0",
)

# Tracks subscribers per symbol
subscriptions: Dict[str, Set[WebSocket]] = {}

# Background polling task
poll_task: Optional[asyncio.Task] = None

# Inactivity shutdown watcher
inactivity_task: Optional[asyncio.Task] = None

# MT5 init state
mt5_initialized: bool = False

# Signal engine task
signal_task: Optional[asyncio.Task] = None
signal_engine_instance: Optional[SignalEngine] = None


def init_mt5() -> bool:
    """Initialize MetaTrader 5 terminal (optionally via env vars)."""
    global mt5_initialized
    mt5_path = os.environ.get("MT5_PATH")
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")

    initialized = mt5.initialize(mt5_path) if mt5_path else mt5.initialize()
    if not initialized:
        print(f"MT5 initialize failed: {mt5.last_error()}")
        mt5_initialized = False
        return False

    if login and password and server:
        if not mt5.login(int(login), password=password, server=server):
            print(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            mt5_initialized = False
            return False

    mt5_initialized = True
    return True


# -- Timeframe parsing --
TF_MAP = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
    "1w": mt5.TIMEFRAME_W1,
    "1mo": mt5.TIMEFRAME_MN1,
    "1mon": mt5.TIMEFRAME_MN1,
    "1month": mt5.TIMEFRAME_MN1,
    "5min": mt5.TIMEFRAME_M5,
    "15min": mt5.TIMEFRAME_M15,
    "1hour": mt5.TIMEFRAME_H1,
    "4hour": mt5.TIMEFRAME_H4,
    "1day": mt5.TIMEFRAME_D1,
}


def parse_timeframe(tf_str: str):
    key = tf_str.strip().lower()
    return TF_MAP.get(key)


def resolve_symbol(symbol: str) -> str:
    """Return a broker-specific symbol variant if the exact symbol is missing."""
    info = mt5.symbol_info(symbol)
    if info is not None:
        return symbol
    candidates = [s.name for s in mt5.symbols_get() if symbol.upper() in s.name.upper()]
    return candidates[0] if candidates else symbol


async def inactivity_watch():
    """Shutdown MT5 if no subscribers for 60 seconds."""
    global inactivity_task, mt5_initialized
    try:
        await asyncio.sleep(60)
        if total_subscribers() == 0 and mt5_initialized:
            print("No subscribers for 60s. Shutting down MT5.")
            mt5.shutdown()
            mt5_initialized = False
    finally:
        inactivity_task = None


def total_subscribers() -> int:
    return sum(len(s) for s in subscriptions.values())


def ensure_inactivity_watch():
    global inactivity_task
    if total_subscribers() == 0 and inactivity_task is None:
        inactivity_task = asyncio.create_task(inactivity_watch())


async def poll_loop():
    """Continuously poll MT5 ticks for subscribed symbols and broadcast."""
    global poll_task
    try:
        while True:
            # If no subscribers, start inactivity watch and idle
            if total_subscribers() == 0:
                ensure_inactivity_watch()
                await asyncio.sleep(1)
                continue

            # Ensure MT5 is initialized
            if not mt5_initialized:
                if not init_mt5():
                    # Wait a bit before retrying init
                    await asyncio.sleep(5)
                    continue

            # Poll each symbol that has subscribers
            for symbol, sockets in list(subscriptions.items()):
                if not sockets:
                    continue

                # Ensure symbol selected
                if mt5.symbol_info(symbol) is None:
                    # Attempt discovery: pick first symbol that contains symbol name tokens
                    candidates = [s.name for s in mt5.symbols_get() if symbol.upper() in s.name.upper()]
                    if candidates:
                        symbol_selected = candidates[0]
                        if symbol_selected != symbol:
                            subscriptions.setdefault(symbol_selected, set()).update(sockets)
                            subscriptions[symbol].clear()
                            print(f"Using discovered symbol: {symbol_selected}")
                            symbol = symbol_selected
                    else:
                        # No matching symbol; skip broadcasting for this symbol
                        continue

                mt5.symbol_select(symbol, True)
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    continue

                payload = {
                    "symbol": symbol,
                    "time": datetime.fromtimestamp(tick.time).isoformat(),
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "last": getattr(tick, "last", None),
                }

                # Broadcast to all subscribers; remove those that fail
                dead: Set[WebSocket] = set()
                for ws in sockets:
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        dead.add(ws)

                for ws in dead:
                    sockets.discard(ws)

            await asyncio.sleep(1)
    finally:
        poll_task = None


def ensure_polling():
    global poll_task
    if poll_task is None:
        poll_task = asyncio.create_task(poll_loop())


@app.get("/health")
async def health():
    return {
        "mt5_initialized": mt5_initialized,
        "subscribers": total_subscribers(),
        "symbols": {k: len(v) for k, v in subscriptions.items()},
    }


@app.get("/history")
async def history(symbol: str = "XAUUSD", timeframe: str = "15m", count: int = 500):
    """Return OHLCV candles for the given symbol and timeframe.

    - symbol: instrument name (e.g., XAUUSD). Will attempt discovery if exact match is missing.
    - timeframe: one of [1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo].
    - count: number of candles from most recent going backward.
    """
    tf = parse_timeframe(timeframe)
    if tf is None:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {timeframe}")

    # Ensure MT5 initialized
    if not mt5_initialized:
        if not init_mt5():
            raise HTTPException(status_code=500, detail=f"MT5 init failed: {mt5.last_error()}")

    # Resolve and select symbol
    sym = resolve_symbol(symbol)
    if not mt5.symbol_select(sym, True):
        raise HTTPException(status_code=404, detail=f"Symbol not available: {symbol}")

    rates = mt5.copy_rates_from_pos(sym, tf, 0, count)
    if rates is None or len(rates) == 0:
        raise HTTPException(status_code=404, detail=f"No data for {sym} {timeframe}")

    # Convert to JSON serializable list
    candles = []
    for r in rates:
        candles.append(
            {
                "t": datetime.fromtimestamp(int(r["time"]))
                .isoformat(),
                "o": float(r["open"]),
                "h": float(r["high"]),
                "l": float(r["low"]),
                "c": float(r["close"]),
                "v": int(r["real_volume"]) if r["real_volume"] else int(r["tick_volume"]),
            }
        )

    return {"symbol": sym, "timeframe": timeframe, "count": len(candles), "candles": candles}


@app.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket, symbol: str = "XAUUSD"):
    await websocket.accept()

    # Register subscriber
    sockets = subscriptions.setdefault(symbol, set())
    sockets.add(websocket)

    # Cancel inactivity shutdown if any
    global inactivity_task
    if inactivity_task is not None:
        inactivity_task.cancel()
        inactivity_task = None

    # Ensure background polling is running
    ensure_polling()

    try:
        # Keep the connection open; optionally receive pings/commands
        while True:
            # We don't require clients to send anything; this will block
            # until client disconnects. Use timeout to allow periodic checks.
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection healthy
                try:
                    await websocket.send_json({"type": "heartbeat", "ts": datetime.utcnow().isoformat()})
                except Exception:
                    # If send fails, let the disconnect handler clean up
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        # Unregister subscriber
        sockets.discard(websocket)
        if len(sockets) == 0 and symbol in subscriptions and subscriptions[symbol] == set():
            # Clean empty symbol bucket
            subscriptions.pop(symbol, None)

        # If no subscribers at all, schedule inactivity shutdown
        ensure_inactivity_watch()


# --- Signal Engine Integration ---
def fetch_history_df(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    tf = parse_timeframe(timeframe)
    if tf is None:
        raise RuntimeError(f"Unsupported timeframe: {timeframe}")

    # Ensure MT5 initialized
    if not mt5_initialized:
        if not init_mt5():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    # Resolve and select symbol
    sym = resolve_symbol(symbol)
    if not mt5.symbol_select(sym, True):
        raise RuntimeError(f"Symbol not available: {symbol}")

    rates = mt5.copy_rates_from_pos(sym, tf, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {sym} {timeframe}")

    # Build DataFrame
    df = pd.DataFrame(rates)
    df = df.rename(columns={"real_volume": "volume"})
    if "volume" not in df.columns or df["volume"].isna().all():
        df["volume"] = df.get("tick_volume", 0)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "volume"]]
    return df


def fetch_range_df(symbol: str, timeframe: str, start_ts, end_ts) -> pd.DataFrame:
    """Fetch OHLCV candles in a date range [start_ts, end_ts].

    start_ts/end_ts may be strings or pandas/py datetime-like; converted to Python datetime.
    """
    tf = parse_timeframe(timeframe)
    if tf is None:
        raise RuntimeError(f"Unsupported timeframe: {timeframe}")

    # Ensure MT5 initialized
    if not mt5_initialized:
        if not init_mt5():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    # Resolve and select symbol
    sym = resolve_symbol(symbol)
    if not mt5.symbol_select(sym, True):
        raise RuntimeError(f"Symbol not available: {symbol}")

    # Normalize timestamps
    start_dt = pd.to_datetime(start_ts).to_pydatetime()
    end_dt = pd.to_datetime(end_ts).to_pydatetime()

    rates = mt5.copy_rates_range(sym, tf, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        # Fallback: try copy_rates_from with approximate count
        approx_count = 2000
        rates = mt5.copy_rates_from_pos(sym, tf, 0, approx_count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No data for {sym} {timeframe} in range")

    df = pd.DataFrame(rates)
    df = df.rename(columns={"real_volume": "volume"})
    if "volume" not in df.columns or df["volume"].isna().all():
        df["volume"] = df.get("tick_volume", 0)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[(df["time"] >= pd.to_datetime(start_ts)) & (df["time"] <= pd.to_datetime(end_ts))]
    df = df[["time", "open", "high", "low", "close", "volume"]]
    return df


@app.on_event("startup")
async def init_env_and_start_signal_engine():
    # Load .env before anything else (both custom and python-dotenv)
    load_dotenv_file()
    load_dotenv()

    # Ensure signals table exists
    try:
        ensure_signals_table()
        print("Postgres OK: ensured 'public.signals' table exists.")
    except Exception as e:
        print(f"Postgres init failed: {e}")

    # Ensure backtesting tables exist
    try:
        ensure_backtesting_tables()
        print("Postgres OK: ensured backtesting tables exist.")
    except Exception as e:
        print(f"Backtesting tables init failed: {e}")

    # Ensure strategy config tables exist and seed default strategy
    try:
        ensure_strategy_tables()
        create_default_gold_strategy_if_missing()
        print("Postgres OK: ensured strategy tables and default 'Gold Strategy'.")
    except Exception as e:
        print(f"Strategy tables init failed: {e}")

    # Start signal engine
    global signal_task, signal_engine_instance
    signal_engine_instance = SignalEngine(fetch_history_df)
    signal_task = asyncio.create_task(signal_engine_instance.run())


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


@app.get("/signals/recent")
async def recent_signals(limit: int = 20):
    """Return recent high-confidence signals from Postgres."""
    try:
        rows = fetch_recent_signals(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Postgres query failed: {e}")
    return {"count": len(rows), "signals": rows}


# --- Manual Backtesting API ---

@app.post("/api/backtest/manual/generate")
async def backtest_generate(payload: Dict[str, Any]):
    required = ["start_date", "end_date"]
    for k in required:
        if k not in payload:
            raise HTTPException(status_code=400, detail=f"Missing field: {k}")
    start_date = payload["start_date"]
    end_date = payload["end_date"]
    symbol = payload.get("symbol", DEFAULT_SYMBOL)
    timeframe = payload.get("timeframe", DEFAULT_TIMEFRAME)
    initial_balance = payload.get("initial_balance")
    commission_per_trade = payload.get("commission_per_trade")
    spread_adjustment = payload.get("spread_adjustment")

    try:
        result = await asyncio.to_thread(
            run_manual_backtest,
            fetch_range_df,
            start_date,
            end_date,
            symbol,
            timeframe,
            initial_balance,
            commission_per_trade,
            spread_adjustment,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest generation failed: {e}")


@app.get("/api/backtest/manual/runs")
async def backtest_runs(from_date: Optional[str] = None, to_date: Optional[str] = None, min_signal_strength: Optional[float] = None, symbol: Optional[str] = None):
    try:
        rows = fetch_backtesting_runs(from_date, to_date, min_signal_strength, symbol)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch runs failed: {e}")


@app.get("/api/backtest/manual/signals/{manual_run_id}")
async def backtest_signals(manual_run_id: str):
    try:
        signals = fetch_backtesting_signals_by_run(manual_run_id)
        return {"manual_run_id": manual_run_id, "signals": signals}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch signals failed: {e}")


@app.post("/api/backtest/manual/execute")
async def backtest_execute(payload: Dict[str, Any]):
    required = ["manual_run_id", "initial_balance", "risk_per_trade_percent", "commission_per_trade", "slippage_percent"]
    for k in required:
        if k not in payload:
            raise HTTPException(status_code=400, detail=f"Missing field: {k}")
    try:
        result = await asyncio.to_thread(
            execute_manual_run,
            fetch_range_df,
            payload["manual_run_id"],
            float(payload["initial_balance"]),
            float(payload["risk_per_trade_percent"]),
            float(payload["commission_per_trade"]),
            float(payload["slippage_percent"]),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest execution failed: {e}")


# --- Strategy Configuration API ---

@app.get("/api/config/strategies")
async def list_strategies():
    try:
        return {"strategies": get_strategies()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List strategies failed: {e}")


@app.get("/api/config/strategies/{strategy_id}")
async def get_strategy(strategy_id: int):
    try:
        return get_strategy_details(strategy_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Strategy not found or fetch failed: {e}")


@app.patch("/api/config/strategies/{strategy_id}/indicator/{indicator_name}")
async def update_indicator(strategy_id: int, indicator_name: str, payload: Dict[str, Any]):
    if "params" not in payload or not isinstance(payload["params"], dict):
        raise HTTPException(status_code=400, detail="Body must include 'params' as an object")
    try:
        update_indicator_params(strategy_id, indicator_name, payload["params"]) 
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update indicator failed: {e}")


@app.patch("/api/config/strategies/{strategy_id}/weights")
async def update_weights(strategy_id: int, payload: Dict[str, Any]):
    if "weights" not in payload or not isinstance(payload["weights"], dict):
        raise HTTPException(status_code=400, detail="Body must include 'weights' as an object")
    try:
        update_strategy_weights(strategy_id, payload["weights"]) 
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update weights failed: {e}")


@app.patch("/api/config/strategies/{strategy_id}/schedule")
async def update_schedule(strategy_id: int, payload: Dict[str, Any]):
    if "run_interval_seconds" not in payload:
        raise HTTPException(status_code=400, detail="Body must include 'run_interval_seconds'")
    try:
        update_strategy_schedule(strategy_id, int(payload["run_interval_seconds"]))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update schedule failed: {e}")


@app.patch("/api/config/strategies/{strategy_id}/threshold")
async def update_threshold(strategy_id: int, payload: Dict[str, Any]):
    if "signal_threshold" not in payload:
        raise HTTPException(status_code=400, detail="Body must include 'signal_threshold'")
    try:
        update_strategy_threshold(strategy_id, float(payload["signal_threshold"]))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update threshold failed: {e}")


@app.post("/api/config/strategies/{strategy_id}/activate")
async def activate_strategy(strategy_id: int):
    try:
        set_active_strategy(strategy_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Activate strategy failed: {e}")