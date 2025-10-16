"""Async utilities for batching Yahoo Finance fetches."""

from __future__ import annotations

import asyncio
from typing import Dict, Iterable, Tuple, TYPE_CHECKING

import pandas as pd

from app.utils_time import as_utc_index

if TYPE_CHECKING:  # pragma: no cover
    import aiohttp


_TIMEFRAME_ALIASES = {
    "m15": "15m",
    "15min": "15m",
    "15m": "15m",
    "1h": "1h",
    "60m": "1h",
    "h1": "1h",
    "4h": "4h",
    "h4": "4h",
    "240m": "4h",
    "1d": "1d",
    "d1": "1d",
}


def _interval_for_timeframe(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    return _TIMEFRAME_ALIASES.get(tf, tf)


def _estimate_range(count: int, interval: str) -> str:
    interval = interval.lower()
    per_day = {
        "1m": 1440,
        "2m": 720,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "60m": 24,
        "1h": 24,
        "90m": 16,
        "4h": 6,
        "1d": 1,
    }.get(interval, 96)
    days = max(1, int(count / max(1, per_day)) + 1)
    if days <= 30:
        return f"{days}d"
    months = max(1, days // 30)
    if months <= 24:
        return f"{months}mo"
    years = max(1, months // 12)
    return f"{years}y"


def parse_yahoo_chart(payload: Dict[str, object]) -> pd.DataFrame:
    chart = payload.get("chart", {}) if isinstance(payload, dict) else {}
    results = chart.get("result") if isinstance(chart, dict) else None
    if not results:
        return pd.DataFrame()

    data = results[0] or {}
    timestamps = data.get("timestamp") or []
    if not timestamps:
        return pd.DataFrame()

    indicators = data.get("indicators", {})
    quote = (indicators.get("quote") or [{}])[0]

    frame = pd.DataFrame({
        "time": pd.to_datetime(timestamps, unit="s", utc=True),
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    })
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    if "volume" in frame:
        frame["volume"] = frame["volume"].fillna(0)
    return frame.set_index("time")


async def fetch_symbol_async(
    session: "aiohttp.ClientSession",
    symbol: str,
    timeframe: str = "15m",
    count: int = 500,
) -> Tuple[str, pd.DataFrame]:
    interval = _interval_for_timeframe(timeframe)
    lookback = _estimate_range(count, interval)
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}?interval={interval}&range={lookback}"
    )
    async with session.get(url) as response:
        response.raise_for_status()
        payload = await response.json()
    df = parse_yahoo_chart(payload)
    return symbol, as_utc_index(df)


async def fetch_batch(
    symbols: Iterable[str],
    timeframe: str = "15m",
    count: int = 500,
) -> Dict[str, pd.DataFrame]:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [fetch_symbol_async(session, sym, timeframe=timeframe, count=count) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    output: Dict[str, pd.DataFrame] = {}
    for result in results:
        if isinstance(result, Exception):
            continue
        symbol, df = result
        output[symbol] = df
    return output


def fetch_batch_sync(
    symbols: Iterable[str],
    timeframe: str = "15m",
    count: int = 500,
) -> Dict[str, pd.DataFrame]:
    try:
        return asyncio.run(fetch_batch(symbols, timeframe=timeframe, count=count))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(fetch_batch(symbols, timeframe=timeframe, count=count))


def fetch_history(symbol: str, timeframe: str = "15m", count: int = 500) -> pd.DataFrame:
    batch = fetch_batch_sync([symbol], timeframe=timeframe, count=count)
    return batch.get(symbol, pd.DataFrame())
