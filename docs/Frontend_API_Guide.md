# Frontend API Guide

This guide explains how your frontend can interact with the backend for:
- Live prices streaming
- Latest signal for a specific symbol (with indicator values and contributions)
- Live technical indicator values (computed from latest Yahoo data)
- Last 5 signal calls for a specific symbol
- Manually triggering signal computation and receiving the result, while database saving respects the threshold

## Canonical Symbols

Use the following canonical symbols. Any alias is normalized to these; anything else returns 400.

Allowed: `XAUUSD`, `USDCAD`, `USDJPY`, `GBPUSD`, `AUDUSD`, `AUS200`, `UK100`, `DJ30`, `SPX`, `NAS100`, `GER40`, `FRA40`.

## Live Prices (WebSocket)

- Endpoint: `GET ws://<host>/ws/prices?symbol=XAUUSD`
- Messages:
  - `{"type":"tick","symbol":"XAUUSD","last":<number>,"bid":<number>,"ask":<number>,"ts":"<iso>"}`
  - `{"type":"heartbeat","ts":"<iso>"}` every ~30s
  - On error (disallowed symbol): `{"type":"error","error":"symbol_not_allowed","allowed":[...]} `

Example (browser):

```js
const ws = new WebSocket("ws://localhost:8000/ws/prices?symbol=XAUUSD");
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "tick") {
    // update UI: msg.last, msg.bid, msg.ask, msg.ts
  }
};
ws.onclose = () => console.log("price stream closed");
```

## Latest Signal (by symbol)

- Endpoint: `GET /signals/latest?symbol=XAUUSD`
- Returns the most recent DB signal for the symbol, including indicator values and contributions.

Response (example):

```json
{
  "symbol": "XAUUSD",
  "signal": {
    "timestamp": "2025-01-01T12:34:56Z",
    "symbol": "XAUUSD",
    "timeframe": "15m",
    "side": "buy",
    "signal_type": "BUY",
    "strength": 62.5,
    "final_signal_strength": 72.5,
    "strategy": "combined",
    "indicators": {"RSI": {"rsi": 29.3}, "MACD": {"macd": 0.12, "signal": 0.08}, ...},
    "indicator_contributions": {"RSI": 10, "MACD": 10, "SMA_EMA": 20, "BBANDS": 12, "STOCH": 10, "ATR": 10},
    "primary_timeframe": "15m",
    "confirmation_timeframe": "1h",
    "trend_timeframe": "4h",
    "h1_trend_direction": "Bullish",
    "h4_trend_direction": "Bullish",
    "alignment_boost": 10,
    "entry_price": 2365.2,
    "stop_loss_price": 2357.0,
    "take_profit_price": 2380.0,
    "risk_reward_ratio": 1.5,
    "volatility_state": "Normal",
    "is_valid": true
  }
}
```

## Live Technical Indicators

- Endpoint: `GET /indicators/live?symbol=XAUUSD&timeframe=15m&count=200`
- Computes indicators on the latest Yahoo OHLCV for the timeframe; returns last values and evaluation directions.

Response (example):

```json
{
  "symbol": "XAUUSD",
  "timeframe": "15m",
  "timestamp": "2025-01-01T12:45:00Z",
  "indicators": {
    "rsi": 31.2,
    "macd": 0.10,
    "macd_signal": 0.08,
    "sma_short": 2362.8,
    "sma_long": 2358.1,
    "ema_short": 2363.1,
    "ema_long": 2359.0,
    "bb_low": 2355.0,
    "bb_mid": 2360.0,
    "bb_high": 2365.0,
    "stoch_k": 22.4,
    "stoch_d": 28.7,
    "atr": 12.1
  },
  "evaluation": {
    "RSI": "buy",
    "MACD": "buy",
    "SMA": "buy",
    "EMA": "buy",
    "BBANDS": "none",
    "STOCH": "none",
    "ATR": "buy"
  }
}
```

## Last 5 Signals (by symbol)

- Endpoint: `GET /signals/recent?symbol=XAUUSD&limit=5`
- Optional: `min_strength=<number>` filters by `final_signal_strength`.

Response (example):

```json
{
  "count": 5,
  "signals": [
    {"timestamp": "...", "symbol": "XAUUSD", "signal_type": "BUY", "final_signal_strength": 68.2, ...},
    {"timestamp": "...", "symbol": "XAUUSD", "signal_type": "SELL", "final_signal_strength": 55.1, ...}
  ]
}
```

## Manual Signal Compute

- Endpoint: `POST /api/signals/compute?symbols=XAUUSD` (comma-separated allowed)
- Behavior:
  - Always returns a summary per symbol, even if strength is low.
  - Saves into DB only when `final_signal_strength >= signal_threshold` (threshold is strategy-configured; default ≥ 50%).

Response (example):

```json
{
  "count": 1,
  "results": [
    {
      "symbol": "XAUUSD",
      "timeframe": "15m",
      "indicator_valid": "6/7",
      "indicator_valid_pct": 85.7,
      "dirs": {"buy": 4, "sell": 1, "none": 2},
      "base_strength": 62.5,
      "final_strength": 72.5,
      "had_signal": true,
      "insert_status": "ok", // or "skipped" when below threshold
      "snapshot_status": "ok",
      "signal_frequency": {"signals": 10, "attempts": 40, "percent": 25.0},
      "fetch": [
        {"tf": "15m", "status": "ok", "len": 500, "last": "2025-01-01 12:45:00+00:00"},
        {"tf": "1h", "status": "ok", "len": 500, "last": "2025-01-01 12:00:00+00:00"},
        {"tf": "4h", "status": "ok", "len": 500, "last": "2025-01-01 12:00:00+00:00"}
      ]
    }
  ]
}
```

## Notes

- Canonical symbols are enforced across all endpoints; send one of the allowed list.
- The latest signal endpoint returns the row as saved in `public.signals` including `indicators` and `indicator_contributions`.
- The manual compute endpoint responds even when strength is low; database inserts respect the configured `signal_threshold`.