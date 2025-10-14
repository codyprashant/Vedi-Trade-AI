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
  - `{"symbol":"XAUUSD","time":"<iso>","bid":<number>,"previousClose":<number>,"marketState":"<string>","regularMarketPrice":<number>}`
  - `{"type":"heartbeat","ts":"<iso>"}` every ~30s
  - On error (disallowed symbol): `{"type":"error","error":"symbol_not_allowed","allowed":[...]} `

### WebSocket Response Fields

- `symbol`: The canonical symbol (e.g., "XAUUSD")
- `time`: ISO timestamp of the data
- `bid`: Current bid price (from Yahoo Finance lastPrice)
- `previousClose`: Previous day's closing price (may be null)
- `marketState`: Current market state (e.g., "REGULAR", "CLOSED", "PRE", "POST", may be null)
- `regularMarketPrice`: Regular market price (may be null)

Example (browser):

```js
const ws = new WebSocket("ws://localhost:8000/ws/prices?symbol=XAUUSD");
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.symbol && msg.bid !== undefined) {
    // update UI: msg.symbol, msg.bid, msg.time, msg.previousClose, msg.marketState, msg.regularMarketPrice
    console.log(`${msg.symbol}: Bid=${msg.bid}, PrevClose=${msg.previousClose}, Market=${msg.marketState}`);
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

## Strategy Configuration

The system supports dynamic strategy configuration through dedicated API endpoints. These allow you to manage multiple strategies, update their parameters, and control which strategy is active.

### List All Strategies

- Endpoint: `GET /api/config/strategies`
- Returns basic information about all available strategies.

Response (example):

```json
{
  "strategies": [
    {
      "id": 1,
      "name": "Gold Strategy",
      "is_active": true,
      "signal_threshold": 0.9,
      "run_interval_seconds": 60
    }
  ]
}
```

### Get Strategy Details

- Endpoint: `GET /api/config/strategies/{strategy_id}`
- Returns complete strategy configuration including indicators and weights.

Response (example):

```json
{
  "strategy": {
    "id": 1,
    "name": "Gold Strategy",
    "description": "Default gold strategy",
    "timeframes": ["M15", "H1", "H4"],
    "signal_threshold": 0.9,
    "run_interval_seconds": 60,
    "is_active": true
  },
  "indicator_params": {
    "RSI": { "period": 14, "overbought": 70, "oversold": 30 },
    "MACD": { "fast": 12, "slow": 26, "signal": 9 },
    "SMA": { "short": 20, "long": 50 },
    "EMA": { "short": 20, "long": 50 },
    "BBANDS": { "length": 20, "stddev": 2 },
    "STOCH": { "k": 14, "d": 3, "overbought": 80, "oversold": 20 },
    "ATR": { "length": 14 },
    "PRICE_ACTION": { "engulfing_lookback": 5, "pinbar_lookback": 5 }
  },
  "weights": {
    "RSI": 0.15,
    "MACD": 0.15,
    "STOCH": 0.1,
    "BBANDS": 0.1,
    "SMA_EMA": 0.2,
    "MTF": 0.15,
    "ATR_STABILITY": 0.075,
    "PRICE_ACTION": 0.075
  }
}
```

### Update Indicator Parameters

- Endpoint: `PATCH /api/config/strategies/{strategy_id}/indicator/{indicator_name}`
- Updates parameters for a specific indicator.

Request body (example for RSI):

```json
{
  "params": {
    "period": 14,
    "overbought": 75,
    "oversold": 25
  }
}
```

### Update Strategy Weights

- Endpoint: `PATCH /api/config/strategies/{strategy_id}/weights`
- Updates the contribution weights for different indicators.

Request body:

```json
{
  "weights": {
    "RSI": 0.2,
    "MACD": 0.2,
    "STOCH": 0.1,
    "BBANDS": 0.1,
    "SMA_EMA": 0.15,
    "MTF": 0.15,
    "ATR_STABILITY": 0.05,
    "PRICE_ACTION": 0.05
  }
}
```

### Update Run Schedule

- Endpoint: `PATCH /api/config/strategies/{strategy_id}/schedule`
- Updates how frequently the strategy engine runs.

Request body:

```json
{
  "run_interval_seconds": 120
}
```

### Update Signal Threshold

- Endpoint: `PATCH /api/config/strategies/{strategy_id}/threshold`
- Updates the minimum strength required for signal generation.

Request body:

```json
{
  "signal_threshold": 0.85
}
```

### Activate Strategy

- Endpoint: `POST /api/config/strategies/{strategy_id}/activate`
- Activates the specified strategy (deactivates all others).
- No request body required.

Response:

```json
{
  "status": "ok"
}
```

## Yahoo Finance fast_info Object Reference

The WebSocket price data is sourced from Yahoo Finance's `fast_info` object. Below are the available keys and their descriptions:

### Core Price Fields
- `lastPrice`: The most recent trading price (used as `bid` in our WebSocket response)
- `regularMarketPrice`: The regular market price during trading hours
- `previousClose`: Previous day's closing price
- `ask`: Ask price (often null for many symbols)
- `bid`: Bid price (often null for many symbols, different from our WebSocket `bid`)

### Market Information
- `marketState`: Current market state
  - `"REGULAR"`: Regular trading hours
  - `"CLOSED"`: Market is closed
  - `"PRE"`: Pre-market trading
  - `"POST"`: After-hours trading
- `currency`: The currency of the instrument (e.g., "USD")
- `exchange`: The exchange where the instrument is traded
- `quoteType`: Type of quote (e.g., "EQUITY", "CURRENCY", "FUTURE")

### Technical Analysis Fields
- `fiftyDayAverage`: 50-day moving average
- `twoHundredDayAverage`: 200-day moving average
- `yearHigh`: 52-week high price
- `yearLow`: 52-week low price

### Volume and Trading
- `regularMarketVolume`: Volume during regular trading hours
- `averageVolume`: Average trading volume
- `averageVolume10days`: 10-day average volume

### Market Cap and Valuation (for stocks)
- `marketCap`: Market capitalization
- `enterpriseValue`: Enterprise value
- `priceToBook`: Price-to-book ratio
- `forwardPE`: Forward price-to-earnings ratio
- `trailingPE`: Trailing price-to-earnings ratio

### Dividend Information (for dividend-paying stocks)
- `dividendYield`: Annual dividend yield
- `dividendRate`: Annual dividend rate
- `exDividendDate`: Ex-dividend date

### Notes
- Many fields may be `null` or unavailable depending on the symbol type and market conditions
- Our WebSocket implementation currently uses `lastPrice`, `previousClose`, `marketState`, and `regularMarketPrice`
- The `bid` field in our WebSocket response comes from `lastPrice`, not the Yahoo Finance `bid` field (which is often null)

## Notes

- Canonical symbols are enforced across all endpoints; send one of the allowed list.
- The latest signal endpoint returns the row as saved in `public.signals` including `indicators` and `indicator_contributions`.
- The manual compute endpoint responds even when strength is low; database inserts respect the configured `signal_threshold`.
- Strategy configuration changes take effect immediately for new signal computations.
- Only one strategy can be active at a time; activating a strategy automatically deactivates others.