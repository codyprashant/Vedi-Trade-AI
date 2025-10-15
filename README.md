# PriceTracker (Yahoo Backend)

FastAPI backend powered by Yahoo Finance for OHLCV history, live price streaming via WebSocket, and a multi-symbol signal engine with indicator snapshots. React + Vite frontend in `frontend/`.

## Features
- Yahoo-based history fetch with 4h resampling fallback.
- WebSocket live prices broadcast (120s polling throttle).
- Signal engine monitors multiple symbols and saves per-symbol snapshots.
- Endpoints for recent signals and latest indicator snapshots.
- OpenAPI (Swagger) available at runtime and exportable to `docs/openapi.json`.

## Tech Stack
- Python 3.10+
- FastAPI + Uvicorn
- Pandas, pandas-ta
- yfinance
- PostgreSQL via `psycopg2-binary`

## Environment
Create a `.env` at project root with:

```
POSTGRES_HOST=<host>
POSTGRES_PORT=<port>
POSTGRES_DB=<db>
POSTGRES_USER=<user>
POSTGRES_PASSWORD=<password>
# Optional CORS override (comma-separated or "*")
CORS_ALLOW_ORIGINS=http://localhost:3000
```

## Install
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Backend
- Windows (PowerShell):
  - `./scripts/start_yahoo_backend_windows.ps1 -Port 8001 -Reload`
- Linux (bash):
  - `PORT=8001 bash ./scripts/start_backend_linux.sh`

Base URL: `http://localhost:8001`

## Core Endpoints

### System & Data
- `GET /health` — server status and signal engine status.
- `GET /history?symbol=SPX&timeframe=4h&count=60` — historical candles.
- `WS /ws/prices?symbol=XAUUSD` — subscribe to live quotes.

### Signals & Analysis
- `GET /signals/recent?limit=20&min_strength=50` — recent signals from Postgres.
- `GET /signals/latest?symbol=XAUUSD` — latest signal for specific symbol.
- `POST /api/signals/compute?symbols=XAUUSD` — manually trigger signal computation.

### Indicators
- `GET /indicators/latest?symbols=XAUUSD,SPX` — latest indicator snapshots per symbol.
- `GET /indicators/live?symbol=XAUUSD&timeframe=15m` — live technical indicators.

### Strategy Configuration
- `GET /api/config/strategies` — list all trading strategies.
- `GET /api/config/strategies/{id}` — get strategy details.
- `PATCH /api/config/strategies/{id}/indicator/{name}` — update indicator parameters.
- `PATCH /api/config/strategies/{id}/weights` — update strategy indicator weights.
- `PATCH /api/config/strategies/{id}/schedule` — update run frequency.
- `PATCH /api/config/strategies/{id}/threshold` — update signal threshold.
- `POST /api/config/strategies/{id}/activate` — activate a trading strategy.

### Backtesting
- `POST /api/backtest/run` — run unified backtest with strategy.
- `GET /api/backtest/roi` — calculate ROI for backtest results.
- `GET /api/backtest/{backtest_id}/results` — get detailed backtest results.
- `GET /api/backtest/list` — list all backtests.

### Documentation
- `GET /openapi.json` — live OpenAPI spec.
- `GET /docs` — Swagger UI.

## Additional Documentation

### Signal Processing
- **[Weighted Voting System](docs/Weighted_Voting_System.md)** — Comprehensive guide to the weighted voting decision aggregation system for technical indicators

## Export Swagger for Frontend
Generate a static OpenAPI JSON file to `docs/openapi.json`:

```
python scripts/export_openapi.py
```

Frontend can load either `http://localhost:8001/openapi.json` or the static `docs/openapi.json`.

## Frontend
Dev server (Windows):

```
./scripts/start_frontend_windows.ps1
```

Or manually in `frontend/`:

```
cd frontend
npm install
npm run dev
```

Vite dev server runs on `http://localhost:3000`.

## Yahoo Finance fast_info Response Structure

The WebSocket live price data is sourced from Yahoo Finance's `fast_info` object. Below is the complete response structure for different asset types:

### XAUUSD (Gold Futures - GC=F)

The `fast_info` object for XAUUSD provides 20 data fields:

```json
{
  "currency": "USD",                          // Base currency (string)
  "dayHigh": 4190.89990234375,               // Today's high price (float)
  "dayLow": 4105.0,                          // Today's low price (float)
  "exchange": "CMX",                         // Exchange code (string)
  "fiftyDayAverage": 3607.9079931640626,     // 50-day moving average (float)
  "lastPrice": 4159.60009765625,             // Most recent price (float) - Used as 'bid' in WebSocket
  "lastVolume": 396022,                      // Last trade volume (int)
  "marketCap": null,                         // Market capitalization (null for futures)
  "open": 4130.89990234375,                  // Today's opening price (float)
  "previousClose": 4175.2998046875,          // Previous day's close (float)
  "quoteType": "FUTURE",                     // Asset type (string)
  "regularMarketPreviousClose": 4108.60009765625, // Regular market previous close (float)
  "shares": null,                            // Outstanding shares (null for futures)
  "tenDayAverageVolume": 2007,               // 10-day average volume (int)
  "threeMonthAverageVolume": 3591,           // 3-month average volume (int)
  "timezone": "America/New_York",            // Market timezone (string)
  "twoHundredDayAverage": 3240.0207050978534, // 200-day moving average (float)
  "yearChange": 0.5437740468984122,          // Year-to-date change percentage (float)
  "yearHigh": 4111.0,                        // 52-week high (float)
  "yearLow": 2554.199951171875               // 52-week low (float)
}
```

## WebSocket Live Price Streaming

### Current WebSocket Response Format

Our WebSocket endpoint (`/ws/prices?symbol=XAUUSD`) now provides enhanced real-time data:

```json
{
  "symbol": "XAUUSD",
  "time": "2025-10-14T22:13:00.915806+00:00",
  "bid": 4160.90,
  "previousClose": 4175.30,
  "indicators": {
    "rsi_9": 52.38,
    "rsi_14": 53.63,
    "macd": 2.36,
    "macd_signal": -0.86,
    "sma_20": 4161.29,
    "sma_50": 4152.56,
    "sma_200": 4113.44,
    "ema_9": 4160.80,
    "ema_21": 4159.25,
    "bb_low": 4153.57,
    "bb_mid": 4161.29,
    "bb_high": 4169.01,
    "stoch_k": 51.50,
    "stoch_d": 52.15,
    "atr": 7.66
  },
  "evaluation": {
    "RSI": "neutral",
    "MACD": "neutral",
    "SMA": "neutral",
    "EMA": "neutral",
    "BBANDS": "neutral",
    "STOCH": "neutral",
    "ATR": "neutral"
  },
  "latestSignal": {
    "timestamp": "2025-01-01T12:34:56Z",
    "signal_type": "BUY",
    "final_signal_strength": 72.5,
    "entry_price": 2365.20,
    "stop_loss_price": 2357.00,
    "take_profit_price": 2380.00
  },
  "signalHistory": [
    // Array of last 10 signals with complete specifications
  ]
}
```

### WebSocket Response Mapping

Our WebSocket response maps the following fields from `fast_info`:

| WebSocket Field | fast_info Field | Description |
|----------------|-----------------|-------------|
| `bid` | `lastPrice` | Current market price (limited to 2 decimal places) |
| `previousClose` | `previousClose` | Previous day's closing price |
| `indicators` | *Computed* | Live technical indicators (21 indicators, 2 decimal precision) |
| `evaluation` | *Computed* | Indicator evaluations ("neutral" instead of "none") |
| `latestSignal` | *Database* | Most recent trading signal for the symbol |
| `signalHistory` | *Database* | Last 10 signals with full specifications |

### Recent Improvements

1. **Enhanced Evaluations**: Changed "none" to "neutral" for better user experience
2. **Precision Control**: All indicator values limited to 2 decimal places
3. **Streamlined Response**: Removed `marketState` and `regularMarketPrice` (not available for futures)
4. **Signal Integration**: Added latest signal and signal history directly in WebSocket response
5. **Complete Technical Analysis**: 21 technical indicators with real-time evaluations

### Important Notes

1. **Removed Fields**: `marketState` and `regularMarketPrice` have been removed as they are not provided by Yahoo Finance for futures symbols like XAUUSD (GC=F).

2. **Asset Type Differences**: Different asset types (stocks, forex, indices, futures) may have different available fields in their `fast_info` response.

3. **Data Types**: 
   - Prices are returned as `float` values with 2 decimal precision
   - Volumes are returned as `int` values  
   - Evaluations use "neutral" instead of "none" for clarity
   - Strings are used for metadata (currency, exchange, timezone, etc.)

4. **Symbol Resolution**: 
   - `XAUUSD` → `GC=F` (Gold futures)
   - `USDCAD` → `USDCAD=X` (Forex pair)
   - `SPX` → `^GSPC` (S&P 500 index)

### Testing fast_info Response

To test the complete `fast_info` response for any symbol, run:

```bash
python test_fastinfo_complete.py
```

This generates a detailed JSON report (`fastinfo_test_results.json`) with the complete response structure for multiple asset types.

## Notes
- Polling throttle set to 120 seconds for Yahoo limits.
- Timeframe keys `4h` and `h4` map consistently to `4h` resampling.
- Indicator snapshots table is ensured at startup.

## License
Proprietary. All rights reserved unless explicitly licensed otherwise.