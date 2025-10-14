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
- `POST /api/backtest/manual/generate` — generate historical signals.
- `GET /api/backtest/manual/runs` — list backtesting runs.
- `GET /api/backtest/manual/signals/{run_id}` — get signals for specific run.

### Documentation
- `GET /openapi.json` — live OpenAPI spec.
- `GET /docs` — Swagger UI.

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

## Notes
- Polling throttle set to 120 seconds for Yahoo limits.
- Timeframe keys `4h` and `h4` map consistently to `4h` resampling.
- Indicator snapshots table is ensured at startup.

## License
Proprietary. All rights reserved unless explicitly licensed otherwise.