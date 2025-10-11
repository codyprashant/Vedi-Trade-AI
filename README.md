# VediGoldAI

GoldSignalEngine v2.3 Adaptive with full Manual Backtesting module and REST API.

This repository provides:
- Real-time adaptive signal engine (MT5-based).
- Manual backtesting mode over historical data with parameterized runs.
- Supabase storage for signals and backtesting metadata.
- REST API endpoints for generating runs, listing runs, retrieving run signals, and executing trade simulations.

## Features
- Adaptive multi-timeframe signal generation with indicator contributions.
- Manual backtesting: generate signals on historical OHLCV with `manual_run_id` tracking.
- Batch insertion and efficient run metadata storage.
- Simulation engine that computes PnL, win rate, drawdown using M1 data.
- FastAPI server exposing endpoints to manage backtests and analytics.

## Tech Stack
- Python 3.10+
- FastAPI
- MetaTrader5 (MT5) via `MetaTrader5` Python module
- Pandas, NumPy
- Supabase (PostgREST)

## Quick Start

### 1) Environment
Create a `.env` file at project root with:

```
SUPABASE_URL=<your_supabase_url>
SUPABASE_SERVICE_ROLE_KEY=<service_role_or_anon_key>
MT5_LOGIN=<your_mt5_login>
MT5_PASSWORD=<your_mt5_password>
MT5_SERVER=<your_mt5_server>
MT5_PATH=<optional_path_to_terminal>
```

- `SUPABASE_*` is required for DB access to `signals`, `backtesting_signals`, and `backtesting_runs` tables.
- `MT5_*` values are required to initialize MT5 and fetch historical OHLCV data.

### 2) Install
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` is not present, install:
```
pip install fastapi uvicorn supabase metaTrader5 pandas numpy python-dotenv requests
```

### 3) Run the API Server
```
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

- On startup, the server ensures the `backtesting_signals` and `backtesting_runs` tables exist.

## Core Endpoints

Base URL: `http://localhost:8000`

- `POST /api/backtest/manual/generate`
  - Initiates a manual backtest run.
  - Request JSON:
    - `symbol` (string), `timeframe` (string, e.g., `M15`), `start_date` (ISO), `end_date` (ISO)
    - `initial_balance` (number), `commission_per_lot` (number, optional), `spread_adjustment` (number, optional)
    - `manual_run_id` (string, optional; will be auto-generated if omitted)
  - Response: `{ manual_run_id: string, inserted_signals: number }`

- `GET /api/backtest/manual/runs`
  - Fetches backtesting run metadata.
  - Response: `[{ manual_run_id, symbol, timeframe, start_date, end_date, initial_balance, commission_per_lot, spread_adjustment, status, created_at }]`

- `GET /api/backtest/manual/signals/{manual_run_id}`
  - Retrieves signals for a specific manual run.
  - Response: `{ manual_run_id: string, signals: Signal[] }`

- `POST /api/backtest/manual/execute`
  - Simulates trade execution over M1 data and computes PnL metrics.
  - Request JSON: `{ manual_run_id: string }`
  - Response: `{ manual_run_id, results: { total_trades, win_rate, total_pnl, max_drawdown, equity_curve } }`

- `GET /signals/recent?limit=50`
  - Recent live signals.
  - Response: `{ count: number, signals: Signal[] }`

Where `Signal` includes fields like: `id, timestamp, symbol, timeframe, side, strength, strategy, indicators, contributions, indicator_contributions, signal_type`, plus trade plan fields if present.

## Backtesting Flow
1. Generate run with `POST /api/backtest/manual/generate` using desired parameters.
2. Inspect runs via `GET /api/backtest/manual/runs` and drill into signals via `GET /api/backtest/manual/signals/{manual_run_id}`.
3. Execute simulation with `POST /api/backtest/manual/execute` to compute PnL metrics.

## Sample Requests

Generate a run:
```
curl -X POST http://localhost:8000/api/backtest/manual/generate \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "XAUUSD",
    "timeframe": "M15",
    "start_date": "2023-01-01T00:00:00Z",
    "end_date": "2023-03-01T00:00:00Z",
    "initial_balance": 10000,
    "commission_per_lot": 7,
    "spread_adjustment": 0.5
  }'
```

List runs:
```
curl http://localhost:8000/api/backtest/manual/runs
```

Signals for a run:
```
curl http://localhost:8000/api/backtest/manual/signals/<manual_run_id>
```

Execute simulation:
```
curl -X POST http://localhost:8000/api/backtest/manual/execute \
  -H "Content-Type: application/json" \
  -d '{ "manual_run_id": "<manual_run_id>" }'
```

## Development Notes
- Tables ensured on startup: `public.backtesting_signals`, `public.backtesting_runs`.
- Batch insertion used for signal generation for efficiency.
- Simulation uses M1 data to approximate execution and computes equity curve.
- Optional analytics (Sharpe ratio, max consecutive wins/losses, comparisons) can be added later.

## GitHub Deployment
If you’re setting this up locally and pushing to GitHub:
1. Initialize git locally if not already:
   - `git init && git add . && git commit -m "Initial commit"`
2. Create a GitHub repository named `VediGoldAI` (via GitHub UI or GitHub CLI `gh`).
3. Add remote and push:
   - `git remote add origin https://github.com/<your-username>/VediGoldAI.git`
   - `git push -u origin main`

If using a Personal Access Token (PAT), ensure it has `repo` permissions.

## License
Proprietary. All rights reserved unless explicitly licensed otherwise.