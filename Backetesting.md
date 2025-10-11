# Backtesting Frontend Agent — Full Prompt

You are building a Backtesting UI for the GoldSignalEngine-v2.3-Adaptive backend. Your job is to design and implement a complete frontend that allows users to:
- Configure and run manual backtests over historical ranges
- Browse and filter backtesting runs
- Inspect signals generated per run
- Execute a trade simulation for a specific run and review performance analytics

The backend is implemented with FastAPI and exposes the following REST API endpoints. Use these to wire the full experience. Assume the base URL is `http://localhost:8000` unless otherwise specified.

---

## API Overview

### 1) Generate Backtest
- Method: `POST`
- Path: `/api/backtest/manual/generate`
- Purpose: Generate signals on historical data for a given range using the adaptive engine. Persists signals to Postgres in `public.backtesting_signals` and registers run metadata in `public.backtesting_runs`.
- Request JSON (fields):
  - `start_date` (string, required, ISO-8601): e.g., `"2025-01-01T00:00:00Z"`
  - `end_date` (string, required, ISO-8601): e.g., `"2025-01-15T00:00:00Z"`
  - `symbol` (string, optional, default `XAUUSD`): e.g., `"XAUUSD"`
  - `timeframe` (string, optional, default `M15`): valid values include `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d` (case-insensitive aliases supported)
  - `initial_balance` (number, optional): starting balance (used for context; not directly used in generation)
  - `commission_per_trade` (number, optional): per-trade commission applied in later simulation
  - `spread_adjustment` (number, optional): spread factor (currently stored/contextual; generation does not adjust prices)
- Typical 200 Response JSON:
  ```json
  {
    "manual_run_id": "manual_run_20250101_120000",
    "signals_generated": 128,
    "average_confidence": 72.5,
    "status": "completed"
  }
  ```
- Error: 500 with `{ "detail": "Backtest generation failed: ..." }` for backend issues; 400 for missing required fields.

### 2) List Runs
- Method: `GET`
- Path: `/api/backtest/manual/runs`
- Purpose: Fetch metadata for previously generated runs.
- Query Params (optional):
  - `from_date` (string, ISO-8601 or date): filter runs by start_date >= from_date
  - `to_date` (string, ISO-8601 or date): filter runs by end_date <= to_date
  - `min_signal_strength` (number): include only runs that have at least one signal with final strength >= this threshold
  - `symbol` (string): filter by instrument symbol
- Typical 200 Response JSON (array of run objects):
  ```json
  [
    {
      "manual_run_id": "manual_run_20250101_120000",
      "start_date": "2025-01-01T00:00:00Z",
      "end_date": "2025-01-15T00:00:00Z",
      "symbol": "XAUUSD",
      "timeframe": "M15",
      "signals_generated": 128,
      "average_confidence": 72.5,
      "average_rr_ratio": 1.55
    }
  ]
  ```
- Error: 500 with `{ "detail": "Fetch runs failed: ..." }`.

### 3) Fetch Signals for a Run
- Method: `GET`
- Path: `/api/backtest/manual/signals/{manual_run_id}`
- Purpose: Retrieve all signals for a specific run.
- Path Params:
  - `manual_run_id` (string): e.g., `"manual_run_20250101_120000"`
- Typical 200 Response JSON:
  ```json
  {
    "manual_run_id": "manual_run_20250101_120000",
    "signals": [
      {
        "timestamp": "2025-01-03T14:30:00Z",
        "symbol": "XAUUSD",
        "signal_type": "BUY",
        "entry_price": 2350.12,
        "stop_loss_price": 2342.80,
        "take_profit_price": 2365.20,
        "final_signal_strength": 85.0,
        "volatility_state": "Normal",
        "risk_reward_ratio": 2.0,
        "indicator_contributions": {
          "RSI": 8,
          "MACD": 12,
          "STOCH": 6,
          "BBANDS": 6,
          "SMA_EMA": 10,
          "MTF": 15,
          "ATR_STABILITY": 8,
          "PRICE_ACTION": 10
        }
      }
    ]
  }
  ```
- Error: 500 with `{ "detail": "Fetch signals failed: ..." }`.

### 4) Execute Simulation
- Method: `POST`
- Path: `/api/backtest/manual/execute`
- Purpose: Simulate trade execution for a run using M1 data after each signal timestamp. Calculates P/L with commission and slippage.
- Request JSON (fields):
  - `manual_run_id` (string, required)
  - `initial_balance` (number, required)
  - `risk_per_trade_percent` (number, required): e.g., `2` for 2% risk per trade
  - `commission_per_trade` (number, required)
  - `slippage_percent` (number, required): applied as a percentage of risk amount
- Typical 200 Response JSON:
  ```json
  {
    "manual_run_id": "manual_run_20250101_120000",
    "initial_balance": 10000,
    "final_balance": 11250.5,
    "net_profit_percent": 12.505,
    "total_trades": 96,
    "wins": 55,
    "losses": 41,
    "win_rate_percent": 57.3,
    "max_drawdown_percent": 7.8,
    "account_blown": false,
    "profit_factor": 1.45,
    "average_rr_ratio": 1.52,
    "result_summary": "Profitable run with moderate drawdown and stable performance.",
    "critical_event": null
  }
  ```
- Error: 500 with `{ "detail": "Backtest execution failed: ..." }`.

---

## UI Requirements

Design and implement a responsive frontend with the following pages and components:

### A) Run Generator
- Form fields:
  - Date range: `start_date`, `end_date` (ISO; provide date+time pickers and UTC toggle)
  - `symbol` dropdown (default `XAUUSD`)
  - `timeframe` dropdown (default `M15`; map the friendly labels to backend keys: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`)
  - Optional numeric inputs: `initial_balance`, `commission_per_trade`, `spread_adjustment`
- Actions:
  - `Generate` button: POST to `/api/backtest/manual/generate`
  - Show progress state (spinner) while request is in-flight
  - On success: Display `manual_run_id`, `signals_generated`, `average_confidence`
  - On error: Display backend error message
- Validation:
  - `start_date` < `end_date`
  - Non-negative numbers for balances/fees/spread
  - Prevent submission without required fields

### B) Runs Dashboard
- Table with columns: `manual_run_id`, `symbol`, `timeframe`, `start_date`, `end_date`, `signals_generated`, `average_confidence`, `average_rr_ratio`
- Filters:
  - Date range (`from_date`, `to_date`)
  - `symbol`
  - `min_signal_strength` (slider or input)
- UX:
  - Paginate if needed
  - Click a run row to navigate to Signals Viewer for that run

### C) Signals Viewer
- For selected `manual_run_id`:
  - Table columns: `timestamp`, `signal_type`, `entry_price`, `stop_loss_price`, `take_profit_price`, `final_signal_strength`, `volatility_state`, `risk_reward_ratio`
  - Expandable row / side panel: show `indicator_contributions` object
- Controls:
  - Sort by time, confidence, RR, side
  - Filter by `signal_type` (BUY/SELL) and confidence threshold
- Optional visualization:
  - Confidence histogram and RR distribution

### D) Execute Simulation
- Form fields:
  - `manual_run_id` (auto-populated from context or selectable)
  - `initial_balance`
  - `risk_per_trade_percent`
  - `commission_per_trade`
  - `slippage_percent`
- Actions:
  - `Run Simulation` button: POST to `/api/backtest/manual/execute`
  - Show progress state while request is in-flight
  - On success: render metrics (`final_balance`, `net_profit_percent`, `total_trades`, `win_rate_percent`, `max_drawdown_percent`, `profit_factor`, `average_rr_ratio`, `account_blown`, `result_summary`)
- Optional visualization:
  - If equity curve is added later, show an equity chart

---

## Integration Details & Assumptions

- Base URL: `http://localhost:8000`
- Auth: None required (local dev). Add tokens if needed in future.
- Expected content types: JSON for request/response; set `Content-Type: application/json` on POSTs.
- Timeframe mapping: The backend accepts `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d` (case-insensitive aliases like `15min`, `1hour` supported internally); prefer canonical forms in UI.
- Date handling: Use ISO strings. Internally the backend converts using pandas; ensure user inputs are valid ISO times.
- Error display: Show `detail` message from backend when present.
- Performance: Generation and execution are synchronous calls wrapped in background threads on the server; the UI should reflect loading state and not assume instant execution for large ranges.

---

## Sample Requests (for testing)

### Generate
```bash
curl -X POST http://localhost:8000/api/backtest/manual/generate \
  -H "Content-Type: application/json" \
  -d '{
    "start_date":"2025-01-01T00:00:00Z",
    "end_date":"2025-01-15T00:00:00Z",
    "symbol":"XAUUSD",
    "timeframe":"M15",
    "initial_balance":10000,
    "commission_per_trade":0.5,
    "spread_adjustment":0.0003
  }'
```

### List Runs
```bash
curl "http://localhost:8000/api/backtest/manual/runs?from_date=2025-01-01&to_date=2025-01-31&symbol=XAUUSD"
```

### Fetch Signals
```bash
curl http://localhost:8000/api/backtest/manual/signals/manual_run_20250101_120000
```

### Execute Simulation
```bash
curl -X POST http://localhost:8000/api/backtest/manual/execute \
  -H "Content-Type: application/json" \
  -d '{
    "manual_run_id":"manual_run_20250101_120000",
    "initial_balance":10000,
    "risk_per_trade_percent":2,
    "commission_per_trade":0.5,
    "slippage_percent":0.02
  }'
```

---

## Deliverables

Implement the Backtesting frontend with:
- Run Generator page with validation and clear UX feedback
- Runs Dashboard with filters, sorting, and navigation
- Signals Viewer with rich table and contributions details
- Execute Simulation page with results presentation and error handling

Aim for clean component architecture, robust state management, and tidy data fetching. Provide small helper utilities to format numbers, dates, and confidence/volatility badges. Add basic dark/light theme support if available.