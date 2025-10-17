# Vedi-Trade-AI — System Overview (Hybrid)

_Last generated: 2025-10-17 13:45 UTC_

This document provides a **developer + operations** overview of the Vedi-Trade-AI backend: modules, endpoints, data flow, environment configuration, database schema, and the full signal lifecycle. It is designed to be dropped in at: **`/docs/system_overview.md`**.

---
## 1) Project Overview

- **Language/Stack:** Python 3.10+, FastAPI, Uvicorn, Pandas + pandas-ta, yfinance, PostgreSQL (psycopg2), WebSocket streaming.
- **Purpose:** Fetch OHLCV & live quotes, compute multi-indicator signals, store snapshots & signals, support unified backtesting, and provide real-time feeds for a frontend.
- **Key Principles:** UTC time hygiene, closed-bar alignment, safe JSON persistence, weighted voting with multi-timeframe confirmation, optional analytics (A/B, Thompson sampling, auto-threshold, weight learning, rollback).

---
## 2) Architecture & Modules

**Top-level modules (selected):**
- `app/yahoo_server.py` — FastAPI app, REST+WebSocket endpoints, orchestration.
- `app/signal_engine.py` — Core pipeline: fetch → indicators → aggregate → threshold → persist.
- `app/indicators.py` — Indicator computation (RSI, MACD, SMA/EMA, BBANDS, STOCH, ATR), trend/momentum helpers.
- `app/mtf_confirmation.py` — Multi-timeframe agreement/guard logic.
- `app/threshold_manager.py` — Static/adaptive thresholding (EWMA bounds).
- `app/analytics/` — A/B regime, Thompson sampling, weight learner, auto-threshold, rollback.
- `app/db.py` — Connection pool, schema ensure, CRUD for signals, snapshots, strategies, backtests, summaries.
- `app/utils_time.py` — UTC normalization, last-closed-bar helpers, retry utilities.
- `app/nan_safety.py` / `app/sanity_filter.py` — Robustness utilities for NaN/infinite values, sanity screens.
- `app/data_fetchers/yahoo_fetcher.py` — Historical & live pulls from Yahoo Finance.
- `backtest/backtest_engine.py` — Unified backtester reusing prod signal path.
- `app/logging_config.py` — Structured logging setup.
- `app/env_loader.py` — Lightweight .env reader.
- `docs/` — Additional documentation (Weighted Voting, Backtesting).

### Mermaid — High-Level Data & Control Flow
```mermaid
flowchart LR
  subgraph Client["Frontend (Futuristic AI Dashboard)"]
    A1[Symbol Selector]
    A2[Real-time Dashboard]
    A3[Signals Explorer]
    A4[Manual Trigger]
    A5[Backtesting UI]
    A6[Strategy Config]
    A7[Trace Viewer]
  end

  subgraph API["FastAPI Backend (yahoo_server.py)"]
    B1[/GET /history/]
    B2[/GET /indicators/latest/]
    B3[/GET /indicators/live/]
    B4[/GET /signals/recent/]
    B5[/GET /signals/latest/]
    B6[/POST /api/signals/compute/]
    B7[/POST /api/backtest/run/]
    B8[/GET /api/backtest/roi/]
    B9[/GET /api/backtest/{id}/results/]
    B10[/GET /api/backtest/list/]
    B11[/GET /api/config/strategies/]
    B12[/GET /api/config/strategies/{id}/]
    B13[/POST /api/config/strategies/{id}/activate/]
    B14[/"WS /ws/prices"/]
    B15[/GET /api/trace/{signal_id}/]
    B16[/GET /api/indicators/contributions/{signal_id}/]
  end

  subgraph Engine["Core Logic"]
    C1[SignalEngine]
    C2[Indicators (pandas-ta)]
    C3[MTF Confirmation]
    C4[Threshold Manager]
    C5[Analytics: A/B, TS, Learner, Rollback]
  end

  subgraph Data["Data Sources & Storage"]
    D1[(PostgreSQL)]
    D2[[Yahoo Finance]]
  end

  A2 -- subscribe --> B14
  A1 --> B2
  A1 --> B5
  A3 --> B4
  A4 --> B6
  A5 --> B7
  A5 --> B9
  A5 --> B10
  A5 --> B8
  A6 --> B11
  A6 --> B12
  A6 --> B13
  A7 --> B15
  A7 --> B16

  B1 ==> C2
  B2 ==> C2
  B3 ==> C2
  B6 ==> C1
  B7 ==> C1
  B5 ==> C1

  C1 <==> C2
  C1 --> C3
  C1 --> C4
  C1 --> C5

  C1 --> D1
  C2 --> D1
  B14 --> C1
  C2 --> D2
```

---
## 3) REST & WebSocket Endpoint Specs

> **Base URL:** `http://localhost:8000` (configurable)  
> **OpenAPI:** `/openapi.json`, Swagger UI at `/docs`.

### System & Data
- **GET `/health`** — Service status + engine stats.

- **GET `/history`** — Historical OHLCV.
  - **Query:** `symbol` (e.g., `XAUUSD`), `timeframe` (e.g., `15m`, `1h`, `4h`), `count` or `start/end`.
  - **Response:** `{ "symbol": "...", "timeframe": "...", "candles": [{t, o, h, l, c, v}] }`

### Indicators
- **GET `/indicators/latest`**
  - **Query:** `symbol`
  - **Sample**:
```json
{
  "symbol": "XAUUSD",
  "timeframe": "15m",
  "timestamp": "2025-10-16T20:15:00Z",
  "indicators": {
    "rsi14": 57.2,
    "macd": {
      "line": 1.23,
      "signal": 0.98,
      "hist": 0.25
    },
    "sma20": 2361.1,
    "ema21": 2363.5,
    "bb": {
      "mid": 2362.0,
      "upper": 2370.1,
      "lower": 2353.9
    },
    "stoch": {
      "k": 76.1,
      "d": 72.4
    },
    "atr14": 12.5
  },
  "evaluation": {
    "trend": "Bullish",
    "momentum": "Positive",
    "confidence": 0.72
  },
  "strategy": "Gold Strategy"
}
```
- **GET `/indicators/live`**
  - **Query:** `symbol`, `timeframe`
  - Returns on-the-fly computed indicators without persistence.

### Signals
- **GET `/signals/recent`**
  - **Query:** filters like `symbol`, `limit`, `min_strength`.
- **GET `/signals/latest`**
  - **Query:** `symbol` (optional; latest per symbol if omitted)
  - **Sample**:
```json
{
  "symbol": "XAUUSD",
  "timeframe": "15m",
  "side": "buy",
  "strength": 74.5,
  "strategy": "Gold Strategy",
  "timestamp": "2025-10-16T20:15:00Z",
  "primary_timeframe": "M15",
  "confirmation_timeframe": "H1",
  "trend_timeframe": "H4",
  "h1_trend_direction": "Bullish",
  "indicator_contributions": {
    "RSI": 0.2,
    "MACD": 0.25,
    "EMA": 0.15,
    "Others": 0.4
  }
}
```
- **POST `/api/signals/compute`** — Trigger one-off computation (single or multiple).
  - **Query:** `symbols=CSV` (e.g., `XAUUSD,EURUSD`)
  - **Sample Request:
```json
{
  "method": "POST",
  "url": "/api/signals/compute?symbols=XAUUSD"
}
```
  - **Sample Response:**
```json
{
  "status": "ok",
  "results": [
    {
      "symbol": "XAUUSD",
      "timeframe": "15m",
      "side": "buy",
      "strength": 72.8,
      "inserted": true
    }
  ]
}
```
### Backtesting
- **POST `/api/backtest/run`**
  - **Body:** `{strategy_id, symbol, start_date, end_date, investment, timeframe}`
  - **Sample:
```json
{
  "strategy_id": 1,
  "symbol": "XAUUSD",
  "start_date": "2025-08-01",
  "end_date": "2025-09-01",
  "investment": 10000,
  "timeframe": "15m"
}
```
- **GET `/api/backtest/{backtest_id}/results`**
  - **Sample:
```json
{
  "backtest_id": 42,
  "equity_curve": [
    {
      "t": "2025-08-01",
      "equity": 10000
    },
    {
      "t": "2025-09-01",
      "equity": 11260
    }
  ],
  "trades": [
    {
      "time": "2025-08-04T10:15Z",
      "side": "buy",
      "entry": 2344.2,
      "exit": 2352.9,
      "pnl_pct": 0.37,
      "result": "profit"
    }
  ]
}
```
- **GET `/api/backtest/roi`**
  - **Query:** `backtest_id`, `amount`

- **GET `/api/backtest/list`** — List previous runs.

### Strategy Configuration
- **GET `/api/config/strategies`**
- **GET `/api/config/strategies/{strategy_id}`**
- **POST `/api/config/strategies/{strategy_id}/activate`**

### Trace & Contributions
- **GET `/api/trace/{signal_id}`** — Trace events for a signal.
- **GET `/api/indicators/contributions/{signal_id}`** — Stored indicator contributions JSON.

### WebSocket
- **WS `/ws/prices?symbol=XAUUSD`**
  - **Server Tick Payload (example):**
```json
{
  "type": "tick",
  "symbol": "XAUUSD",
  "price": 2365.92,
  "timestamp": "2025-10-16T20:16:05Z"
}
```

---

## 4) Database Schema (PostgreSQL)

> Tables are auto-created/ensured by `app/db.py` on startup/access.

- **backtest_signals**
  - `id SERIAL PRIMARY KEY`
  - `backtest_id INT NOT NULL REFERENCES public.backtests(id) ON DELETE CASCADE`
  - `signal_time TIMESTAMP NOT NULL`
  - `direction TEXT NOT NULL CHECK (direction IN ('buy'`
  - `'sell'))`
  - `entry_price FLOAT NOT NULL`
  - `exit_price FLOAT`
  - `profit_pct FLOAT NOT NULL DEFAULT 0`
  - `result TEXT NOT NULL CHECK (result IN ('profit'`
  - `'loss'`
  - `'open'))`
  - `confidence FLOAT DEFAULT 0.5`
  - `reason TEXT`
  - `created_at TIMESTAMP DEFAULT NOW()`
- **backtests**
  - `id SERIAL PRIMARY KEY`
  - `strategy_id INT NOT NULL`
  - `symbol TEXT NOT NULL`
  - `timeframe TEXT NOT NULL`
  - `start_date TIMESTAMP NOT NULL`
  - `end_date TIMESTAMP NOT NULL`
  - `investment FLOAT NOT NULL DEFAULT 10000`
  - `total_return_pct FLOAT NOT NULL DEFAULT 0`
  - `efficiency_pct FLOAT NOT NULL DEFAULT 0`
  - `created_at TIMESTAMP DEFAULT NOW()`
- **indicator_snapshots**
  - `id bigint generated by default as identity primary key`
  - `timestamp timestamptz not null`
  - `symbol text not null`
  - `timeframe text not null`
  - `indicators jsonb not null`
  - `evaluation jsonb`
  - `strategy text`
  - `created_at timestamptz default now()`
- **signal_performance_daily**
  - `id SERIAL PRIMARY KEY`
  - `evaluation_date DATE NOT NULL`
  - `symbol TEXT NOT NULL`
  - `timeframe TEXT NOT NULL`
  - `signals_evaluated INT DEFAULT 0`
  - `new_profits INT DEFAULT 0`
  - `new_losses INT DEFAULT 0`
  - `daily_roi_pct FLOAT DEFAULT 0.0`
  - `created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()`
  - `UNIQUE(evaluation_date`
  - `symbol`
  - `timeframe)`
- **signal_performance_summary**
  - `id SERIAL PRIMARY KEY`
  - `strategy_id INT`
  - `symbol TEXT NOT NULL`
  - `timeframe TEXT NOT NULL`
  - `total_signals INT DEFAULT 0`
  - `win_count INT DEFAULT 0`
  - `loss_count INT DEFAULT 0`
  - `open_count INT DEFAULT 0`
  - `avg_profit_pct FLOAT DEFAULT 0.0`
  - `total_roi_pct FLOAT DEFAULT 0.0`
  - `efficiency_pct FLOAT DEFAULT 0.0`
  - `last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()`
  - `UNIQUE(symbol`
  - `timeframe)`
- **signal_results**
  - `id SERIAL PRIMARY KEY`
  - `signal_id BIGINT NOT NULL REFERENCES public.signals(id)`
  - `result TEXT NOT NULL CHECK (result IN ('profit'`
  - `'loss'`
  - `'open'))`
  - `exit_price FLOAT`
  - `profit_pct FLOAT`
  - `evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()`
  - `evaluation_notes TEXT`
  - `UNIQUE(signal_id)`
- **signals**
  - `id bigint generated by default as identity primary key`
  - `timestamp timestamptz not null`
  - `symbol text not null`
  - `timeframe text not null`
  - `side text not null check (side in ('buy'`
  - `'sell'))`
  - `strength numeric not null`
  - `strategy text not null`
  - `indicators jsonb not null`
  - `contributions jsonb not null`
  - `indicator_contributions jsonb`
  - `signal_type text`
  - `primary_timeframe text`
  - `confirmation_timeframe text`
  - `trend_timeframe text`
  - `h1_trend_direction text check (h1_trend_direction in ('Bullish'`
  - `'Bearish'))`
  - `h4_trend_direction text check (h4_trend_direction in ('Bullish'`
  - `'Bearish'))`
  - `alignment_boost numeric`
  - `final_signal_strength numeric`
  - `entry_price numeric`
  - `stop_loss_price numeric`
  - `take_profit_price numeric`
  - `stop_loss_distance_pips numeric`
  - `take_profit_distance_pips numeric`
  - `risk_reward_ratio numeric`
  - `volatility_state text`
  - `is_valid boolean`
- **strategies**
  - `id bigint generated by default as identity primary key`
  - `name text unique not null`
  - `description text`
  - `is_active boolean default false`
  - `primary_timeframe text not null`
  - `confirmation_timeframe text not null`
  - `trend_timeframe text not null`
  - `run_interval_seconds integer not null default 600`
  - `signal_threshold numeric not null`
  - `created_at timestamptz default now()`
  - `updated_at timestamptz default now()`
- **strategy_indicators**
  - `id bigint generated by default as identity primary key`
  - `strategy_id bigint not null references public.strategies(id) on delete cascade`
  - `indicator_name text not null`
  - `params jsonb not null`
  - `unique (strategy_id`
  - `indicator_name)`
- **strategy_weights**
  - `id bigint generated by default as identity primary key`
  - `strategy_id bigint not null references public.strategies(id) on delete cascade`
  - `weights jsonb not null`
  - `unique (strategy_id)`

> **Notes:**
> - `signals` stores per-signal metadata, strengths, multi-timeframe fields, and JSONB of `indicators`, `contributions`, `indicator_contributions`.
> - `indicator_snapshots` stores latest computed indicator snapshots (with optional evaluation and strategy tag).
> - `strategies`, `strategy_indicators`, `strategy_weights` define configurable strategy space.
> - Backtesting tables: `backtests`, `backtest_signals`, `signal_results` hold simulation, per-trade outcomes, and summaries.
> - Performance rollups: `signal_performance_daily`, `signal_performance_summary` for analytics.

---

## 5) Environment Variables (with Dummy Values)

Create a `.env` file in project root. The system accepts both canonical names and some lowercase synonyms.

```dotenv
# ---------- Database ----------
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vedi_trading
DB_USER=vedi_user
DB_PASSWORD=change_me_strong

# (Synonyms supported internally)
POSTGRES_USER=${DB_USER}
POSTGRES_PASSWORD=${DB_PASSWORD}
POSTGRES_DB=${DB_NAME}

# ---------- Server / CORS / Logging ----------
PORT=8000
CORS_ALLOW_ORIGINS=http://localhost:3000
VEDI_LOG_LEVEL=info

# ---------- Optional: AWS / Polygon (only for polygon_downloader utilities) ----------
POLYGON_ENDPOINT_URL=https://polygon.example.com
POLYGON_REGION=eu-west-1
POLYGON_ACCESS_KEY_ID=AKIAEXAMPLE
POLYGON_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxx
POLYGON_AWS_ACCESS_KEY_ID=${POLYGON_ACCESS_KEY_ID}
POLYGON_AWS_SECRET_ACCESS_KEY=${POLYGON_SECRET_ACCESS_KEY}
POLYGON_AWS_SESSION_TOKEN=
```

> If any of `DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD` are missing, startup will fail with a credential error.  
> `CORS_ALLOW_ORIGINS` accepts a CSV or `*` for wide-open dev usage.

---

## 6) Typical Signal Lifecycle

1. **Trigger** — Periodic scheduler or manual `POST /api/signals/compute?symbols=...` kicks off the pipeline.
2. **Fetch** — Yahoo fetcher loads OHLCV (UTC-indexed) for the requested `symbol/timeframe` ensuring **closed bars** only.
3. **Indicators** — `indicators.py` computes RSI/MACD/SMA/EMA/BBANDS/STOCH/ATR etc. NaNs are sanitized.
4. **Aggregation** — Weighted voting (configurable `WEIGHTS`) blends indicators into a composite score; trend vs momentum ratios (`TREND_WEIGHT_RATIO` / `MOMENTUM_WEIGHT_RATIO`) applied.
5. **Multi-Timeframe** — Primary (M15) is confirmed by trend TF (H1/H4) via `mtf_confirmation.py`.
6. **Thresholding** — Static or adaptive EWMA thresholds select **buy / sell / neutral**; confidence zones (e.g., strong ≥ 0.7).
7. **Persistence** — Insert row into `signals` and upsert `indicator_snapshots` for latest snapshot; attach `indicator_contributions` for transparency.
8. **Broadcast** — WebSocket `/ws/prices` emits latest ticks; REST endpoints expose `signals/latest`, `indicators/latest`.
9. **Backtesting** — `backtest_engine.py` replays the same pipeline over historical windows; results saved into `backtests`/* tables.
10. **Analytics** — Optional A/B regimes, Thompson sampling exploration, weight learning; performance rollups updated.

---

## 7) Deployment Notes

- Run with Uvicorn: `uvicorn app.yahoo_server:app --host 0.0.0.0 --port $PORT`
- Ensure Postgres reachable and `.env` loaded (the app also ships a minimal env loader).
- Open firewall for `PORT` and restrict CORS in production.
- Consider adding:
  - API key/JWT middleware
  - Redis caching for `/history` and heavy endpoints
  - Structured logs to file/ELK
  - Alembic migrations for schema evolution

---

## 8) Appendix — Sample Requests/Responses

**Indicators (latest):**
```http
GET /indicators/latest?symbol=XAUUSD
```
```json
{
  "symbol": "XAUUSD",
  "timeframe": "15m",
  "timestamp": "2025-10-16T20:15:00Z",
  "indicators": {
    "rsi14": 57.2,
    "macd": {
      "line": 1.23,
      "signal": 0.98,
      "hist": 0.25
    },
    "sma20": 2361.1,
    "ema21": 2363.5,
    "bb": {
      "mid": 2362.0,
      "upper": 2370.1,
      "lower": 2353.9
    },
    "stoch": {
      "k": 76.1,
      "d": 72.4
    },
    "atr14": 12.5
  },
  "evaluation": {
    "trend": "Bullish",
    "momentum": "Positive",
    "confidence": 0.72
  },
  "strategy": "Gold Strategy"
}
```

**Signals (latest):**
```http
GET /signals/latest?symbol=XAUUSD
```
```json
{
  "symbol": "XAUUSD",
  "timeframe": "15m",
  "side": "buy",
  "strength": 74.5,
  "strategy": "Gold Strategy",
  "timestamp": "2025-10-16T20:15:00Z",
  "primary_timeframe": "M15",
  "confirmation_timeframe": "H1",
  "trend_timeframe": "H4",
  "h1_trend_direction": "Bullish",
  "indicator_contributions": {
    "RSI": 0.2,
    "MACD": 0.25,
    "EMA": 0.15,
    "Others": 0.4
  }
}
```

**Manual compute:**
```http
POST /api/signals/compute?symbols=XAUUSD
```
```json
{
  "status": "ok",
  "results": [
    {
      "symbol": "XAUUSD",
      "timeframe": "15m",
      "side": "buy",
      "strength": 72.8,
      "inserted": true
    }
  ]
}
```

**Backtest run:**
```http
POST /api/backtest/run
```
```json
{
  "strategy_id": 1,
  "symbol": "XAUUSD",
  "start_date": "2025-08-01",
  "end_date": "2025-09-01",
  "investment": 10000,
  "timeframe": "15m"
}
```

**Backtest results:**
```http
GET /api/backtest/42/results
```
```json
{
  "backtest_id": 42,
  "equity_curve": [
    {
      "t": "2025-08-01",
      "equity": 10000
    },
    {
      "t": "2025-09-01",
      "equity": 11260
    }
  ],
  "trades": [
    {
      "time": "2025-08-04T10:15Z",
      "side": "buy",
      "entry": 2344.2,
      "exit": 2352.9,
      "pnl_pct": 0.37,
      "result": "profit"
    }
  ]
}
```

**WebSocket tick:**
```http
WS /ws/prices?symbol=XAUUSD
```
```json
{
  "type": "tick",
  "symbol": "XAUUSD",
  "price": 2365.92,
  "timestamp": "2025-10-16T20:16:05Z"
}
```

---

**End of document.**
