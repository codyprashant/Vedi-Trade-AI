# VediGold AI — Comprehensive Documentation

## Overview

VediGold AI is a FastAPI-based trading analytics and signal generation engine tailored for `XAUUSD` (Gold). It analyzes multi-timeframe price data, computes a diverse set of technical indicators, and produces validated trade signals. It also provides manual backtesting to generate signals over historical ranges and a trade execution simulator to evaluate performance.

The system exposes REST endpoints for health, historical data, signals, and backtesting, plus a WebSocket feed for live tick streaming. Signals and backtesting artifacts are persisted in Postgres (e.g., Supabase) for downstream analytics.

## Key Features

- Multi-timeframe signal generation on `M15`, validated against `H1` trend; additive `H1`/`H4` alignment boosts increase confidence.
- Rich indicator suite: RSI, MACD, SMA/EMA, Bollinger Bands, Stochastic, ATR, and price action patterns.
- Strategy engine combining trend and momentum contributions with configurable weights and thresholds.
- Volatility-aware trade planning: entry, stop-loss, and take-profit distances adapt to ATR and volatility state.
- Live WebSocket tick streaming and REST history retrieval from MetaTrader 5.
- Manual backtesting over historical periods with batch signal generation and a 2-day M1 execution simulator.
- Postgres persistence for signals and backtesting runs/signals.
- OpenAPI documentation export (`docs/openapi.json`) using the FastAPI schema.

## Architecture

- Server: `app/server.py`
  - FastAPI application with REST and WebSocket endpoints.
  - Manages MT5 initialization and an inactivity watcher that shuts down MT5 if no subscribers for 60 seconds.
  - Starts the `SignalEngine` loop at startup; stops it gracefully on shutdown.
- Decision Engine: `app/signal_engine.py`
  - Periodically fetches `M15`, `H1`, `H4` data and computes indicators on `M15`.
  - Evaluates strategy strength and validates M15 signals against H1 trend; applies additive alignment boosts (H1 and H4).
  - Classifies volatility via H1 ATR vs rolling mean; skips extreme volatility.
  - Calculates trade plan (entry/SL/TP) and persists high-confidence signals.
- Indicators & Strategies: `app/indicators.py`
  - Implements indicator computations using `pandas_ta`.
  - Evaluates per-indicator directional signals and computes strategy strength for `trend`, `momentum`, and `combined`.
  - Provides multi-timeframe helpers and price action heuristics.
- Backtesting: `app/backtesting.py`
  - Generates signals across a historical date range, enforcing alignment and volatility rules.
  - Simulates trade execution with M1 data for up to 2 days, tracking PnL, win rate, drawdown, and profit factor.
- Persistence: `app/db.py`
  - Connection pool to Postgres; ensures table schemas exist.
  - Inserts and queries for live signals and backtesting runs/signals.
- Configuration: `app/config.py`
  - Defaults for symbol/timeframes and indicator parameters.
  - Confidence weights and thresholds used by the decision engine.

## Configuration

- Defaults
  - `DEFAULT_SYMBOL = "XAUUSD"`
  - `DEFAULT_TIMEFRAME = "15m"` (primary analysis timeframe)
  - `PRIMARY_TIMEFRAME = "M15"`, `CONFIRMATION_TIMEFRAME = "H1"`, `TREND_TIMEFRAME = "H4"`
  - `DEFAULT_HISTORY_COUNT = 500`
- Indicator parameters (`INDICATOR_PARAMS`)
  - RSI: `length=14`, `overbought=70`, `oversold=30`
  - MACD: `fast=12`, `slow=26`, `signal=9`
  - SMA: `short=50`, `long=200`
  - EMA: `short=20`, `long=50`
  - Bollinger Bands: `length=20`, `std=2`
  - Stochastic: `k=14`, `d=3`, `oversold=20`, `overbought=80`
  - ATR: `length=14`, `min_ratio=0.002` (≥0.2% of price)
- Indicator weights (`WEIGHTS`)
  - `RSI=15`, `MACD=20`, `SMA_EMA=15`, `BBANDS=10`, `STOCH=10`
  - `MTF=10` (H1 alignment), `ATR_STABILITY=10` (Normal volatility), `PRICE_ACTION=10`
- Thresholds
  - `SIGNAL_THRESHOLD = 90` (final strength required to persist a signal)
 - Alignment boosts (configurable)
  - `ALIGNMENT_BOOST_H1 = 10` (applied when M15 aligns with H1)
  - `ALIGNMENT_BOOST_H4 = 5` (applied when H4 agrees with H1 and M15 aligns with H1)

## Indicators Library

- RSI: oversold (<30) → buy; overbought (>70) → sell.
- MACD: cross-over/cross-under of MACD vs signal line determines side.
- SMA/EMA: short-vs-long cross-over/cross-under signals; both contribute to trend.
- Bollinger Bands: close ≤ lower band with RSI oversold → buy; close ≥ upper band with RSI overbought → sell.
- Stochastic: K/D cross with bounds (oversold/overbought) defines side.
- ATR: acts as filter via `min_ratio` and provides volatility and distance sizing.
- Price Action (`price_action_direction`):
  - Engulfing patterns on last two candles (bullish/bearish) add directional score.
  - Pin bar detection using wick/body ratios.
  - Trend continuation via EMA20 slope and close position.
  - Fallback: majority-of-closes with EMA20 filter over last `lookback=5` candles.

## Strategies & Decision Engine

- Strategy strength computation (`compute_strategy_strength`):
  - Trend: direction by majority among `SMA`, `EMA`, `MACD`; contributions from `SMA_EMA`, `MACD`, and ATR filter.
  - Momentum: direction by majority among `RSI`, `STOCH`, `BBANDS`; contributions from each when aligned.
  - Combined: majority across all (excluding ATR for direction); contributions from all, ATR acts as filter.
- Best strategy (`best_signal`): selects the strategy with the highest computed strength.
- Multi-timeframe validation:
  - `H1` trend via EMA(50/200); M15 side must align with `H1` or the signal is skipped (engine loop).
  - Additive alignment boosts:
    - `+ALIGNMENT_BOOST_H1` when `M15` aligns with `H1`.
    - `+ALIGNMENT_BOOST_H4` when `H4` agrees with `H1` and `M15` aligns with `H1`.
    - No penalties on misalignment; lack of alignment simply yields no boost.
- Volatility classification (using H1 ATR vs mean50):
  - `Extreme` if ATR > 3× mean → skip.
  - `High` if ATR > 1.2× mean; `Low` if ATR < 0.8× mean; else `Normal`.
- Confidence contributions (final strength basis):
  - Sum of weights when indicator directions align with `m15_side` plus `MTF`/`ATR_STABILITY`/`PRICE_ACTION`.
  - Alignment boosts added on top: `final_strength = min(100, base_strength + H1_boost + H4_boost)`.
  - Must be ≥ `SIGNAL_THRESHOLD` to persist.

## Signal Generation Flow

![Signal Generation Flow](signal_flowchart.svg)

1. Fetch `M15`, `H1`, `H4` history concurrently.
2. Compute indicators on `M15`; evaluate per-indicator directions.
3. Compute strategy strengths (`trend`, `momentum`, `combined`) and select the best.
4. Validate `M15` side against `H1` trend (engine run skips if misaligned; manual compute summarizes).
5. Classify volatility using `H1` ATR vs mean; skip extreme volatility.
6. Compute trade plan (entry/SL/TP) distances based on ATR and volatility state.
7. Compute contributions (base) and additive alignment boosts (`+H1`, `+H4`).
8. Persist the signal to Postgres if final strength ≥ threshold.

## Trade Plan Computation

- Entry price: last M15 close ± `0.1 × ATR(H1)` (subtract for buy, add for sell).
- Stop-loss distance multiplier by volatility:
  - High: `sl_mult = 2.0`, `rr = 1.2`
  - Low: `sl_mult = 1.0`, `rr = 1.8`
  - Normal: `sl_mult = 1.5`, `rr = 1.5`
- SL bounds: clamp SL distance to `0.25%–1.2%` of price.
- TP distance by R:R with bounds `0.4%–2.0%` of price.
- Pips conversion: assumes `XAUUSD pip = 0.01`; `pips = distance / 0.01`.

## Persistence Layer

- `public.signals` columns (core fields):
  - `timestamp`, `symbol`, `timeframe`, `side`, `strength`, `strategy`, `indicators`, `contributions`
  - `indicator_contributions`, `signal_type`, `primary_timeframe`, `confirmation_timeframe`, `trend_timeframe`
  - `h1_trend_direction`, `h4_trend_direction`, `alignment_boost`, `final_signal_strength`
  - `entry_price`, `stop_loss_price`, `take_profit_price`, `stop_loss_distance_pips`, `take_profit_distance_pips`
  - `risk_reward_ratio`, `volatility_state`, `is_valid`
- `public.backtesting_runs` columns:
  - `manual_run_id`, `start_date`, `end_date`, `symbol`, `timeframe`, `signals_generated`
  - `average_confidence`, `average_rr_ratio`, `run_duration_seconds`, `status`, `created_at`
- `public.backtesting_signals` columns:
  - `manual_run_id`, `timestamp`, `symbol`, `signal_type`, `entry_price`, `stop_loss_price`, `take_profit_price`
  - `final_signal_strength`, `volatility_state`, `risk_reward_ratio`, `indicator_contributions`, `created_at`, `source_mode`

## APIs

- `GET /health`: MT5 init status, subscriber counts, symbol map.
- `GET /history?symbol=XAUUSD&timeframe=15m&count=500`: OHLCV candles; timeframe supports `1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo`.
- `GET /signals/recent?limit=20`: Recent high-confidence signals from Postgres.
- WebSocket `GET /ws/prices?symbol=XAUUSD`: Streams live tick JSON payloads; sends heartbeats.

### Strategy Configuration Management

The system supports dynamic strategy configuration through dedicated API endpoints, allowing real-time adjustment of trading parameters without server restarts.

#### Strategy Management
- `GET /api/config/strategies`: List all available strategies with basic information (id, name, active status, threshold, run interval).
- `GET /api/config/strategies/{strategy_id}`: Retrieve complete strategy details including indicator parameters, weights, and configuration.
- `POST /api/config/strategies/{strategy_id}/activate`: Activate a specific strategy (deactivates all others).

#### Parameter Configuration
- `PATCH /api/config/strategies/{strategy_id}/indicator/{indicator_name}`: Update parameters for specific indicators (RSI periods, MACD settings, Bollinger Band parameters, etc.).
- `PATCH /api/config/strategies/{strategy_id}/weights`: Modify contribution weights for different indicators and components (RSI, MACD, SMA_EMA, BBANDS, STOCH, MTF, ATR_STABILITY, PRICE_ACTION).
- `PATCH /api/config/strategies/{strategy_id}/threshold`: Adjust the minimum signal strength required for persistence (`SIGNAL_THRESHOLD`).
- `PATCH /api/config/strategies/{strategy_id}/schedule`: Configure the signal engine run frequency (`run_interval_seconds`).

#### Configuration Features
- **Real-time Updates**: Changes take effect immediately for new signal computations.
- **Multi-strategy Support**: Manage multiple trading strategies with different configurations.
- **Parameter Validation**: API endpoints validate parameter ranges and types.
- **Atomic Operations**: Each configuration change is applied atomically to maintain consistency.

#### Configurable Parameters
- **Indicator Settings**: RSI periods and thresholds, MACD fast/slow/signal periods, SMA/EMA lengths, Bollinger Band parameters, Stochastic settings, ATR periods, price action lookback windows.
- **Strategy Weights**: Individual contribution weights for each indicator and component (must sum to appropriate totals).
- **Operational Settings**: Signal threshold (0.0-1.0), run interval (seconds), alignment boost values.
- **Risk Management**: Volatility classification thresholds, trade plan multipliers, R:R ratio bounds.

### Manual Backtesting API

- `POST /api/backtest/manual/generate`
  - Body: `start_date`, `end_date` (ISO), optional `symbol`, `timeframe`, `initial_balance`, `commission_per_trade`, `spread_adjustment`.
  - Behavior: fetches ranges (with buffer) for `M15`, `H1`, `H4` and iterates `M15` to evaluate signals using the decision engine logic; enforces H1 alignment and skips extreme volatility; computes trade plans; persists batch signals; records run metadata with averages and duration.
  - Returns: summary `{ manual_run_id, signals_generated, average_confidence, status }`.
- `GET /api/backtest/manual/runs`
  - Query: optional `from_date`, `to_date`, `min_signal_strength`, `symbol`.
  - Returns: list of runs; optional filtering by existence of signals meeting `min_signal_strength`.
- `GET /api/backtest/manual/signals/{manual_run_id}`
  - Returns: ordered signals for the specified manual run.
- `POST /api/backtest/manual/execute`
  - Body: `manual_run_id`, `initial_balance`, `risk_per_trade_percent`, `commission_per_trade`, `slippage_percent`.
  - Behavior: simulates trade outcomes using M1 data for up to 2 days after each signal timestamp; determines first touch SL/TP; applies costs; updates balance and equity curve; computes metrics.
  - Returns: performance metrics including net profit %, win rate, max drawdown %, profit factor, average R:R, and summary.

## Execution Simulator Details

- Position sizing: `position_size = risk_amount / sl_distance`; `risk_amount = balance × (risk_per_trade_percent/100)`.
- Outcome resolution: iterates M1 candles, checks `low/high` against `SL/TP` depending on side; SL assumed if TP not reached within 2 days.
- Costs: commission per trade plus slippage calculated as a percentage of risk amount.
- Metrics:
  - `net_profit_percent = (final_balance - initial_balance) / initial_balance × 100`
  - `win_rate_percent = wins / total_trades × 100`
  - `max_drawdown_percent`: peak-to-trough drawdown over equity curve.
  - `profit_factor = sum(profits) / sum(losses)` (losses as positives).
  - `average_rr_ratio`: average of `|TP - entry| / |entry - SL|`.

## MT5 Integration

- Initialization uses environment variables when available: `MT5_PATH`, `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`.
- Symbol resolution: attempts to find broker-specific variants if exact symbol is missing.
- Inactivity shutdown: MT5 terminates if no WebSocket subscribers for 60 seconds.

## OpenAPI & Documentation

- Export OpenAPI: run `python scripts/export_openapi.py` to generate `docs/openapi.json`.
- The OpenAPI schema includes all REST and WebSocket endpoints documented above.
- `docs/openapi.json` is git-ignored to avoid churn; regenerate as needed.

## Operational Notes

- Signal loop interval: ~5 seconds between iterations, with retries on errors.
- Caching: H1/H4 trend and ATR values cached between iterations; recomputed when timestamps advance.
- Safety checks: minimum historical data length enforced; extreme volatility skipped.
- Pips assumption: `XAUUSD pip = 0.01` used uniformly for SL/TP pip conversions.

## Extensibility & Tuning

- Adjust indicator parameters and weights in `app/config.py` for sensitivity changes.
- Add new indicators by extending `compute_indicators` and `evaluate_signals`, and include them in strategy contributions.
- Modify volatility thresholds or R:R profiles to fit different instruments.
- Extend price action heuristics with additional patterns for domain-specific signals.

## Security & Secrets

- Environment variables for DB: `user`, `password`, `host`, `port`, `dbname` or `DB_USER/DB_PASSWORD/DB_HOST/DB_PORT/DB_NAME`.
- Ensure `.env` is present in deployment environments; `.env` is excluded by `.gitignore`.

## Summary

VediGold AI combines multi-timeframe technical analysis with configurable strategy weighting and volatility-aware trade planning. It provides both real-time signal generation and a robust manual backtesting suite to validate assumptions. All artifacts are persisted to Postgres for analysis, and the FastAPI server offers a clean API surface for integration.