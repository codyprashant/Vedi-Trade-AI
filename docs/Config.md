# Strategy Configuration UI Prompt (Gold Strategy)

Design a configuration UI to manage strategies and their indicator settings stored in the database. This UI drives the “Gold Strategy” but supports multiple strategies. Use the endpoints below to read and update configs. The signal engine and backtesting read directly from these configs.

## Goals

- Manage strategies: list, view details, activate one as the current strategy
- Edit indicator parameters under a specific strategy
- Edit strategy weights for contributions
- Configure run frequency (seconds) for the strategy engine
- Configure signal efficiency threshold (0–1 scale, e.g., 0.9 for 90%)

## Screens

- Strategy List
  - Shows: `name`, `is_active`, `signal_threshold`, `run_interval_seconds`
  - Actions: View, Activate

- Strategy Detail (Gold Strategy)
  - Tabs or sections:
    - Overview: `name`, `description`, `is_active`, `timeframes`
    - Indicator Params: forms for RSI, MACD, SMA, EMA, BBANDS, STOCH, ATR, PRICE_ACTION
    - Weights: key-value editor for contribution weights (e.g., `RSI`, `MACD`, `SMA_EMA`, `MTF`, `ATR_STABILITY`, `PRICE_ACTION`)
    - Schedule: `run_interval_seconds` (number input)
    - Threshold: `signal_threshold` (0–1 float; UI may display percentage and convert to float)

## Field Guide (Common)

- `timeframes`: array of strings, e.g., `["M15","H1","H4"]`
- `signal_threshold`: float 0–1; 0.9 means 90% efficiency
- `run_interval_seconds`: integer seconds between engine iterations
- `is_active`: boolean marking the strategy used by the engine

## Indicator Params (Examples)

- RSI: `{ "period": 14, "overbought": 70, "oversold": 30 }`
- MACD: `{ "fast": 12, "slow": 26, "signal": 9 }`
- SMA: `{ "short": 20, "long": 50 }`
- EMA: `{ "short": 20, "long": 50 }`
- BBANDS: `{ "length": 20, "stddev": 2 }`
- STOCH: `{ "k": 14, "d": 3, "overbought": 80, "oversold": 20 }`
- ATR: `{ "length": 14 }`
- PRICE_ACTION: `{ "engulfing_lookback": 5, "pinbar_lookback": 5 }`

## Weights (Examples)

- `{ "RSI": 0.15, "MACD": 0.15, "STOCH": 0.1, "BBANDS": 0.1, "SMA_EMA": 0.2, "MTF": 0.15, "ATR_STABILITY": 0.075, "PRICE_ACTION": 0.075 }`

## API Endpoints

- List strategies
  - GET `/api/config/strategies`
  - Response: `{ "strategies": [{ "id": 1, "name": "Gold Strategy", "is_active": true, "signal_threshold": 0.9, "run_interval_seconds": 60 }] }`

- Get strategy details
  - GET `/api/config/strategies/{strategy_id}`
  - Response shape:
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

- Update indicator params
  - PATCH `/api/config/strategies/{strategy_id}/indicator/{indicator_name}`
  - Body: `{ "params": { /* indicator-specific params */ } }`
  - Response: `{ "status": "ok" }`

- Update weights
  - PATCH `/api/config/strategies/{strategy_id}/weights`
  - Body: `{ "weights": { /* contribution weights */ } }`
  - Response: `{ "status": "ok" }`

- Update schedule (run frequency)
  - PATCH `/api/config/strategies/{strategy_id}/schedule`
  - Body: `{ "run_interval_seconds": 60 }`
  - Response: `{ "status": "ok" }`

- Update signal threshold
  - PATCH `/api/config/strategies/{strategy_id}/threshold`
  - Body: `{ "signal_threshold": 0.9 }`
  - Response: `{ "status": "ok" }`

- Activate strategy
  - POST `/api/config/strategies/{strategy_id}/activate`
  - Response: `{ "status": "ok" }`

## UI Implementation Notes

- Represent `signal_threshold` in the UI as a percentage slider; convert to float for backend (e.g., 90% → 0.9)
- Validate ranges (RSI thresholds 0–100, STOCH 0–100, BBANDS stddev typically 1–3)
- Use optimistic UI updates and refresh strategy detail after patches
- Prevent editing when strategy is not found; show clear error messaging

## Quick UI Checklist

- Strategy list with activate action
- Strategy detail with tabs: Overview, Indicator Params, Weights, Schedule, Threshold
- Form editors for each indicator (field types: number inputs)
- Save and revert buttons per section
- Global success/error toasts on PATCH/POST