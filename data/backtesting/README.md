# Backtesting CSV Data

Place historical OHLCV CSV files here for manual backtests.

## File naming

- Per symbol and timeframe: `data/backtesting/<SYMBOL>_<TIMEFRAME>.csv`
  - Examples: `XAUUSD_15m.csv`, `XAUUSD_1h.csv`, `BTCUSD_15m.csv`

## Required columns

CSV must include at least these columns (case-insensitive accepted):

- `time` (ISO datetime; timezone optional, UTC preferred)
- `open`
- `high`
- `low`
- `close`
- `volume`

Accepted aliases will be normalized automatically if present:

- `Datetime`, `date` → `time`
- `Open` → `open`, `High` → `high`, `Low` → `low`, `Close` → `close`, `Volume` → `volume`
- `o` → `open`, `h` → `high`, `l` → `low`, `c` → `close`, `v` → `volume`

## Notes

- For `source_mode=csv` runs, the primary timeframe CSV is required.
- H1 and H4 are derived by resampling the primary CSV if not provided separately.
- Ensure data is continuous and sorted; gaps will reduce signal counts.