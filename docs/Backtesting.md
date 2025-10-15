# VediTrading AI - Backtesting System Documentation

## Overview

The VediTrading AI backtesting system provides comprehensive historical testing capabilities for trading strategies using the unified `BacktestEngine`. The system simulates real trading conditions by generating signals on historical data and executing trades with realistic market conditions, slippage, and commission costs.

## Architecture

### Unified BacktestEngine

The `BacktestEngine` class serves as the central component for all backtesting operations, providing:

- **Historical Data Fetching**: Retrieves OHLCV data from Yahoo Finance
- **Signal Generation**: Uses the same strategy logic as live trading
- **Trade Simulation**: Realistic execution with slippage and commission
- **Performance Analysis**: Comprehensive metrics calculation
- **Data Persistence**: Stores results in PostgreSQL for analysis

### Key Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    BacktestEngine Workflow                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Initialize Parameters                                       │
│     ├── Strategy ID, Symbol, Date Range                        │
│     ├── Investment Amount, Timeframe                           │
│     └── Risk Management Settings                               │
│                                                                 │
│  2. Fetch Market Data                                          │
│     ├── Yahoo Finance OHLCV Data                              │
│     ├── Data Validation & Cleaning                            │
│     └── Timeframe Conversion                                  │
│                                                                 │
│  3. Generate Signals                                           │
│     ├── Technical Indicator Calculation                       │
│     ├── Strategy Evaluation                                   │
│     ├── Multi-Timeframe Confirmation                          │
│     └── Signal Validation                                     │
│                                                                 │
│  4. Simulate Trades                                            │
│     ├── Position Sizing Calculation                           │
│     ├── Entry/Exit Price Determination                        │
│     ├── Stop Loss & Take Profit Logic                         │
│     ├── Slippage & Commission Application                     │
│     └── Trade Result Recording                                │
│                                                                 │
│  5. Calculate Performance Metrics                              │
│     ├── Total Return & ROI                                    │
│     ├── Win Rate & Profit Factor                              │
│     ├── Maximum Drawdown                                      │
│     ├── Sharpe Ratio                                          │
│     └── Efficiency Percentage                                 │
│                                                                 │
│  6. Store Results                                              │
│     ├── Backtest Summary (public.backtests)                   │
│     ├── Individual Signals (public.backtest_signals)          │
│     └── Performance Metrics                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Database Schema

### Backtests Table
```sql
CREATE TABLE public.backtests (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    investment DECIMAL(15,2) NOT NULL,
    total_return_pct DECIMAL(8,4) NOT NULL,
    efficiency_pct DECIMAL(8,4) NOT NULL,
    win_count INTEGER DEFAULT 0,
    loss_count INTEGER DEFAULT 0,
    open_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Backtest Signals Table
```sql
CREATE TABLE public.backtest_signals (
    id SERIAL PRIMARY KEY,
    backtest_id INTEGER REFERENCES public.backtests(id),
    timestamp TIMESTAMPTZ NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DECIMAL(10,5) NOT NULL,
    exit_price DECIMAL(10,5),
    profit_pct DECIMAL(8,4),
    result VARCHAR(20),
    confidence DECIMAL(5,2),
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## API Endpoints

### 1. Run Backtest
**Endpoint**: `POST /api/backtest/run`

**Parameters**:
- `strategy_id` (int): Strategy identifier
- `symbol` (str): Trading symbol (e.g., "XAUUSD")
- `start_date` (str): Start date (YYYY-MM-DD)
- `end_date` (str): End date (YYYY-MM-DD)
- `investment` (float): Initial investment amount (default: 10000)
- `timeframe` (str): Analysis timeframe (default: "15m")

**Example Request**:
```bash
curl -X POST "http://localhost:8001/api/backtest/run?strategy_id=1&symbol=XAUUSD&start_date=2024-01-01&end_date=2024-01-31&investment=10000&timeframe=1h"
```

**Response**:
```json
{
  "success": true,
  "backtest": {
    "backtest_id": 123,
    "strategy_id": 1,
    "symbol": "XAUUSD",
    "timeframe": "1h",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T00:00:00Z",
    "investment": 10000.0,
    "total_return_pct": 5.75,
    "efficiency_pct": 85.5,
    "win_count": 12,
    "loss_count": 3,
    "open_count": 0
  }
}
```

### 2. Calculate ROI
**Endpoint**: `GET /api/backtest/roi`

**Parameters**:
- `backtest_id` (int): Backtest identifier
- `amount` (float): Investment amount for projection

**Example Request**:
```bash
curl "http://localhost:8001/api/backtest/roi?backtest_id=123&amount=5000"
```

**Response**:
```json
{
  "backtest_id": 123,
  "initial": 5000.0,
  "final": 5287.5,
  "return_pct": 5.75,
  "profit": 287.5,
  "symbol": "XAUUSD",
  "timeframe": "1h",
  "efficiency_pct": 85.5
}
```

### 3. List Backtests
**Endpoint**: `GET /api/backtest/list`

**Response**:
```json
{
  "count": 5,
  "backtests": [
    {
      "id": 123,
      "symbol": "XAUUSD",
      "timeframe": "1h",
      "start_date": "2024-01-01T00:00:00Z",
      "end_date": "2024-01-31T00:00:00Z",
      "investment": 10000.0,
      "total_return_pct": 5.75,
      "efficiency_pct": 85.5,
      "created_at": "2024-02-01T10:30:00Z"
    }
  ]
}
```

### 4. Get Backtest Results
**Endpoint**: `GET /api/backtest/{backtest_id}/results`

**Response**:
```json
{
  "summary": {
    "backtest_id": 123,
    "total_return_pct": 5.75,
    "efficiency_pct": 85.5,
    "win_count": 12,
    "loss_count": 3
  },
  "signals": [
    {
      "timestamp": "2024-01-05T10:00:00Z",
      "side": "buy",
      "entry_price": 2000.0,
      "exit_price": 2020.0,
      "profit_pct": 1.0,
      "result": "profit",
      "confidence": 85.0,
      "reason": "Strong momentum signal"
    }
  ]
}
```

## Usage Examples

### Python SDK Usage

```python
from backtest.backtest_engine import BacktestEngine

# Initialize backtest engine
engine = BacktestEngine(
    strategy_id=1,
    symbol="XAUUSD",
    start_date="2024-01-01",
    end_date="2024-01-31",
    investment=10000,
    timeframe="1h"
)

# Run the backtest
results = engine.run_backtest()

print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Efficiency: {results['efficiency_pct']:.2f}%")
print(f"Win Rate: {results['win_count']}/{results['win_count'] + results['loss_count']}")
```

### Command Line Usage

```bash
# Run a backtest via API
curl -X POST "http://localhost:8001/api/backtest/run" \
  -G \
  -d "strategy_id=1" \
  -d "symbol=XAUUSD" \
  -d "start_date=2024-01-01" \
  -d "end_date=2024-01-31" \
  -d "investment=10000" \
  -d "timeframe=1h"

# Get ROI projection
curl "http://localhost:8001/api/backtest/roi?backtest_id=123&amount=5000"

# List all backtests
curl "http://localhost:8001/api/backtest/list"
```

## Performance Metrics

### Key Metrics Calculated

1. **Total Return Percentage**: Overall profit/loss as percentage of initial investment
2. **Efficiency Percentage**: Ratio of profitable trades to total trades
3. **Win/Loss Counts**: Number of profitable vs losing trades
4. **Average Profit per Trade**: Mean profit across all trades
5. **Maximum Drawdown**: Largest peak-to-trough decline
6. **Sharpe Ratio**: Risk-adjusted return metric

### Calculation Formulas

```python
# Total Return Percentage
total_return_pct = ((final_balance - initial_investment) / initial_investment) * 100

# Efficiency Percentage  
efficiency_pct = (win_count / total_trades) * 100

# Profit Factor
profit_factor = total_profits / total_losses

# Sharpe Ratio
sharpe_ratio = (mean_return - risk_free_rate) / std_deviation_returns
```

## Risk Management

### Position Sizing
The system uses a risk-based position sizing algorithm:

```python
risk_amount = account_balance * (risk_per_trade_percent / 100)
position_size = risk_amount / stop_loss_distance_pips
```

### Stop Loss & Take Profit
- **Dynamic SL/TP**: Based on ATR (Average True Range) multiples
- **Risk-Reward Ratio**: Configurable target ratios (typically 1:2 or 1:3)
- **Volatility Adjustment**: SL/TP distances adapt to market volatility

### Slippage & Commission
- **Slippage**: Configurable slippage simulation (default: 1-2 pips)
- **Commission**: Broker commission costs included in calculations
- **Spread**: Bid-ask spread consideration for realistic execution

## Testing & Validation

### Comprehensive Test Suite

The backtesting system includes 6 comprehensive test cases:

1. **Signal Generation Consistency**: Validates signal generation logic
2. **Trade Simulation (Profit)**: Tests profitable trade scenarios
3. **Trade Simulation (Loss)**: Tests loss scenarios and risk management
4. **Efficiency Calculation**: Validates performance metric calculations
5. **ROI Endpoint**: Tests API endpoint functionality
6. **Storage & Retrieval**: Validates data persistence

### Running Tests

```bash
# Run all backtest tests
python -m pytest test/test_backtest_engine.py -v

# Run specific test
python -m pytest test/test_backtest_engine.py::TestBacktestEngine::test_backtest_signal_generation_consistency -v
```

## Best Practices

### Data Quality
- **Historical Data Validation**: Ensure complete OHLCV data coverage
- **Missing Data Handling**: Skip periods with insufficient data
- **Outlier Detection**: Filter unrealistic price movements

### Strategy Validation
- **Out-of-Sample Testing**: Reserve data for validation
- **Walk-Forward Analysis**: Progressive testing methodology
- **Multiple Timeframes**: Test across different market conditions

### Performance Analysis
- **Statistical Significance**: Ensure sufficient trade sample size
- **Market Regime Analysis**: Test across different market conditions
- **Correlation Analysis**: Understand strategy behavior patterns

## Troubleshooting

### Common Issues

**No Market Data Available**:
- Verify symbol format and availability
- Check date range validity
- Ensure Yahoo Finance connectivity

**Signal Generation Errors**:
- Validate strategy configuration
- Check indicator parameter settings
- Review error logs for specific issues

**Performance Calculation Issues**:
- Verify trade execution logic
- Check position sizing calculations
- Validate SL/TP price levels

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

engine = BacktestEngine(...)
results = engine.run_backtest()
```

## Future Enhancements

### Planned Features
- **Multi-Asset Portfolio Testing**: Test strategies across multiple symbols
- **Monte Carlo Simulation**: Statistical robustness testing
- **Optimization Engine**: Automated parameter optimization
- **Advanced Metrics**: Additional performance and risk metrics
- **Visualization Dashboard**: Interactive backtest result analysis

### Integration Opportunities
- **Live Trading Integration**: Seamless transition from backtest to live
- **Risk Management System**: Advanced position sizing and risk controls
- **Alert System**: Automated notifications for backtest completion
- **Reporting Engine**: Automated backtest report generation