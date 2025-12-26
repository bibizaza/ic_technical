# Herculis Technical Score

Python module to compute Technical Scores directly from price data, replacing Bloomberg BQL computation.

## Features

- **Save Bloomberg computational power** - Compute indicators locally from price data
- **Full control** - Understand and modify calculations
- **Bug fixes** - Corrected Parabolic SAR logic (was inverted in BQL)
- **Transparency** - See component scores and raw indicator values

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from herculis_technical_score import compute_technical_score
import pandas as pd

# Prepare price data
prices_df = pd.DataFrame({
    'Date': [...],
    'Price': [...],
    'High': [...],
    'Low': [...]
})

# Compute score
result = compute_technical_score(prices_df, ticker="SPX")

print(f"Technical Score: {result['technical_score']}")  # 0-100
print(f"Components: {result['components']}")  # Individual indicator scores
```

### Streamlit Integration

```python
from herculis_technical_score import compute_all_scores

# Compute scores for all tickers in Excel file
scores_df = compute_all_scores('input_ic_v2_global_ticker_V3.xlsx')

# Display in Streamlit
st.dataframe(scores_df)
```

## Technical Indicators

The module computes 8 technical indicators with configured weights:

| Indicator | Weight | Type | Description |
|-----------|--------|------|-------------|
| **SMA** | 15% | Trend-following | Simple Moving Averages (5,10,20,50,100,200) |
| **EMA** | 15% | Trend-following | Exponential Moving Averages (10,20,50,100,200) |
| **RSI** | 10% | **Contrarian** | Relative Strength Index (14-period) |
| **DMI/ADX** | 15% | Trend-following | Directional Movement Index |
| **Parabolic SAR** | 10% | Trend-following | Stop and Reverse (FIXED logic) |
| **MACD** | 15% | Trend-following | Moving Average Convergence Divergence |
| **Stochastics** | 10% | Momentum | %K and %D oscillator |
| **MAE** | 10% | **Contrarian** | Moving Average Envelope |

### Scoring Logic

Each indicator produces a score from 0.0 to 1.0:
- **1.0** = Most bullish
- **0.5** = Neutral
- **0.0** = Most bearish

Indicators are combined using configured weights to produce a final score (0-100).

### Key Fixes

**Parabolic SAR Logic (CORRECTED):**
- BQL had inverted logic (contrarian instead of trend-following)
- Our implementation uses correct trend-following logic:
  - SAR < Price = Uptrend = Bullish (score +0.75)
  - SAR > Price = Downtrend = Bearish (score 0.0)

## Configuration

All parameters can be customized in `config.py`:

```python
# Example: Change RSI period
RSI_PERIOD = 20  # Default: 14

# Example: Adjust component weights
WEIGHTS = {
    'sma': 0.20,      # Increase SMA weight
    'rsi': 0.05,      # Decrease RSI weight
    # ... must sum to 1.0
}
```

## Project Structure

```
herculis-technical-score/
├── config.py                      # All parameters and weights
├── src/
│   ├── __init__.py
│   ├── scoring.py                 # Main module
│   ├── utils.py                   # Helper functions
│   └── indicators/
│       ├── __init__.py
│       ├── moving_averages.py     # SMA, EMA
│       ├── rsi.py                 # RSI
│       ├── dmi.py                 # DMI, ADX, ADXR
│       ├── parabolic_sar.py       # Parabolic SAR
│       ├── macd.py                # MACD
│       ├── stochastics.py         # Stochastic Oscillator
│       └── mae.py                 # Moving Average Envelope
├── tests/
│   └── test_scoring.py            # Test suite
└── requirements.txt

## Testing

Run tests to validate against BQL scores:

```bash
python -m pytest tests/
```

## API Reference

### `compute_technical_score(prices_df, ticker, include_components=True)`

Compute technical score for a single ticker.

**Parameters:**
- `prices_df` (pd.DataFrame): Price data with columns ['Date', 'Price', 'High', 'Low']
- `ticker` (str): Asset identifier
- `include_components` (bool): Include component breakdowns

**Returns:**
```python
{
    'ticker': 'SPX',
    'technical_score': 75.5,      # Final score (0-100)
    'components': {               # Component scores (0.0-1.0)
        'sma': 0.85,
        'ema': 0.80,
        'rsi': 0.50,             # Neutral (contrarian)
        'dmi': 0.75,
        'parabolic': 0.75,
        'macd': 0.75,
        'stochastics': 0.60,
        'mae': 0.70
    },
    'raw_indicators': {          # Raw indicator values
        'rsi': 55.3,
        'macd': {...},
        ...
    }
}
```

### `compute_all_scores(excel_path, tickers=None)`

Compute scores for multiple tickers from Excel file.

**Parameters:**
- `excel_path` (str): Path to Excel file
- `tickers` (list, optional): List of tickers (auto-detect if None)

**Returns:** pd.DataFrame with columns ['ticker', 'technical_score', 'sma_score', ...]

## License

Internal use only - Herculis Partners

## Authors

Developed for Herculis Partners technical analysis workflow.
```
