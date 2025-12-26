# Herculis Assessment

Automatic asset classification system that assigns 5-level assessments (Bearish → Cautious → Neutral → Constructive → Bullish) based on DMAS scores, price structure, and momentum.

## Features

- **Automatic Classification** - Eliminates manual assessment decisions
- **Multi-factor Analysis** - Combines DMAS, price structure, and momentum
- **Transparent Reasoning** - See all adjustments and their rationale
- **Configurable Rules** - Customize thresholds and adjustment logic
- **Batch Processing** - Classify multiple assets efficiently

## Assessment Levels

| Level | Value | Color | Description |
|-------|-------|-------|-------------|
| **Bullish** | +2 | 🟢 Green | Strong positive outlook |
| **Constructive** | +1 | 🟩 Light Green | Moderate positive outlook |
| **Neutral** | 0 | 🟡 Gold | No clear direction |
| **Cautious** | -1 | 🟠 Orange | Moderate negative outlook |
| **Bearish** | -2 | 🔴 Red | Strong negative outlook |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from herculis_assessment import classify
import pandas as pd

# Prepare price data (needs 200+ days)
prices = pd.Series([...])  # Your price series

# Current DMAS and 1-week-ago DMAS
dmas_current = 72.5
dmas_1w_ago = 68.0

# Classify
result = classify(
    ticker='SPX Index',
    prices=prices,
    dmas=dmas_current,
    dmas_1w=dmas_1w_ago
)

print(f"Assessment: {result.assessment.name}")
print(f"Adjustments: {result.adjustments}")
```

### Batch Classification

```python
from herculis_assessment import classify_all

tickers = ['SPX Index', 'NKY Index', 'CSI 300 Index']
prices_dict = {ticker: pd.Series([...]) for ticker in tickers}
dmas_dict = {'SPX Index': 72.5, 'NKY Index': 28.3, ...}
dmas_1w_dict = {'SPX Index': 68.0, 'NKY Index': 35.0, ...}

results_df = classify_all(tickers, prices_dict, dmas_dict, dmas_1w_dict)
print(results_df)
```

## Classification Logic

### Step 1: Base Assessment (from DMAS)

DMAS score determines the starting assessment:

| DMAS Range | Base Assessment |
|------------|----------------|
| ≥ 70 | Bullish |
| 55-69 | Constructive |
| 45-54 | Neutral |
| 30-44 | Cautious |
| < 30 | Bearish |

### Step 2: Downgrade Rules (applied in order)

1. **Price < 200d MA** → Hard cap at Cautious (cannot be above Cautious)
2. **Price < 100d MA** → Downgrade 1 level
3. **Price < 50d MA** → Downgrade 1 level (stacks with #2)
4. **DMAS dropped ≥10 pts WoW** → Downgrade 1 level

### Step 3: Upgrade Rules (applied after downgrades)

1. **Perfect structure (Price > 50d > 100d > 200d) AND DMAS ≥65** → Upgrade to Bullish
2. **DMAS gained ≥10 pts WoW** → Upgrade 1 level

### Step 4: Clamping

Final assessment is clamped to [-2, +2] range (Bearish to Bullish).

## Examples

### Example 1: Strong Uptrend

```
DMAS: 75 → Base: Bullish
Price > 50d > 100d > 200d (perfect structure)
WoW change: +5 pts

Adjustments: None
Final: Bullish ✓
```

### Example 2: High DMAS but Broken Structure

```
DMAS: 72 → Base: Bullish
Price < 200d MA

Adjustments:
  1. Price < 200d MA → Capped at Cautious
  2. Price < 100d MA → Downgrade 1 level
  3. Price < 50d MA → Downgrade 1 level

Final: Bearish (capped at minimum)
```

### Example 3: Moderate DMAS with Perfect Structure

```
DMAS: 66 → Base: Constructive
Price > 50d > 100d > 200d (perfect structure)
DMAS ≥ 65

Adjustments:
  1. Perfect structure + DMAS 66 → Upgrade to Bullish

Final: Bullish ✓
```

### Example 4: Improving Momentum

```
DMAS: 55 → Base: Constructive
WoW change: +12 pts

Adjustments:
  1. DMAS gained 12 pts WoW → Upgrade 1 level

Final: Bullish
```

## Configuration

All thresholds can be customized in `src/config.py`:

```python
# DMAS thresholds for base classification
DMAS_THRESHOLDS = {
    'bullish': 70,
    'constructive': 55,
    'neutral': 45,
    'cautious': 30,
}

# Moving average periods
MA_PERIODS = {
    'short': 50,
    'medium': 100,
    'long': 200
}

# Adjustment parameters
ADJUSTMENT_CONFIG = {
    'wow_change_threshold': 10,        # ±10 points triggers adjustment
    'structure_upgrade_min_dmas': 65,  # Min DMAS for structure upgrade
    'close_to_ma_threshold': 2.0,      # Within 2% = "close to MA"
}
```

## Project Structure

```
herculis-assessment/
├── src/
│   ├── __init__.py            # Package exports
│   ├── config.py              # All thresholds and parameters
│   ├── structure.py           # MA structure analysis
│   └── classifier.py          # Main classification logic
├── tests/
│   └── test_classifier.py     # Comprehensive test suite
├── example_integration.py     # Integration examples
├── requirements.txt
└── README.md
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Run specific test class:

```bash
python -m pytest tests/test_classifier.py::TestDowngradeRules -v
```

## API Reference

### `classify(ticker, prices, dmas, dmas_1w)`

Classify a single asset into assessment level.

**Parameters:**
- `ticker` (str): Asset identifier
- `prices` (pd.Series): Price series (needs 200+ data points)
- `dmas` (float): Current DMAS score (0-100)
- `dmas_1w` (float, optional): DMAS from 1 week ago

**Returns:**
```python
ClassificationResult(
    ticker='SPX Index',
    assessment=Assessment.BULLISH,          # Final level
    base_assessment=Assessment.CONSTRUCTIVE, # Before adjustments
    dmas=72.5,
    dmas_wow_change=4.5,
    structure=StructureAnalysis(...),       # MA structure details
    adjustments=[                           # List of adjustments
        "Perfect structure + DMAS 72.5 → Upgrade to Bullish"
    ],
    description="Base: Constructive (DMAS 72.5) → 1 adjustment(s)"
)
```

### `classify_all(tickers, prices_dict, dmas_dict, dmas_1w_dict)`

Classify multiple assets.

**Parameters:**
- `tickers` (list[str]): List of asset identifiers
- `prices_dict` (dict): Ticker → price series mapping
- `dmas_dict` (dict): Ticker → current DMAS mapping
- `dmas_1w_dict` (dict, optional): Ticker → 1-week-ago DMAS mapping

**Returns:** pd.DataFrame with columns:
- `ticker`: Asset identifier
- `assessment`: Assessment label (string)
- `assessment_value`: Integer value (-2 to +2)
- `base_assessment`: Base assessment before adjustments
- `dmas`: Current DMAS score
- `dmas_wow_change`: Week-over-week DMAS change
- `price_vs_50d`: % distance from 50d MA
- `price_vs_100d`: % distance from 100d MA
- `price_vs_200d`: % distance from 200d MA
- `adjustments_count`: Number of adjustments applied
- `description`: Summary text

### `analyze_structure(prices)`

Analyze price structure relative to moving averages.

**Parameters:**
- `prices` (pd.Series): Price series (needs 200+ data points)

**Returns:**
```python
StructureAnalysis(
    score=2,                    # -1 to +2
    price_vs_50d=5.2,           # % above 50d MA
    price_vs_100d=8.1,          # % above 100d MA
    price_vs_200d=12.3,         # % above 200d MA
    above_50d=True,
    above_100d=True,
    above_200d=True,
    perfect_structure=True,     # Price > 50d > 100d > 200d
    description="Perfect uptrend structure (Price > 50d > 100d > 200d)"
)
```

## Streamlit Integration

```python
# In your Streamlit app:
from herculis_assessment import classify, ASSESSMENT_LABELS, ASSESSMENT_COLORS

result = classify('SPX Index', prices, dmas_current, dmas_1w_ago)

# Color-coded display
st.markdown(
    f"<h2 style='color: {ASSESSMENT_COLORS[result.assessment]};'>"
    f"{ASSESSMENT_LABELS[result.assessment]}</h2>",
    unsafe_allow_html=True
)

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("DMAS", f"{result.dmas:.1f}", f"{result.dmas_wow_change:+.1f}")
with col2:
    st.metric("vs 50d MA", f"{result.structure.price_vs_50d:+.1f}%")
with col3:
    st.metric("vs 200d MA", f"{result.structure.price_vs_200d:+.1f}%")

# Show adjustments
if result.adjustments:
    with st.expander("Adjustment Details"):
        for adj in result.adjustments:
            st.write(f"• {adj}")
```

## Design Principles

1. **Conservative Downgrades** - Multiple structural issues compound (can downgrade multiple levels)
2. **Selective Upgrades** - Only exceptional conditions trigger upgrades (perfect structure + high DMAS)
3. **Structure Priority** - Price below 200d MA hard-caps assessment regardless of DMAS
4. **Momentum Matters** - Large WoW changes (±10 pts) trigger adjustments
5. **Transparency** - All adjustments are logged with clear reasoning

## Common Patterns

### Pattern 1: High DMAS, Broken Structure → Cautious/Bearish
- DMAS suggests strength, but price action shows weakness
- Multiple downgrades from MA violations
- **Interpretation:** Distribution phase, be cautious

### Pattern 2: Moderate DMAS, Perfect Structure → Bullish
- DMAS moderate but structure is pristine
- Upgrade from perfect alignment
- **Interpretation:** Healthy trend, follow structure

### Pattern 3: Neutral DMAS, Improving Momentum → Constructive
- Base assessment neutral, but momentum turning positive
- Upgrade from WoW gain
- **Interpretation:** Early recovery signal

## License

Internal use only - Herculis Partners

## Authors

Developed for Herculis Partners investment process automation.
