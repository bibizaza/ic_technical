# MARS Momentum Scoring - Aggregation Methods

This document explains the different aggregation methods available for combining the 5 absolute momentum components into a final score.

## Overview

The MARS system computes 5 absolute momentum components:
1. **Pure Momentum** - Blended 12m/6m/3m returns
2. **Trend Smoothness** - Fraction of positive days
3. **Sharpe Ratio** - Risk-adjusted returns
4. **Idiosyncratic Momentum** - Self-driven momentum (residual vs benchmark)
5. **ADX** - Trend strength indicator

Each component is converted to a percentile rank (0-100) based on its 5-year rolling distribution.

## Aggregation Methods

### 1. **Top 2** (Default)
```python
score = generate_spx_score_history(prices_df, agg_method="top2")
```

**Logic**: Average the **top 2** component percentiles at each date.

**Use case**:
- Default method for SPX
- Focuses on the strongest signals
- Adaptive - automatically selects the best performing components
- Robust to weak/noisy components

**Example**: If components are [95, 88, 62, 45, 30], score = (95 + 88) / 2 = 91.5

---

### 2. **Top 3**
```python
score = generate_spx_score_history(prices_df, agg_method="top3")
```

**Logic**: Average the **top 3** component percentiles at each date.

**Use case**:
- More stable than top2 (less sensitive to outliers)
- Good for assets with consistently strong momentum
- Middle ground between top2 and weighted

**Example**: If components are [95, 88, 62, 45, 30], score = (95 + 88 + 62) / 3 = 81.7

---

### 3. **Weighted Average**
```python
# Use default weights
score = generate_spx_score_history(prices_df, agg_method="weighted")

# Use custom weights
custom_weights = {
    "pure": 0.50,    # Pure momentum
    "smooth": 0.20,  # Trend smoothness
    "sharpe": 0.15,  # Sharpe ratio
    "idio": 0.10,    # Idiosyncratic momentum
    "adx": 0.05,     # ADX
}
score = generate_spx_score_history(prices_df, agg_method="weighted", weights=custom_weights)
```

**Default weights**:
```python
DEFAULT_WEIGHTS = {
    "pure": 0.40,    # 40% - Pure momentum (most important)
    "smooth": 0.20,  # 20% - Trend smoothness
    "sharpe": 0.20,  # 20% - Sharpe ratio
    "idio": 0.10,    # 10% - Idiosyncratic momentum
    "adx": 0.10,     # 10% - ADX
}
```

**Use case**:
- When you have domain knowledge about component importance
- More stable than top-k methods
- Can be tuned per asset class

**Example**: With default weights and components [95, 88, 62, 45, 30]:
```
score = 0.40×95 + 0.20×88 + 0.20×62 + 0.10×45 + 0.10×30
     = 38 + 17.6 + 12.4 + 4.5 + 3
     = 75.5
```

---

### 4. **Dynamic LASSO** (Advanced)
```python
# For CSI: Learn weights via walk-forward validation
from mars_engine.lasso_weighting import perform_walk_forward_validation

weights_history = perform_walk_forward_validation(
    prices_df=prices_df,
    raw_components_history=raw_components,  # Raw component time series
    target_col="CSI",
    training_window_years=5,
    testing_window_years=1,
    forward_return_days=63,
)

# Use latest learned weights
latest_weights = weights_history.iloc[-1][['pure', 'smooth', 'sharpe', 'idio', 'adx']].to_dict()
score = generate_csi_score_history(prices_df, agg_method="weighted", weights=latest_weights)
```

**Logic**:
1. Train LASSO regression models on rolling windows
2. Learn which components best predict forward returns
3. Use time-varying weights (adaptive to regime changes)
4. Automatic feature selection (weights can go to zero)

**Use case**:
- **Recommended for CSI** (emerging market with regime changes)
- Assets with time-varying dynamics
- When you want data-driven, adaptive weighting
- Requires sufficient history (5+ years)

**Advantages**:
- Adapts to changing market regimes
- Automatic feature selection via L1 regularization
- Learns from forward return predictions
- More sophisticated than fixed weights

**Disadvantages**:
- More complex implementation
- Requires longer history
- Can overfit if not validated properly
- Walk-forward validation is computationally expensive

---

## Recommendation by Asset

| Asset | Recommended Method | Rationale |
|-------|-------------------|-----------|
| **SPX** | `top2` (default) | Mature market, stable dynamics, strong signals |
| **CSI** | `lasso` (dynamic) | Emerging market, regime changes, adaptive weighting needed |
| **Other Equities** | `top2` or `weighted` | Depends on market maturity |
| **Commodities** | `top3` | More stable, less noise-sensitive |
| **Crypto** | `top2` | High volatility, focus on strongest signals |

---

## Implementation Examples

### Example 1: SPX with default (top2)
```python
from mars_engine import generate_spx_score_history, load_prices_for_mars

prices_df = load_prices_for_mars("data.xlsx")
spx_score = generate_spx_score_history(prices_df)  # Uses top2 by default
print(f"Latest SPX MARS score: {spx_score.iloc[-1]:.1f}")
```

### Example 2: CSI with custom weighted
```python
from mars_engine import generate_csi_score_history, load_prices_for_mars

prices_df = load_prices_for_mars("data.xlsx")

# Emphasize pure momentum for CSI
csi_weights = {
    "pure": 0.50,
    "smooth": 0.15,
    "sharpe": 0.15,
    "idio": 0.10,
    "adx": 0.10,
}

csi_score = generate_csi_score_history(
    prices_df,
    agg_method="weighted",
    weights=csi_weights
)
print(f"Latest CSI MARS score: {csi_score.iloc[-1]:.1f}")
```

### Example 3: Comparing methods
```python
from mars_engine import generate_spx_score_history, load_prices_for_mars
import matplotlib.pyplot as plt

prices_df = load_prices_for_mars("data.xlsx")

# Compute with different methods
top2 = generate_spx_score_history(prices_df, agg_method="top2")
top3 = generate_spx_score_history(prices_df, agg_method="top3")
weighted = generate_spx_score_history(prices_df, agg_method="weighted")

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(top2.index, top2, label="Top 2", linewidth=2)
plt.plot(top3.index, top3, label="Top 3", linewidth=2)
plt.plot(weighted.index, weighted, label="Weighted", linewidth=2)
plt.legend()
plt.title("SPX MARS Score - Aggregation Method Comparison")
plt.ylabel("Score (0-100)")
plt.grid(alpha=0.3)
plt.show()
```

---

## Technical Notes

### Cache Invalidation
The MARS scorer caches results by data fingerprint. When you change `agg_method` or `weights`, the cache key changes automatically, so you'll get the correct recalculated scores.

### Backward Compatibility
All existing code continues to work - default behavior is `agg_method="top2"` which matches the original implementation.

### Performance
- `top2` and `top3`: Fast (simple sorting)
- `weighted`: Fast (dot product)
- `lasso`: Slow (requires training, use cached weights in production)

---

## Future Enhancements

1. **Dynamic LASSO Integration**: Fully integrate walk-forward validation into CSI workflow
2. **Ensemble Methods**: Combine multiple aggregation methods
3. **Regime-Aware Weighting**: Switch methods based on detected market regime
4. **Component Importance Analysis**: Tools to understand which components drive scores
