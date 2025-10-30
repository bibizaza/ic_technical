# Component Calculation Comparison: Reference vs Implementation

## 1. Pure Momentum

### Reference (pure_momentum.py):
```python
weights = {'12m': 0.4, '6m': 0.4, '3m': 0.2}
weighted = sum(returns[p] * weights[p] for p in valid)
return weighted / total_w  # Normalized sum
```

### My Implementation:
```python
r12m + 0.5 * r6m + 0.25 * r3m  # Weights: 1.0, 0.5, 0.25
```

### Status: ❌ WRONG - Different weights!

---

## 2. Trend Smoothness

### Reference (trend_smoothness.py):
```python
returns = data.pct_change().iloc[-window_days:]
positive_days = (returns > 0).sum()
return float(positive_days) / window_days
```

### My Implementation:
```python
daily = close.pct_change()
return daily.gt(0).rolling(126, min_periods=126).mean()
```

### Status: ✅ Looks OK (both compute fraction of positive days)

---

## 3. Sharpe Ratio

### Reference (risk_adjusted_return.py):
```python
returns = data.pct_change().iloc[-window_days:]
mean = returns.mean()
std = returns.std()
sharpe = (mean * 252) / (std * np.sqrt(252))
```

### My Implementation:
```python
lr = np.log(close / close.shift(1))
mean = lr.rolling(126).mean()
std  = lr.rolling(126).std()
return (mean / std) * np.sqrt(252.0)
```

### Status: ⚠️ DIFFERENT - I use log returns, reference uses simple returns!

---

## 4. Idiosyncratic Momentum

### Reference (idiosyncratic_momentum.py):
```python
# 12-month regression window
lb_days = int(lookback_months * 21)  # Default 12 * 21 = 252
rw_days = int(residual_window_months * 21)  # Default 6 * 21 = 126

# Regression on returns
returns = data[[target] + basket_tickers].pct_change().dropna()
reg_data = returns.iloc[-lb_days:]
bench_returns = reg_data[basket_tickers].mean(axis=1)
asset_returns = reg_data[target]

model = LinearRegression().fit(X, y)
resid = y - model.predict(X)

# Cumulative return of last 6 months of residuals
resid_window = resid_series.iloc[-rw_days:]
cumulative_return = (1 + resid_window).prod() - 1
```

### My Implementation:
```python
# Rolling OLS over 126 days
ra = np.log(close / close.shift(1))
rb = np.log(benchmark / benchmark.shift(1))
# ... rolling covariance calculation ...
resid_lr = ra - (alpha + beta * rb)
resid_simple = np.expm1(resid_lr)
idio = (1.0 + resid_simple).cumprod() - 1.0
```

### Status: ⚠️ DIFFERENT - I use cumprod over full history, reference uses last 6 months only!

---

## 5. ADX

### Reference (trend_strength_adx.py):
```python
# Standard ADX(14) calculation
# Uses RMA (Wilder smoothing)
```

### My Implementation:
```python
# Uses simple rolling mean, not RMA
adx = dx.rolling(period, min_periods=period).mean()
```

### Status: ⚠️ DIFFERENT - Should use RMA/EMA, not simple mean!

---

## Summary of Issues:

1. ❌ **Pure Momentum**: Wrong weights (1.0/0.5/0.25 vs 0.4/0.4/0.2)
2. ⚠️ **Sharpe**: Log returns vs simple returns
3. ❌ **Idiosyncratic**: Cumulative over full history vs last 6 months only
4. ⚠️ **ADX**: Simple mean vs RMA smoothing

These discrepancies would cause LASSO to learn completely different patterns!
