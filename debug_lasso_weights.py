#!/usr/bin/env python3
"""
Debug script to inspect actual LASSO weights being used for CSI.
"""
import pandas as pd
from mars_engine.data_loader import load_prices_for_mars
from mars_engine.csi_lasso_scorer import _train_csi_lasso_weights_cached

# Load data
excel_path = "data/data.xlsx"
print("Loading price data...")
prices_df = load_prices_for_mars(excel_path)
print(f"✓ Loaded {len(prices_df)} rows\n")

# Train LASSO weights
print("Training LASSO weights with walk-forward validation...")
lasso_weights_df = _train_csi_lasso_weights_cached(excel_path)

if lasso_weights_df.empty:
    print("❌ ERROR: LASSO weights DataFrame is EMPTY!")
    print("This means the fallback to DEFAULT_WEIGHTS is being triggered.")
    exit(1)

print(f"✓ LASSO training completed: {len(lasso_weights_df)} folds\n")

# Display all folds
print("=" * 80)
print("LASSO WEIGHTS HISTORY (All Folds)")
print("=" * 80)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:.6f}')
print(lasso_weights_df)
print()

# Latest fold weights
print("=" * 80)
print("LATEST FOLD WEIGHTS (Currently Applied)")
print("=" * 80)
latest_weights = lasso_weights_df.iloc[-1]
print(latest_weights)
print()

# Check if weights are all zeros
total_weight = abs(latest_weights[['pure', 'smooth', 'sharpe', 'idio', 'adx']]).sum()
print(f"\nTotal absolute weight sum: {total_weight:.6f}")

if total_weight < 0.001:
    print("⚠️  WARNING: Weights are effectively ZERO!")
    print("This will trigger DEFAULT_WEIGHTS fallback in compute_csi_score_with_lasso()")
else:
    print("✓ Non-zero weights detected")

# Check for normalization
normalized_sum = sum([
    latest_weights.get('pure', 0.0),
    latest_weights.get('smooth', 0.0),
    latest_weights.get('sharpe', 0.0),
    latest_weights.get('idio', 0.0),
    latest_weights.get('adx', 0.0)
])
print(f"Sum of weights (before normalization): {normalized_sum:.6f}")
