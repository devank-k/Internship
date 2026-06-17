import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.datasets import get_rdataset
 
# ─────────────────────────────────────────────────────────────────
# DATASET: AirPassengers — classic multiplicative seasonal TS
# Monthly airline passengers 1949–1960
# ─────────────────────────────────────────────────────────────────
ap = get_rdataset('AirPassengers')
airline = pd.Series(
    ap.data['value'].values,
    index=pd.date_range('1949-01', periods=144, freq='ME')
)
airline.name = 'Passengers'
 
# ─────────────────────────────────────────────────────────────────
# ADDITIVE DECOMPOSITION
# ─────────────────────────────────────────────────────────────────
add_decomp = seasonal_decompose(airline, model='additive', period=12)
 
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Additive Decomposition — Airline Passengers', fontsize=14, fontweight='bold')
 
airline.plot(ax=axes[0], color='steelblue'); axes[0].set_title('Original')
add_decomp.trend.plot(ax=axes[1], color='darkorange'); axes[1].set_title('Trend')
add_decomp.seasonal.plot(ax=axes[2], color='green'); axes[2].set_title('Seasonality')
add_decomp.resid.plot(ax=axes[3], color='red', style='o', markersize=2)
axes[3].axhline(0, linestyle='--', color='black', linewidth=0.8)
axes[3].set_title('Residuals')
 
plt.tight_layout()
plt.show()
 
# ─────────────────────────────────────────────────────────────────
# MULTIPLICATIVE DECOMPOSITION
# ─────────────────────────────────────────────────────────────────
mul_decomp = seasonal_decompose(airline, model='multiplicative', period=12)
 
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Multiplicative Decomposition — Airline Passengers', fontsize=14, fontweight='bold')
 
airline.plot(ax=axes[0], color='steelblue'); axes[0].set_title('Original')
mul_decomp.trend.plot(ax=axes[1], color='darkorange'); axes[1].set_title('Trend')
mul_decomp.seasonal.plot(ax=axes[2], color='green'); axes[2].set_title('Seasonal Factor')
mul_decomp.resid.plot(ax=axes[3], color='red', style='o', markersize=2)
axes[3].axhline(1, linestyle='--', color='black', linewidth=0.8)
axes[3].set_title('Residuals (ratio, should be ~1.0)')
 
plt.tight_layout()
plt.show()
 
# ─────────────────────────────────────────────────────────────────
# INTERPRETING THE COMPONENTS
# ─────────────────────────────────────────────────────────────────
print("=== Trend Extremes ===")
print(f"Trend start: {mul_decomp.trend.dropna().iloc[0]:.1f}")
print(f"Trend end:   {mul_decomp.trend.dropna().iloc[-1]:.1f}")
print()
print("=== Seasonal Factors (Multiplicative) ===")
seasonal_by_month = mul_decomp.seasonal[:12].values
months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']
for m, s in zip(months, seasonal_by_month):
    bar = '█' * int(s * 20)
    print(f"  {m}: {s:.3f}  {bar}")
 
print()
print("=== Residual Statistics ===")
resid = mul_decomp.resid.dropna()
print(f"  Mean:   {resid.mean():.4f}  (ideally ~1.0 for multiplicative)")
print(f"  Std:    {resid.std():.4f}")
print(f"  Min:    {resid.min():.4f}")
print(f"  Max:    {resid.max():.4f}")
 
# ─────────────────────────────────────────────────────────────────
# CHECKING IF RESIDUALS ARE WHITE NOISE
# ─────────────────────────────────────────────────────────────────
from statsmodels.stats.diagnostic import acorr_ljungbox
 
lb_result = acorr_ljungbox(resid, lags=12, return_df=True)
print("\n=== Ljung-Box Test on Residuals (p > 0.05 = white noise) ===")
print(lb_result[['lb_stat', 'lb_pvalue']].to_string())