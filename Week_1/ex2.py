import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
np.random.seed(42)
n = 300
 
# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTION: Plot series + rolling stats
# ─────────────────────────────────────────────────────────────────
def plot_rolling_stats(series, title, window=30, ax=None):
    """
    Plot the series alongside its rolling mean and std.
    Stationary → flat rolling mean + flat rolling std.
    Non-stationary → drifting rolling mean or expanding/contracting std.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
 
    rolling_mean = series.rolling(window=window).mean()
    rolling_std  = series.rolling(window=window).std()
 
    ax.plot(series.index, series.values,
            color='steelblue', alpha=0.6, label='Observed', linewidth=1)
    ax.plot(rolling_mean.index, rolling_mean,
            color='darkorange', linewidth=2, label=f'Rolling Mean (w={window})')
    ax.fill_between(rolling_std.index,
                    rolling_mean - rolling_std,
                    rolling_mean + rolling_std,
                    alpha=0.2, color='darkorange', label='±1 Rolling Std')
 
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    return ax
 
# ─────────────────────────────────────────────────────────────────
# EXAMPLE 1: Stationary Series (mean-reverting, constant variance)
# ─────────────────────────────────────────────────────────────────
stationary = pd.Series(
    np.random.normal(0, 1, n),
    index=pd.date_range('2020-01-01', periods=n, freq='D')
)
 
# ─────────────────────────────────────────────────────────────────
# EXAMPLE 2: Non-Stationary — Trending (non-constant mean)
# ─────────────────────────────────────────────────────────────────
trending = pd.Series(
    np.cumsum(np.random.normal(0.05, 1, n)),  # random walk with drift
    index=pd.date_range('2020-01-01', periods=n, freq='D')
)
 
# ─────────────────────────────────────────────────────────────────
# EXAMPLE 3: Non-Stationary — Changing Variance (heteroskedastic)
# ─────────────────────────────────────────────────────────────────
expanding_var = pd.Series(
    [np.random.normal(0, 0.1 + 0.01*i) for i in range(n)],
    index=pd.date_range('2020-01-01', periods=n, freq='D')
)
 
# ─────────────────────────────────────────────────────────────────
# EXAMPLE 4: Non-Stationary — Seasonality (non-constant mean cycle)
# ─────────────────────────────────────────────────────────────────
seasonal_only = pd.Series(
    5 * np.sin(2 * np.pi * np.arange(n) / 52) + np.random.normal(0, 0.5, n),
    index=pd.date_range('2020-01-01', periods=n, freq='D')
)
 
# ─────────────────────────────────────────────────────────────────
# PLOT ALL FOUR
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 16))
fig.suptitle('Rolling Mean & Std — Stationarity Diagnostics', fontsize=14, fontweight='bold')
 
plot_rolling_stats(stationary,    '✅ STATIONARY: White Noise (flat mean, flat std)', ax=axes[0])
plot_rolling_stats(trending,      '❌ NON-STATIONARY: Upward Trend (mean drifts up)', ax=axes[1])
plot_rolling_stats(expanding_var, '❌ NON-STATIONARY: Growing Variance (std expands)', ax=axes[2])
plot_rolling_stats(seasonal_only, '❌ NON-STATIONARY: Seasonality (mean oscillates)', ax=axes[3])
 
plt.tight_layout()
plt.show()