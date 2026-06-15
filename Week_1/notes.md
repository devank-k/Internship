# Time Series Analysis — Foundations & Concepts
 
> A comprehensive guide covering data types, time series classifications, decomposition, and stationarity — with practical pandas and statsmodels examples.
 
---
 
## Table of Contents
 
1. [Types of Data](#1-types-of-data)
   - Time Series vs Cross-Sectional vs Pooled vs Panel
   - Real Dataset Examples in pandas
2. [Types of Time Series](#2-types-of-time-series)
   - Univariate vs Multivariate
   - Regular vs Irregular
   - Sparse vs Dense
3. [Components of a Time Series](#3-components-of-a-time-series)
   - Trend, Seasonality, Cycles, Residuals
   - Additive vs Multiplicative Decomposition
   - Code with statsmodels
4. [Stationarity](#4-stationarity)
   - Conceptual Intuition
   - Rolling Mean & Std Visualization
   - What Non-Stationarity Looks Like
---
 
## 1. Types of Data
 
Understanding the structure of your data before modeling is fundamental. The four primary data structures you will encounter in quantitative analysis are: **time series**, **cross-sectional**, **pooled**, and **panel** data.
 
---
 
### 1.1 Time Series Data
 
Observations on a **single entity** recorded at **multiple time points**, ordered chronologically.
 
| Date       | Temperature (°C) |
|------------|-----------------|
| 2024-01-01 | 12.3            |
| 2024-01-02 | 11.8            |
| 2024-01-03 | 13.1            |
 
**Key properties:**
- Observations are **not independent** — past values influence future values
- Order matters: shuffling destroys information
- The time index is a core part of the data, not just a label
```python
import pandas as pd
import yfinance as yf  # or use built-in datasets
 
# Example: Stock price — classic univariate time series
df_ts = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=5, freq='D'),
    'price': [100.2, 101.5, 99.8, 102.3, 103.1]
})
df_ts.set_index('date', inplace=True)
print(df_ts)
# date         price
# 2020-01-01  100.2
# 2020-01-02  101.5
# ...
 
# Using a real dataset: airline passengers (classic benchmark)
from statsmodels.datasets import co2
co2_data = co2.load_pandas().data
co2_data = co2_data.ffill()
print(co2_data.head())
# Mauna Loa weekly CO2 readings — a pure time series
```
 
**Real-world examples:**
- Daily stock prices of AAPL
- Monthly inflation rate of India
- Hourly temperature readings from a weather station
- Weekly COVID-19 case counts
---
 
### 1.2 Cross-Sectional Data
 
Observations on **multiple entities** at a **single point in time**.
 
| Country | GDP (USD Bn) | Population (M) | Literacy Rate (%) |
|---------|-------------|----------------|-------------------|
| India   | 3730        | 1400           | 77.7              |
| USA     | 25460       | 335            | 99.0              |
| Brazil  | 1920        | 215            | 93.0              |
 
**Key properties:**
- No time dimension — a single snapshot
- Observations are (ideally) **independent** of each other
- Order of rows is irrelevant
```python
import pandas as pd
 
# Cross-sectional: country stats at one point in time
df_cross = pd.DataFrame({
    'country': ['India', 'USA', 'Brazil', 'Germany', 'Japan'],
    'gdp_bn_usd': [3730, 25460, 1920, 4260, 4230],
    'population_m': [1400, 335, 215, 84, 124],
    'literacy_rate': [77.7, 99.0, 93.0, 99.0, 99.0]
})
print(df_cross)
 
# Using a real dataset: UCI Adult Income (single survey year)
# Each row is a person, no repeated measurements over time
from sklearn.datasets import fetch_openml
adult = fetch_openml(name='adult', version=2, as_frame=True)
df_adult = adult.frame
print(df_adult.head())
# age, education, occupation, income — all at one point in time
```
 
**Real-world examples:**
- A survey of 500 households in one month
- GDP, HDI, and trade balance of all countries in 2023
- Test scores of all students in a school on exam day
---
 
### 1.3 Pooled (Cross-Sectional) Data
 
Multiple **cross-sections stacked together** across different time periods, but **without tracking the same entities** over time. Also called "repeated cross-sections."
 
| Year | Country | GDP   |
|------|---------|-------|
| 2021 | India   | 3150  |
| 2021 | USA     | 23000 |
| 2022 | India   | 3390  |
| 2022 | USA     | 25460 |
 
**Key properties:**
- Has both a time and an entity dimension
- BUT the entities at `t=1` may not be the same as at `t=2`
- More data than pure cross-section, but less rich than panel data
```python
import pandas as pd
import numpy as np
 
# Pooled: different countries surveyed across different years
# (same countries here, but could be different respondents each year)
years = [2020, 2020, 2021, 2021, 2022, 2022]
countries = ['India', 'USA', 'India', 'USA', 'India', 'USA']
gdp = [2700, 21000, 3150, 23000, 3390, 25460]
 
df_pooled = pd.DataFrame({
    'year': years,
    'country': countries,
    'gdp_bn_usd': gdp
})
print(df_pooled)
 
# Real example: World Bank annual development indicators
# Stacked year by year — this is pooled cross-sectional
import pandas_datareader.wb as wb  # pip install pandas-datareader
# df_wb = wb.download(indicator='NY.GDP.PCAP.CD',
#                     country=['IND', 'USA', 'BRA'],
#                     start=2015, end=2022)
```
 
**Real-world examples:**
- Household income surveys conducted every 3 years (different households each time)
- Annual country-level World Bank indicators
- Quarterly company reports across different industries
---
 
### 1.4 Panel Data (Longitudinal Data)
 
Observations on the **same entities** tracked across **multiple time periods**. Also called **longitudinal data**.
 
| Entity  | Year | Revenue | Employees |
|---------|------|---------|-----------|
| Apple   | 2020 | 274.5   | 147000    |
| Apple   | 2021 | 365.8   | 154000    |
| Google  | 2020 | 182.5   | 135301    |
| Google  | 2021 | 257.6   | 156500    |
 
**Key properties:**
- Both a **time dimension** and an **entity dimension** are tracked
- Controls for **unobserved heterogeneity** (fixed or random effects)
- Rich structure — enables causal inference that pure time series or cross-sections cannot
```python
import pandas as pd
 
# Panel data: same companies tracked over multiple years
data = {
    'company': ['Apple','Apple','Apple','Google','Google','Google'],
    'year':    [2019,   2020,   2021,   2019,    2020,    2021],
    'revenue': [260.2,  274.5,  365.8,  161.9,  182.5,  257.6],
    'employees':[137000,147000,154000, 118899, 135301, 156500]
}
df_panel = pd.DataFrame(data)
 
# Set a MultiIndex for proper panel structure
df_panel.set_index(['company', 'year'], inplace=True)
print(df_panel)
# company  year
# Apple    2019    260.2  137000
#          2020    274.5  147000
#          2021    365.8  154000
# Google   2019    161.9  118899
# ...
 
# Reshaping: wide to long (common for panel data wrangling)
df_wide = pd.DataFrame({
    'company': ['Apple', 'Google'],
    'rev_2019': [260.2, 161.9],
    'rev_2020': [274.5, 182.5],
    'rev_2021': [365.8, 257.6]
})
df_long = df_wide.melt(id_vars='company',
                       value_vars=['rev_2019','rev_2020','rev_2021'],
                       var_name='year', value_name='revenue')
df_long['year'] = df_long['year'].str.extract(r'(\d+)').astype(int)
print(df_long.sort_values(['company','year']))
 
# Accessing a single entity's time series from panel data
apple_ts = df_panel.loc['Apple']['revenue']
print(apple_ts)  # Now a pure time series for Apple
```
 
**Real-world examples:**
- NIFTY 50 stock prices — 50 stocks, daily, over years
- WHO patient cohort — same patients measured monthly
- IMF World Economic Outlook — 190 countries tracked annually
---
 
### Quick Comparison Table
 
| Property              | Time Series | Cross-Sectional | Pooled         | Panel           |
|-----------------------|-------------|-----------------|----------------|-----------------|
| Entities              | One         | Many            | Many           | Many (same)     |
| Time periods          | Many        | One             | Many           | Many            |
| Same entity over time | ✅          | ❌              | ❌             | ✅              |
| Controls for entity FE| ❌          | ❌              | Limited        | ✅              |
| Example               | AAPL prices | 2023 census     | Annual surveys | S&P 500 stocks  |
 
---
 
## 2. Types of Time Series
 
### 2.1 Univariate vs Multivariate
 
#### Univariate Time Series
 
A **single variable** tracked over time. The goal is typically to forecast future values of that same variable using only its own past.
 
```
t=1   t=2   t=3   t=4   t=5
100   104   99    107   111      ← only one column
```
 
```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import co2
 
# CO2 concentration — univariate
co2_data = co2.load_pandas().data.ffill()
print(co2_data.head())
#             co2
# 1958-03-29  316.1
# 1958-04-05  317.3
 
# Another classic: airline passengers
from statsmodels.datasets import get_rdataset
ap = get_rdataset('AirPassengers')
airline = pd.Series(ap.data['value'].values,
                    index=pd.date_range('1949-01', periods=144, freq='ME'))
print(airline.head())
# One column: monthly passenger count
```
 
**Use cases:** Forecasting — predict tomorrow's temperature using only past temperatures. Methods: ARIMA, Exponential Smoothing, Prophet.
 
---
 
#### Multivariate Time Series
 
**Multiple variables** all evolving over time, potentially influencing each other.
 
```
t=1   t=2   t=3     ← time steps
[T=30, H=60, P=1012]
[T=31, H=58, P=1010]
[T=29, H=65, P=1008]
↑ temperature, humidity, pressure — all tracked together
```
 
```python
import pandas as pd
import numpy as np
 
# Multivariate: weather variables moving together
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=365, freq='D')
 
df_multi = pd.DataFrame({
    'temperature': 20 + 10*np.sin(np.linspace(0, 2*np.pi, 365)) + np.random.normal(0,1,365),
    'humidity':    60 - 5*np.sin(np.linspace(0, 2*np.pi, 365)) + np.random.normal(0,2,365),
    'pressure':    1013 + np.random.normal(0, 3, 365)
}, index=dates)
 
print(df_multi.head())
#             temperature  humidity    pressure
# 2023-01-01    20.24       60.48     1014.87
# ...
 
# Compute correlation matrix to understand relationships
print(df_multi.corr())
 
# Real example: macroeconomic indicators (GDP, unemployment, inflation)
# All move together but tell different parts of the same story
```
 
**Use cases:** VAR (Vector Autoregression) models, Granger causality, deep learning (LSTM). Example: predicting electricity demand using temperature + time-of-day + day-of-week.
 
---
 
### 2.2 Regular vs Irregular Time Series
 
#### Regular (Fixed-Frequency) Time Series
 
Observations arrive at **equal, predictable intervals**.
 
```python
import pandas as pd
 
# Regular: one observation every day, no gaps
regular = pd.date_range('2024-01-01', periods=10, freq='D')
print(regular)
# DatetimeIndex(['2024-01-01', '2024-01-02', ..., '2024-01-10'])
 
# Common regular frequencies in pandas
# 'T' or 'min' → minutely
# 'H'          → hourly
# 'D'          → daily (calendar days)
# 'B'          → business days only
# 'W'          → weekly (Sunday end)
# 'ME'         → month end
# 'QE'         → quarter end
# 'YE'         → year end
 
# Create a regular monthly time series
monthly = pd.Series(
    [100, 105, 98, 112, 115, 108, 120, 125, 118, 130, 128, 135],
    index=pd.date_range('2023-01', periods=12, freq='ME')
)
print(monthly)
```
 
#### Irregular Time Series
 
Observations arrive at **unequal or unpredictable intervals**.
 
```python
import pandas as pd
 
# Irregular: event-driven (transactions, earthquakes, sensor faults)
timestamps = pd.to_datetime([
    '2024-01-01 08:03:22',
    '2024-01-01 09:47:05',  # ~1.7 hours later
    '2024-01-01 09:47:11',  # 6 seconds later
    '2024-01-02 14:22:00',  # next day
    '2024-01-05 07:15:44',  # 3 days later
])
 
values = [200, 185, 183, 210, 195]
irregular_ts = pd.Series(values, index=timestamps)
print(irregular_ts)
 
# Converting irregular to regular via resampling/interpolation
# Resample to hourly, filling gaps with the last known value
regular_from_irr = irregular_ts.resample('h').ffill()
print(regular_from_irr.head(10))
 
# Or interpolate to fill in-between values
regular_interp = irregular_ts.resample('h').interpolate(method='time')
```
 
**Handling strategies for irregular TS:**
- `resample()` with `.mean()`, `.sum()`, `.ffill()`, `.bfill()`
- `asof()` for lookups at arbitrary times
- Time-weighted interpolation
---
 
### 2.3 Sparse vs Dense Time Series
 
#### Dense Time Series
 
**Most time points have observations**. This is the typical case in financial, weather, and economic data.
 
```python
import pandas as pd
import numpy as np
 
# Dense: nearly every slot is filled
dense = pd.Series(
    np.random.randn(100),
    index=pd.date_range('2024-01-01', periods=100, freq='D')
)
density = dense.notna().mean()
print(f"Density: {density:.0%}")  # Density: 100%
```
 
#### Sparse Time Series
 
**Most time points have no observation** — zeros, NaNs, or missing values dominate.
 
```python
import pandas as pd
import numpy as np
 
# Sparse: product sales — most days nothing is sold for niche items
np.random.seed(0)
dates = pd.date_range('2024-01-01', periods=365, freq='D')
 
# Only 5% of days have sales events
sales_mask = np.random.rand(365) < 0.05
sales_values = np.where(sales_mask, np.random.randint(1, 50, 365), 0)
 
sparse_ts = pd.Series(sales_values, index=dates)
non_zero_pct = (sparse_ts > 0).mean()
print(f"Non-zero entries: {non_zero_pct:.1%}")  # ~5%
print(sparse_ts[sparse_ts > 0].head())
 
# Sparse TS challenges:
# 1. Standard decomposition fails (no signal in all those zeros)
# 2. Aggregating to weekly/monthly often helps
weekly = sparse_ts.resample('W').sum()
print(weekly.head(10))
 
# Real sparse examples:
# - Earthquake occurrences (rare events)
# - Fraud transactions (< 0.1% of all transactions)
# - Hospital admissions for rare diseases
# - Server error logs
```
 
**Key distinction:**
 
| Type   | Description                              | Example                        |
|--------|------------------------------------------|--------------------------------|
| Dense  | Most time steps have values              | Stock prices, temperature      |
| Sparse | Most time steps are zero or missing      | Fraud events, rare diseases    |
| Regular| Equal time intervals                     | Monthly CPI, daily close price |
| Irregular| Unequal time intervals               | User clicks, sensor triggers   |
 
---
 
## 3. Components of a Time Series
 
Any time series can be decomposed into constituent parts. Understanding these components is essential before choosing a forecasting model.
 
### 3.1 The Four Components
 
#### Trend (T)
 
The **long-term direction** of the series — upward, downward, or flat — after removing short-term fluctuations.
 
- Rising GDP over decades
- Increasing internet users globally
- Declining landline phone subscriptions
#### Seasonality (S)
 
**Regular, periodic fluctuations** that repeat at fixed, known intervals (daily, weekly, monthly, yearly).
 
- Retail sales spike every December
- Ice cream sales peak every summer
- Electricity demand peaks every weekday morning
#### Cyclical Component (C)
 
**Longer-term oscillations** that are irregular and NOT of fixed period. Usually driven by economic or business cycles.
 
- Business expansion and contraction (~3–10 year cycles)
- Housing market boom-bust cycles
- (Harder to model — often grouped with the residual)
#### Residual / Noise (R or ε)
 
What remains after removing trend, seasonality, and cycles. Ideally **white noise** — random, uncorrelated errors with zero mean.
 
If residuals show patterns → your decomposition is incomplete or your model is misspecified.
 
---
 
### 3.2 Additive vs Multiplicative Decomposition
 
#### Additive Model
 
Use when the **magnitude of seasonality is constant** regardless of the trend level.
 
```
Y(t) = T(t) + S(t) + R(t)
```
 
If trend is at 100 or 1000, the seasonal swing is always the same size (e.g., ±20 units).
 
#### Multiplicative Model
 
Use when **seasonal swings grow proportionally** with the trend level.
 
```
Y(t) = T(t) × S(t) × R(t)
```
 
If trend doubles, the seasonal amplitude also roughly doubles. Common in economic and financial data.
 
**How to decide:** Plot the series. If the seasonal peaks and troughs get larger as the series rises → multiplicative. If they stay constant → additive.
 
---
 
### 3.3 Code: Decomposition with statsmodels
 
```python
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
plt.savefig('additive_decomposition.png', dpi=150)
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
plt.savefig('multiplicative_decomposition.png', dpi=150)
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
```
 
---
 
### 3.4 Reading the Decomposition Plots
 
```
Original series
───────────────────────────────────────────────────────────
Look for: overall direction, visible bumps, growing amplitude?
 
Trend component
───────────────────────────────────────────────────────────
Look for: smooth upward/downward drift
Red flag: large jumps or discontinuities in trend
 
Seasonal component
───────────────────────────────────────────────────────────
Additive: should be a fixed repeating wave (constant amplitude)
Multiplicative: shows the ratio/factor (values near 1.0 = little effect)
Red flag: amplitude changes dramatically over time → wrong model
 
Residuals
───────────────────────────────────────────────────────────
Should look like: random scatter around 0 (additive) or 1 (multiplicative)
Red flag: structure/patterns in residuals → missed components or wrong model
```
 
---
 
## 4. Stationarity
 
### 4.1 What is Stationarity?
 
A time series is **stationary** if its statistical properties do not change over time. Specifically:
 
| Property        | Formal Definition                          | Informal Meaning                   |
|-----------------|--------------------------------------------|------------------------------------|
| Constant Mean   | E[X(t)] = μ for all t                      | Series fluctuates around same level|
| Constant Variance| Var[X(t)] = σ² for all t                 | Spread doesn't grow or shrink      |
| Constant Autocovariance| Cov[X(t), X(t+k)] depends only on k | Correlations are stable over time  |
 
**Why stationarity matters:** Most classical time series models (ARMA, ARIMA) assume stationarity. Fitting them on non-stationary data gives spurious results.
 
**Strict vs Weak (Covariance) Stationarity:**
- Strict: entire distribution is time-invariant
- Weak/Covariance: only the first two moments (mean, variance) are constant — sufficient for most practical use
---
 
### 4.2 Visual Intuition: Rolling Mean and Standard Deviation
 
The most immediate tool to detect non-stationarity is plotting rolling statistics.
 
```python
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
plt.savefig('stationarity_rolling_stats.png', dpi=150)
plt.show()
```
 
---
 
### 4.3 What Non-Stationarity Looks Like in Plots
 
#### Upward (or Downward) Trend
 
```
Value
  │                               ●●●
  │                          ●●●●●
  │                     ●●●●●
  │                ●●●●●
  │           ●●●●●
  │      ●●●●●
  │ ●●●●●
  └──────────────────────────────── Time
 
Rolling mean: slopes upward — clearly NON-STATIONARY
```
 
**Signature in rolling stats:** Rolling mean trends in one direction. Rolling std may also increase.
 
**Fix:** First differencing → `df['diff'] = df['value'].diff()`
 
---
 
#### Changing (Non-Constant) Variance
 
```
Value
  │     ●      ●             ●●●       ●●●●●●
  │ ●●●●●●●●●●●●●●●●●●●●●● ●●●●●● ●●●●●●●●●●●●●●
  │     ●      ●             ●●●       ●●●●●●
  └──────────────────────────────── Time
         ← tight spread →         ← wide spread →
 
Rolling std: increases over time — NON-STATIONARY (heteroskedastic)
```
 
**Signature in rolling stats:** Rolling mean may be flat, but rolling std grows.
 
**Fix:** Log transformation → `df['log_val'] = np.log(df['value'])` followed by differencing.
 
---
 
#### Seasonality Without Stationarity
 
```
Value
  │   ▲       ▲       ▲       ▲
  │  / \     / \     / \     / \
  │ /   \   /   \   /   \   /   \
  │/     \ /     \ /     \ /     \
  │       V       V       V       V
  └──────────────────────────────── Time
 
Rolling mean: oscillates — NON-STATIONARY
```
 
**Signature in rolling stats:** Rolling mean moves up and down in a periodic pattern.
 
**Fix:** Seasonal differencing → `df['seas_diff'] = df['value'].diff(12)` (for monthly data)
 
---
 
#### Structural Break
 
```
Value
  │          ●●●●●●●●●●●●●●●
  │ ●●●●●●●●●│
  │          └── sudden level shift
  └──────────────────────────────── Time
 
Rolling mean: jumps discontinuously — NON-STATIONARY
```
 
**Signature in rolling stats:** Rolling mean has an abrupt step change at a specific date.
 
**Fix:** Include a dummy/indicator variable for the break point, or split the series.
 
---
 
### 4.4 Formal Stationarity Tests
 
Visual inspection is necessary but not sufficient. Formal tests provide statistical confirmation.
 
```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
 
def stationarity_report(series, name="Series", signif=0.05):
    """
    Run ADF and KPSS tests and print an interpretable report.
    
    ADF  (Augmented Dickey-Fuller):
        H0: unit root exists (non-stationary)
        Reject H0 (p < 0.05) → evidence of stationarity
    
    KPSS (Kwiatkowski-Phillips-Schmidt-Shin):
        H0: series is stationary
        Reject H0 (p < 0.05) → evidence of non-stationarity
    
    Both agree:
        ADF rejects & KPSS does NOT reject → STATIONARY  ✅
        ADF does NOT reject & KPSS rejects → NON-STATIONARY ❌
        Both reject → trend-stationary (has deterministic trend)
        Neither rejects → inconclusive (more data needed)
    """
    print(f"\n{'='*55}")
    print(f"  Stationarity Tests: {name}")
    print(f"{'='*55}")
 
    # ADF Test
    adf_result = adfuller(series.dropna(), autolag='AIC')
    adf_pval   = adf_result[1]
    adf_stat   = adf_result[0]
 
    print(f"\n  📊 ADF Test")
    print(f"     Test Statistic : {adf_stat:.4f}")
    print(f"     p-value        : {adf_pval:.4f}")
    print(f"     Critical Values: ", end='')
    for key, val in adf_result[4].items():
        print(f"{key} = {val:.3f}", end='  ')
    print()
    if adf_pval < signif:
        print(f"     → p={adf_pval:.4f} < {signif}: Reject H0 → Likely STATIONARY ✅")
    else:
        print(f"     → p={adf_pval:.4f} ≥ {signif}: Fail to reject H0 → Likely NON-STATIONARY ❌")
 
    # KPSS Test
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    kpss_pval   = kpss_result[1]
    kpss_stat   = kpss_result[0]
 
    print(f"\n  📊 KPSS Test")
    print(f"     Test Statistic : {kpss_stat:.4f}")
    print(f"     p-value        : {kpss_pval:.4f}")
    if kpss_pval < signif:
        print(f"     → p={kpss_pval:.4f} < {signif}: Reject H0 → Likely NON-STATIONARY ❌")
    else:
        print(f"     → p={kpss_pval:.4f} ≥ {signif}: Fail to reject H0 → Likely STATIONARY ✅")
 
    # Joint Conclusion
    print(f"\n  📋 Joint Conclusion:")
    adf_stationary  = adf_pval < signif
    kpss_stationary = kpss_pval >= signif
 
    if adf_stationary and kpss_stationary:
        print("     ✅ Both tests agree → STATIONARY")
    elif not adf_stationary and not kpss_stationary:
        print("     ❌ Both tests agree → NON-STATIONARY")
    elif adf_stationary and not kpss_stationary:
        print("     ⚠️  Tests disagree: ADF says stationary, KPSS says not → likely trend-stationary")
    else:
        print("     ⚠️  Tests disagree: KPSS says stationary, ADF says not → inconclusive, check manually")
 
# ─────────────────────────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────────────────────────
np.random.seed(42)
n = 300
idx = pd.date_range('2020-01-01', periods=n, freq='D')
 
white_noise   = pd.Series(np.random.normal(0, 1, n), index=idx)
random_walk   = pd.Series(np.cumsum(np.random.normal(0, 1, n)), index=idx)
 
stationarity_report(white_noise,  "White Noise (stationary)")
stationarity_report(random_walk,  "Random Walk (non-stationary)")
 
# Differenced random walk — should become stationary
rw_diff = random_walk.diff().dropna()
stationarity_report(rw_diff, "Random Walk After 1st Differencing")
```
 
---
 
### 4.5 Making a Non-Stationary Series Stationary
 
```python
import numpy as np
import pandas as pd
 
np.random.seed(0)
n = 200
 
# Non-stationary: random walk with positive drift
rw = pd.Series(np.cumsum(np.random.normal(0.1, 1, n)),
               index=pd.date_range('2020-01-01', periods=n, freq='D'))
 
# ── Method 1: First Differencing (removes trend) ─────────────────
rw_diff1 = rw.diff().dropna()
# If still non-stationary → second differencing
rw_diff2 = rw.diff().diff().dropna()
 
# ── Method 2: Log Transform (stabilizes variance) ─────────────────
# Only valid when all values > 0
pos_series = pd.Series(np.exp(np.cumsum(np.random.normal(0.01, 0.05, n))),
                        index=pd.date_range('2020-01-01', periods=n, freq='D'))
log_ts   = np.log(pos_series)
log_diff = np.log(pos_series).diff().dropna()  # log return
 
# ── Method 3: Seasonal Differencing (removes seasonality) ─────────
from statsmodels.datasets import get_rdataset
ap = get_rdataset('AirPassengers')
airline = pd.Series(
    ap.data['value'].values,
    index=pd.date_range('1949-01', periods=144, freq='ME')
)
 
# Seasonal difference (period=12 for monthly)
airline_seas_diff = airline.diff(12).dropna()
 
# Often: log + seasonal diff + regular diff for economic TS
airline_transformed = np.log(airline).diff(12).diff(1).dropna()
 
print("Before: non-stationary airline series")
print(f"  Mean std in first half:  {airline[:72].std():.2f}")
print(f"  Mean std in second half: {airline[72:].std():.2f}")
 
print("\nAfter log + seasonal diff + first diff:")
print(f"  Mean: {airline_transformed.mean():.4f}  (should be ~0)")
print(f"  Std:  {airline_transformed.std():.4f}  (should be stable)")
```
 
---
 
### 4.6 Stationarity Decision Flowchart
 
```
         START
           │
           ▼
    Plot the series
    + rolling stats
           │
    ┌──────┴──────┐
    │             │
  Looks         Looks
 stationary  non-stationary
    │             │
    ▼             ▼
Run ADF      Is there a TREND?
+ KPSS        │           │
tests        YES           NO
             │             │
         1st diff      Is there SEASONALITY?
             │             │           │
             │            YES           NO
             │             │             │
             │        Seasonal        Changing
             │          diff           variance?
             │             │             │
             │             │            YES
             │             │             │
             │             │        Log transform
             │             │
             └──────┬──────┘
                    ▼
              Re-test with ADF + KPSS
                    │
              ┌─────┴─────┐
            Pass         Fail
              │             │
         Proceed to    Try another
          modeling    transformation
                       or check for
                      structural breaks
```
 
---
 
## Summary Reference Card
 
### Data Structure Cheatsheet
 
```python
# Time Series
ts = pd.Series([...], index=pd.date_range(...))
 
# Cross-Sectional
cs = pd.DataFrame({'entity': [...], 'feature': [...]})
 
# Panel — MultiIndex
panel = df.set_index(['entity_col', 'time_col'])
 
# Access one entity's time series from panel
entity_ts = panel.loc['EntityName']['value_col']
```
 
### Decomposition Cheatsheet
 
```python
from statsmodels.tsa.seasonal import seasonal_decompose
 
# Additive: constant seasonal amplitude
result_add = seasonal_decompose(ts, model='additive', period=12)
 
# Multiplicative: growing seasonal amplitude
result_mul = seasonal_decompose(ts, model='multiplicative', period=12)
 
# Access components
trend      = result_mul.trend
seasonal   = result_mul.seasonal
residuals  = result_mul.resid
```
 
### Stationarity Tests Cheatsheet
 
```python
from statsmodels.tsa.stattools import adfuller, kpss
 
# ADF: p < 0.05 → stationary
adf_p = adfuller(ts, autolag='AIC')[1]
 
# KPSS: p < 0.05 → NON-stationary
kpss_p = kpss(ts, regression='c', nlags='auto')[1]
 
# Quick fix for non-stationarity
ts_diff      = ts.diff().dropna()        # 1st difference
ts_log_diff  = np.log(ts).diff().dropna() # log + diff
ts_seas_diff = ts.diff(12).dropna()      # seasonal diff (monthly)
```
 
### Visual Diagnosis Guide
 
| Rolling Stats Pattern                          | Diagnosis          | Fix                      |
|------------------------------------------------|--------------------|--------------------------|
| Flat mean, flat std                            | ✅ Stationary       | None needed              |
| Mean slopes up/down                            | ❌ Trend            | `diff(1)`                |
| Std grows over time                            | ❌ Heteroskedastic  | `log()` then `diff(1)`   |
| Mean oscillates periodically                   | ❌ Seasonal         | `diff(period)`           |
| Mean has sudden step change                    | ❌ Structural break | Add dummy variable        |
 
---
 
## Dependencies
 
```bash
pip install pandas numpy matplotlib statsmodels scikit-learn
# Optional
pip install yfinance pandas-datareader
```
 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.datasets import get_rdataset, co2
```
 