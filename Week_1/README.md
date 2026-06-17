# Delhi Climate Time Series Analysis

## Overview

This project analyzes the **Daily Delhi Climate Dataset** to identify periodicity and patterns in temperature data using time series decomposition and Fast Fourier Transform (FFT).

## Dataset

**File**: `DailyDelhiClimate.csv`

**Columns**:
- `date`: Date of observation
- `meantemp`: Mean temperature in °C
- `humidity`: Humidity percentage
- `wind_speed`: Wind speed
- `meanpressure`: Mean pressure

**Time Period**: 2013-2017 (approximately 1462 days)

## Analysis Performed

### 1. Data Exploration
- Load and parse CSV with datetime index
- Display basic statistics and data info
- Visualize temperature trends over time and by month

### 2. Linear Regression Model
- **Features**: humidity, wind_speed, meanpressure
- **Target**: meantemp
- Train-test split: 80-20
- Predicts mean temperature based on meteorological variables

### 3. Seasonal Decomposition
Uses additive decomposition with period=365:
- **Trend**: Slow long-term changes in baseline temperature
- **Seasonal**: Repeating yearly (365-day) pattern
- **Residual**: Random noise and unexplained variations

**Equation**: `Original Temperature = Trend + Seasonal + Residual`

### 4. FFT (Fast Fourier Transform) Periodicity Analysis
- Converts time-domain signal to frequency domain
- Identifies dominant repeating cycles in the data
- **Key Finding**: Dominant period = **365.50 days** (annual cycle)
- Secondary peak at 180 days indicates harmonic (asymmetry in the yearly pattern)

## Key Findings

✓ **Strong Annual Seasonality** (365.5 days)
- Temperature follows predictable yearly cycle
- Hot in summer months, cold in winter months
- Pattern repeats consistently every year

✓ **Non-Stationary Time Series**
- Presence of trend component makes it non-stationary
- Would require differencing or detrending for ARIMA-type models

✓ **Small Residuals** (range: -5 to +5°C)
- Model captures most patterns well
- Remaining variation is random weather noise

✓ **Additive Model Fit**
- Components add together (not multiplicative)
- Indicates seasonal amplitude stays relatively constant across years


## Time Series Classification

**Type**: Additive Time Series with Strong Seasonality and Trend

- **Stationary?** No (due to trend component)
- **Periodicity**: 365.5 days (annual)
- **Pattern Complexity**: Simple (dominated by one strong seasonal component)

```

## Results

- **Dominant Period**: 365.50 days ± 0.5 days
- **FFT Peak Power**: 4.67e+07
- **Model Fit**: Good (residuals close to zero)
- **Secondary Harmonics**: Detected at 180 days, 122 days (fractional periods)

