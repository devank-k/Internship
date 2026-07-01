# Seasonality Tests in Time Series Forecasting
## Kruskal-Wallis Test by Season Group & ACF at Seasonal Lags

---

## Table of Contents

- [What is Seasonality?](#what-is-seasonality)
- [Why Test for Seasonality Before Modelling?](#why-test-for-seasonality-before-modelling)
- [Where Seasonality Tests Fit in the Pipeline](#where-seasonality-tests-fit-in-the-pipeline)
- [Two Approaches to Detecting Seasonality](#two-approaches-to-detecting-seasonality)
- [Method 1 — Kruskal-Wallis Test by Season Group](#method-1--kruskal-wallis-test-by-season-group)
  - [How it Works](#how-it-works)
  - [Assumptions](#assumptions)
  - [Limitation](#limitation)
- [Method 2 — ACF at Seasonal Lags](#method-2--acf-at-seasonal-lags)
  - [How it Works](#how-it-works-1)
  - [Which Lags to Check](#which-lags-to-check)
  - [Limitation](#limitation-1)
- [Side-by-Side Comparison](#side-by-side-comparison)
- [Code Implementation](#code-implementation)
  - [Kruskal-Wallis Seasonality Test](#kruskal-wallis-seasonality-test)
  - [ACF at Seasonal Lags](#acf-at-seasonal-lags)
  - [Full Pipeline — Both Methods with Verdict](#full-pipeline--both-methods-with-verdict)
  - [On a Real DataFrame](#on-a-real-dataframe)
- [Interpreting Results](#interpreting-results)
- [What to Do After Detecting Seasonality](#what-to-do-after-detecting-seasonality)
- [Common Mistakes](#common-mistakes)
- [References](#references)

---

## What is Seasonality?

Seasonality is a **regular, predictable pattern that repeats at a fixed
period** s. Unlike a trend (which moves in one direction) or noise (which
is random), seasonal patterns repeat with the same shape every s periods.

```
No seasonality:   5, 4, 6, 3, 5, 7, 4, 6, 5, 4, 3, 6 ...  (no pattern)

With seasonality (s=4, quarterly):
  Q1:  3,  3,  4,  3        ← always low in Q1
  Q2:  8,  9,  8,  9        ← always high in Q2
  Q3:  6,  7,  6,  7        ← moderate in Q3
  Q4:  4,  3,  4,  4        ← low again in Q4
```

The key word is **regular** — the pattern returns at the same interval every
time. A one-off spike or a slowly shifting mean is not seasonality.

**Common seasonal periods:**

| Data frequency | Season | Period s |
|----------------|--------|----------|
| Monthly | Annual | 12 |
| Quarterly | Annual | 4 |
| Weekly | Annual | 52 |
| Daily | Weekly | 7 |
| Daily | Annual | 365 |
| Hourly | Daily | 24 |
| Hourly | Weekly | 168 |

---

## Why Test for Seasonality Before Modelling?

Ignoring seasonality in a model produces systematically wrong forecasts that
repeat the same error every cycle. Testing first tells you:

1. **Whether seasonal terms are needed** — prevents over-fitting seasonal
   parameters into a series that has no seasonality
2. **What the seasonal period is** — confirms whether to use s=7, s=12,
   or s=52 in SARIMA
3. **How strong the seasonality is** — weak seasonality may not need
   explicit modelling; strong seasonality dominates the forecast
4. **Whether to apply seasonal differencing** — `series.diff(s)` removes
   deterministic seasonality before fitting ARIMA

---

## Where Seasonality Tests Fit in the Pipeline

```
Raw time series
      ↓
Trend tests (Mann-Kendall, Cox-Stuart)
      ↓
Stationarity tests (ADF, KPSS, PP)
      ↓
Seasonality tests (Kruskal-Wallis + ACF at seasonal lags)  ← here
      ↓
      ├── No seasonality → ARIMA(p,d,q)
      └── Seasonality detected → SARIMA(p,d,q)(P,D,Q)[s]
                               → or STL decomposition
                               → or seasonal differencing + ARIMA
```

---

## Two Approaches to Detecting Seasonality

The two methods attack the same question from completely different angles:

| | Kruskal-Wallis | ACF at Seasonal Lags |
|---|---|---|
| **Approach** | Statistical test on group medians | Correlation analysis |
| **Question** | Do season groups have different distributions? | Is the series correlated with itself s steps ago? |
| **Output** | H statistic, p-value | Correlation values at lags s, 2s, 3s |
| **Needs** | Season labels (month, quarter, weekday) | Just the series and a guess at s |
| **Detects** | Deterministic seasonality (fixed pattern) | Stochastic seasonality (autocorrelation at lag s) |
| **Blind to** | Trending series (needs detrending first) | Trend (can inflate seasonal ACF) |

Running both together gives a more complete picture than either alone.

---

## Method 1 — Kruskal-Wallis Test by Season Group

### How it Works

Kruskal-Wallis is a non-parametric one-way ANOVA on ranks. Applied to
seasonality, it groups observations by their season label (e.g. month 1,
month 2, ..., month 12) and tests whether the groups have the same median.

**Null hypothesis:** All season groups have the same median.
**Alternative:** At least one season group has a different median.

If January always has lower values than July, the group medians will differ
significantly and the test will reject the null — confirming seasonality.

**Steps:**

1. Add a season label column: `df['month'] = df['date'].dt.month`
2. Group observations by season label
3. Rank all observations globally (ignoring groups)
4. Compute the H statistic: how much the average rank in each group
   deviates from the overall average rank
5. Compare H to a chi-squared distribution with (k−1) degrees of freedom,
   where k = number of season groups

**H statistic formula:**

```
H = [12 / (n(n+1))] × Σⱼ [nⱼ × (R̄ⱼ − R̄)²]

where:
  n   = total number of observations
  nⱼ  = number of observations in group j
  R̄ⱼ = mean rank in group j
  R̄  = overall mean rank = (n+1)/2
  k   = number of groups (e.g. 12 for monthly)
```

A large H means group rank means differ a lot → seasonality present.
Under the null, H follows χ²(k−1).

### Assumptions

- Observations within each group are independent
- At least 5 observations per season group (otherwise exact test needed)
- Series should be **detrended first** — a trend artificially inflates
  ranks in later seasons, producing false positives
- Measures deterministic seasonality only (fixed seasonal effect)

### Limitation

Kruskal-Wallis detects that *some* groups differ but does not tell you
*which* groups are different or *what shape* the seasonal pattern has.
It also requires you to already know the seasonal period s to assign labels.

---

## Method 2 — ACF at Seasonal Lags

### How it Works

If a series has seasonality with period s, observations s periods apart are
similar — warm Julys are followed by warm Julys, slow Januaries by slow
Januaries. This similarity shows up as significant autocorrelation at lags
s, 2s, 3s, ...

**Testing the seasonal ACF:**

1. Compute ACF up to at least 2–3 full seasons
2. Check whether `r(s)`, `r(2s)`, `r(3s)` exceed the 95% CI `±1.96/√n`
3. If multiple seasonal lags are significant and decay across multiples of s
   → seasonal autocorrelation is confirmed

**The seasonal ACF pattern:**

```
ACF — monthly data with annual seasonality (s=12)

r(k)
 1.0 |█
 0.5 |
 0.3 |            *            ← r(12) significant
 0.1 |···············*·········· ← r(24) smaller
 0.0 |────────────────────────────────────────── CI
−0.1 |
     lag: 0  2  4  6  8 10 12 14 16 18 20 22 24

Spikes at lag 12 and 24 — seasonal period = 12
```

**Quantifying seasonal strength:**

The magnitude of `r(s)` relative to the confidence interval tells you how
strong the seasonality is:

```
|r(s)| < CI           → no seasonal autocorrelation
CI < |r(s)| < 0.3     → weak seasonality
0.3 ≤ |r(s)| < 0.6   → moderate seasonality
|r(s)| ≥ 0.6          → strong seasonality
```

### Which Lags to Check

For a suspected period s, check lags s, 2s, 3s:

```python
# monthly data — check lags 12, 24, 36
seasonal_lags = [12, 24, 36]

# daily data with weekly seasonality — check lags 7, 14, 21
seasonal_lags = [7, 14, 21]

# hourly data with daily seasonality — check lags 24, 48, 72
seasonal_lags = [24, 48, 72]
```

If you are unsure of s, plot the full ACF and look for the first large spike
beyond lag 1 — that lag is likely s.

### Limitation

ACF at seasonal lags is inflated by trends. A non-stationary trending series
will show large ACF at all lags including seasonal ones. Always detrend or
first-difference before checking seasonal ACF to avoid false positives.

---

## Side-by-Side Comparison

| Feature | Kruskal-Wallis | ACF at Seasonal Lags |
|---------|---------------|----------------------|
| Null hypothesis | All seasons have equal median | No autocorrelation at lag s |
| Requires known s | Yes — to assign group labels | No — can discover s from plot |
| Output | H, p-value, verdict | r(s), r(2s), r(3s), significance |
| Seasonal strength | Not directly | Yes — magnitude of r(s) |
| Sensitive to trend | Yes — detrend first | Yes — difference first |
| Detects | Fixed seasonal mean shift | Cyclical autocorrelation |
| Best for | Confirming a known period | Discovering the period |

---

## Code Implementation

### Kruskal-Wallis Seasonality Test

```python
import numpy as np
import pandas as pd
from scipy.stats import kruskal

def kruskal_seasonality(series, dates, period='month', alpha=0.05):
    """
    Test for seasonality using Kruskal-Wallis on season groups.

    Parameters
    ----------
    series : array-like
        Time series values.
    dates : array-like
        Corresponding datetime index.
    period : str
        Season label — 'month', 'quarter', 'dayofweek', 'hour'.
    alpha : float
        Significance level (default 0.05).
    """
    df = pd.DataFrame({'value': series, 'date': pd.to_datetime(dates)})
    df = df.sort_values('date').reset_index(drop=True)

    # assign season label
    if   period == 'month':     df['season'] = df['date'].dt.month
    elif period == 'quarter':   df['season'] = df['date'].dt.quarter
    elif period == 'dayofweek': df['season'] = df['date'].dt.dayofweek
    elif period == 'hour':      df['season'] = df['date'].dt.hour
    else:
        raise ValueError("period must be 'month', 'quarter', 'dayofweek', or 'hour'")

    # group observations by season
    groups = [grp['value'].values for _, grp in df.groupby('season')]
    n_groups = len(groups)

    # check minimum group size
    min_size = min(len(g) for g in groups)
    if min_size < 5:
        print(f"Warning: smallest group has {min_size} obs — results unreliable.")

    # Kruskal-Wallis test
    h_stat, p_value = kruskal(*groups)
    verdict = "SEASONALITY DETECTED" if p_value < alpha else "NO SEASONALITY"

    # group medians for inspection
    medians = df.groupby('season')['value'].median().round(3).to_dict()

    print(f"\n{'='*54}")
    print(f"  Kruskal-Wallis Seasonality Test  [{period}]")
    print(f"{'='*54}")
    print(f"  H statistic  : {h_stat:.4f}")
    print(f"  p-value      : {p_value:.4f}")
    print(f"  Groups (k)   : {n_groups}  (df = {n_groups-1})")
    print(f"  Min group n  : {min_size}")
    print(f"  {'─'*50}")
    print(f"  Verdict      : {verdict}")
    print(f"\n  Group medians:")
    for season, med in medians.items():
        bar = '█' * int(abs(med) / max(abs(v) for v in medians.values()) * 20)
        print(f"    {period[0].upper()}{season:>2} : {med:>8.3f}  {bar}")
    print(f"{'='*54}\n")

    return {
        "h_stat"  : round(h_stat, 4),
        "p_value" : round(p_value, 4),
        "verdict" : verdict,
        "medians" : medians,
    }
```

---

### ACF at Seasonal Lags

```python
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

def seasonal_acf_test(series, period=12, n_cycles=3, alpha=0.05):
    """
    Test for seasonality using ACF at seasonal lags.

    Parameters
    ----------
    series : array-like
        Time series values (should be detrended/differenced first).
    period : int
        Suspected seasonal period (e.g. 12 for monthly, 7 for daily).
    n_cycles : int
        Number of seasonal cycles to check (default 3).
    alpha : float
        Significance level (default 0.05).
    """
    series = pd.Series(series).dropna()
    n      = len(series)
    ci     = 1.96 / np.sqrt(n)
    maxlag = period * n_cycles

    acf_vals = acf(series, nlags=maxlag, fft=True)

    # check ACF at seasonal lags
    seasonal_lags  = [period * k for k in range(1, n_cycles + 1)]
    seasonal_acf   = {lag: round(acf_vals[lag], 4) for lag in seasonal_lags}
    significant    = {lag: abs(v) > ci for lag, v in seasonal_acf.items()}
    n_sig          = sum(significant.values())

    # strength of r(s)
    rs = abs(seasonal_acf[period])
    if   rs < ci:         strength = "none"
    elif rs < 0.3:        strength = "weak"
    elif rs < 0.6:        strength = "moderate"
    else:                 strength = "strong"

    verdict = "SEASONALITY DETECTED" if n_sig >= 1 else "NO SEASONALITY"

    print(f"\n{'='*54}")
    print(f"  ACF Seasonal Lags Test  [period={period}]")
    print(f"{'='*54}")
    print(f"  n          : {n}")
    print(f"  95% CI     : ±{ci:.3f}")
    print(f"  {'─'*50}")
    print(f"  {'Lag':<8} {'ACF r(k)':<12} {'|r| > CI':<12} {'Significant'}")
    print(f"  {'─'*50}")
    for lag in seasonal_lags:
        r   = seasonal_acf[lag]
        sig = '★ YES' if significant[lag] else 'no'
        bar = '█' * int(abs(r) / 1.0 * 20)
        print(f"  lag {lag:<4}  {r:<12.4f} {str(abs(r) > ci):<12} {sig:<8} {bar}")
    print(f"  {'─'*50}")
    print(f"  r(s={period}) strength : {strength}")
    print(f"  Verdict          : {verdict}")
    print(f"{'='*54}\n")

    # plot ACF with seasonal lag markers
    fig, ax = plt.subplots(figsize=(12, 4))
    lags = np.arange(maxlag + 1)
    ax.bar(lags, acf_vals, color=['#D85A30' if l in seasonal_lags else '#3B8BD4'
                                   for l in lags], width=0.6, alpha=0.8)
    ax.axhline( ci, linestyle='--', color='#BA7517', linewidth=1, label='95% CI')
    ax.axhline(-ci, linestyle='--', color='#BA7517', linewidth=1)
    ax.axhline(0,   linestyle='-',  color='grey',    linewidth=0.5)
    for lag in seasonal_lags:
        ax.axvline(lag, linestyle=':', color='#D85A30', alpha=0.4)
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
    ax.set_title(f'ACF — seasonal lags (s={period}) highlighted in red')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    return {
        "period"         : period,
        "seasonal_acf"   : seasonal_acf,
        "significant"    : significant,
        "strength"       : strength,
        "verdict"        : verdict,
        "ci"             : round(ci, 4),
    }
```

---

### Full Pipeline — Both Methods with Verdict

```python
def seasonality_tests(series, dates, period, period_name='month', alpha=0.05):
    """
    Run both Kruskal-Wallis and ACF seasonal lag tests.
    Print a combined verdict.
    """
    print(f"\n{'#'*56}")
    print(f"  SEASONALITY TESTS — period={period} ({period_name})")
    print(f"{'#'*56}")

    # detrend before testing (remove linear trend)
    from scipy.signal import detrend
    series_dt = detrend(pd.Series(series).dropna().values)

    # test 1 — Kruskal-Wallis
    kw = kruskal_seasonality(series_dt, dates, period=period_name, alpha=alpha)

    # test 2 — ACF at seasonal lags
    ac = seasonal_acf_test(series_dt, period=period, alpha=alpha)

    # combined verdict
    kw_sig = kw['p_value'] < alpha
    ac_sig = ac['significant'].get(period, False)

    if kw_sig and ac_sig:
        final = f"STRONG EVIDENCE of seasonality (period={period}) — both tests agree"
        action = f"Use SARIMA(p,d,q)(P,D,Q)[{period}] or STL decomposition"
    elif kw_sig or ac_sig:
        final = f"WEAK EVIDENCE of seasonality — one test significant"
        action = "Inspect visually; consider seasonal differencing as a precaution"
    else:
        final = "NO SEASONALITY DETECTED — both tests agree"
        action = "Use ARIMA without seasonal terms"

    print(f"\n{'='*56}")
    print(f"  Combined Verdict")
    print(f"{'='*56}")
    print(f"  Kruskal-Wallis : {'significant' if kw_sig else 'not significant'}"
          f"  (p={kw['p_value']:.4f})")
    print(f"  ACF lag {period:<5}  : {'significant' if ac_sig else 'not significant'}"
          f"  (r={ac['seasonal_acf'].get(period, 0):.4f})")
    print(f"  {'─'*52}")
    print(f"  Final    : {final}")
    print(f"  Action   : {action}")
    print(f"{'='*56}\n")

    return {"kruskal": kw, "acf": ac, "final": final, "action": action}
```

---

### On a Real DataFrame

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# monthly data — test for annual seasonality (s=12)
results = seasonality_tests(
    series      = df['meantemp'],
    dates       = df['date'],
    period      = 12,
    period_name = 'month',
)

# daily data — test for weekly seasonality (s=7)
results = seasonality_tests(
    series      = df['value'],
    dates       = df['date'],
    period      = 7,
    period_name = 'dayofweek',
)

# if unsure of period — plot ACF and look for the first large spike
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['meantemp'], lags=60)
# look for the first significant spike beyond lag 1 → that lag is s
```

---

## Interpreting Results

**Both tests significant — strong seasonality:**
The series has a repeating pattern at period s. Both the group medians
(Kruskal-Wallis) and the autocorrelation structure (ACF) confirm it. Add
seasonal terms to the model.

**Only Kruskal-Wallis significant:**
The seasonal groups have different medians but the correlation structure is
not strong. This suggests a **deterministic** seasonal effect (fixed offsets
per season) rather than a stochastic one. Seasonal dummies or STL
decomposition may be more appropriate than SARIMA.

**Only ACF seasonal lag significant:**
The series shows autocorrelation at lag s but the group medians do not differ
significantly. This suggests a **stochastic** seasonal component — the
seasonality fluctuates in strength over time. SARIMA handles this better than
seasonal dummies.

**Neither significant:**
No evidence of seasonality at the tested period. Either there is truly no
seasonality, or the period is different from what was tested. Try other
periods or inspect the raw ACF plot visually.

---

## What to Do After Detecting Seasonality

**Option 1 — Seasonal differencing**
Removes deterministic seasonality, leaving the non-seasonal structure:

```python
df['seasonal_diff'] = df['value'].diff(period)   # e.g. diff(12) for monthly

# re-run stationarity and trend tests on the differenced series
# then fit ARIMA on the result
```

**Option 2 — SARIMA**
Models both the non-seasonal and seasonal structure simultaneously:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA(p,d,q)(P,D,Q)[s]
model = SARIMAX(
    df['value'],
    order=(1, 1, 1),            # non-seasonal terms
    seasonal_order=(1, 1, 1, 12)  # seasonal terms + period
).fit()
print(model.summary())
```

**Option 3 — STL decomposition**
Separates the series into trend, seasonal, and residual components.
The residual is then modelled with ARIMA:

```python
from statsmodels.tsa.seasonal import STL

stl    = STL(df['value'], period=12).fit()
trend  = stl.trend
seasonal = stl.seasonal
resid  = stl.resid

stl.plot()

# model the residual
from statsmodels.tsa.arima.model import ARIMA
resid_model = ARIMA(resid, order=(1, 0, 0)).fit()
```

**Option 4 — Seasonal dummies**
Add binary dummy variables for each season (except one as baseline):

```python
dummies = pd.get_dummies(df['month'], prefix='m', drop_first=True)
df = pd.concat([df, dummies], axis=1)
# use as exogenous variables in SARIMAX(exog=dummies)
```

---

## Common Mistakes

**1. Not detrending before Kruskal-Wallis**
A linear trend pushes later seasons to higher ranks systematically. This
makes the test reject the null even when there is no true seasonality —
only a trend. Always detrend or first-difference before running the test.

```python
from scipy.signal import detrend
series_dt = detrend(series)    # remove linear trend first
kruskal_seasonality(series_dt, dates, period='month')
```

**2. Testing the wrong period**
If your data has weekly seasonality but you test for monthly groups, the
test will fail to detect it. When unsure, plot the full ACF first and look
for the first large spike beyond lag 1.

```python
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, lags=60)   # look for the first spike → that is s
```

**3. Too few seasons of data**
Kruskal-Wallis needs at least 5 observations per season group. For monthly
data, this means at least 5 × 12 = 60 observations (5 years). Fewer than
this and the test has low power.

**4. Checking only lag s in ACF**
True seasonality produces spikes at s, 2s, 3s, ... If only lag s is
significant and 2s and 3s are not, it may be coincidental. Require at
least 2 of the 3 seasonal lags to be significant for confidence.

**5. Confusing seasonality with cyclicality**
Seasonality has a **fixed period** (every 12 months, every 7 days).
Cyclicality is a longer-term wave with a **variable period** (economic
cycles of 3–10 years). ACF and Kruskal-Wallis test for fixed seasonality
only — cyclicality requires spectral analysis.

**6. Applying seasonal differencing when there is no seasonality**
Over-differencing destroys structure. Only apply `diff(s)` if tests confirm
seasonality at period s. Check that the series is not already stationary
without seasonal differencing.

---

## References

- Kruskal, W.H. & Wallis, W.A. (1952). *Use of ranks in one-criterion
  variance analysis.* Journal of the American Statistical Association,
  47(260), 583–621.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice* (3rd ed.). Chapter 9.
  https://otexts.com/fpp3/seasonality.html
- Cleveland, R.B., Cleveland, W.S., McRae, J.E. & Terpenning, I. (1990).
  *STL: A seasonal-trend decomposition procedure based on loess.*
  Journal of Official Statistics, 6(1), 3–33.
- scipy kruskal:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
- statsmodels STL:
  https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html
- statsmodels SARIMAX:
  https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html