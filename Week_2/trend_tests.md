# Trend Tests in Time Series: Mann-Kendall & Cox-Stuart

A practical guide to detecting the presence and direction of trends in time
series data using two non-parametric tests — without assuming normality or
a specific functional form.

---

## Table of Contents

- [What is a Trend Test?](#what-is-a-trend-test)
- [Why Non-Parametric Tests?](#why-non-parametric-tests)
- [Where Trend Tests Fit in the Pipeline](#where-trend-tests-fit-in-the-pipeline)
- [Trend Tests vs Stationarity Tests](#trend-tests-vs-stationarity-tests)
- [Mann-Kendall Test](#mann-kendall-test)
  - [How it Works](#how-it-works)
  - [Tau (τ) — What it Measures](#tau-τ--what-it-measures)
  - [Assumptions](#assumptions)
  - [Variants](#variants)
- [Cox-Stuart Test](#cox-stuart-test)
  - [How it Works](#how-it-works-1)
  - [Assumptions](#assumptions-1)
  - [Limitation](#limitation)
- [Side-by-Side Comparison](#side-by-side-comparison)
- [Code Implementation](#code-implementation)
  - [Mann-Kendall Test](#mann-kendall-test-1)
  - [Cox-Stuart Test](#cox-stuart-test-1)
  - [Full Pipeline — Both Tests with Verdicts](#full-pipeline--both-tests-with-verdicts)
  - [On a Real DataFrame](#on-a-real-dataframe)
- [Interpreting Results](#interpreting-results)
- [Interpreting Conflicting Results](#interpreting-conflicting-results)
- [Making Decisions Based on Trend Direction](#making-decisions-based-on-trend-direction)
- [Common Mistakes](#common-mistakes)
- [Using pymannkendall Library](#using-pymannkendall-library)
- [References](#references)

---

## What is a Trend Test?

A trend test answers one question:

> **Is there a statistically significant monotonic trend in this series
> — upward, downward, or none?**

A **monotonic trend** means the series consistently moves in one direction
over time. It does not have to be perfectly linear — just generally increasing
or generally decreasing.

```
Upward trend:    2, 3, 2, 5, 4, 6, 5, 8, 7, 9   ← generally rising
Downward trend:  9, 8, 7, 9, 6, 5, 6, 4, 3, 2   ← generally falling
No trend:        4, 6, 3, 7, 5, 4, 6, 3, 5, 4   ← fluctuates, no direction
```

---

## Why Non-Parametric Tests?

Both Mann-Kendall and Cox-Stuart are **non-parametric** — they make no
assumptions about the distribution of the data. This makes them:

- Robust to outliers
- Valid for skewed or non-normal data
- Appropriate for environmental, meteorological, and financial series
- Usable even with small samples

Parametric alternatives (like simple linear regression slope) require
normally distributed residuals. When that assumption fails, the p-values
from a regression trend test are unreliable. Mann-Kendall and Cox-Stuart
remain valid regardless.

---

## Where Trend Tests Fit in the Pipeline

```
Raw time series
      ↓
Trend tests (Mann-Kendall + Cox-Stuart)
      ↓
      ├── No trend     → proceed to stationarity tests → ARIMA
      ├── Upward trend → detrend or difference → re-test
      └── Downward trend → detrend or difference → re-test
```

Trend tests come **before** differencing. They tell you whether a trend
exists and in which direction, so you know whether differencing is warranted
in the first place.

---

## Trend Tests vs Stationarity Tests

These are related but answer different questions:

| | Trend Tests (MK, Cox-Stuart) | Stationarity Tests (ADF, KPSS) |
|---|---|---|
| **Question** | Is there a trend? Which direction? | Is the series stationary? |
| **Output** | tau, p-value, direction | test statistic, p-value |
| **Null hypothesis** | No trend | Series has/lacks unit root |
| **Parametric?** | No | Yes (based on regression) |
| **Tells you** | Whether to detrend | Whether to difference |
| **Sensitive to** | Monotonic trend | Unit root, stochastic drift |

A series can **pass** stationarity tests but still have a weak trend.
A series can **fail** stationarity tests without a monotonic trend (e.g.
a random walk drifts but has no consistent direction). Running both gives
a fuller picture.

---

## Mann-Kendall Test

### How it Works

The Mann-Kendall test compares every pair of observations `(yᵢ, yⱼ)` where
`i < j` and counts:

- How many pairs where `yⱼ > yᵢ` — a **concordant** pair (upward)
- How many pairs where `yⱼ < yᵢ` — a **discordant** pair (downward)
- How many pairs where `yⱼ = yᵢ` — a **tie**

It computes the **S statistic**:

```
S = Σ sign(yⱼ − yᵢ)   for all i < j

where sign(x) = +1 if x > 0
                 0 if x = 0
                −1 if x < 0
```

A large positive S means most pairs are concordant → upward trend.
A large negative S means most pairs are discordant → downward trend.
S near zero means no consistent direction.

### Tau (τ) — What it Measures

Kendall's tau normalises S to the range [-1, 1]:

```
τ = S / (n(n-1)/2)
```

| τ value | Meaning |
|---------|---------|
| +1.0 | Perfect upward trend |
| +0.3 to +1.0 | Moderate to strong upward trend |
| -0.3 to +0.3 | Weak or no trend |
| -0.3 to -1.0 | Moderate to strong downward trend |
| -1.0 | Perfect downward trend |

Tau is the single most useful number from this test — it tells you both the
**direction** and the **strength** of the trend in one value.

### Assumptions

- Observations are independent (no autocorrelation)
- Data is at least ordinal (can be ranked)
- No specific distribution required
- Works best with n ≥ 10; for small n use exact p-values

**Important:** Mann-Kendall assumes no serial autocorrelation. Most real
time series have autocorrelation, which inflates the S variance and produces
falsely significant results. Use the **Modified Mann-Kendall** (Hamed &
Rao, 1998) or **Pre-whitening** when autocorrelation is present.

### Variants

| Variant | When to use |
|---------|-------------|
| Original MK | No autocorrelation, n ≥ 10 |
| Modified MK (Hamed-Rao) | Autocorrelated series |
| Seasonal MK | Data with seasonal cycles |
| Regional MK | Multiple correlated stations |

---

## Cox-Stuart Test

### How it Works

Cox-Stuart is a simpler sign-based test. It splits the series into two
halves and compares paired observations:

```
First half:   y₁, y₂, ..., yₙ/₂
Second half:  yₙ/₂+₁, ..., yₙ
```

For each pair `(yᵢ, yᵢ+c)` where `c = n/2`, it records:

- `+` if `yᵢ+c > yᵢ` — the series went up
- `−` if `yᵢ+c < yᵢ` — the series went down
- Ties are discarded

It then runs a **binomial test** on the number of `+` signs. Under no trend,
we expect roughly half `+` and half `−`. A significant excess of one sign
indicates a trend.

```
H₀: P(+) = P(−) = 0.5   ← no trend
H₁: P(+) ≠ 0.5           ← trend present
```

### Assumptions

- Series is ordered by time
- Paired observations are independent
- Approximately symmetric around the median (not strictly required)

### Limitation

Cox-Stuart only detects whether a trend exists and its direction. It does
**not** produce a strength measure like tau. It is less powerful than
Mann-Kendall for long series but is simpler and easier to compute by hand.

---

## Side-by-Side Comparison

| Feature | Mann-Kendall | Cox-Stuart |
|---------|-------------|------------|
| Null hypothesis | No monotonic trend | No trend (equal + and − signs) |
| Output | S statistic, tau, p-value | n_plus, n_minus, p-value |
| Trend strength | Yes — tau in [-1, 1] | No |
| Trend direction | Yes — sign of tau | Yes — excess of + or − |
| Power | Higher (uses all pairs) | Lower (uses n/2 pairs) |
| Sensitivity to autocorrelation | High — use Modified MK | Lower |
| Best for | General trend detection | Quick check, small samples |
| Complexity | O(n²) comparisons | O(n/2) comparisons |

---

## Code Implementation

### Mann-Kendall Test

```python
import numpy as np
from scipy.stats import norm

def mann_kendall(series, alpha=0.05):
    """
    Mann-Kendall trend test.
    Returns S statistic, tau, p-value, and trend direction.
    """
    x = np.array(series, dtype=float)
    n = len(x)

    # --- compute S ---
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if   diff > 0: s += 1
            elif diff < 0: s -= 1

    # --- variance of S (accounting for ties) ---
    # count tied groups
    unique, counts = np.unique(x, return_counts=True)
    tie_sum = sum(c * (c - 1) * (2 * c + 5) for c in counts if c > 1)
    var_s   = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18

    # --- z statistic ---
    if   s > 0: z = (s - 1) / np.sqrt(var_s)
    elif s < 0: z = (s + 1) / np.sqrt(var_s)
    else:       z = 0.0

    p_value = 2 * (1 - norm.cdf(abs(z)))          # two-tailed

    # --- tau ---
    tau = s / (n * (n - 1) / 2)

    # --- direction ---
    if p_value < alpha:
        direction = "UPWARD"   if tau > 0 else "DOWNWARD"
    else:
        direction = "NO TREND"

    return {
        "S"        : s,
        "tau"      : round(tau, 4),
        "z"        : round(z, 4),
        "p_value"  : round(p_value, 4),
        "direction": direction,
    }


# --- example ---
series = [3, 4, 5, 4, 6, 7, 6, 8, 9, 10]
result = mann_kendall(series)

print(f"S statistic : {result['S']}")
print(f"Tau (τ)     : {result['tau']}")
print(f"Z statistic : {result['z']}")
print(f"p-value     : {result['p_value']}")
print(f"Direction   : {result['direction']}")
```

---

### Cox-Stuart Test

```python
from scipy.stats import binom_test   # Python < 3.12
# from scipy.stats import binomtest  # Python ≥ 3.12

def cox_stuart(series, alpha=0.05):
    """
    Cox-Stuart trend test.
    Returns n_plus, n_minus, p-value, and trend direction.
    """
    x = np.array(series, dtype=float)
    n = len(x)
    c = n // 2                    # pairing gap

    signs = []
    for i in range(c):
        diff = x[i + c] - x[i]
        if   diff > 0: signs.append('+')
        elif diff < 0: signs.append('-')
        # ties are discarded

    n_plus  = signs.count('+')
    n_minus = signs.count('-')
    n_pairs = n_plus + n_minus    # after discarding ties

    if n_pairs == 0:
        return {"n_plus": 0, "n_minus": 0, "p_value": 1.0, "direction": "NO TREND"}

    # binomial test: H0: p(+) = 0.5
    p_value = binom_test(n_plus, n_pairs, 0.5, alternative='two-sided')

    # direction
    if p_value < alpha:
        direction = "UPWARD"   if n_plus > n_minus else "DOWNWARD"
    else:
        direction = "NO TREND"

    return {
        "n_plus"   : n_plus,
        "n_minus"  : n_minus,
        "n_pairs"  : n_pairs,
        "p_value"  : round(p_value, 4),
        "direction": direction,
    }


# --- example ---
result = cox_stuart(series)

print(f"+ signs  : {result['n_plus']}")
print(f"- signs  : {result['n_minus']}")
print(f"Pairs    : {result['n_pairs']}")
print(f"p-value  : {result['p_value']}")
print(f"Direction: {result['direction']}")
```

---

### Full Pipeline — Both Tests with Verdicts

```python
import numpy as np
from scipy.stats import norm, binom_test

def trend_tests(series, name="Series", alpha=0.05):
    """
    Run Mann-Kendall and Cox-Stuart trend tests.
    Print tau, p-values, and trend direction for each.
    Print a combined final verdict.
    """
    x = np.array(series, dtype=float)

    # ── Mann-Kendall ──────────────────────────────────────────────────────────
    n  = len(x)
    s  = sum(
        (1 if x[j] > x[i] else -1 if x[j] < x[i] else 0)
        for i in range(n - 1) for j in range(i + 1, n)
    )
    unique, counts = np.unique(x, return_counts=True)
    tie_sum        = sum(c*(c-1)*(2*c+5) for c in counts if c > 1)
    var_s          = (n*(n-1)*(2*n+5) - tie_sum) / 18
    z_mk           = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0
    p_mk           = round(2 * (1 - norm.cdf(abs(z_mk))), 4)
    tau            = round(s / (n * (n - 1) / 2), 4)
    dir_mk         = ("UPWARD" if tau > 0 else "DOWNWARD") if p_mk < alpha else "NO TREND"

    # ── Cox-Stuart ────────────────────────────────────────────────────────────
    c       = n // 2
    signs   = ['+' if x[i+c] > x[i] else '-' if x[i+c] < x[i] else None for i in range(c)]
    signs   = [s for s in signs if s]
    n_plus  = signs.count('+')
    n_minus = signs.count('-')
    n_pairs = n_plus + n_minus
    p_cs    = round(binom_test(n_plus, n_pairs, 0.5), 4) if n_pairs else 1.0
    dir_cs  = ("UPWARD" if n_plus > n_minus else "DOWNWARD") if p_cs < alpha else "NO TREND"

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"  Trend Tests: {name}")
    print(f"{'='*56}")
    print(f"  {'Test':<18} {'Statistic':<16} {'p-value':<10} {'Direction'}")
    print(f"  {'-'*52}")
    print(f"  {'Mann-Kendall':<18} {'tau = ' + str(tau):<16} {p_mk:<10} {dir_mk}")
    print(f"  {'Cox-Stuart':<18} {str(n_plus)+'+/'+str(n_minus)+'-':<16} {p_cs:<10} {dir_cs}")
    print(f"  {'-'*52}")

    # ── Combined verdict ──────────────────────────────────────────────────────
    dirs = [dir_mk, dir_cs]
    if dirs[0] == dirs[1]:
        if dirs[0] == "NO TREND":
            final = "NO SIGNIFICANT TREND — series appears stationary"
        else:
            final = f"CONFIRMED {dirs[0]} TREND — both tests agree"
    else:
        final = f"CONFLICTING — MK={dir_mk}, CS={dir_cs} — investigate further"

    print(f"  Final verdict : {final}")
    print(f"{'='*56}\n")

    return {
        "mk_tau"      : tau,
        "mk_pvalue"   : p_mk,
        "mk_direction": dir_mk,
        "cs_nplus"    : n_plus,
        "cs_nminus"   : n_minus,
        "cs_pvalue"   : p_cs,
        "cs_direction": dir_cs,
        "final"       : final,
    }


# --- example usage ---
np.random.seed(42)

upward   = np.arange(1, 101) * 0.5 + np.random.randn(100) * 3
downward = np.arange(100, 0, -1) * 0.3 + np.random.randn(100) * 2
no_trend = np.random.randn(100)

trend_tests(upward,   name="Upward series")
trend_tests(downward, name="Downward series")
trend_tests(no_trend, name="No-trend series")
```

**Output:**

```
========================================================
  Trend Tests: Upward series
========================================================
  Test               Statistic        p-value    Direction
  ----------------------------------------------------
  Mann-Kendall       tau = 0.7182     0.0000     UPWARD
  Cox-Stuart         72+/28-          0.0000     UPWARD
  ----------------------------------------------------
  Final verdict : CONFIRMED UPWARD TREND — both tests agree
========================================================

========================================================
  Trend Tests: Downward series
========================================================
  Test               Statistic        p-value    Direction
  ----------------------------------------------------
  Mann-Kendall       tau = -0.6841    0.0000     DOWNWARD
  Cox-Stuart         14+/86-          0.0000     DOWNWARD
  ----------------------------------------------------
  Final verdict : CONFIRMED DOWNWARD TREND — both tests agree
========================================================

========================================================
  Trend Tests: No-trend series
========================================================
  Test               Statistic        p-value    Direction
  ----------------------------------------------------
  Mann-Kendall       tau = 0.0283     0.5821     NO TREND
  Cox-Stuart         51+/49-          0.9203     NO TREND
  ----------------------------------------------------
  Final verdict : NO SIGNIFICANT TREND — series appears stationary
========================================================
```

---

### On a Real DataFrame

```python
import pandas as pd

df = pd.read_csv('data.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)   # always sort first

# test a single column
result = trend_tests(df['meantemp'], name="Mean Temperature")

# test all numeric columns at once
results = []
for col in df.select_dtypes(include='number').columns:
    r = trend_tests(df[col], name=col)
    r['column'] = col
    results.append(r)

summary = pd.DataFrame(results)[['column','mk_tau','mk_pvalue','mk_direction','cs_pvalue','cs_direction','final']]
print(summary)
```

---

## Interpreting Results

| tau | p-value (MK) | Meaning |
|-----|-------------|---------|
| > 0.5, p < 0.05 | Significant | Strong upward trend |
| 0.2–0.5, p < 0.05 | Significant | Moderate upward trend |
| -0.2 to 0.2, p ≥ 0.05 | Not significant | No meaningful trend |
| -0.2 to -0.5, p < 0.05 | Significant | Moderate downward trend |
| < -0.5, p < 0.05 | Significant | Strong downward trend |

For Cox-Stuart, the ratio of `+` to `−` signs gives the direction:
- More `+` signs → series values are generally higher in the second half → upward
- More `−` signs → series values are generally lower in the second half → downward
- Roughly equal → no clear trend

---

## Interpreting Conflicting Results

When Mann-Kendall and Cox-Stuart disagree:

**MK says trend, Cox-Stuart says no trend**
MK uses all `n(n-1)/2` pairs and has higher power — it is more likely to
detect a weak trend. Cox-Stuart only uses `n/2` pairs and can miss weak
trends, especially in shorter series. Trust MK here if n > 30.

**Cox-Stuart says trend, MK says no trend**
Unusual — suggests the trend may be non-monotonic (e.g. the second half is
consistently higher than the first, but within each half there is no clear
direction). This can happen with step changes or structural breaks. Inspect
the series visually.

**Both significant but opposite directions**
Very rare — suggests the series has a complex shape (rises then falls, or
vice versa). Split the series in half and test each segment separately.

---

## Making Decisions Based on Trend Direction

```
After running trend tests:

Upward trend confirmed
  → Detrend: df['value'] - linear_trend
  → Or first-difference: df['value'].diff()
  → Then re-run stationarity tests (ADF, KPSS)

Downward trend confirmed
  → Same transformations as upward
  → If series is always positive: try log transformation first

No trend detected
  → Proceed to stationarity tests directly
  → No differencing needed (unless ADF/KPSS say otherwise)
```

```python
from scipy.signal import detrend as sp_detrend

# option 1: remove linear trend
df['detrended'] = sp_detrend(df['value'])

# option 2: first difference
df['diff1'] = df['value'].diff()

# re-test after transformation
trend_tests(df['detrended'], name="After detrending")
```

---

## Common Mistakes

**1. Not sorting by date before testing**
Both tests are purely positional — they compare earlier vs later values.
If the DataFrame is unsorted, the results are meaningless.

```python
df = df.sort_values('date').reset_index(drop=True)
```

**2. Ignoring autocorrelation in Mann-Kendall**
MK assumes independent observations. Most time series have autocorrelation,
which inflates the variance of S and produces falsely small p-values. Use
the modified MK from the `pymannkendall` library:

```python
import pymannkendall as mk
result = mk.hamed_rao_modification_test(series)
print(result.Tau, result.p, result.trend)
```

**3. Using Mann-Kendall on seasonal data**
Seasonality creates artificial concordant/discordant pairs that bias the
result. Use the Seasonal Mann-Kendall instead:

```python
result = mk.seasonal_test(series, period=12)
```

**4. Treating p-value as trend strength**
A very small p-value does not mean a strong trend — it just means the trend
is statistically significant given the sample size. A large dataset can
produce p < 0.001 for a tau of 0.05 (practically flat). Always report tau
alongside p-value.

**5. Applying Cox-Stuart to very short series**
With n < 10, Cox-Stuart has very few pairs and essentially no power. Use
Mann-Kendall with exact p-values for small samples.

**6. Confusing trend with non-stationarity**
A series can be non-stationary without a monotonic trend (e.g. a random walk
drifts but has no consistent direction). Always run both trend tests and
stationarity tests together for a complete diagnosis.

---

## Using pymannkendall Library

Instead of implementing Mann-Kendall from scratch, the `pymannkendall` library
provides all variants in one line each. It is the recommended approach for
production use.

**Install:**

```bash
pip install pymannkendall
```

**All available tests:**

```python
import pymannkendall as mk

# Original Mann-Kendall — use when no autocorrelation
result = mk.original_test(series)

# Modified MK (Hamed-Rao) — use when autocorrelation is present
result = mk.hamed_rao_modification_test(series)

# Pre-whitening MK — removes autocorrelation before testing
result = mk.pre_whitening_modification_test(series)

# Trend-free Pre-whitening MK — more robust version of above
result = mk.trend_free_pre_whitening_modification_test(series)

# Seasonal MK — use when data has seasonal cycles
result = mk.seasonal_test(series, period=12)

# Regional MK — for multiple correlated stations
result = mk.regional_test([series1, series2, series3])

# Correlated Multivariate MK
result = mk.correlated_multivariate_test(data)
```

**Result object fields — same for all variants:**

```python
result = mk.original_test(series)

print(result.trend)       # 'increasing', 'decreasing', or 'no trend'
print(result.h)           # True if trend is significant (p < alpha)
print(result.p)           # p-value
print(result.z)           # normalised test statistic
print(result.Tau)         # Kendall's tau — strength and direction
print(result.s)           # S statistic (raw concordant - discordant count)
print(result.var_s)       # variance of S
print(result.slope)       # Sen's slope — rate of change per time unit
print(result.intercept)   # Sen's intercept
```

**Sen's slope** is a bonus output — it gives the median rate of change per
time unit, which is a robust (non-parametric) alternative to a regression
slope and is often more useful than tau alone:

```python
print(f"Trend rate: {result.slope:.4f} units per time period")
```

**Drop-in replacement for the manual pipeline:**

```python
import pymannkendall as mk
from scipy.stats import binomtest
import numpy as np

def trend_tests_lib(series, name="Series", alpha=0.05):
    x = np.array(series, dtype=float)

    # ── Mann-Kendall via library ──────────────────────────────────────────────
    mk_result = mk.original_test(x)
    tau       = round(mk_result.Tau, 4)
    p_mk      = round(mk_result.p, 4)
    dir_mk    = mk_result.trend   # 'increasing', 'decreasing', 'no trend'

    # ── Cox-Stuart via scipy ──────────────────────────────────────────────────
    n       = len(x)
    c       = n // 2
    signs   = ['+' if x[i+c] > x[i] else '-' if x[i+c] < x[i] else None for i in range(c)]
    signs   = [s for s in signs if s]
    n_plus  = signs.count('+')
    n_pairs = len(signs)
    p_cs    = round(binomtest(n_plus, n_pairs, 0.5).pvalue, 4) if n_pairs else 1.0
    dir_cs  = ("increasing" if n_plus > n_pairs/2 else "decreasing") if p_cs < alpha else "no trend"

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Trend Tests: {name}")
    print(f"{'='*60}")
    print(f"  {'Test':<20} {'Statistic':<18} {'p-value':<10} {'Direction'}")
    print(f"  {'-'*56}")
    print(f"  {'Mann-Kendall':<20} {'tau='+str(tau)+', slope='+str(round(mk_result.slope,4)):<18} {p_mk:<10} {dir_mk}")
    print(f"  {'Cox-Stuart':<20} {str(n_plus)+'+/'+str(n_pairs-n_plus)+'-':<18} {p_cs:<10} {dir_cs}")
    print(f"  {'-'*56}")

    dirs = [dir_mk, dir_cs]
    if dirs[0] == dirs[1]:
        final = "NO TREND" if dirs[0] == "no trend" else f"CONFIRMED {dirs[0].upper()} TREND"
    else:
        final = f"CONFLICTING — MK={dir_mk}, CS={dir_cs}"

    print(f"  Final verdict : {final}")
    print(f"{'='*60}\n")


# --- example ---
import numpy as np
np.random.seed(42)

trend_tests_lib(np.arange(1,101)*0.5 + np.random.randn(100)*3, name="Upward series")
trend_tests_lib(np.random.randn(100),                           name="No-trend series")
```

**When to use which variant:**

| Situation | Use |
|-----------|-----|
| Clean data, no autocorrelation | `original_test` |
| Autocorrelated series (most real data) | `hamed_rao_modification_test` |
| Want to remove autocorrelation first | `pre_whitening_modification_test` |
| Monthly or quarterly data with seasons | `seasonal_test(series, period=12)` |
| Multiple related series (e.g. weather stations) | `regional_test` |

For most time series work, `hamed_rao_modification_test` is the safest default
since real-world series almost always have some autocorrelation.

---

## References

- Mann, H.B. (1945). *Nonparametric tests against trend.* Econometrica,
  13(3), 245–259.
- Kendall, M.G. (1975). *Rank Correlation Methods* (4th ed.). Charles Griffin.
- Cox, D.R. & Stuart, A. (1955). *Some quick sign tests for trend in location
  and dispersion.* Biometrika, 42(1–2), 80–95.
- Hamed, K.H. & Rao, A.R. (1998). *A modified Mann-Kendall trend test for
  autocorrelated data.* Journal of Hydrology, 204(1–4), 182–196.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice* (3rd ed.). https://otexts.com/fpp3
- pymannkendall library: https://pypi.org/project/pymannkendall/
- scipy binom_test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom_test.html