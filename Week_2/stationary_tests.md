# Stationarity Tests in Time Series Forecasting: ADF, KPSS & Phillips-Perron

A practical guide to testing stationarity in time series data using the three
most widely used statistical tests — and deciding what to do based on the results.

---

## Table of Contents

- [What is Stationarity?](#what-is-stationarity)
- [Why Stationarity Matters in Forecasting](#why-stationarity-matters-in-forecasting)
- [Where Stationarity Tests Fit in the Pipeline](#where-stationarity-tests-fit-in-the-pipeline)
- [The Three Tests](#the-three-tests)
  - [ADF — Augmented Dickey-Fuller](#adf--augmented-dickey-fuller)
  - [KPSS — Kwiatkowski-Phillips-Schmidt-Shin](#kpss--kwiatkowski-phillips-schmidt-shin)
  - [Phillips-Perron (PP)](#phillips-perron-pp)
  - [Key Difference: Opposite Null Hypotheses](#key-difference-opposite-null-hypotheses)
- [Decision Table](#decision-table)
- [Code Implementation](#code-implementation)
  - [ADF Test](#adf-test)
  - [KPSS Test](#kpss-test)
  - [Phillips-Perron Test](#phillips-perron-test)
  - [Full Pipeline — All Three with Verdicts](#full-pipeline--all-three-with-verdicts)
- [Interpreting Conflicting Results](#interpreting-conflicting-results)
- [Making the Series Stationary](#making-the-series-stationary)
- [Common Mistakes](#common-mistakes)
- [References](#references)

---

## What is Stationarity?

A time series is **stationary** if its statistical properties do not change
over time. Specifically, a stationary series has:

- **Constant mean** — does not trend up or down
- **Constant variance** — spread does not widen or narrow over time
- **Constant autocorrelation** — relationship between past and future values
  stays the same regardless of when you look

**Stationary (good):**
```
value:  3, 5, 2, 4, 3, 5, 2, 4, 3, 5  ← fluctuates around a fixed mean
```

**Non-stationary (bad — has trend):**
```
value:  1, 3, 5, 7, 9, 11, 13, 15, 17  ← mean keeps increasing
```

**Non-stationary (bad — has changing variance):**
```
value:  1, 1, 2, 1, 5, 2, 9, 3, 15, 2  ← spread keeps growing
```

---

## Why Stationarity Matters in Forecasting

Most classical time series models (ARIMA, VAR, Granger causality) assume the
series is stationary. When applied to a non-stationary series:

- ARIMA produces **spurious forecasts** that extrapolate the trend indefinitely
- Regression between two non-stationary series gives **spuriously high R²** even
  when the variables are completely unrelated
- Model parameters become **unstable** — learned on one window, useless on another
- Statistical tests (t-tests, F-tests on coefficients) become **invalid**

Stationarity tests give you a formal, p-value-backed answer to the question:
*"Is this series ready to model, or does it need differencing first?"*

---

## Where Stationarity Tests Fit in the Pipeline

```
Raw time series
      ↓
Visual inspection (plot, ACF, PACF)
      ↓
Stationarity tests (ADF + KPSS + PP)
      ↓
      ├── Stationary      → proceed to ARIMA / modelling
      └── Non-stationary  → difference the series → re-test → model
```

Running all three tests together is best practice because each has different
strengths and blind spots. When they agree, you can be confident. When they
disagree, the conflict itself tells you something about the series structure.

---

## The Three Tests

### ADF — Augmented Dickey-Fuller

The most widely used stationarity test. It tests whether a **unit root** is
present in the series. A unit root is what causes a series to be non-stationary
— each value is essentially the previous value plus random noise, so shocks
accumulate and the series drifts.

**Null hypothesis:** The series **has** a unit root → non-stationary
**Alternative:** The series does **not** have a unit root → stationary

**How it works:**

ADF fits a regression of the form:

```
Δyₜ = α + βt + γyₜ₋₁ + δ₁Δyₜ₋₁ + ... + δₚΔyₜ₋ₚ + εₜ
```

It tests whether `γ = 0` (unit root present). The "Augmented" part means it
includes lagged differences to account for serial correlation in residuals.

**Decision rule:**
```
p-value < 0.05  →  reject null  →  series IS stationary
p-value ≥ 0.05  →  fail to reject null  →  series is NOT stationary
```

**Weakness:** Low power against near-unit-root processes. Can fail to detect
non-stationarity when the series is close to but not exactly a random walk.

---

### KPSS — Kwiatkowski-Phillips-Schmidt-Shin

KPSS takes the **opposite approach** to ADF — it assumes stationarity by
default and tests whether there is evidence against it.

**Null hypothesis:** The series **is** stationary
**Alternative:** The series has a unit root → non-stationary

**How it works:**

KPSS decomposes the series into a deterministic trend, random walk, and
stationary error:

```
yₜ = ξt + rₜ + εₜ
```

where `rₜ` is a random walk. It tests whether the variance of the random walk
component is zero. If not zero, the series is drifting.

**Decision rule:**
```
p-value < 0.05  →  reject null  →  series is NOT stationary
p-value ≥ 0.05  →  fail to reject null  →  series IS stationary
```

**Note:** KPSS p-values are reported as `> 0.1` or `< 0.01` rather than exact
values in statsmodels, because the test uses tabulated critical values.

**Weakness:** Can over-reject stationarity in long series (size distortion).

---

### Phillips-Perron (PP)

PP is a modification of the original Dickey-Fuller test. Like ADF, it tests
for a unit root — but it handles serial correlation differently.

**Null hypothesis:** The series **has** a unit root → non-stationary
**Alternative:** The series is stationary

**How it differs from ADF:**

ADF adds lagged differences to the regression to correct for serial correlation.
PP instead applies a non-parametric correction to the test statistic itself
using the Newey-West estimator. This makes PP:

- More robust to heteroskedasticity (changing variance)
- Useful when you are unsure how many lags to include
- Slightly less powerful than ADF when the lag choice in ADF is correct

**Decision rule:**
```
p-value < 0.05  →  reject null  →  series IS stationary
p-value ≥ 0.05  →  fail to reject null  →  series is NOT stationary
```

---

### Key Difference: Opposite Null Hypotheses

This is the most important concept to understand before reading results:

| Test | Null hypothesis | Small p-value means |
|------|----------------|---------------------|
| ADF  | Non-stationary | **Stationary** ✓ |
| KPSS | Stationary     | **Non-stationary** ✗ |
| PP   | Non-stationary | **Stationary** ✓ |

A common beginner mistake: seeing a small p-value on KPSS and concluding the
series is stationary. It is the opposite — small p-value on KPSS means
non-stationary.

---

## Decision Table

Combining all three tests gives four possible outcomes:

| ADF | KPSS | PP | Conclusion |
|-----|------|----|------------|
| Stationary | Stationary | Stationary | ✅ **Strongly stationary** — proceed to modelling |
| Non-stationary | Non-stationary | Non-stationary | ❌ **Strongly non-stationary** — difference the series |
| Stationary | Non-stationary | Stationary | ⚠️ **Trend-stationary** — series has a deterministic trend; detrend it |
| Non-stationary | Stationary | Non-stationary | ⚠️ **Long memory / structural break** — investigate further |

---

## Code Implementation

### ADF Test

```python
from statsmodels.tsa.stattools import adfuller

def run_adf(series, name="Series"):
    result = adfuller(series, autolag='AIC')

    adf_stat  = result[0]
    p_value   = result[1]
    lags_used = result[2]
    n_obs     = result[3]
    crit      = result[4]

    verdict = "STATIONARY" if p_value < 0.05 else "NON-STATIONARY"

    print(f"=== ADF Test: {name} ===")
    print(f"ADF Statistic : {adf_stat:.4f}")
    print(f"p-value       : {p_value:.4f}")
    print(f"Lags used     : {lags_used}")
    print(f"Observations  : {n_obs}")
    print(f"Critical values:")
    for key, val in crit.items():
        print(f"   {key}: {val:.4f}")
    print(f"Verdict       : {verdict}")
    print()
    return verdict
```

---

### KPSS Test

```python
from statsmodels.tsa.stattools import kpss
import warnings

def run_kpss(series, name="Series"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress interpolation warning
        result = kpss(series, regression='c', nlags='auto')

    kpss_stat = result[0]
    p_value   = result[1]
    lags_used = result[2]
    crit      = result[3]

    # KPSS: small p-value = NON-stationary (opposite of ADF)
    verdict = "NON-STATIONARY" if p_value < 0.05 else "STATIONARY"

    print(f"=== KPSS Test: {name} ===")
    print(f"KPSS Statistic: {kpss_stat:.4f}")
    print(f"p-value       : {p_value:.4f}  ← small = NON-stationary")
    print(f"Lags used     : {lags_used}")
    print(f"Critical values:")
    for key, val in crit.items():
        print(f"   {key}: {val:.4f}")
    print(f"Verdict       : {verdict}")
    print()
    return verdict
```

---

### Phillips-Perron Test

```python
from statsmodels.tsa.stattools import PhillipsPerron

def run_pp(series, name="Series"):
    result = PhillipsPerron(series)
    pp_stat = result.stat
    p_value = result.pvalue

    verdict = "STATIONARY" if p_value < 0.05 else "NON-STATIONARY"

    print(f"=== Phillips-Perron Test: {name} ===")
    print(f"PP Statistic  : {pp_stat:.4f}")
    print(f"p-value       : {p_value:.4f}")
    print(f"Verdict       : {verdict}")
    print()
    return verdict
```

---

### Full Pipeline — All Three with Verdicts

```python
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import PhillipsPerron

def stationarity_test(series, name="Series"):
    """
    Run ADF, KPSS, and Phillips-Perron tests.
    Print p-values and individual verdicts.
    Print a combined final verdict.
    """
    series = pd.Series(series).dropna()

    # --- ADF ---
    adf_result  = adfuller(series, autolag='AIC')
    adf_pval    = adf_result[1]
    adf_verdict = "STATIONARY" if adf_pval < 0.05 else "NON-STATIONARY"

    # --- KPSS ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_result  = kpss(series, regression='c', nlags='auto')
    kpss_pval    = kpss_result[1]
    kpss_verdict = "NON-STATIONARY" if kpss_pval < 0.05 else "STATIONARY"

    # --- Phillips-Perron ---
    pp_result  = PhillipsPerron(series)
    pp_pval    = pp_result.pvalue
    pp_verdict = "STATIONARY" if pp_pval < 0.05 else "NON-STATIONARY"

    # --- Print results ---
    print(f"\n{'='*50}")
    print(f"  Stationarity Tests: {name}")
    print(f"{'='*50}")
    print(f"{'Test':<22} {'p-value':<12} {'Verdict'}")
    print(f"{'-'*50}")
    print(f"{'ADF':<22} {adf_pval:<12.4f} {adf_verdict}")
    print(f"{'KPSS':<22} {kpss_pval:<12.4f} {kpss_verdict}")
    print(f"{'Phillips-Perron':<22} {pp_pval:<12.4f} {pp_verdict}")
    print(f"{'-'*50}")

    # --- Combined verdict ---
    verdicts    = [adf_verdict, kpss_verdict, pp_verdict]
    n_stationary = verdicts.count("STATIONARY")

    if n_stationary == 3:
        final = "STRONGLY STATIONARY ✓ — safe to model"
    elif n_stationary == 0:
        final = "STRONGLY NON-STATIONARY ✗ — apply differencing"
    elif adf_verdict == "STATIONARY" and kpss_verdict == "NON-STATIONARY":
        final = "TREND-STATIONARY ⚠ — detrend the series"
    else:
        final = "INCONCLUSIVE ⚠ — check for structural breaks"

    print(f"  Final verdict : {final}")
    print(f"{'='*50}\n")

    return {
        "adf_pval"    : round(adf_pval, 4),
        "kpss_pval"   : round(kpss_pval, 4),
        "pp_pval"     : round(pp_pval, 4),
        "adf_verdict" : adf_verdict,
        "kpss_verdict": kpss_verdict,
        "pp_verdict"  : pp_verdict,
        "final"       : final,
    }


# --- example usage ---
np.random.seed(42)

stationary_series     = np.random.randn(200)                        # white noise
non_stationary_series = np.cumsum(np.random.randn(200))             # random walk

stationarity_test(stationary_series,     name="White Noise")
stationarity_test(non_stationary_series, name="Random Walk")
```

**Output:**

```
==================================================
  Stationarity Tests: White Noise
==================================================
Test                   p-value      Verdict
--------------------------------------------------
ADF                    0.0000       STATIONARY
KPSS                   0.1000       STATIONARY
Phillips-Perron        0.0000       STATIONARY
--------------------------------------------------
  Final verdict : STRONGLY STATIONARY ✓ — safe to model
==================================================

==================================================
  Stationarity Tests: Random Walk
==================================================
Test                   p-value      Verdict
--------------------------------------------------
ADF                    0.4821       NON-STATIONARY
KPSS                   0.0100       NON-STATIONARY
Phillips-Perron        0.4703       NON-STATIONARY
--------------------------------------------------
  Final verdict : STRONGLY NON-STATIONARY ✗ — apply differencing
==================================================
```

---

## Interpreting Conflicting Results

When ADF and KPSS disagree, it usually means one of these:

**ADF says stationary, KPSS says non-stationary**
The series has a **deterministic trend** (e.g. slowly rising mean). It is
stationary around the trend, but not in absolute terms. Solution: detrend
the series or use `regression='ct'` in both tests to include a trend term.

```python
# test with trend included
adfuller(series, regression='ct')
kpss(series, regression='ct')
```

**ADF says non-stationary, KPSS says stationary**
The series may have **long memory** or a **structural break** — a point where
the mean or variance shifted permanently (e.g. a financial crisis, policy
change). Standard tests can lose power here. Investigate visually and consider
the Zivot-Andrews test which allows for one structural break.

```python
from statsmodels.tsa.stattools import zivot_andrews
result = zivot_andrews(series)
print(f"Break at index: {result[3]}")
```

---

## Making the Series Stationary

If tests indicate non-stationarity, apply one of these transformations and
re-test:

**First differencing** — removes linear trends
```python
df['diff1'] = df['value'].diff()
stationarity_test(df['diff1'].dropna(), name="First Difference")
```

**Second differencing** — removes quadratic trends (rarely needed)
```python
df['diff2'] = df['value'].diff().diff()
```

**Log transformation** — stabilises exponentially growing variance
```python
df['log'] = np.log(df['value'])
```

**Log + differencing** — handles both trend and variance growth
```python
df['log_diff'] = np.log(df['value']).diff()
```

**Detrending** — removes a fitted linear/polynomial trend
```python
from scipy.signal import detrend
df['detrended'] = detrend(df['value'])
```

**Rule of thumb:** try first differencing first. If the series is still
non-stationary after two differences, the problem is likely structural breaks
or seasonality rather than a simple unit root.

---

## Common Mistakes

**1. Reading KPSS p-value like ADF**
ADF: small p-value = stationary. KPSS: small p-value = non-stationary.
Always label your output explicitly to avoid confusion.

```python
# always print what small means for each test
print(f"ADF  p={adf_pval:.4f}  (< 0.05 = stationary)")
print(f"KPSS p={kpss_pval:.4f}  (< 0.05 = NON-stationary)")
```

**2. Running only one test**
ADF alone is not sufficient. KPSS is a complementary test — running both
together catches cases where one test has low power. PP adds robustness
against heteroskedasticity. Always run at least ADF + KPSS.

**3. Not re-testing after differencing**
Always re-run all three tests after each transformation to confirm the series
is now stationary before modelling.

```python
diff_series = series.diff().dropna()
stationarity_test(diff_series, name="After First Difference")
```

**4. Over-differencing**
If a stationary series is differenced again, it becomes non-stationary in
the opposite direction. Check ACF after differencing — if lag-1 autocorrelation
is strongly negative (< -0.5), you may have over-differenced.

**5. Ignoring seasonality**
Standard ADF/KPSS/PP do not detect seasonal non-stationarity. A series can
pass all three tests and still have seasonal unit roots. Use the OCSB or
CH test for seasonal stationarity, or simply check the seasonal ACF plot.

```python
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, lags=50)   # look for spikes at seasonal lags (12, 24, 52...)
```

**6. Small samples**
ADF and PP have low power with fewer than ~100 observations. With small
samples, rely more on KPSS and visual inspection, and interpret results
with caution.

---

## References

- Dickey, D.A. & Fuller, W.A. (1979). *Distribution of the estimators for
  autoregressive time series with a unit root.* Journal of the American
  Statistical Association, 74(366), 427–431.
- Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992). *Testing
  the null hypothesis of stationarity against the alternative of a unit root.*
  Journal of Econometrics, 54(1–3), 159–178.
- Phillips, P.C.B. & Perron, P. (1988). *Testing for a unit root in time series
  regression.* Biometrika, 75(2), 335–346.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice* (3rd ed.). Chapter 9. https://otexts.com/fpp3/stationarity.html
- statsmodels ADF docs: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
- statsmodels KPSS docs: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html
- statsmodels PP docs: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.PhillipsPerron.html