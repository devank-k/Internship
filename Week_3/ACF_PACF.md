# ACF & PACF in Time Series Forecasting

A detailed guide to autocorrelation and partial autocorrelation — what they
measure, how they are computed, and how to use them to identify the right
forecasting model.

---

## Table of Contents

- [What is Autocorrelation?](#what-is-autocorrelation)
- [Why Autocorrelation Matters in Forecasting](#why-autocorrelation-matters-in-forecasting)
- [Lag-k Correlation — How it is Computed](#lag-k-correlation--how-it-is-computed)
  - [Step-by-step Example](#step-by-step-example)
  - [The Correlogram](#the-correlogram)
- [ACF — Autocorrelation Function](#acf--autocorrelation-function)
  - [Formula](#formula)
  - [What ACF Captures](#what-acf-captures)
  - [Confidence Bands](#confidence-bands)
- [PACF — Partial Autocorrelation Function](#pacf--partial-autocorrelation-function)
  - [Formula](#formula-1)
  - [What PACF Captures](#what-pacf-captures)
  - [How PACF is Computed — Yule-Walker / Levinson-Durbin](#how-pacf-is-computed--yule-walker--levinson-durbin)
- [ACF vs PACF — The Critical Difference](#acf-vs-pacf--the-critical-difference)
- [Signature Patterns for Model Identification](#signature-patterns-for-model-identification)
  - [AR(p) — Autoregressive](#arp--autoregressive)
  - [MA(q) — Moving Average](#maq--moving-average)
  - [ARMA(p,q)](#armapq)
  - [White Noise](#white-noise)
  - [Seasonal Pattern](#seasonal-pattern)
  - [Non-stationary Series](#non-stationary-series)
- [ARIMA Order Selection Using ACF and PACF](#arima-order-selection-using-acf-and-pacf)
- [Code Implementation](#code-implementation)
  - [Plotting ACF and PACF](#plotting-acf-and-pacf)
  - [Computing Raw Values](#computing-raw-values)
  - [Full Diagnostic Pipeline](#full-diagnostic-pipeline)
- [Common Mistakes](#common-mistakes)
- [References](#references)

---

## What is Autocorrelation?

Autocorrelation is the correlation of a time series with a **delayed copy of
itself**. It answers one question:

> If the value was high (or low) k periods ago, does that tell me anything
> about today's value?

A series with no memory — pure random noise — has zero autocorrelation at all
lags. A series with a trend, cycle, or seasonal pattern will have non-zero
autocorrelation that reveals its underlying structure.

**Intuition with examples:**

```
White noise (no memory):
  t:   1   2   3   4   5   6   7   8   9   10
  y:   3  -1   4   0   2  -2   1   3  -1   2
  Knowing y at t=3 tells you nothing about y at t=7 → r(k) ≈ 0

AR(1) series (strong memory):
  t:   1   2   3   4   5   6   7   8   9   10
  y:   2   1.6 1.3 1.0 0.8 0.7 0.5 0.4 0.3  0.2
  High values tend to be followed by high values → r(1) ≈ 0.8
```

The key insight: **autocorrelation is not about the values themselves, but
about whether past values predict future values.** If they do, a time series
model can exploit that predictability.

---

## Why Autocorrelation Matters in Forecasting

Most machine learning models treat observations as independent. Time series
models explicitly model the *dependence* between observations across time.
ACF and PACF are the primary tools for:

1. **Diagnosing whether a series has structure worth modelling** — a flat ACF
   means the series is white noise; no model will outperform a naive mean.

2. **Identifying the correct model order** — the shape of ACF and PACF plots
   tells you whether to use AR, MA, or ARMA terms and how many.

3. **Verifying residuals after fitting** — a good model should produce
   residuals with no significant autocorrelation. Checking the residual ACF
   tells you whether the model has captured all the structure.

4. **Detecting seasonality** — spikes at multiples of the seasonal period
   (e.g. lags 12, 24, 36 for monthly data) signal a seasonal component.

5. **Checking stationarity assumptions** — a slowly decaying ACF that never
   reaches zero is a sign of a non-stationary series, even before running
   formal tests.

---

## Lag-k Correlation — How it is Computed

The lag-k autocorrelation is computed exactly like a Pearson correlation
coefficient, except both variables are the same series — one at time t and
one shifted k steps back.

Given a series `y₁, y₂, ..., yₙ`:

1. Form two vectors by shifting the series by k steps:
   - `A = [y_{k+1}, y_{k+2}, ..., y_n]` (current values)
   - `B = [y_1,   y_2,   ..., y_{n-k}]` (values k steps earlier)

2. Compute their mean-centred covariance and normalise by variance.

The result is a number between -1 and +1:
- `+1` — when the series is high, its value k steps ago was also high
- `-1` — when the series is high, its value k steps ago was low
- `0`  — no linear relationship between current and lagged values

### Step-by-step Example

```
Series: y = [2, 4, 3, 5, 4, 6, 5, 7]
Mean ȳ = 4.5
Variance = 2.5

Lag k = 1:
  Pairs: (4,2), (3,4), (5,3), (4,5), (6,4), (5,6), (7,5)
  Centred:
    (4−4.5)(2−4.5) = (−0.5)(−2.5) = +1.25
    (3−4.5)(4−4.5) = (−1.5)(−0.5) = +0.75
    (5−4.5)(3−4.5) = (+0.5)(−1.5) = −0.75
    (4−4.5)(5−4.5) = (−0.5)(+0.5) = −0.25
    (6−4.5)(4−4.5) = (+1.5)(−0.5) = −0.75
    (5−4.5)(6−4.5) = (+0.5)(+1.5) = +0.75
    (7−4.5)(5−4.5) = (+2.5)(+0.5) = +1.25

  Sum = 1.25 + 0.75 − 0.75 − 0.25 − 0.75 + 0.75 + 1.25 = 2.25
  r(1) = 2.25 / ((8−1) × 2.5) = 2.25 / 17.5 ≈ 0.129
```

A longer series with a stronger trend would give a much higher r(1) — the
above is a short example with moderate structure.

### The Correlogram

When you plot r(k) against k for k = 0, 1, 2, ..., maxLag, you get a
**correlogram** — the visual representation of ACF or PACF. The x-axis
is the lag, the y-axis is the correlation value. Horizontal dashed lines
mark the 95% confidence band (see below).

---

## ACF — Autocorrelation Function

### Formula

```
r(k) = Cov(yₜ, yₜ₋ₖ) / Var(y)

     = [ Σₜ (yₜ − ȳ)(yₜ₋ₖ − ȳ) ] / [ Σₜ (yₜ − ȳ)² ]

where the sum runs from t = k+1 to n
```

By definition, `r(0) = 1` always — a series is perfectly correlated with
itself at lag zero.

### What ACF Captures

ACF at lag k measures the **total** correlation between `yₜ` and `yₜ₋ₖ`.

This total correlation includes:
- The **direct** effect: `yₜ₋ₖ` directly influencing `yₜ`
- All **indirect** effects: the chain `yₜ₋ₖ → yₜ₋ₖ₊₁ → ... → yₜ`

This is both ACF's strength and its weakness. It tells you that correlation
exists at lag k, but not whether it is direct or inherited from shorter lags.

**Example — why indirect effects matter:**

In an AR(1) process with `φ = 0.8`:

```
yₜ = 0.8 × yₜ₋₁ + ε

r(1) = 0.80   ← direct effect
r(2) = 0.64   ← because yₜ₋₂ → yₜ₋₁ → yₜ  (0.8²)
r(3) = 0.51   ← 0.8³
r(k) = 0.8ᵏ  ← exponentially decaying
```

ACF cannot distinguish "lag 2 has correlation 0.64" from "lag 2 appears to
have correlation 0.64 only because of the lag 1 effect". PACF is the tool
for that.

### Confidence Bands

Under the null hypothesis of no autocorrelation, the standard error of r(k)
is approximately `1/√n`. The 95% confidence bands are:

```
CI = ± 1.96 / √n
```

Spikes that exceed ±1.96/√n are statistically significant at the 5% level.

For n = 100: CI = ±0.196
For n = 200: CI = ±0.139
For n = 500: CI = ±0.088

Larger samples have tighter bands, making it easier to detect small but real
autocorrelation.

---

## PACF — Partial Autocorrelation Function

### Formula

PACF at lag k is the correlation between `yₜ` and `yₜ₋ₖ` after removing
the linear influence of all intermediate lags `yₜ₋₁, yₜ₋₂, ..., yₜ₋ₖ₊₁`:

```
α(k) = Corr(yₜ − Ŷₜ, yₜ₋ₖ − Ŷₜ₋ₖ)

where Ŷₜ     = projection of yₜ onto span{yₜ₋₁, ..., yₜ₋ₖ₊₁}
      Ŷₜ₋ₖ  = projection of yₜ₋ₖ onto span{yₜ₋₁, ..., yₜ₋ₖ₊₁}
```

In plain language: regress `yₜ` on `yₜ₋₁, ..., yₜ₋ₖ₊₁`, take the residual,
do the same for `yₜ₋ₖ`, and correlate the two residuals. What remains is the
**direct** effect of lag k only.

### What PACF Captures

PACF isolates the direct contribution of lag k, with everything else already
accounted for.

**Same AR(1) example with φ = 0.8:**

```
α(1) = 0.80   ← the direct AR(1) coefficient
α(2) ≈ 0.00   ← once lag 1 is accounted for, lag 2 adds nothing
α(3) ≈ 0.00
α(k) ≈ 0.00   for k > 1
```

PACF cuts off sharply at lag 1 — which is the defining signature of an AR(1)
process. The "cut-off" tells you the true AR order directly.

### How PACF is Computed — Yule-Walker / Levinson-Durbin

PACF is computed iteratively using the Levinson-Durbin algorithm:

```
Step 1:  α(1) = r(1)

Step 2:  α(2) = [r(2) − r(1)²] / [1 − r(1)²]

Step k:  α(k) = [r(k) − Σⱼ₌₁ᵏ⁻¹ φₖ₋₁,ⱼ · r(k−j)] /
                [1    − Σⱼ₌₁ᵏ⁻¹ φₖ₋₁,ⱼ · r(j)  ]

where the φ coefficients are updated at each step
```

Each step fits an AR model of order k using the previous step's coefficients.
The PACF value at lag k is the last coefficient added (`φₖₖ`), which
represents the incremental contribution of lag k after all shorter lags are
already in the model.

---

## ACF vs PACF — The Critical Difference

This is the most important concept to internalise:

| | ACF | PACF |
|---|---|---|
| **Measures** | Total correlation at lag k | Direct correlation at lag k |
| **Includes** | Direct + indirect effects | Direct effect only |
| **AR process** | Decays gradually | Cuts off sharply |
| **MA process** | Cuts off sharply | Decays gradually |
| **Analogy** | Gross effect (like unadjusted R²) | Net effect (like a regression coefficient) |

The analogy to regression is exact: just as a raw correlation between two
variables conflates direct and indirect effects, ACF conflates the direct
and indirect lag effects. PACF is the time-series equivalent of a partial
regression coefficient — it controls for confounds.

---

## Signature Patterns for Model Identification

This is the core practical use of ACF and PACF plots in forecasting. The
patterns below assume the series is already stationary.

### AR(p) — Autoregressive

An AR(p) process: `yₜ = φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ + εₜ`

```
ACF:   decays exponentially (or with damped oscillation if φ < 0)
PACF:  cuts off sharply after lag p — significant at lags 1…p, near zero after

Decision: PACF tells you p directly.
           Count the significant spikes in PACF → that is your AR order.
```

**AR(1) with φ = 0.8:**
```
ACF:  1.00, 0.80, 0.64, 0.51, 0.41, 0.33 ...  (decays as 0.8ᵏ)
PACF: 1.00, 0.80, 0.00, 0.00, 0.00, 0.00 ...  (cuts off at lag 1)
```

**AR(2):**
```
ACF:  decays, possibly with oscillation
PACF: significant at lags 1 and 2, zero after → AR order = 2
```

### MA(q) — Moving Average

An MA(q) process: `yₜ = εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θqεₜ₋q`

```
ACF:   cuts off sharply after lag q — significant at lags 1…q, near zero after
PACF:  decays exponentially (MA is the "dual" of AR)

Decision: ACF tells you q directly.
           Count the significant spikes in ACF → that is your MA order.
```

**MA(1) with θ = 0.7:**
```
ACF:  1.00, 0.47, 0.00, 0.00, 0.00 ...  (cuts off at lag 1)
PACF: 1.00, 0.47, -0.22, 0.10, ... ...  (decays slowly)
```

The MA(1) ACF spike at lag 1 is `θ/(1+θ²)`, not θ itself.

### ARMA(p,q)

```
ACF:   decays after lag q
PACF:  decays after lag p
Both:  show tails rather than clean cut-offs

Decision: harder to identify by eye.
           Use information criteria (AIC, BIC) to select p and q jointly.
```

When both ACF and PACF decay gradually, ARMA is likely. In practice, try a
grid of (p,q) combinations and choose by AIC:

```python
from statsmodels.tsa.arima.model import ARIMA
import itertools

best_aic, best_order = np.inf, None
for p, q in itertools.product(range(4), range(4)):
    try:
        m = ARIMA(series, order=(p, 0, q)).fit()
        if m.aic < best_aic:
            best_aic, best_order = m.aic, (p, 0, q)
    except:
        pass
print(f"Best order: {best_order}, AIC: {best_aic:.2f}")
```

### White Noise

A purely random series with no structure.

```
ACF:  all values within ±1.96/√n — no significant spikes
PACF: same — all within confidence bands

Decision: no model needed; series cannot be forecast beyond its mean.
```

Verifying residuals should produce this pattern — if the residual ACF/PACF
has significant spikes, the model has not captured all the structure.

### Seasonal Pattern

```
ACF:  significant spikes at multiples of the season length s
      (lags s, 2s, 3s, ...)
PACF: significant spike at lag s, smaller at 2s

Decision: use SARIMA(p,d,q)(P,D,Q)[s] or seasonal differencing.

Monthly data (s=12):  spikes at lags 12, 24, 36 ...
Weekly data  (s=7):   spikes at lags 7, 14, 21 ...
Quarterly    (s=4):   spikes at lags 4, 8, 12 ...
```

### Non-stationary Series

```
ACF:  decays very slowly — still significant at high lags (10, 20, 30 ...)
      or never reaches zero within the plotted range
PACF: large spike at lag 1 (close to 1.0), near zero after

Decision: series needs differencing.
           Apply .diff(), re-test with ADF/KPSS, then re-plot ACF/PACF.
```

This is the visual equivalent of a failed ADF test. The slow decay in ACF
before differencing and the clean pattern after is one of the clearest
diagnostics in time series analysis.

---

## ARIMA Order Selection Using ACF and PACF

The classical Box-Jenkins methodology uses ACF and PACF as the primary
identification tool for ARIMA(p, d, q) models:

```
Step 1 — Determine d (differencing order)
  Plot ACF of raw series.
  If ACF decays slowly → difference the series.
  Re-plot ACF after differencing.
  Repeat until ACF decays quickly → series is stationary.
  d = number of differences applied.

Step 2 — Determine p (AR order)
  Plot PACF of the stationary (differenced) series.
  Count significant spikes before the first cut-off → p.

Step 3 — Determine q (MA order)
  Plot ACF of the stationary (differenced) series.
  Count significant spikes before the first cut-off → q.

Step 4 — Fit ARIMA(p, d, q) and check residuals
  Residual ACF should look like white noise.
  If residual ACF has spikes → p or q is too low; increase by 1 and refit.
  If AIC/BIC improves by dropping terms → p or q is too high.
```

**Summary identification table:**

| ACF shape | PACF shape | Model |
|-----------|------------|-------|
| Decays exponentially | Cuts off at lag p | AR(p) |
| Cuts off at lag q | Decays exponentially | MA(q) |
| Both decay | Both decay | ARMA(p,q) |
| Both flat | Both flat | White noise |
| Spikes at lag s, 2s, 3s | Spike at lag s | Seasonal |
| Slow decay (non-stationary) | Spike near 1.0 at lag 1 | Needs differencing |

---

## Code Implementation

### Plotting ACF and PACF

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv('data.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
series = df['meantemp']

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# raw series
series.plot(ax=axes[0], title='Original series', color='#3B8BD4')

# ACF
plot_acf(series, lags=40, ax=axes[1], title='ACF', color='#1D9E75',
         vlines_kwargs={'colors': '#1D9E75'})

# PACF
plot_pacf(series, lags=40, ax=axes[2], title='PACF', method='ywmle',
          color='#7F77DD', vlines_kwargs={'colors': '#7F77DD'})

plt.tight_layout()
plt.show()
```

The `method='ywmle'` argument uses the Yule-Walker method for PACF — the
most common and stable choice. Other options: `'ols'` (OLS regression at
each lag), `'ld'` (Levinson-Durbin).

### Computing Raw Values

```python
from statsmodels.tsa.stattools import acf, pacf

# compute ACF values
acf_values, acf_ci = acf(series, nlags=40, alpha=0.05)
# acf_ci contains the 95% confidence intervals for each lag

# compute PACF values
pacf_values, pacf_ci = pacf(series, nlags=40, alpha=0.05, method='ywmle')

# print significant lags
ci = 1.96 / len(series)**0.5
sig_acf  = [k for k, v in enumerate(acf_values)  if k > 0 and abs(v) > ci]
sig_pacf = [k for k, v in enumerate(pacf_values) if k > 0 and abs(v) > ci]

print(f"Significant ACF lags  : {sig_acf[:10]}")
print(f"Significant PACF lags : {sig_pacf[:10]}")
```

### Full Diagnostic Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

def acf_pacf_diagnostic(series, name="Series", max_lag=40):
    """
    Full ACF/PACF diagnostic with pattern identification.
    Prints significant lags, suggests model type,
    and plots the original series + ACF + PACF.
    """
    series = pd.Series(series).dropna()
    n   = len(series)
    ci  = 1.96 / np.sqrt(n)

    acf_vals  = acf(series,  nlags=max_lag, fft=True)
    pacf_vals = pacf(series, nlags=max_lag, method='ywmle')

    # significant lags (excluding lag 0)
    sig_acf  = [k for k in range(1, max_lag+1) if abs(acf_vals[k])  > ci]
    sig_pacf = [k for k in range(1, max_lag+1) if abs(pacf_vals[k]) > ci]

    # slow decay check (non-stationarity signal)
    slow_decay = all(abs(acf_vals[k]) > ci for k in range(1, min(10, max_lag)))

    # suggest model
    acf_cutoff  = (len(sig_acf)  <= 3 and (not sig_acf  or max(sig_acf)  <= 3))
    pacf_cutoff = (len(sig_pacf) <= 3 and (not sig_pacf or max(sig_pacf) <= 3))

    if slow_decay:
        suggestion = "NON-STATIONARY — apply differencing first"
    elif not sig_acf and not sig_pacf:
        suggestion = "WHITE NOISE — no autocorrelation detected"
    elif any(k in sig_acf for k in [7, 12, 24, 52]):
        suggestion = "SEASONAL — consider SARIMA or seasonal differencing"
    elif pacf_cutoff and not acf_cutoff:
        p = max(sig_pacf) if sig_pacf else 0
        suggestion = f"AR({p}) — PACF cuts off, ACF decays"
    elif acf_cutoff and not pacf_cutoff:
        q = max(sig_acf) if sig_acf else 0
        suggestion = f"MA({q}) — ACF cuts off, PACF decays"
    else:
        suggestion = "ARMA — both ACF and PACF decay; use AIC grid search"

    # print summary
    print(f"\n{'='*54}")
    print(f"  ACF/PACF Diagnostic: {name}")
    print(f"{'='*54}")
    print(f"  n                 : {n}")
    print(f"  95% CI            : ±{ci:.3f}")
    print(f"  Significant ACF   : lags {sig_acf[:8]}")
    print(f"  Significant PACF  : lags {sig_pacf[:8]}")
    print(f"  Slow ACF decay    : {slow_decay}")
    print(f"  {'─'*50}")
    print(f"  Suggestion        : {suggestion}")
    print(f"{'='*54}\n")

    # plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(f"ACF/PACF Diagnostic: {name}", fontsize=13)

    series.plot(ax=axes[0], color='#3B8BD4', linewidth=1)
    axes[0].set_title('Original series', fontsize=11)
    axes[0].set_ylabel('Value')

    plot_acf(series, lags=max_lag, ax=axes[1], title='ACF',
             color='#1D9E75', vlines_kwargs={'colors': '#1D9E75'})

    plot_pacf(series, lags=max_lag, ax=axes[2], title='PACF',
              method='ywmle', color='#7F77DD',
              vlines_kwargs={'colors': '#7F77DD'})

    plt.tight_layout()
    plt.savefig(f'{name.replace(" ", "_")}_acf_pacf.png', dpi=150)
    plt.show()

    return {
        "sig_acf"    : sig_acf,
        "sig_pacf"   : sig_pacf,
        "slow_decay" : slow_decay,
        "suggestion" : suggestion,
        "ci"         : round(ci, 4),
    }


# --- example usage ---
import numpy as np
np.random.seed(42)

# AR(1)
ar1 = np.zeros(200)
for i in range(1, 200): ar1[i] = 0.8 * ar1[i-1] + np.random.randn()
acf_pacf_diagnostic(ar1, name="AR(1) φ=0.8")

# MA(1)
eps  = np.random.randn(201)
ma1  = eps[1:] + 0.7 * eps[:-1]
acf_pacf_diagnostic(ma1, name="MA(1) θ=0.7")

# real data
df = pd.read_csv('data.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
acf_pacf_diagnostic(df['meantemp'], name="Mean Temperature")
```

---

## Common Mistakes

**1. Plotting ACF/PACF before differencing**
If the series is non-stationary, the ACF will decay slowly and the pattern
is meaningless for model identification. Always check stationarity (ADF,
KPSS) and difference if needed before reading the ACF/PACF shape.

```python
# wrong — non-stationary series
plot_acf(df['value'])

# correct — after differencing
plot_acf(df['value'].diff().dropna())
```

**2. Misreading the cut-off**
A cut-off at lag q means ACF is significant for lags 1 to q and
non-significant from lag q+1 onward. If ACF is significant at lags 1, 2, 3
and then drops — that is MA(3), not MA(1).

**3. Confusing AR and MA signatures**
- AR: PACF cuts off, ACF decays → use PACF to find p
- MA: ACF cuts off, PACF decays → use ACF to find q

The AR pattern in ACF looks like gradual decay. The MA pattern in ACF looks
like a cliff (sharp drop). Remembering "MA = ACF cliff" avoids most confusion.

**4. Over-trusting visual identification for ARMA**
When both ACF and PACF decay, the exact p and q cannot be read visually.
Use AIC/BIC grid search rather than guessing from the plot.

**5. Not checking residual ACF**
After fitting an ARIMA model, always plot the residual ACF. If it still
has significant spikes, the model has not captured all the structure and
should be revised.

```python
model  = ARIMA(series, order=(1,0,0)).fit()
resid  = model.resid
plot_acf(resid, lags=40)        # should look like white noise
```

**6. Using too few lags**
Seasonal series need ACF plotted out to at least 2 full seasons. For
monthly data with annual seasonality, plot at least 24–36 lags — a lag-12
spike is invisible if you only plot 10 lags.

```python
plot_acf(series, lags=48)       # for monthly data
```

**7. Confusing the confidence band formula**
The ±1.96/√n band assumes all previous lags are zero — appropriate when
checking if any autocorrelation exists at all. After fitting a model, the
Ljung-Box test is more appropriate than eyeballing the residual ACF against
the same bands.

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(model.resid, lags=[10, 20], return_df=True)
print(lb)   # p-values should all be > 0.05
```

---

## References

- Box, G.E.P., Jenkins, G.M., Reinsel, G.C. & Ljung, G.M. (2015).
  *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
  — The original reference for ACF/PACF-based model identification.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice* (3rd ed.). Chapter 9. https://otexts.com/fpp3/acf.html
- Shumway, R.H. & Stoffer, D.S. (2017). *Time Series Analysis and Its
  Applications* (4th ed.). Springer. Chapter 3.
- statsmodels ACF/PACF docs:
  https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html
- statsmodels plot_acf:
  https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html