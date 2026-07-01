# Autocorrelation Tests in Time Series Forecasting
## Ljung-Box, Durbin-Watson & Breusch-Godfrey — Q-stat, p-values, decisions

---

## Table of Contents

- [What Are Autocorrelation Tests?](#what-are-autocorrelation-tests)
- [Why Test Residuals for Autocorrelation?](#why-test-residuals-for-autocorrelation)
- [Where These Tests Fit in the Pipeline](#where-these-tests-fit-in-the-pipeline)
- [The Three Tests](#the-three-tests)
  - [Ljung-Box Test](#ljung-box-test)
  - [Durbin-Watson Test](#durbin-watson-test)
  - [Breusch-Godfrey Test](#breusch-godfrey-test)
  - [Key Differences at a Glance](#key-differences-at-a-glance)
- [Code Implementation](#code-implementation)
  - [Ljung-Box Test](#ljung-box-test-1)
  - [Durbin-Watson Test](#durbin-watson-test-1)
  - [Breusch-Godfrey Test](#breusch-godfrey-test-1)
  - [Full Pipeline — All Three with Verdicts](#full-pipeline--all-three-with-verdicts)
  - [On a Real Model](#on-a-real-model)
- [Interpreting Results](#interpreting-results)
- [What to Do When Tests Fail](#what-to-do-when-tests-fail)
- [Common Mistakes](#common-mistakes)
- [References](#references)

---

## What Are Autocorrelation Tests?

Autocorrelation tests check whether the **residuals** of a fitted model are
serially correlated — i.e. whether the errors at time t are related to
errors at earlier times.

A well-specified model should produce residuals that are **white noise** —
random, uncorrelated, with no exploitable structure left behind. If residuals
are autocorrelated, it means:

- The model has missed some pattern in the data
- Forecasts are sub-optimal — the leftover structure could have been used
- Standard errors and confidence intervals are invalid

```
Good model residuals (white noise):
  t:       1     2     3     4     5     6     7
  resid:  +0.3  -0.1  +0.5  -0.2  +0.1  -0.4  +0.2
  Pattern: random, no direction, no memory

Bad model residuals (autocorrelated):
  t:       1     2     3     4     5     6     7
  resid:  +0.3  +0.5  +0.7  +0.9  +0.6  +0.4  +0.2
  Pattern: positive errors cluster → model consistently underpredicts
```

---

## Why Test Residuals for Autocorrelation?

1. **Model adequacy** — confirms the model has captured all the structure.
   Leftover autocorrelation means p or q is too low.

2. **Validity of inference** — OLS and ARIMA standard errors assume
   uncorrelated residuals. If they are correlated, t-statistics and
   confidence intervals are wrong.

3. **Forecast improvement** — autocorrelated residuals can be modelled
   themselves, reducing forecast error.

4. **Regulatory / reporting requirement** — in econometrics and finance,
   model diagnostics routinely include autocorrelation test results.

---

## Where These Tests Fit in the Pipeline

```
Fit model (ARIMA, OLS regression, etc.)
      ↓
Extract residuals: resid = model.resid
      ↓
Run autocorrelation tests
      ↓
      ├── All tests pass (p > 0.05) → model is adequate
      └── Any test fails (p < 0.05) → residuals have autocorrelation
                                     → increase AR or MA order
                                     → or add seasonal terms
                                     → refit and re-test
```

These tests are always run **after fitting**, on residuals — not on the
raw series. Testing the raw series for autocorrelation is what ACF/PACF
and stationarity tests are for.

---

## The Three Tests

### Ljung-Box Test

The most widely used residual autocorrelation test in time series.

**Null hypothesis:** No autocorrelation in the residuals up to lag h.
**Alternative:** Autocorrelation exists at one or more lags up to h.

**How it works:**

Ljung-Box computes the Q statistic — a weighted sum of squared
autocorrelations across the first h lags:

```
Q = n(n+2) × Σₖ₌₁ʰ [ r²(k) / (n−k) ]

where:
  n    = number of residuals
  r(k) = autocorrelation of residuals at lag k
  h    = number of lags tested
  Q    ~ χ²(h) under H₀
```

The weights `n/(n−k)` give slightly more importance to earlier lags. The
original Box-Pierce test used equal weights `r²(k)` — Ljung-Box is the
improved, bias-corrected version and is preferred in practice.

**Choosing h (number of lags):**

```
Rule of thumb: h = min(10, n/5)  for non-seasonal series
               h = min(2s, n/5) for seasonal series (s = season period)

Common choices:
  h = 10   general check
  h = 20   longer-range dependence
  h = [10, 20, 30]  run at multiple horizons
```

**Decision rule:**
```
p-value > 0.05  →  fail to reject H₀  →  residuals are white noise ✓
p-value ≤ 0.05  →  reject H₀          →  residuals are autocorrelated ✗
```

**Strength:** Tests multiple lags jointly. Works for ARIMA residuals.
Sensitive to autocorrelation at any lag up to h.

**Weakness:** Can miss autocorrelation at a specific lag if other lags
dilute the signal. Does not identify which lag is the problem.

---

### Durbin-Watson Test

A classic test designed specifically for **first-order autocorrelation**
(lag 1 only) in regression residuals.

**Null hypothesis:** No first-order autocorrelation (ρ = 0).
**Alternative:** First-order autocorrelation exists (ρ ≠ 0).

**How it works:**

Durbin-Watson computes the ratio of the sum of squared successive
differences to the total sum of squared residuals:

```
DW = Σₜ₌₂ⁿ (eₜ − eₜ₋₁)² / Σₜ₌₁ⁿ eₜ²

where eₜ = residual at time t
```

**Interpreting the DW statistic:**

```
DW ≈ 2(1 − ρ̂)   where ρ̂ is the estimated lag-1 autocorrelation

DW = 0   → strong positive autocorrelation  (ρ = +1)
DW = 2   → no autocorrelation               (ρ =  0)  ← want this
DW = 4   → strong negative autocorrelation  (ρ = −1)
```

**Decision regions (approximate):**

```
0.0 ────── 1.5 ────── 2.5 ────── 4.0
     ↑           ↑           ↑
  positive    no auto     negative
  autocorr   correlation  autocorr

1.5 < DW < 2.5  → inconclusive / no autocorrelation
DW < 1.5        → positive autocorrelation ✗
DW > 2.5        → negative autocorrelation ✗
```

The exact critical values depend on n and the number of regressors k.
Use the tabulated dL and dU values for a precise test, or the p-value
from `statsmodels`.

**Strength:** Simple, widely understood, works for OLS regression.
**Weakness:** Tests lag 1 only. Invalid when the model includes a lagged
dependent variable as a regressor (use Breusch-Godfrey instead). Does not
produce a p-value from first principles — uses tabulated bounds.

---

### Breusch-Godfrey Test

A more general test that handles higher-order autocorrelation and works
correctly when the model includes lagged dependent variables — where
Durbin-Watson breaks down.

**Null hypothesis:** No autocorrelation up to order p in the residuals.
**Alternative:** Autocorrelation exists at one or more orders up to p.

**How it works:**

1. Fit the original model, obtain residuals `eₜ`.
2. Regress `eₜ` on the original regressors plus lagged residuals
   `eₜ₋₁, eₜ₋₂, ..., eₜ₋ₚ`.
3. Compute `LM = (n − p) × R²` from this auxiliary regression.
4. Under H₀, `LM ~ χ²(p)`.

```
Auxiliary regression:
  eₜ = α + β₁X₁ₜ + ... + βₖXₖₜ + γ₁eₜ₋₁ + ... + γₚeₜ₋ₚ + υₜ

LM statistic = (n − p) × R²  of this regression
p-value from χ²(p) distribution
```

**Decision rule:**
```
p-value > 0.05  →  no autocorrelation up to order p ✓
p-value ≤ 0.05  →  autocorrelation detected up to order p ✗
```

**Strength:** Works with lagged dependent variables. Tests multiple orders
jointly. More powerful than Durbin-Watson for higher-order autocorrelation.
**Weakness:** Requires specifying the maximum lag order p in advance.
Less commonly used for pure ARIMA diagnostics (Ljung-Box preferred there).

---

### Key Differences at a Glance

| Feature | Ljung-Box | Durbin-Watson | Breusch-Godfrey |
|---------|-----------|---------------|-----------------|
| Null hypothesis | No autocorr up to lag h | No lag-1 autocorr | No autocorr up to order p |
| Lags tested | Multiple (1 to h) | Lag 1 only | Multiple (1 to p) |
| Output | Q statistic, p-value | DW statistic (0–4) | LM statistic, p-value |
| Works with lagged DV | Yes | No | Yes |
| Primary use | ARIMA residual check | OLS regression | OLS with lagged regressors |
| Detects negative autocorr | Yes | Yes | Yes |
| Identifies which lag | No | No | No |

**When to use which:**

```
ARIMA / time series model residuals  → Ljung-Box (primary choice)
OLS regression, no lagged DV         → Durbin-Watson (quick check)
OLS with lagged Y as regressor        → Breusch-Godfrey (DW invalid here)
Want to test higher-order in OLS     → Breusch-Godfrey
Comprehensive check on any model     → run all three
```

---

## Code Implementation

### Ljung-Box Test

```python
import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

def ljung_box_test(residuals, lags=[10, 20, 30], alpha=0.05):
    """
    Ljung-Box test for autocorrelation in residuals.
    Prints Q statistic and p-value at each lag horizon.
    """
    residuals = pd.Series(residuals).dropna()
    n         = len(residuals)

    result = acorr_ljungbox(residuals, lags=lags, return_df=True)

    print(f"\n{'='*54}")
    print(f"  Ljung-Box Test   (n={n})")
    print(f"  H₀: No autocorrelation up to lag h")
    print(f"{'='*54}")
    print(f"  {'Lag h':<10} {'Q statistic':<16} {'p-value':<12} {'Verdict'}")
    print(f"  {'─'*50}")

    verdicts = {}
    for lag in lags:
        q    = result.loc[lag, 'lb_stat']
        p    = result.loc[lag, 'lb_pvalue']
        v    = 'PASS ✓' if p > alpha else 'FAIL ✗ (autocorr)'
        verdicts[lag] = p > alpha
        print(f"  h={lag:<8} Q={q:<14.4f} p={p:<12.4f} {v}")

    overall = 'PASS ✓ residuals ~ white noise' if all(verdicts.values()) \
              else 'FAIL ✗ autocorrelation detected'
    print(f"  {'─'*50}")
    print(f"  Overall : {overall}")
    print(f"{'='*54}\n")

    return result


# --- example ---
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

np.random.seed(42)
series = np.cumsum(np.random.randn(200))          # random walk
model  = ARIMA(series, order=(1, 1, 0)).fit()
ljung_box_test(model.resid, lags=[10, 20, 30])
```

---

### Durbin-Watson Test

```python
from statsmodels.stats.stattools import durbin_watson

def durbin_watson_test(residuals):
    """
    Durbin-Watson test for first-order autocorrelation.
    Prints DW statistic and interpretation.
    """
    residuals = pd.Series(residuals).dropna()
    dw        = durbin_watson(residuals)

    # interpret
    if   dw < 1.5: interp = "POSITIVE autocorrelation ✗  (DW < 1.5)"
    elif dw > 2.5: interp = "NEGATIVE autocorrelation ✗  (DW > 2.5)"
    else:          interp = "NO autocorrelation ✓         (1.5 < DW < 2.5)"

    # estimated rho
    rho_hat = 1 - dw / 2

    print(f"\n{'='*54}")
    print(f"  Durbin-Watson Test")
    print(f"  H₀: No first-order autocorrelation (ρ = 0)")
    print(f"{'='*54}")
    print(f"  DW statistic  : {dw:.4f}")
    print(f"  Estimated ρ   : {rho_hat:.4f}  (DW ≈ 2(1−ρ))")
    print(f"  {'─'*50}")
    print(f"  Verdict       : {interp}")
    print(f"\n  Reference scale:")
    print(f"    0.0 ── 1.5 ── [2.0] ── 2.5 ── 4.0")
    print(f"    +autocorr   no autocorr   −autocorr")
    print(f"{'='*54}\n")

    return {"dw": round(dw, 4), "rho_hat": round(rho_hat, 4), "verdict": interp}


# --- example ---
durbin_watson_test(model.resid)
```

---

### Breusch-Godfrey Test

```python
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

def breusch_godfrey_test(model_fit, max_lag=5, alpha=0.05):
    """
    Breusch-Godfrey LM test for autocorrelation up to max_lag.
    Requires a fitted statsmodels model object.
    Prints LM statistic and p-value.
    """
    lm_stat, p_value, f_stat, f_p = acorr_breusch_godfrey(model_fit, nlags=max_lag)

    verdict = "PASS ✓ no autocorrelation" if p_value > alpha \
              else "FAIL ✗ autocorrelation detected"

    print(f"\n{'='*54}")
    print(f"  Breusch-Godfrey Test   (max lag={max_lag})")
    print(f"  H₀: No autocorrelation up to order {max_lag}")
    print(f"{'='*54}")
    print(f"  LM statistic  : {lm_stat:.4f}")
    print(f"  LM p-value    : {p_value:.4f}")
    print(f"  F statistic   : {f_stat:.4f}")
    print(f"  F p-value     : {f_p:.4f}")
    print(f"  {'─'*50}")
    print(f"  Verdict       : {verdict}")
    print(f"{'='*54}\n")

    return {
        "lm_stat" : round(lm_stat, 4),
        "p_value" : round(p_value, 4),
        "f_stat"  : round(f_stat, 4),
        "f_pvalue": round(f_p, 4),
        "verdict" : verdict,
    }


# --- example (OLS regression) ---
import statsmodels.api as sm

X = sm.add_constant(np.arange(200))
y = 0.5 * np.arange(200) + np.random.randn(200)
ols_model = sm.OLS(y, X).fit()
breusch_godfrey_test(ols_model, max_lag=5)
```

---

### Full Pipeline — All Three with Verdicts

```python
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic  import acorr_ljungbox, acorr_breusch_godfrey
from statsmodels.stats.stattools    import durbin_watson

def autocorrelation_tests(model_fit, lags=[10, 20], max_bg_lag=5, alpha=0.05):
    """
    Run Ljung-Box, Durbin-Watson, and Breusch-Godfrey on model residuals.
    Print Q-stat, DW, LM statistics, p-values, and a combined verdict.
    """
    resid = pd.Series(model_fit.resid).dropna()
    n     = len(resid)

    # ── Ljung-Box ─────────────────────────────────────────────────────────────
    lb     = acorr_ljungbox(resid, lags=lags, return_df=True)
    lb_sig = any(lb['lb_pvalue'] <= alpha)

    # ── Durbin-Watson ─────────────────────────────────────────────────────────
    dw      = durbin_watson(resid)
    dw_sig  = not (1.5 < dw < 2.5)
    rho_hat = round(1 - dw / 2, 4)

    # ── Breusch-Godfrey ───────────────────────────────────────────────────────
    lm_stat, lm_p, f_stat, f_p = acorr_breusch_godfrey(model_fit, nlags=max_bg_lag)
    bg_sig  = lm_p <= alpha

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Autocorrelation Diagnostic Tests   (n={n})")
    print(f"{'='*60}")

    print(f"\n  ── Ljung-Box (H₀: no autocorr up to lag h) ──")
    print(f"  {'Lag h':<8} {'Q stat':<14} {'p-value':<12} {'Result'}")
    print(f"  {'─'*52}")
    for lag in lags:
        q = lb.loc[lag, 'lb_stat']
        p = lb.loc[lag, 'lb_pvalue']
        r = 'PASS ✓' if p > alpha else 'FAIL ✗'
        print(f"  h={lag:<6} {q:<14.4f} {p:<12.4f} {r}")

    print(f"\n  ── Durbin-Watson (H₀: no lag-1 autocorr) ──")
    dw_v = 'PASS ✓' if not dw_sig else 'FAIL ✗'
    print(f"  DW statistic : {dw:.4f}   (ρ̂ ≈ {rho_hat})   {dw_v}")
    print(f"  Scale: 0[+autocorr]──1.5──[2.0]──2.5──[−autocorr]4")

    print(f"\n  ── Breusch-Godfrey (H₀: no autocorr up to lag {max_bg_lag}) ──")
    bg_v = 'PASS ✓' if not bg_sig else 'FAIL ✗'
    print(f"  LM stat : {lm_stat:.4f}   p = {lm_p:.4f}   {bg_v}")
    print(f"  F  stat : {f_stat:.4f}   p = {f_p:.4f}")

    # ── Combined verdict ──────────────────────────────────────────────────────
    n_fail = sum([lb_sig, dw_sig, bg_sig])
    print(f"\n  {'─'*56}")
    if n_fail == 0:
        final = "ALL TESTS PASS ✓ — residuals are white noise"
    elif n_fail == 1:
        failed = ['Ljung-Box' if lb_sig else None,
                  'Durbin-Watson' if dw_sig else None,
                  'Breusch-Godfrey' if bg_sig else None]
        failed = [f for f in failed if f]
        final  = f"MARGINAL — {failed[0]} flags autocorrelation; review model"
    else:
        final = f"FAIL ({n_fail}/3 tests) ✗ — significant autocorrelation in residuals"

    print(f"  Final verdict : {final}")
    print(f"{'='*60}\n")

    return {
        "ljung_box"       : lb.to_dict(),
        "dw"              : round(dw, 4),
        "rho_hat"         : rho_hat,
        "bg_lm_stat"      : round(lm_stat, 4),
        "bg_p_value"      : round(lm_p, 4),
        "lb_significant"  : lb_sig,
        "dw_significant"  : dw_sig,
        "bg_significant"  : bg_sig,
        "final"           : final,
    }
```

---

### On a Real Model

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(42)

# ── Example 1: ARIMA on a time series ────────────────────────────────────────
series = np.cumsum(np.random.randn(300))        # random walk

model_arima = ARIMA(series, order=(1, 1, 1)).fit()
print(model_arima.summary())

results = autocorrelation_tests(model_arima, lags=[10, 20, 30], max_bg_lag=5)

# ── Example 2: OLS regression ─────────────────────────────────────────────────
df = pd.read_csv('data.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

X = sm.add_constant(df[['humidity', 'wind_speed', 'meanpressure']])
y = df['meantemp']

ols_model = sm.OLS(y, X).fit()
autocorrelation_tests(ols_model, lags=[10, 20], max_bg_lag=5)

# ── Quick single-line checks ──────────────────────────────────────────────────
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools   import durbin_watson

# Ljung-Box only
lb = acorr_ljungbox(model_arima.resid, lags=[10], return_df=True)
print(lb)

# Durbin-Watson only
dw = durbin_watson(model_arima.resid)
print(f"DW = {dw:.4f}")
```

---

## Interpreting Results

**All tests pass (p > 0.05, DW ≈ 2):**
Residuals are white noise. The model has captured all exploitable structure.
Forecasts and standard errors are valid.

**Ljung-Box fails at h=10 but passes at h=20:**
Low-lag autocorrelation present — likely a missing AR or MA term at lag 1–5.
Increase p or q by 1 and refit.

**Only Durbin-Watson fails (DW < 1.5):**
Strong positive lag-1 autocorrelation. Positive errors are followed by
positive errors — model consistently undershoots then overshoots.
Add an AR(1) term or MA(1) term.

**Only Breusch-Godfrey fails:**
Higher-order autocorrelation (lag 2–5) exists that DW missed. Often
indicates a missing seasonal component or insufficient AR order.

**All three tests fail:**
The model is substantially misspecified. Residuals carry significant
structure. Common causes: wrong differencing order, missing seasonal terms,
or wrong AR/MA order. Re-examine ACF/PACF of residuals.

---

## What to Do When Tests Fail

**Step 1 — Plot the residual ACF/PACF:**
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(model.resid,  lags=40)
plot_pacf(model.resid, lags=40)
# identify which lags are significant → tells you what to add
```

**Step 2 — Increase AR or MA order based on the pattern:**

| Residual ACF/PACF pattern | Fix |
|--------------------------|-----|
| PACF spike at lag 1 | Increase AR order: p → p+1 |
| ACF spike at lag 1 | Add MA term: q → q+1 |
| Spikes at lag s, 2s | Add seasonal terms (P, D, Q, s) |
| Spike at lag 2 only | Add AR(2) or check for 2nd differencing |
| Both ACF and PACF decay | Try ARMA — increase both p and q |

**Step 3 — Refit and re-run tests:**
```python
# original model failed
model_v1 = ARIMA(series, order=(1, 1, 0)).fit()
autocorrelation_tests(model_v1)   # FAIL

# increase MA order
model_v2 = ARIMA(series, order=(1, 1, 1)).fit()
autocorrelation_tests(model_v2)   # check again

# compare AIC
print(f"AIC v1: {model_v1.aic:.2f}")
print(f"AIC v2: {model_v2.aic:.2f}")   # lower is better
```

---

## Common Mistakes

**1. Running tests on the raw series instead of residuals**
Ljung-Box on a raw series tests whether the series has autocorrelation —
which is expected and not a problem. These tests are only meaningful on
model residuals.

```python
# wrong
acorr_ljungbox(series, lags=[10])

# correct
acorr_ljungbox(model.resid, lags=[10])
```

**2. Using Durbin-Watson when the model includes lagged Y**
DW is invalid when lagged dependent variables appear as regressors —
it is biased toward 2 (no autocorrelation) and will miss real problems.
Use Breusch-Godfrey instead.

```python
# model includes y_{t-1} as a regressor → DW invalid
X = sm.add_constant(pd.DataFrame({'y_lag1': y.shift(1), 'x': x}))
ols = sm.OLS(y, X).fit()
# use BG, not DW
breusch_godfrey_test(ols, max_lag=5)
```

**3. Testing only one lag for Ljung-Box**
A single lag value can miss autocorrelation patterns. Always test at
multiple horizons — `lags=[10, 20, 30]` — to detect both short-range and
long-range dependence.

**4. Treating a borderline DW as conclusive**
DW has an inconclusive region where neither H₀ nor H₁ can be accepted.
If DW falls in this region, supplement with Ljung-Box or Breusch-Godfrey.

**5. Ignoring failed tests after model selection**
AIC and BIC optimise fit, not residual whiteness. A model selected by AIC
can still have autocorrelated residuals. Always run diagnostic tests even
after automated model selection.

**6. Confusing autocorrelation tests with heteroskedasticity tests**
Ljung-Box, DW, and BG test for autocorrelation in residuals.
They do not test for changing variance (heteroskedasticity).
Use the Breusch-Pagan or White test for that separately.

---

## References

- Ljung, G.M. & Box, G.E.P. (1978). *On a measure of lack of fit in time
  series models.* Biometrika, 65(2), 297–303.
- Durbin, J. & Watson, G.S. (1950). *Testing for serial correlation in least
  squares regression I.* Biometrika, 37(3–4), 409–428.
- Breusch, T.S. (1978). *Testing for autocorrelation in dynamic linear
  models.* Australian Economic Papers, 17(31), 334–355.
- Godfrey, L.G. (1978). *Testing against general autoregressive and moving
  average error models when the regressors include lagged dependent
  variables.* Econometrica, 46(6), 1293–1301.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice* (3rd ed.). Chapter 9.
  https://otexts.com/fpp3/residuals.html
- statsmodels acorr_ljungbox:
  https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
- statsmodels durbin_watson:
  https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.durbin_watson.html
- statsmodels acorr_breusch_godfrey:
  https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html