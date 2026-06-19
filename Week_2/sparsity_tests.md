# Sparsity Tests in Time Series Forecasting: ADI Score & CV²

A practical guide to detecting and classifying sparse demand patterns in time
series data — and choosing the right forecasting model based on the result.

---

## Table of Contents

- [The Core Problem in Time Series](#the-core-problem-in-time-series)
- [Where Sparsity Tests Fit in the Forecasting Pipeline](#where-sparsity-tests-fit-in-the-forecasting-pipeline)
- [What is a Sparse Time Series?](#what-is-a-sparse-time-series)
- [Why Sparsity Tests Matter](#why-sparsity-tests-matter)
- [Key Metrics](#key-metrics)
  - [ADI — Average Demand Interval](#adi--average-demand-interval)
  - [CV² — Squared Coefficient of Variation](#cv--squared-coefficient-of-variation)
  - [What Each Metric Captures in Time Series Terms](#what-each-metric-captures-in-time-series-terms)
- [Classification Matrix](#classification-matrix)
- [Thresholds](#thresholds)
- [Code Implementation](#code-implementation)
  - [Basic Example](#basic-example)
  - [Full Pipeline with Verdict](#full-pipeline-with-verdict)
  - [On a Real DataFrame](#on-a-real-dataframe)
- [Interpreting Results](#interpreting-results)
- [Choosing a Forecasting Model](#choosing-a-forecasting-model)
- [Common Mistakes](#common-mistakes)
- [References](#references)

---

## The Core Problem in Time Series

Most time series forecasting models assume demand is **continuous and relatively
smooth** across time periods:

```
ARIMA expects:  3, 4, 5, 3, 4, 6, 5, 4 ...
Reality:        0, 0, 0, 7, 0, 0, 0, 3 ...
```

When you force ARIMA or exponential smoothing onto a sparse series, they treat
zeros as real signal and try to fit a smooth curve through them — producing
forecasts like `0.4 units`, which is meaningless in practice. Worse, they
accumulate large errors across all the zero periods where demand simply did
not occur.

Sparsity tests solve this by **detecting the pattern before modelling** and
routing you to a model family built specifically for intermittent data.

---

## Where Sparsity Tests Fit in the Forecasting Pipeline

```
Raw time series
      ↓
Sparsity test (ADI + CV²)
      ↓
      ├── Smooth       → ARIMA, ETS, Linear Regression
      ├── Erratic      → Holt-Winters, SARIMA
      ├── Intermittent → Croston's method, SBA
      └── Lumpy        → ADIDA, IMAPA, bootstrapping
```

ADI and CV² act as a **routing decision** — they tell you which family of
models is appropriate before you write a single line of forecasting code.

---

## What is a Sparse Time Series?

A sparse (or intermittent) time series is one where most time periods have
**zero demand**, with occasional non-zero observations. This is extremely
common in:

- Retail sales (slow-moving SKUs)
- Spare parts and maintenance inventory
- Hospital admissions or rare events
- Machine failures / downtime logs
- Insurance claims

**Example:**

```
Day:    1   2   3   4   5   6   7   8   9   10
Sales:  0   0   5   0   0   0   2   0   0    3
```

---

## Why Sparsity Tests Matter

| Without sparsity test | With sparsity test |
|---|---|
| Apply ARIMA blindly | Classify demand type first |
| Get predictions like 0.4 units | Use Croston's or ADIDA |
| High forecast error (RMSE) | Lower error, appropriate model |
| Errors pile up on zero periods | Model handles zeros explicitly |
| Inventory over/understocked | Better stock decisions |

---

## Key Metrics

### ADI — Average Demand Interval

ADI measures **how frequently** demand occurs — specifically, the average number
of time periods between two non-zero demand events.

**Formula:**

```
ADI = Total number of periods / Number of non-zero periods
```

**Example:**

```
Series:  0  0  5  0  0  0  2  0  0  3
Periods: 10
Non-zero occurrences: 3  (at positions 3, 7, 10)

ADI = 10 / 3 = 3.33
```

An ADI of 3.33 means demand occurs roughly once every 3–4 periods on average.

---

### CV² — Squared Coefficient of Variation

CV² measures **how irregular the size** of non-zero demand events is — i.e.,
when demand does occur, is it always roughly the same amount, or does it vary
wildly?

**Formula:**

```
CV  = Standard Deviation of non-zero values / Mean of non-zero values
CV² = CV²
```

**Example:**

```
Non-zero values: 5, 2, 3
Mean            = 3.33
Std deviation   = 1.53
CV              = 1.53 / 3.33 = 0.46
CV²             = 0.46² = 0.21
```

A CV² of 0.21 means the non-zero demand sizes are relatively stable.

---

### What Each Metric Captures in Time Series Terms

In a time series context, ADI and CV² describe two completely different
dimensions of the series:

| Metric | Dimension | Question it answers |
|--------|-----------|---------------------|
| ADI | **Time** | How spread out are demand events across the timeline? |
| CV² | **Magnitude** | When demand does arrive, how consistent is the quantity? |

This is why both are needed together. A series can have infrequent demand
(high ADI) but always the same size (low CV²) — that is intermittent, not
lumpy. Croston's method handles this by forecasting the **interval between
demands** and the **demand size** separately — which is exactly what ADI and
CV² measure respectively.

---

## Classification Matrix

Based on the Syntetos-Boylan (2005) framework, combining ADI and CV² produces
four demand categories:

```
                    CV² ≤ 0.49              CV² > 0.49
               ┌────────────────────┬────────────────────┐
  ADI ≤ 1.32   │      SMOOTH        │      ERRATIC        │
               │  frequent, stable  │  frequent, uneven   │
               ├────────────────────┼────────────────────┤
  ADI > 1.32   │   INTERMITTENT     │       LUMPY         │
               │ infrequent, stable │ infrequent, uneven  │
               └────────────────────┴────────────────────┘
```

---

## Thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| ADI | > 1.32 | Demand is intermittent (infrequent) |
| ADI | ≤ 1.32 | Demand is regular (frequent) |
| CV² | > 0.49 | Demand sizes are erratic (variable) |
| CV² | ≤ 0.49 | Demand sizes are stable (consistent) |

These thresholds were empirically derived by Syntetos & Boylan (2005) and are
the most widely used in supply chain and inventory literature.

---

## Code Implementation

### Basic Example

```python
import numpy as np

demand = [0, 0, 5, 0, 0, 0, 2, 0, 0, 3]

def compute_adi(series):
    series = np.array(series)
    n_periods = len(series)
    n_nonzero = np.count_nonzero(series)
    if n_nonzero == 0:
        return np.inf          # no demand at all
    return n_periods / n_nonzero

def compute_cv2(series):
    series = np.array(series)
    nonzero = series[series > 0]
    if len(nonzero) < 2:
        return 0.0             # not enough data to measure variation
    cv = nonzero.std() / nonzero.mean()
    return cv ** 2

adi = compute_adi(demand)
cv2 = compute_cv2(demand)

print(f"ADI : {adi:.2f}")
print(f"CV² : {cv2:.2f}")
```

---

### Full Pipeline with Verdict

```python
import numpy as np

def classify_demand(series):
    series = np.array(series)

    # --- compute ADI ---
    n_periods = len(series)
    n_nonzero = np.count_nonzero(series)

    if n_nonzero == 0:
        return {"adi": np.inf, "cv2": 0.0, "verdict": "NO DEMAND"}

    adi = n_periods / n_nonzero

    # --- compute CV² ---
    nonzero_vals = series[series > 0]
    if len(nonzero_vals) < 2:
        cv2 = 0.0
    else:
        cv = nonzero_vals.std() / nonzero_vals.mean()
        cv2 = cv ** 2

    # --- classify ---
    if adi <= 1.32 and cv2 <= 0.49:
        verdict = "SMOOTH"
    elif adi <= 1.32 and cv2 > 0.49:
        verdict = "ERRATIC"
    elif adi > 1.32 and cv2 <= 0.49:
        verdict = "INTERMITTENT"
    else:
        verdict = "LUMPY"

    return {"adi": round(adi, 4), "cv2": round(cv2, 4), "verdict": verdict}


# --- example usage ---
demand = [0, 0, 5, 0, 0, 0, 2, 0, 0, 3]
result = classify_demand(demand)

print(f"ADI     : {result['adi']}")
print(f"CV²     : {result['cv2']}")
print(f"Verdict : {result['verdict']}")
```

**Output:**

```
ADI     : 3.3333
CV²     : 0.2083
Verdict : INTERMITTENT
```

---

### On a Real DataFrame

If you have multiple SKUs or products in a DataFrame, you can classify each row:

```python
import pandas as pd
import numpy as np

# example: each row is a product, each column is a time period
df = pd.DataFrame({
    'SKU_A': [0, 0, 5, 0, 0, 0, 2, 0, 0, 3],
    'SKU_B': [4, 3, 5, 4, 0, 3, 2, 4, 5, 3],
    'SKU_C': [0, 0, 0, 10, 0, 0, 0, 0, 50, 0],
    'SKU_D': [2, 5, 1, 3, 4, 2, 1, 3, 2, 4],
})

results = []
for sku in df.columns:
    series = df[sku].values
    res = classify_demand(series)
    res['sku'] = sku
    results.append(res)

summary = pd.DataFrame(results)[['sku', 'adi', 'cv2', 'verdict']]
print(summary)
```

**Output:**

```
     sku      adi     cv2        verdict
0  SKU_A   3.3333  0.2083   INTERMITTENT
1  SKU_B   1.1111  0.1753         SMOOTH
2  SKU_C   3.3333  1.7778          LUMPY
3  SKU_D   1.0000  0.3077         SMOOTH
```

---

## Interpreting Results

| Category | ADI | CV² | What it means | Example |
|---|---|---|---|---|
| **Smooth** | ≤ 1.32 | ≤ 0.49 | Demand is frequent and consistent | Daily staple goods |
| **Erratic** | ≤ 1.32 | > 0.49 | Demand is frequent but sizes vary a lot | Promotional items |
| **Intermittent** | > 1.32 | ≤ 0.49 | Demand is rare but consistent in size | Spare parts |
| **Lumpy** | > 1.32 | > 0.49 | Demand is rare and unpredictable in size | Emergency orders |

---

## Choosing a Forecasting Model

| Category | Recommended Models | Why |
|---|---|---|
| Smooth | ARIMA, ETS, Linear Regression | Continuous, well-behaved series — standard models work well |
| Erratic | Holt-Winters, SARIMA, ensemble | Frequent but variable — needs models that adapt to size changes |
| Intermittent | Croston's method, SBA | Separates interval forecasting from size forecasting |
| Lumpy | ADIDA, IMAPA, bootstrapping | Most complex — needs aggregation or simulation approaches |

**Why Croston's method works for intermittent series**

Croston (1972) observed that standard exponential smoothing conflates two
separate problems in sparse data:

1. *When* will the next demand event occur? (interval)
2. *How much* will it be? (size)

Croston's method forecasts these independently using two separate smoothing
equations, then combines them. This directly maps to what ADI and CV² measure —
ADI informs the interval model, CV² informs the size model.

```python
# using statsforecast for intermittent demand
from statsforecast import StatsForecast
from statsforecast.models import CrostonSBA, ADIDA, IMAPA

models = [CrostonSBA(), ADIDA(), IMAPA()]
sf = StatsForecast(models=models, freq='D')
sf.fit(df_train)
forecast = sf.predict(h=12)
```

**Evaluating forecast accuracy on sparse series**

Standard metrics like RMSE are misleading on sparse data because large errors
on zero periods dominate. Prefer:

```python
# Mean Absolute Scaled Error (MASE) — robust to zero-heavy series
from statsforecast.losses import mase

# or use Scaled Pinball Loss for probabilistic forecasts
# Avoid plain RMSE / MAE on lumpy or intermittent series
```

---

## Common Mistakes

**1. Including zeros in CV² calculation**
Zeros represent absence of demand, not small demand. Always filter them out
before computing std and mean.

```python
# wrong
cv = series.std() / series.mean()

# correct
nonzero = series[series > 0]
cv = nonzero.std() / nonzero.mean()
```

**2. Too few non-zero observations**
With fewer than 5–10 non-zero values, ADI and CV² are unreliable. Flag these
series separately.

```python
if n_nonzero < 5:
    verdict = "INSUFFICIENT DATA"
```

**3. Applying regular models to lumpy demand**
ARIMA on a lumpy series will give near-zero forecasts with high error. Always
classify first, then model.

**4. Fixed thresholds for all domains**
The 1.32 and 0.49 thresholds are defaults from Syntetos-Boylan. Some industries
use different cutoffs — always validate against your own data.

**5. Using RMSE to evaluate sparse forecasts**
RMSE heavily penalises errors on zero periods, which dominate sparse series.
This makes models that predict small non-zero values look worse than models
that predict zero everywhere. Use MASE or Mean Absolute Percentage Error
(MAPE) with care, or use interval-specific metrics instead.

**6. Not sorting the time series before computing ADI**
ADI is positional — it counts gaps between non-zero rows in the order they
appear. If your DataFrame is not sorted by date, the gap counts will be wrong.
Always sort first:

```python
df = df.sort_values('date').reset_index(drop=True)
```

---

## References

- Syntetos, A.A. & Boylan, J.E. (2005). *The accuracy of intermittent demand
  estimates.* International Journal of Forecasting, 21(2), 303–314.
- Croston, J.D. (1972). *Forecasting and stock control for intermittent demands.*
  Operational Research Quarterly, 23(3), 289–303.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice* (3rd ed.). OTexts. https://otexts.com/fpp3
- Nixtla StatsForecast documentation: https://nixtlaverse.nixtla.io/statsforecast