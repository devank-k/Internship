# Reading ACF & PACF Plots
## Spikes, Decay Patterns — Identifying AR(p) vs MA(q) Order

---

## The One Rule to Memorise

```
PACF tells you p  →  AR order
ACF  tells you q  →  MA order
```

Everything else is a variation of this rule. The plot that **cuts off** (drops
sharply to zero) tells you the order. The plot that **decays** (tails off
gradually) just confirms you are looking at that process.

---

## What "Cuts Off" and "Decays" Mean

Before reading any plot, you need to know what these two words mean visually.

**Cuts off** — spikes are significant up to some lag k, then drop abruptly
inside the confidence band and stay there:

```
value
 1.0 |█
 0.5 |█ █
 0.0 |─────────────────────────  ← confidence band
-0.5 |
     lag: 0  1  2  3  4  5  6

Significant at lags 1 and 2, then nothing → cut-off at lag 2
```

**Decays (tails off)** — spikes start large and gradually shrink, crossing
the confidence band somewhere far out:

```
value
 1.0 |█
 0.8 |  █
 0.6 |    █
 0.4 |      █
 0.2 |        █  ·
 0.0 |─────────────────────────  ← confidence band
     lag: 0  1  2  3  4  5  6

Never drops sharply — gradual exponential decrease
```

The confidence band is always `±1.96 / √n`. Spikes outside it are
significant. Spikes inside it are noise.

---

## Pattern 1 — AR(p)

**What generates it:** `yₜ = φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ + εₜ`

```
ACF                             PACF
────────────────────            ────────────────────
█                               █
  ▇                               █
    ▅                               ·
      ▃                               ·
        ▂                               ·
          ·   ·   ·                       ·   ·   ·
─────────────────────           ─────────────────────
Decays exponentially            Cuts off after lag p
(or with damped sine wave)      ↑ count these spikes = p
```

**How to read it:**
- ACF: tails off — ignore for order identification
- PACF: count significant spikes → that number is p

**AR(1) example (φ = 0.8):**
```
ACF lags:   1.00, 0.80, 0.64, 0.51, 0.41, 0.33, ...  (never cuts off)
PACF lags:  1.00, 0.80, 0.00, 0.00, 0.00, 0.00, ...  (cuts off at lag 1)

Read: PACF has 1 spike → AR(1)
```

**AR(2) example (φ₁ = 0.6, φ₂ = −0.3):**
```
ACF:   decays with oscillation (negative φ₂ causes alternating signs)
PACF:  significant at lags 1 and 2, zero after

Read: PACF has 2 spikes → AR(2)
```

**Tip:** If ACF alternates in sign while decaying (positive, negative,
positive ...), φ is negative. Still AR — just with a negative coefficient.

---

## Pattern 2 — MA(q)

**What generates it:** `yₜ = εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θqεₜ₋q`

```
ACF                             PACF
────────────────────            ────────────────────
█                               █
  █                               ▇
    █                               ▅
      ·                               ▃
        ·   ·   ·                       ▂   ·   ·
─────────────────────           ─────────────────────
Cuts off after lag q            Decays exponentially
↑ count these spikes = q        (ignore for order)
```

**How to read it:**
- ACF: count significant spikes → that number is q
- PACF: tails off — ignore for order identification

MA is the exact mirror image of AR. Where AR has PACF cut-off, MA has ACF
cut-off. Where AR has ACF decay, MA has PACF decay.

**MA(1) example (θ = 0.7):**
```
ACF lags:   1.00, 0.47, 0.00, 0.00, 0.00, 0.00, ...  (cuts off at lag 1)
PACF lags:  1.00, 0.47, −0.22, 0.10, −0.05, ...      (decays slowly)

Read: ACF has 1 spike → MA(1)
Note: ACF spike at lag 1 = θ/(1+θ²) = 0.7/1.49 = 0.47, not 0.7 itself
```

**MA(2) example:**
```
ACF:   significant at lags 1 and 2, zero after
PACF:  decays (alternating signs possible)

Read: ACF has 2 spikes → MA(2)
```

---

## Pattern 3 — ARMA(p,q)

**What generates it:** both AR and MA terms present.

```
ACF                             PACF
────────────────────            ────────────────────
█                               █
  ▇                               ▇
    ▅                               ▅
      ▃                               ▃
        ▂                               ▂
          ·   ·   ·                       ·   ·
─────────────────────           ─────────────────────
Both tail off gradually         Both tail off gradually
```

**How to read it:**
- When both ACF and PACF decay without a clean cut-off → ARMA
- Cannot determine p and q visually — use AIC/BIC grid search

**What to do:**
```python
import itertools
from statsmodels.tsa.arima.model import ARIMA

best_aic, best_order = np.inf, None
for p, q in itertools.product(range(5), range(5)):
    try:
        m = ARIMA(series, order=(p, d, q)).fit()
        if m.aic < best_aic:
            best_aic   = m.aic
            best_order = (p, d, q)
    except:
        pass

print(f"Best order: {best_order}, AIC: {best_aic:.2f}")
```

---

## Pattern 4 — White Noise

**What generates it:** pure random series, no structure.

```
ACF                             PACF
────────────────────            ────────────────────
█                               █

  ·   ·   ·   ·   ·               ·   ·   ·   ·   ·
─────────────────────           ─────────────────────
All within CI bands             All within CI bands
```

**How to read it:**
- No significant spikes in either plot beyond lag 0
- Series has no autocorrelation — no time series model can help
- If this is a **residual** ACF → good, model captured everything
- If this is the **original** series → series is unpredictable, forecast
  with the mean

With n observations, you expect about 5% of spikes to cross the CI by
chance even in white noise. So 1–2 borderline spikes out of 20 lags is
not necessarily meaningful.

---

## Pattern 5 — Non-stationary Series

**What generates it:** random walk, unit root, trending series.

```
ACF                             PACF
────────────────────            ────────────────────
█                               █  (spike near 1.0)
  █
    █                             ·   ·   ·   ·
      █
        █
          █   █   █
─────────────────────           ─────────────────────
Decays very slowly              Single large spike at lag 1
still significant at lag 10+    (close to 1.0), near zero after
```

**How to read it:**
- ACF still large and significant even at lags 10, 20, 30 — never dies out
- PACF has one dominant spike at lag 1 (value near 1.0)
- This is not an AR(1) — the PACF spike is much closer to 1.0 than a
  stationary AR(1) would produce

**What to do:**
```python
# first difference and re-plot
series_diff = series.diff().dropna()
plot_acf(series_diff, lags=40)
plot_pacf(series_diff, lags=40)
```

After differencing, the ACF/PACF should show one of the stationary
patterns above.

---

## Pattern 6 — Seasonal Series

**What generates it:** repeating pattern every s periods.

```
ACF — monthly data with annual seasonality (s=12)
────────────────────────────────────────
█
  ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
                                        █     ← lag 12
                                          ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
                                                                                █  ← lag 24
─────────────────────────────────────────────────────────────────────────────────
```

**How to read it:**
- Significant spikes at lags s, 2s, 3s, ... (multiples of season)
- May also have non-seasonal AR/MA structure between the seasonal spikes
- Spike height typically decreases at 2s, 3s, ... (decays across seasons)

**What to do:**
```python
# seasonal differencing first
series_sdiff = series.diff(12).dropna()   # for monthly data
plot_acf(series_sdiff, lags=48)

# then fit SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(series, order=(p,d,q), seasonal_order=(P,D,Q,12))
```

**Common seasonal periods:**
```
Daily data with weekly season  : s = 7
Monthly data with annual season: s = 12
Quarterly data                 : s = 4
Hourly data with daily season  : s = 24
Business data with annual      : s = 52 (weekly)
```

---

## The Decision Flowchart

```
Plot ACF and PACF of the stationary series
              │
              ▼
   Both plots flat (no sig spikes)?
   ├── YES → White noise. No model needed.
   └── NO  ↓

   ACF decays slowly (still large at lag 10+)?
   ├── YES → Non-stationary. Difference and restart.
   └── NO  ↓

   Spikes at multiples of s (12, 7, 4...)?
   ├── YES → Seasonal. Use SARIMA.
   └── NO  ↓

   PACF cuts off, ACF decays?
   ├── YES → AR(p). p = number of PACF spikes.
   └── NO  ↓

   ACF cuts off, PACF decays?
   ├── YES → MA(q). q = number of ACF spikes.
   └── NO  ↓

   Both ACF and PACF decay?
   └── ARMA(p,q). Use AIC/BIC grid search.
```

---

## Quick Reference Table

| ACF shape | PACF shape | Model | What to read |
|-----------|------------|-------|--------------|
| Exponential decay | Cuts off at lag p | AR(p) | Count PACF spikes |
| Cuts off at lag q | Exponential decay | MA(q) | Count ACF spikes |
| Both decay | Both decay | ARMA(p,q) | Use AIC grid |
| Both flat | Both flat | White noise | No model |
| Slow decay (never dies) | Single spike ≈1.0 | Non-stationary | Difference first |
| Spikes at s, 2s, 3s | Spike at s | Seasonal | Use SARIMA |
| Decays with alternating signs | Cuts off | AR with negative φ | Count PACF spikes |
| Cuts off with alternating | Decays | MA with negative θ | Count ACF spikes |

---

## Tricky Cases and What They Mean

**Spike at lag 1 only in ACF, decaying PACF → MA(1)**
The most common MA pattern. Often mistaken for AR because the ACF spike
at lag 1 looks like an AR(1) ACF. Check PACF — if it decays, it is MA.

**Spike at lag 1 only in PACF, decaying ACF → AR(1)**
The most common AR pattern. PACF cuts off cleanly after lag 1.

**Alternating signs in ACF decay → AR with negative φ**
Negative autoregressive coefficients produce oscillating ACF. Still decays
overall — still AR. PACF will cut off; count the spikes.

**Significant spike only at lag 2 in PACF, not lag 1 → AR(2) with φ₁ ≈ 0**
The AR(2) model has a zero first coefficient and a non-zero second. Unusual
but valid. Fit ARIMA(2,d,0) and check if φ₁ is significant.

**Single borderline spike in both ACF and PACF at lag 1**
Could be ARMA(1,1) or just noise. Try both ARIMA(1,d,0) and ARIMA(0,d,1)
and compare AIC. If sample is small (n < 100), the pattern is unreliable —
rely on AIC rather than visual identification.

**Gradual decay in ACF, then a sudden spike at lag 12**
Non-seasonal AR structure plus seasonal component. Model as
SARIMA(p,d,q)(1,0,0)[12] — handle both levels.

---

## Reading the Residual ACF After Fitting

After fitting any ARIMA model, the residuals should look like white noise.
This is how you verify the model is correct:

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

model = ARIMA(series, order=(1, 1, 0)).fit()
residuals = model.resid

plot_acf(residuals, lags=40)
# all spikes should be inside ±1.96/√n
```

**Residual ACF still has spikes → model is wrong:**

| Residual pattern | Fix |
|-----------------|-----|
| Spike at lag 1 in ACF | Add MA(1) term → ARIMA(p,d,1) |
| Spike at lag 1 in PACF | Increase AR order → ARIMA(p+1,d,q) |
| Spikes at lag s, 2s | Add seasonal terms → SARIMA |
| Spikes at many lags | Series may need another difference |
| Random single spike | Likely noise — run Ljung-Box to confirm |

**Ljung-Box test — formal check instead of eyeballing:**
```python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(model.resid, lags=[10, 20, 30], return_df=True)
print(lb)
# all p-values should be > 0.05 → residuals are white noise
```

---

## Full Code — Plot and Read in One Step

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

def read_acf_pacf(series, name="Series", lags=40):
    series = pd.Series(series).dropna()
    n  = len(series)
    ci = 1.96 / np.sqrt(n)

    av = acf(series,  nlags=lags, fft=True)
    pv = pacf(series, nlags=lags, method='ywmle')

    sig_acf  = [k for k in range(1, lags+1) if abs(av[k]) > ci]
    sig_pacf = [k for k in range(1, lags+1) if abs(pv[k]) > ci]

    # pattern detection
    acf_decays_slow = all(abs(av[k]) > ci for k in range(1, min(8, lags)))
    acf_cuts_off    = len(sig_acf) > 0 and max(sig_acf) <= 3 and len(sig_acf) <= 3
    pacf_cuts_off   = len(sig_pacf) > 0 and max(sig_pacf) <= 3 and len(sig_pacf) <= 3
    both_flat       = len(sig_acf) == 0 and len(sig_pacf) == 0
    seasonal        = any(k in sig_acf for k in [4, 7, 12, 24, 52])

    if acf_decays_slow:
        pattern = "NON-STATIONARY — difference first"
    elif both_flat:
        pattern = "WHITE NOISE — no model needed"
    elif seasonal:
        pattern = f"SEASONAL — spikes at {[k for k in sig_acf if k in [4,7,12,24,52]]}"
    elif pacf_cuts_off and not acf_cuts_off:
        p = max(sig_pacf)
        pattern = f"AR({p}) — PACF cuts off at lag {p}, ACF decays"
    elif acf_cuts_off and not pacf_cuts_off:
        q = max(sig_acf)
        pattern = f"MA({q}) — ACF cuts off at lag {q}, PACF decays"
    else:
        pattern = "ARMA — both decay; run AIC grid search"

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  CI       : ±{ci:.3f}   n={n}")
    print(f"  Sig ACF  : {sig_acf[:8]}")
    print(f"  Sig PACF : {sig_pacf[:8]}")
    print(f"  Pattern  : {pattern}")
    print(f"{'='*50}\n")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    plot_acf(series,  lags=lags, ax=axes[0], title=f'ACF — {name}')
    plot_pacf(series, lags=lags, ax=axes[1], title=f'PACF — {name}',
              method='ywmle')
    plt.tight_layout()
    plt.show()

    return pattern
```

---

## References

- Box, G.E.P. & Jenkins, G.M. (1970). *Time Series Analysis: Forecasting
  and Control.* Holden-Day.
  — Original source of ACF/PACF-based model identification.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice* (3rd ed.). Chapters 9–10.
  https://otexts.com/fpp3/acf.html
- Shumway, R.H. & Stoffer, D.S. (2017). *Time Series Analysis and Its
  Applications.* Springer. Chapter 3.
- statsmodels plot_acf:
  https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html