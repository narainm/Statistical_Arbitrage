# Research Log — Avellaneda & Lee Stat Arb Implementation

This file documents each test iteration of the OU model and s-score computation.
The goal is to keep a clear record of what broke, what was changed, and why,
so the debugging history is traceable.

---

## Test 1 — Baseline (Broken)

**Formula:** `s = -mu_adj / sigma_eq`

**Volume adjustment:** ON (`USE_VOLUME_ADJ = True`)

**Residual path:**
```python
x_series = col.cumsum().values
```

**What the code did:**  
Used `s = -mu / sigma_eq` as the s-score, following the simplified formula in
Appendix A of the paper. That formula is only valid when X_60 = 0, which happens
in the paper because their OLS regression forces the residuals to sum to zero over
the 60-day window. In this implementation, the residuals do NOT sum to zero, so
X_t at the end of the window is not zero. Using -mu / sigma_eq ignores the actual
current position of the cumulative residual.

**Issue observed:**  
The equity curve bled down steadily across the full backtest period. The signal was
structurally wrong: it was only using the long-run mean of the OU process (mu), not
how far the current residual was from that mean.

**Bugs also present (unrelated to formula):**
- `x_t_vals` dict was created but never populated, so the code had dead code paths.
- `return sscore_df, kappa_df` was accidentally indented inside the time loop,
  causing the function to exit after the very first valid `t`. Almost no s-scores
  were computed.

---

## Test 2 — Formula corrected, but X_t still wrong

**Formula changed to:** `s = (x_t - mu_adj) / sigma_eq`

**Volume adjustment:** OFF (`USE_VOLUME_ADJ = False`)

**Residual path:**
```python
x_series = col.cumsum().values
x_t = x_series[-1]  # <-- bug: not stored per ticker
```

**What the code did:**  
Switched to the correct full s-score formula. However, `x_t = x_series[-1]` was
placed in the scoring loop AFTER iterating over all tickers. At that point,
`x_series` was whatever the last ticker in the loop had computed, not the current
ticker being scored. So every ticker got the same (wrong) X_t value.

Also: the `return` bug was still present -- function still exited after first valid t.

**Issue observed:**  
Scores were numerically non-trivial (not zero) but cross-sectionally identical for
all stocks on any given day. Long and short signals were essentially random noise.

---

## Test 3 — Over-normalisation of residuals

**Formula:** `s = (x_t - mu_adj) / sigma_eq`

**Volume adjustment:** OFF

**Residual path (attempted fix):**
```python
col_std = (col - col.mean()) / (col.std() + 1e-8)
x_series = np.cumsum(col_std.values)
x_t_vals[ticker] = x_series[-1]
```

**What the code did:**  
Standardised the daily residual column before computing the cumulative sum. The
intent was to make the scale of X_series more consistent across stocks.

**Issue observed:**  
Over-normalisation. Standardising each stock's 60-day residual window to unit
variance before cumulating collapses the magnitude of X_series. The OU fit still
works, but sigma_eq becomes very small and the s-scores either blow up or collapse
to near-zero depending on rounding. Almost no stocks passed the kappa filter,
and the ones that did generated effectively zero signal. Very few trades were opened.

**Lesson:** The paper does NOT standardise the residuals before cumulating. The
cross-sectional variance difference between stocks is real information. The centring
(subtracting the mean of the cumulative series) is enough to remove drift without
killing scale.

---

## Test 4 — Corrected version (base PCA + OU)

**Formula:** `s = (x_t - mu_adj) / sigma_eq`

**Volume adjustment:** OFF (`USE_VOLUME_ADJ = False`)

**Residual path:**
```python
x_series = col.cumsum().values
x_series = x_series - np.mean(x_series)  # centre, do not standardise
x_t_vals[ticker] = x_series[-1]          # store per ticker BEFORE fit
```

**Scoring step:**
```python
x_t = x_t_vals[ticker]                   # pull this ticker's own X_t
s = (x_t - mu_adj) / sigma_eq
```

**Bug fixes applied:**
1. `x_t_vals` is now populated inside the ticker loop, before `fit_ou_params`.
2. `x_t = x_t_vals[ticker]` is used in scoring, not a stale `x_series` from the
   outer scope.
3. `return sscore_df, kappa_df` is now correctly outside the time loop.
4. Added `MAX_KAPPA = 40.0` filter to reject degenerate fast-reversion fits.
5. Residuals are cumulated and mean-centred. No pre-standardisation.

**Outcome:**  
Gross equity curve showed a clear upward edge. Net equity curve still lost money.
The gross-to-net spread was large, pointing at transaction costs and overtrading
as the primary problem, not signal quality.

---

## Test 5 — Stricter entry, lower transaction costs

**Entry threshold:** 2.0 (raised from 1.25)

**Exit thresholds:** 0.0 / 0.0 (symmetric, exit when s-score crosses zero)

**TC:** 2 bps (lowered from 10 bps, consistent with large-cap liquid names)

**Volume adjustment:** OFF

**Purpose:**  
Attack the turnover problem from two angles simultaneously: fewer trades opened
(stricter entry) and lower cost per trade (realistic for liquid S&P names). By
testing both changes together first, we can tell if the net curve turns positive
at all before trying to separate the effects.

Raising entry from 1.25 to 2.0 means only stocks with a larger anomalous deviation
from their OU equilibrium trigger a position. This should reduce daily churn
significantly.

Setting exit to 0.0 (close when s-score crosses zero rather than stopping at -0.50
/ +0.75) gives positions more room to run before being closed. Tighter exits were
causing rapid round-trips in Test 4.

**Status:** Pending run. This is the current recommended test.

---

## Test 6 — Stricter entry, original transaction costs

**Entry threshold:** 2.0

**Exit thresholds:** 0.0 / 0.0

**TC:** 10 bps (original paper assumption)

**Volume adjustment:** OFF

**Purpose:**  
Same signal configuration as Test 5, but with 10 bps costs. This isolates the
question: does the net improvement in Test 5 (if any) come from trading less, or
just from paying less per trade?

If Test 5 beats Test 4 net but Test 6 still loses, the 2 bps assumption is doing
most of the work. If Test 6 also improves, reduced turnover is the real driver.

**Status:** Pending run. Run after Test 5 results are in.

---

## Test 7 — Medium entry threshold, lower costs

**Entry threshold:** 1.75

**Exit thresholds:** 0.25 / 0.25

**TC:** 2 bps

**Volume adjustment:** OFF

**Purpose:**  
Middle-ground test. If Test 5 with entry=2.0 turns out to be too selective
(very few positions, low exposure, strategy rarely fires), this test checks
whether 1.75 gives a better balance between signal quality and trade frequency.

The exit at 0.25 (rather than 0.0) adds a small buffer so positions don't get
closed on noise around the zero line.

**Status:** Pending run. Run after Tests 5 and 6 to compare trade frequency.

---

## Next Steps

Once Tests 5, 6, and 7 have results, the plan is:

1. Compare annualized turnovers across Test 4, 5, 6, 7 using `print_cost_diagnostics`.
2. Confirm which combination of entry threshold and cost assumption produces a
   positive net Sharpe.
3. If net Sharpe is positive under 2 bps, also check at 5 bps as a more
   conservative cost assumption for mid-cap names.
4. Once a clean net-positive baseline exists, re-enable `USE_VOLUME_ADJ = True`
   and test whether the volume adjustment from section 6 of the paper adds
   incremental value.
5. Consider testing the ETF factor model separately once the PCA baseline is stable.

---

## Test 8 — Rescaled s-score (diagnostic amplification)

**Formula:** `s = 1.5 * (x_t - mu_adj) / sigma_eq`

**Entry threshold:** 2.0

**Exit thresholds:** 0.0 / 0.0

**TC:** 2 bps

**Volume adjustment:** OFF

**Change made:**  
Added `SIGNAL_SCALE = 1.5` constant in `ou_model.py`. The s-score line changed from:
```python
s = (x_t - mu_adj) / sigma_eq
```
to:
```python
s = SIGNAL_SCALE * (x_t - mu_adj) / sigma_eq
```
Nothing else changed. OU parameters (kappa, mu, sigma_eq), residuals, factor model,
and all thresholds are identical to Test 5.

**Purpose:**  
Test 5 ran with avg |s-score| ~0.7 and entry threshold 2.0. At that magnitude, the
score rarely crosses the entry threshold and very few trades fire. Two explanations
are possible:

1. The signal is structurally too weak -- the raw s-scores are genuinely small
   relative to the noise in this universe, and we need to either lower the entry
   threshold or find a way to get bigger deviations.
2. The model is correct but the scale is off due to the mean-centring step or some
   other implementation detail that compresses the scores.

Multiplying by 1.5 is a clean diagnostic: it doesn't change the direction or
ranking of signals across stocks, only their magnitude. If performance improves
meaningfully, that confirms interpretation (2) -- the model is correct but the
scores are too compressed. If it doesn't help, we're dealing with (1) and need
to reconsider the signal design.

**Diagnostics added:**  
`compute_sscore_rolling` now prints mean |s-score|, std, and max |s-score|
immediately after the rolling loop completes, before any backtest runs.
These print regardless of which test case is active.

**Hypothesis:**  
If SIGNAL_SCALE = 1.5 improves net Sharpe materially vs Test 5, the next step is
to understand *why* the raw scores are compressed and fix the root cause rather
than relying on a scaling constant in production.

**Status:** Pending run.
