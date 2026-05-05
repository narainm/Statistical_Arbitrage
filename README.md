# Statistical Arbitrage — Avellaneda & Lee (2010) | v2

Python implementation of the mean-reversion statistical arbitrage strategy from:

> **Avellaneda, M. & Lee, J.H. (2010).** Statistical arbitrage in the US equities market. *Quantitative Finance, 10(7), 761–782.*

This is a research-level implementation that uses the Ornstein-Uhlenbeck model from the paper, a PCA-based factor model, ETF-based factor model, volume-adjusted signals, and optional factor-neutral portfolio construction.

**This is a research project, not a production system. Performance results may not replicate the paper due to survivorship bias, universe size, and data differences. See Limitations below.**

---

## What Changed from v1

| Feature | v1 | v2 |
|---|---|---|
| Signal model | Rolling z-score | OU / AR(1) s-score (Appendix A) |
| PCA window | 60 days | 252 days (matches paper) |
| ETF factor model | No | Yes |
| Volume adjustment | No | Yes (section 6), OFF by default |
| Factor neutrality | No | Optional (linear algebra) |
| Mean centering | No | Yes (equation 18) |
| MAX_KAPPA filter | No | Yes (40.0, rejects degenerate fits) |
| Diagnostics | No | Yes (print_diagnostics in metrics.py) |

---

## Project Structure

```
stat_arb_v2/
├── data.py         — Download prices + volume, sector ETF mapping
├── factors.py      — PCA residuals + ETF regression residuals
├── ou_model.py     — AR(1) fitting, OU params, s-score, volume adjustment
├── signals.py      — Position state machine (bang-bang rule)
├── portfolio.py    — Dollar-neutral weights + optional factor neutrality
├── backtest.py     — P&L simulation with transaction costs
├── metrics.py      — Sharpe, drawdown, rolling Sharpe, regime analysis, diagnostics
├── main.py         — Runs everything, produces charts
├── research_log.md — Test iterations and bug history
└── README.md
```

---

## The OU Model

The paper models the idiosyncratic residual X(t) as an Ornstein-Uhlenbeck process:

```
dX = kappa * (mu - X) * dt + sigma * dW
```

In discrete time this is an AR(1) model:

```
X_{t+1} = a + b * X_t + noise
```

From the AR(1) coefficients we recover:
- `kappa = -log(b) * 252`  — mean-reversion speed
- `mu = a / (1 - b)`       — long-run mean
- `sigma_eq = std(noise) / sqrt(1 - b^2)`  — equilibrium volatility

### S-score formula

The **s-score** is:

```
s = (X_t - mu_adj) / sigma_eq
```

where `mu_adj = mu - mean(mu across all stocks)` (cross-sectional centring, equation 18 in the paper), and `X_t` is the current cumulative residual for this specific stock.

**Important note on the simplified formula.** Appendix A of the paper states that because `X_60 = 0` by construction (their OLS regression forces the residuals to sum to zero within each window), the s-score simplifies to `s = -mu / sigma_eq`. That simplification does **not** apply here. In this implementation, the cumulative residual series is mean-centred (subtract the mean of the window) but `X_t` at the end of the window is not constrained to zero. We therefore use the full formula `s = (X_t - mu_adj) / sigma_eq`.

Using the simplified formula `-mu / sigma_eq` when `X_t != 0` was one of the bugs in earlier versions (see Test 1 in research_log.md).

### Kappa filters

We filter out stocks where kappa is outside the range [8.4, 40.0]:
- `kappa < 8.4`: mean-reversion time > 30 days, too slow to be useful.
- `kappa > 40.0`: mean-reversion time < ~6 days, almost certainly a degenerate fit.

---

## Volume Adjustment

From section 6 of the paper: if a stock rallies on high volume, we trust the move more and fade it less. If it moves on low volume, we're more willing to bet against it.

Implemented as: `s_adjusted = s_raw / vol_ratio`

where `vol_ratio = today_volume / 10-day_average_volume`.

Toggle with `USE_VOLUME_ADJ = True/False` in main.py. **Default is False** until the base PCA + OU signal is validated (see research_log.md).

---

## Factor Neutrality

Beyond dollar-neutrality, we can reduce portfolio exposure to the systematic PCA factors via a least-squares projection:

```
delta_w = -L @ pinv(L^T L) @ exposure
```

where L is the factor loading matrix and `exposure = L^T @ w`.

Toggle with `USE_FACTOR_NEUTRAL = True/False`. **Default is False.**

---

## Diagnostics

`metrics.print_diagnostics(sscore_df, kappa_df)` prints at the start of every run:

- Number of valid s-score rows
- Number of nonzero trading days
- Average absolute s-score
- Mean, median, min, max kappa (post-filter)

This is useful for catching OU model failures early. A near-zero average |s-score| is a sign that the residual computation or cumsum logic has gone wrong.

---

## How to Run

```bash
pip install pandas numpy yfinance scikit-learn matplotlib pyarrow
python main.py
```

First run downloads data and caches it. Subsequent runs load from cache.

**Runtime:** 5–10 minutes (the 252-day rolling PCA is slow across 2,500+ days).

---

## Key Parameters (main.py)

| Param | Default | Notes |
|---|---|---|
| `FACTOR_MODEL` | `"PCA"` | `"PCA"` or `"ETF"` |
| `N_PCA_FACTORS` | `15` | Fixed factor count |
| `PCA_VAR_THRESHOLD` | `None` | Set `0.50` for variance-based selection |
| `PCA_WINDOW` | `252` | Correlation matrix window (matches paper) |
| `OU_WINDOW` | `60` | OU estimation window (matches paper) |
| `USE_VOLUME_ADJ` | `False` | Off until base signal is validated |
| `USE_FACTOR_NEUTRAL` | `False` | Factor-neutral construction |
| `TC_BPS` | `10.0` | Round-trip costs (matches paper) |

---

## Debugging Log / Model Iterations

See `research_log.md` for the full test history. Summary:

**Test 1:** Used `s = -mu / sigma_eq` (simplified formula). This is only valid when `X_60 = 0`, which is not the case here. Equity curve bled down. Also had the `return` indented inside the time loop (function exited after the first valid day).

**Test 2:** Switched to `s = (x_t - mu_adj) / sigma_eq` but `x_t` was picked up from a shared `x_series` variable that belonged to the last ticker processed in the loop. Every stock got the same (wrong) X_t. Signals were numerically non-trivial but cross-sectionally identical.

**Test 3:** Tried standardising the daily residuals before cumulating (`col_std = (col - mean) / std`). This killed signal magnitude — sigma_eq collapsed and almost no trades were opened.

**Test 4 (current):** Cumulate raw residuals, then subtract the mean of the cumulative series (centre, don't standardise). Store `x_t_vals[ticker] = x_series[-1]` inside the ticker loop. Use `x_t = x_t_vals[ticker]` in the scoring step. Fixed the `return` indentation. Added `MAX_KAPPA = 40.0` filter.

---

## Limitations

- **Survivorship bias**: Universe is current S&P constituents, not point-in-time. This inflates performance.
- **Small universe**: ~90 stocks vs 1,400+ in the paper. Less diversification, fewer opportunities.
- **Short borrowing costs**: Not modeled.
- **Execution**: Assumes close-to-close fills. Real execution has slippage beyond the flat 10bps.
- **ETF model simplification**: Each stock regressed on one ETF. The paper discusses multi-ETF approaches.

---

## References

- Avellaneda, M. & Lee, J.H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761–782.
- Khandani, A.E. & Lo, A.W. (2007). What happened to the quants in August 2007? SSRN.
