"""
ou_model.py

Estimates Ornstein-Uhlenbeck parameters from residual return series
and computes the s-score, following Appendix A of Avellaneda & Lee (2010).

The OU process is: dX = kappa * (mu - X) * dt + sigma * dW
We fit it using AR(1) regression on the cumulative residual series X_t.

AR(1): X_{t+1} = a + b * X_t + noise
Then:
  kappa = -log(b) * 252
  mu    = a / (1 - b)
  sigma = std(noise) * sqrt(2 * kappa / (1 - b^2))
  sigma_eq = std(noise) / sqrt(1 - b^2)

s-score = SIGNAL_SCALE * (X_t - mu_adj) / sigma_eq

Note: X_t is the current cumulative residual for each stock. It is NOT
always zero. The earlier simplification s = -mu / sigma_eq only applies
when X_60 = 0, which happens in the paper because they reset the cumsum
each window AND the OLS regression forces residuals to sum to zero. In
this implementation we centre the cumulative series (subtract the mean)
but do NOT force X_t to zero, so we use the full formula.

SIGNAL_SCALE = 1.5 as of Test 8. This is a diagnostic multiplier only --
it amplifies the signal proportionally without changing any OU parameters,
factor exposures, or residual structure. If a higher scale produces better
net performance, it confirms the model is directionally correct but the raw
s-scores are too small relative to the entry thresholds.

-- Research log: see research_log.md for bug history and test iterations --
"""

import numpy as np
import pandas as pd


# Stocks with kappa below this revert too slowly to be useful
# (mean-reversion time > 30 business days out of a 252-day year)
MIN_KAPPA = 8.4

# Stocks with kappa above this are suspicious -- instant mean-reversion
# is almost certainly a data artefact or a degenerate AR(1) fit.
MAX_KAPPA = 40.0

# Test 8 diagnostic: multiply the raw s-score by this constant.
# The model structure, OU parameters, and residuals are untouched.
# Purpose: if avg |s-score| ~0.7 and entry threshold is 2.0, very few
# trades fire. Scaling by 1.5 pushes avg |s-score| to ~1.05 and tests
# whether the signal is directionally correct but simply too small.
# Set back to 1.0 to return to unscaled behaviour.
SIGNAL_SCALE = 1.5


def fit_ou_params(x_series):
    """
    Fit OU parameters to a cumulative residual series x_series.
    x_series should be a 1D numpy array of length ~60 (the estimation window).

    Returns a dict with: kappa, mu, sigma, sigma_eq, b
    Returns None if the series fails the mean-reversion filter.

    Changes vs original:
    - Added MAX_KAPPA = 40.0 upper filter to reject degenerate fast-reversion fits.
    - Comments updated to reflect that s-score now uses ticker-specific X_t.
    """
    x = np.array(x_series, dtype=float)

    if len(x) < 10:
        return None

    # AR(1) regression: X_{t+1} = a + b * X_t
    x_lag = x[:-1]
    x_next = x[1:]

    X_mat = np.column_stack([np.ones(len(x_lag)), x_lag])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X_mat, x_next, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a, b = coeffs

    # b should be in (0, 1) for a stationary mean-reverting process.
    # b >= 1 implies a random walk or explosive process -- skip it.
    if b <= 0 or b >= 1:
        return None

    noise_var = np.var(x_next - (a + b * x_lag))

    kappa = -np.log(b) * 252

    # Lower bound: reversion too slow to be useful over a 60-day window
    if kappa < MIN_KAPPA:
        return None

    # Upper bound: kappa > 40 means the series reverts in under ~6 trading
    # days. This is almost always a sign of an over-fit or bad data window,
    # not a real signal.
    if kappa > MAX_KAPPA:
        return None

    mu = a / (1 - b)
    sigma_eq = np.sqrt(noise_var / (1 - b**2))

    if sigma_eq < 1e-8:
        return None

    sigma = np.sqrt(noise_var * 2 * kappa / (1 - b**2))

    return {
        "kappa": kappa,
        "mu": mu,
        "sigma": sigma,
        "sigma_eq": sigma_eq,
        "b": b,
    }


def compute_sscore_rolling(
    resid_daily,
    ou_window=60,
    centre_means=True,
):
    """
    Compute rolling s-scores for all stocks.

    For each day t, for each stock i:
      1. Take the last ou_window days of daily residuals
      2. Cumulate them: X_series = cumsum(resid[t-window : t])
         Then centre: X_series -= mean(X_series)
      3. Fit OU to that X_series via AR(1)
      4. Store X_t = X_series[-1]  (the most recent value in this window)
      5. s_score = SIGNAL_SCALE * (X_t - mu_adj) / sigma_eq

    Why NOT -mu / sigma_eq?
    The simplified formula s = -mu / sigma_eq from Appendix A only holds
    when X_60 = 0 by construction (the paper's OLS forces the residuals to
    sum exactly to zero). In this implementation that constraint is not
    imposed, so X_t is not necessarily zero. Using the full formula is
    correct and more general.

    centre_means: subtracts the cross-sectional mean of mu estimates before
    computing s-scores. This is equation (18) in the paper and helps remove
    model drift bias.

    LOOKAHEAD NOTE: Only past data enters the window. The s-score at day t
    uses data from [t-window, t-1] only.

    Bug fixes applied (see research_log.md):
    - x_t_vals dict is now populated per-ticker INSIDE the ticker loop.
    - x_t = x_t_vals[ticker] is used in the scoring step, not a stale x_series.
    - return is now OUTSIDE the time loop (was accidentally indented inside).
    - Residuals are cumulated then mean-centred; NOT standardised before cumsum.
    """
    n_days, n_stocks = resid_daily.shape
    tickers = resid_daily.columns

    sscore_df = pd.DataFrame(
        np.nan, index=resid_daily.index, columns=tickers
    )
    kappa_df = pd.DataFrame(
        np.nan, index=resid_daily.index, columns=tickers
    )

    for t in range(ou_window, n_days):
        window_resid = resid_daily.iloc[t - ou_window : t]

        mu_vals = {}
        sigma_eq_vals = {}
        kappa_vals = {}
        x_t_vals = {}  # BUG FIX: store each ticker's current X_t separately

        for ticker in tickers:
            col = window_resid[ticker].dropna()
            if len(col) < 20:
                continue

            # Cumulate residuals within this window, then centre.
            # Centring (subtract mean) reduces the effect of persistent drift
            # without killing signal magnitude. Do NOT standardise col first --
            # that over-normalises and wipes out the signal (Test 3 failure).
            x_series = col.cumsum().values
            x_series = x_series - np.mean(x_series)

            # BUG FIX: store X_t before fitting so we can retrieve it later
            x_t_vals[ticker] = x_series[-1]

            params = fit_ou_params(x_series)
            if params is None:
                continue

            mu_vals[ticker] = params["mu"]
            sigma_eq_vals[ticker] = params["sigma_eq"]
            kappa_vals[ticker] = params["kappa"]

        if len(mu_vals) < 5:
            continue

        # Cross-sectional mean centering (equation 18 in paper).
        # Removes overall model drift bias -- assumes average stock is fairly priced.
        mean_mu = np.mean(list(mu_vals.values())) if centre_means else 0.0

        current_date = resid_daily.index[t]

        for ticker in mu_vals:
            mu_adj = mu_vals[ticker] - mean_mu
            sigma_eq = sigma_eq_vals[ticker]

            # BUG FIX: use this ticker's own stored X_t, not whatever x_series
            # happened to be in scope from the last ticker in the loop above.
            x_t = x_t_vals[ticker]

            # Full s-score formula. s > 0 means stock is above its OU mean
            # (overpriced relative to factor model) -> short candidate.
            # s < 0 means underpriced -> long candidate.
            # SIGNAL_SCALE = 1.5 for Test 8: amplify signal without changing
            # any underlying OU parameters or residual structure.
            s = SIGNAL_SCALE * (x_t - mu_adj) / sigma_eq

            sscore_df.loc[current_date, ticker] = s
            kappa_df.loc[current_date, ticker] = kappa_vals[ticker]

    # Extended s-score diagnostics (Test 8).
    # Prints mean |s|, std, and max |s| so we can confirm the scaling
    # is producing reasonable magnitudes before running the full backtest.
    valid_scores = sscore_df.stack().dropna()
    if len(valid_scores) > 0:
        print(f"\nS-score diagnostics (SIGNAL_SCALE={SIGNAL_SCALE}):")
        print(f"  Mean |s-score|:  {valid_scores.abs().mean():.4f}")
        print(f"  Std of s-score:  {valid_scores.std():.4f}")
        print(f"  Max |s-score|:   {valid_scores.abs().max():.4f}")

    # BUG FIX: this return was previously INSIDE the time loop, causing the
    # function to exit after the very first valid t. Now correctly outside.
    return sscore_df, kappa_df


def apply_volume_adjustment(sscore_df, vol_ratio_df, tickers=None):
    """
    Trading-time signal adjustment from section 6 of the paper.

    The idea: if a stock moves on HIGH volume, the move is probably
    informative and we should fade it less aggressively (shrink the signal).
    If it moves on LOW volume, it looks more like noise and we lean in harder.

    Mechanically: s_adjusted = s_raw / vol_ratio
    where vol_ratio = today_volume / rolling_avg_volume.

    High volume => vol_ratio > 1 => smaller adjusted signal.
    Low volume  => vol_ratio < 1 => larger adjusted signal.

    NOTE: this is OFF by default (USE_VOLUME_ADJ = False in main.py) until
    the base PCA + OU model is validated. Test 2 and beyond isolate the
    base signal first.
    """
    if tickers is None:
        tickers = [t for t in sscore_df.columns if t in vol_ratio_df.columns]

    adjusted = sscore_df.copy()

    for ticker in tickers:
        if ticker not in vol_ratio_df.columns:
            continue
        vr = vol_ratio_df[ticker].reindex(sscore_df.index)
        # Clip to avoid dividing by near-zero or amplifying excessively
        vr = vr.clip(0.2, 5.0)
        adjusted[ticker] = sscore_df[ticker] / vr

    return adjusted
