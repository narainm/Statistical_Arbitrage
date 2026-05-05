"""
metrics.py

Performance metrics, year-by-year breakdown, and basic regime analysis.

Changes vs previous version:
- Added print_cost_diagnostics(): compares gross vs net returns, computes
  average daily turnover, annualized turnover, average daily TC drag,
  and total TC drag over the period.
- Added print_activity_diagnostics(): reports trading-day counts, average
  long/short counts, gross/net exposure, and average daily position changes.
- print_diagnostics() unchanged.
- All other functions unchanged.
"""

import numpy as np
import pandas as pd


def annualised_return(pnl, n=252):
    total = (1 + pnl).prod()
    years = len(pnl) / n
    return total ** (1 / years) - 1 if years > 0 else np.nan


def annualised_vol(pnl, n=252):
    return pnl.std() * np.sqrt(n)


def sharpe(pnl, rf=0.0, n=252):
    ann_ret = annualised_return(pnl, n)
    ann_vol = annualised_vol(pnl, n)
    if ann_vol == 0:
        return np.nan
    return (ann_ret - rf) / ann_vol


def max_drawdown(pnl):
    eq = (1 + pnl).cumprod()
    peak = eq.cummax()
    return ((eq - peak) / peak).min()


def win_rate(pnl):
    return (pnl > 0).mean()


def rolling_sharpe(pnl, window=252):
    roll_ret = pnl.rolling(window).mean() * 252
    roll_vol = pnl.rolling(window).std() * np.sqrt(252)
    return roll_ret / roll_vol


def print_summary(pnl, label="Strategy"):
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  Period:       {pnl.index[0].date()} -> {pnl.index[-1].date()}")
    print(f"  Days:         {len(pnl)}")
    print(f"  Ann. Return:  {annualised_return(pnl):.2%}")
    print(f"  Ann. Vol:     {annualised_vol(pnl):.2%}")
    print(f"  Sharpe:       {sharpe(pnl):.2f}")
    print(f"  Max DD:       {max_drawdown(pnl):.2%}")
    print(f"  Win Rate:     {win_rate(pnl):.2%}")
    print(f"{'='*52}")


def yearly_sharpe(pnl):
    results = {}
    for yr in sorted(pnl.index.year.unique()):
        yr_pnl = pnl[pnl.index.year == yr]
        if len(yr_pnl) < 50:
            continue
        results[yr] = sharpe(yr_pnl)
    return pd.Series(results, name="Sharpe")


def print_diagnostics(sscore_df, kappa_df):
    """
    Quick health check on the OU model output. Run this before the backtest
    so you catch obvious signal failures before waiting for the full run.
    """
    valid_rows = sscore_df.dropna(how="all")
    nonzero_days = (sscore_df.abs().sum(axis=1) > 0).sum()
    avg_abs_sscore = sscore_df.abs().stack().mean()

    kappa_flat = kappa_df.stack().dropna()

    print(f"\n{'='*52}")
    print("  OU Model Diagnostics")
    print(f"{'='*52}")
    print(f"  Valid s-score rows:      {len(valid_rows)}")
    print(f"  Nonzero trading days:    {nonzero_days}")
    print(f"  Avg |s-score|:           {avg_abs_sscore:.4f}")
    print(f"  Mean kappa:              {kappa_flat.mean():.2f}")
    print(f"  Median kappa:            {kappa_flat.median():.2f}")
    print(f"  Min kappa (post-filter): {kappa_flat.min():.2f}")
    print(f"  Max kappa (post-filter): {kappa_flat.max():.2f}")
    print(f"{'='*52}")

    if avg_abs_sscore < 0.01:
        print("  WARNING: average |s-score| is near zero. Signal may be wiped out.")
        print("  Check residual computation and cumsum logic in ou_model.py.")
    if nonzero_days == 0:
        print("  WARNING: no nonzero trading days found. OU model may have failed entirely.")


def print_cost_diagnostics(net_pnl, gross_pnl, weight_df, tc_bps=10.0):
    """
    Compare gross vs net performance and quantify the transaction cost drag.

    This is the key diagnostic for understanding whether turnover is killing
    the strategy. If gross Sharpe looks reasonable but net Sharpe is deeply
    negative, the trade frequency is the problem, not the signal.

    Turnover here = sum of absolute daily weight changes across all stocks.
    Annualized turnover = average daily turnover * 252.
    """
    tc = tc_bps / 10_000

    # Align indices (weight_df may have a different index than pnl)
    common = net_pnl.index.intersection(weight_df.index)
    w = weight_df.loc[common]

    # Daily turnover = total absolute weight change across all stocks
    daily_turnover = w.diff().abs().sum(axis=1).fillna(0)

    # Daily TC drag = turnover * cost rate (should match the backtest)
    daily_tc = daily_turnover * tc

    total_tc_drag = daily_tc.sum()
    avg_daily_tc  = daily_tc.mean()
    ann_turnover  = daily_turnover.mean() * 252

    gross_ann_ret = annualised_return(gross_pnl)
    net_ann_ret   = annualised_return(net_pnl)
    gross_sharpe  = sharpe(gross_pnl)
    net_sharpe    = sharpe(net_pnl)

    print(f"\n{'='*52}")
    print("  Cost & Turnover Diagnostics")
    print(f"{'='*52}")
    print(f"  TC rate:               {tc_bps:.1f} bps per round-trip")
    print(f"  Avg daily turnover:    {daily_turnover.mean():.4f}  (sum |delta w|)")
    print(f"  Annualized turnover:   {ann_turnover:.2f}x")
    print(f"  Avg daily TC drag:     {avg_daily_tc:.6f}  ({avg_daily_tc*252:.2%} ann.)")
    print(f"  Total TC drag:         {total_tc_drag:.4f}  over full period")
    print(f"  ---")
    print(f"  Gross ann. return:     {gross_ann_ret:.2%}")
    print(f"  Net   ann. return:     {net_ann_ret:.2%}")
    print(f"  Gross Sharpe:          {gross_sharpe:.2f}")
    print(f"  Net   Sharpe:          {net_sharpe:.2f}")
    print(f"  Sharpe drag from TC:   {gross_sharpe - net_sharpe:.2f}")
    print(f"{'='*52}")

    if ann_turnover > 50:
        print("  NOTE: annualized turnover is very high (>50x). Strategy is likely")
        print("  trading in and out of positions almost every day. Consider stricter")
        print("  entry thresholds or wider exit bands to reduce churn.")


def print_activity_diagnostics(pos_df, weight_df):
    """
    Report position-level activity stats.

    Useful for checking whether stricter entry thresholds meaningfully
    reduced trade frequency vs the baseline, or whether the strategy is
    still churning at roughly the same rate.
    """
    n_long  = (pos_df == 1).sum(axis=1)
    n_short = (pos_df == -1).sum(axis=1)

    # Days where at least one position was held
    active_days = ((pos_df != 0).sum(axis=1) > 0).sum()

    # Gross and net exposure
    gross_exposure = pos_df.abs().sum(axis=1)  # in position units (pre-weighting)
    net_exposure   = pos_df.sum(axis=1)

    # Daily position changes: how many stocks changed state each day
    pos_changes = (pos_df.diff().abs() > 0).sum(axis=1)

    print(f"\n{'='*52}")
    print("  Position Activity Diagnostics")
    print(f"{'='*52}")
    print(f"  Days with active positions: {active_days}")
    print(f"  Avg longs per day:          {n_long.mean():.1f}")
    print(f"  Avg shorts per day:         {n_short.mean():.1f}")
    print(f"  Peak longs:                 {n_long.max()}")
    print(f"  Peak shorts:                {n_short.max()}")
    print(f"  Avg gross exposure:         {gross_exposure.mean():.1f} positions")
    print(f"  Avg net exposure:           {net_exposure.mean():.2f} positions")
    print(f"  Avg daily position changes: {pos_changes.mean():.1f} stocks/day")
    print(f"  Max position changes (day): {pos_changes.max()} stocks")
    print(f"{'='*52}")


def regime_analysis(pnl, vix=None):
    """
    Split performance into high-vol vs low-vol regimes.

    If VIX data isn't available, we use realised vol of the pnl itself
    as a proxy. Not ideal but gets the point across.

    The paper observes that stat arb tends to work better in high-vol regimes
    when fewer PCA factors are needed to explain most of the variance.
    """
    if vix is not None:
        vol_proxy = vix.reindex(pnl.index).ffill()
    else:
        vol_proxy = pnl.rolling(60).std() * np.sqrt(252)

    median_vol = vol_proxy.median()
    high_vol = pnl[vol_proxy > median_vol]
    low_vol  = pnl[vol_proxy <= median_vol]

    print("\nRegime Analysis (high vs low vol):")
    print(f"  High vol Sharpe: {sharpe(high_vol):.2f}  ({len(high_vol)} days)")
    print(f"  Low  vol Sharpe: {sharpe(low_vol):.2f}  ({len(low_vol)} days)")

    return high_vol, low_vol
