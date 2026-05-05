"""
main.py

Runs the full pipeline. Toggle between PCA and ETF factor models,
with or without volume adjustment and factor neutrality.

Usage:
    python main.py

Caches price data on first run. Subsequent runs are much faster.

To switch test cases, change TEST_CASE below. Everything else
(entry/exit thresholds, transaction costs, volume adj) is pulled
automatically from TEST_CONFIGS.

Changes vs previous version:
- Added TEST_CONFIGS dict with presets for tests 1-7.
- Active test case set via TEST_CASE; config applied automatically.
- Added print_cost_diagnostics and print_activity_diagnostics calls.
- Plot title and output file name now include the test case name.
- Gross P&L also shown in P&L histogram panel.
- Entry threshold line added to s-score plot.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data import (
    download_prices, get_stock_prices, get_etf_prices,
    compute_log_returns, compute_volume_ratio, SECTOR_ETF_MAP
)
from factors import compute_pca_residuals, compute_etf_residuals
from ou_model import compute_sscore_rolling, apply_volume_adjustment
from signals import generate_positions, summarise_positions
from portfolio import build_weight_matrix
from backtest import run_backtest, equity_curve, get_drawdown_series
from metrics import (
    print_summary, yearly_sharpe, rolling_sharpe, regime_analysis,
    print_diagnostics, print_cost_diagnostics, print_activity_diagnostics
)


# ------------------------------------------------------------------
# Test configuration registry
#
# Tests 1-4 are historical and documented for reference only.
# Tests 5-7 are the active research focus: reducing overtrading and
# understanding what fraction of turnover drag is avoidable.
# ------------------------------------------------------------------

TEST_CONFIGS = {
    "test_1_original_bad_sscore": {
        "description": "Original incorrect s-score: s = -mu_adj / sigma_eq, volume adjustment on.",
        "entry": 1.25,
        "exit_long": 0.50,
        "exit_short": 0.75,
        "tc_bps": 10.0,
        "volume_adj": True,
        "notes": "Historical reference only. Do not run unless reproducing old bug.",
    },
    "test_2_xt_formula_bug": {
        "description": "Changed to s = (x_t - mu_adj) / sigma_eq but accidentally reused x_series across tickers.",
        "entry": 1.25,
        "exit_long": 0.50,
        "exit_short": 0.75,
        "tc_bps": 10.0,
        "volume_adj": False,
        "notes": "Historical reference only.",
    },
    "test_3_over_normalized": {
        "description": "Normalized residuals before cumsum, which killed signal magnitude and produced almost no trades.",
        "entry": 1.25,
        "exit_long": 0.50,
        "exit_short": 0.75,
        "tc_bps": 10.0,
        "volume_adj": False,
        "notes": "Historical reference only.",
    },
    "test_4_correct_ou_base": {
        "description": "Correct ticker-specific X_t with lightly centered cumulative residuals. Base PCA + OU version.",
        "entry": 1.25,
        "exit_long": 0.50,
        "exit_short": 0.75,
        "tc_bps": 10.0,
        "volume_adj": False,
        "notes": "Gross has edge, net loses because turnover/costs are too high.",
    },
    "test_5_stricter_entry_lower_costs": {
        "description": (
            "Reduce overtrading by requiring stronger entry signals (2.0 vs 1.25) "
            "and using lower liquid-stock transaction costs (2 bps)."
        ),
        "entry": 2.0,
        "exit_long": 0.0,
        "exit_short": 0.0,
        "tc_bps": 2.0,
        "volume_adj": False,
        "notes": "Current recommended next test.",
    },
    "test_6_strict_entry_original_cost": {
        "description": (
            "Same stricter entry as test 5, but keeps original 10 bps transaction cost. "
            "Tests whether net improvement comes from lower turnover or just lower costs."
        ),
        "entry": 2.0,
        "exit_long": 0.0,
        "exit_short": 0.0,
        "tc_bps": 10.0,
        "volume_adj": False,
        "notes": "Isolates the turnover reduction effect from the cost reduction effect.",
    },
    "test_7_medium_entry_lower_cost": {
        "description": (
            "Middle-ground between paper threshold (1.25) and strict threshold (2.0). "
            "Checks whether 2.0 entry is too selective."
        ),
        "entry": 1.75,
        "exit_long": 0.25,
        "exit_short": 0.25,
        "tc_bps": 2.0,
        "volume_adj": False,
        "notes": "Checks whether 2.0 entry is too selective.",
    },
}

# ------------------------------------------------------------------
# Active test case -- change this line to switch configurations
# ------------------------------------------------------------------
TEST_CASE = "test_5_stricter_entry_lower_costs"

# ------------------------------------------------------------------
# Pull config for the active test (don't edit these directly)
# ------------------------------------------------------------------
_cfg = TEST_CONFIGS[TEST_CASE]
ENTRY_THRESH   = _cfg["entry"]
EXIT_LONG      = _cfg["exit_long"]
EXIT_SHORT     = _cfg["exit_short"]
TC_BPS         = _cfg["tc_bps"]
USE_VOLUME_ADJ = _cfg["volume_adj"]

# ------------------------------------------------------------------
# Fixed settings -- not part of the current test sweep
# ------------------------------------------------------------------
START = "2013-01-01"
END   = "2023-12-31"

FACTOR_MODEL       = "PCA"
N_PCA_FACTORS      = 15
PCA_VAR_THRESHOLD  = None
PCA_WINDOW         = 252
ETF_WINDOW         = 60
OU_WINDOW          = 60

USE_FACTOR_NEUTRAL = False
LEVERAGE           = 1.0


def main():
    print(f"\n{'='*60}")
    print(f"  Running: {TEST_CASE}")
    print(f"  {_cfg['description']}")
    print(f"  Entry: {ENTRY_THRESH} | Exit L/S: {EXIT_LONG}/{EXIT_SHORT} | TC: {TC_BPS} bps")
    if _cfg.get("notes"):
        print(f"  Note: {_cfg['notes']}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    prices, volume = download_prices(start=START, end=END)

    stock_prices = get_stock_prices(prices)
    etf_prices   = get_etf_prices(prices)

    stock_returns = compute_log_returns(stock_prices)
    etf_returns   = compute_log_returns(etf_prices)

    tickers = stock_returns.columns.tolist()
    print(f"\nUniverse: {len(tickers)} stocks | {START} to {END}")
    print(f"Factor model: {FACTOR_MODEL} | Volume adj: {USE_VOLUME_ADJ}")

    # ------------------------------------------------------------------
    # 2. Factor model -> daily idiosyncratic residuals
    # ------------------------------------------------------------------
    if FACTOR_MODEL == "PCA":
        resid_daily = compute_pca_residuals(
            stock_returns,
            estimation_window=PCA_WINDOW,
            n_factors=N_PCA_FACTORS,
            var_threshold=PCA_VAR_THRESHOLD,
        )
    elif FACTOR_MODEL == "ETF":
        etf_aligned = etf_returns.reindex(stock_returns.index).ffill()
        resid_daily = compute_etf_residuals(
            stock_returns,
            etf_aligned,
            sector_map=SECTOR_ETF_MAP,
            estimation_window=ETF_WINDOW,
        )
    else:
        raise ValueError(f"Unknown FACTOR_MODEL: {FACTOR_MODEL}")

    # ------------------------------------------------------------------
    # 3. OU model -> s-scores
    # ------------------------------------------------------------------
    sscore_df, kappa_df = compute_sscore_rolling(
        resid_daily,
        ou_window=OU_WINDOW,
        centre_means=True,
    )

    print_diagnostics(sscore_df, kappa_df)

    # ------------------------------------------------------------------
    # 4. Volume adjustment (controlled by test config)
    # ------------------------------------------------------------------
    if USE_VOLUME_ADJ:
        vol_ratio = compute_volume_ratio(
            volume[tickers].reindex(sscore_df.index),
            window=10,
        )
        sscore_adj = apply_volume_adjustment(sscore_df, vol_ratio, tickers)
    else:
        sscore_adj = sscore_df

    # ------------------------------------------------------------------
    # 5. Generate positions
    # ------------------------------------------------------------------
    pos_df = generate_positions(
        sscore_adj,
        entry_thresh=ENTRY_THRESH,
        exit_long=EXIT_LONG,
        exit_short=EXIT_SHORT,
    )

    n_long, n_short = summarise_positions(pos_df)

    # ------------------------------------------------------------------
    # 6. Portfolio weights
    # ------------------------------------------------------------------
    weight_df = build_weight_matrix(
        pos_df,
        use_factor_neutral=USE_FACTOR_NEUTRAL,
        leverage=LEVERAGE,
    )

    # ------------------------------------------------------------------
    # 7. Backtest
    # ------------------------------------------------------------------
    net_pnl, gross_pnl = run_backtest(weight_df, stock_returns, tc_bps=TC_BPS)

    eq_curve = equity_curve(net_pnl)

    # ------------------------------------------------------------------
    # 8. Performance metrics and diagnostics
    # ------------------------------------------------------------------
    print_summary(net_pnl,   label=f"NET   — {TEST_CASE}")
    print_summary(gross_pnl, label=f"GROSS — {TEST_CASE}")

    print_cost_diagnostics(net_pnl, gross_pnl, weight_df, tc_bps=TC_BPS)
    print_activity_diagnostics(pos_df, weight_df)

    yr_sharpe = yearly_sharpe(net_pnl)
    print("\nYear-by-year Sharpe (net):")
    for yr, s in yr_sharpe.items():
        if np.isnan(s):
            print(f"  {yr}: N/A")
            continue
        bar = "#" * max(0, int(s * 5))
        print(f"  {yr}: {s:+.2f}  {bar}")

    regime_analysis(net_pnl)

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    plot_results(net_pnl, gross_pnl, eq_curve, n_long, n_short, sscore_df, kappa_df)


def plot_results(net_pnl, gross_pnl, eq_curve, n_long, n_short, sscore_df, kappa_df):
    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- Equity curve ---
    ax1 = fig.add_subplot(gs[0, :])
    eq_curve.plot(ax=ax1, color="#2563eb", lw=1.6, label="Net of costs")
    equity_curve(gross_pnl).plot(ax=ax1, color="#93c5fd", lw=1.0, ls="--", label="Gross")
    ax1.axhline(1.0, color="#94a3b8", lw=0.7, ls=":")
    ax1.set_title(
        f"{TEST_CASE}  |  {FACTOR_MODEL}  |  "
        f"entry={ENTRY_THRESH}  exit L/S={EXIT_LONG}/{EXIT_SHORT}  TC={TC_BPS}bps",
        fontweight="bold",
        fontsize=9,
    )
    ax1.set_ylabel("Portfolio value ($1 start)")
    ax1.legend(loc="upper left")

    # --- Rolling Sharpe (net) ---
    ax2 = fig.add_subplot(gs[1, 0])
    rs = rolling_sharpe(net_pnl, 252)
    rs.plot(ax=ax2, color="#16a34a", lw=1.1)
    ax2.axhline(0, color="#94a3b8", lw=0.6)
    ax2.axhline(1, color="#86efac", lw=0.6, ls="--")
    ax2.set_title("Rolling 1Y Sharpe (net)")

    # --- Drawdown ---
    ax3 = fig.add_subplot(gs[1, 1])
    dd = get_drawdown_series(net_pnl)
    dd.plot(ax=ax3, color="#dc2626", lw=1.0)
    ax3.fill_between(dd.index, dd.values, 0, alpha=0.25, color="#dc2626")
    ax3.set_title("Drawdown (net)")

    # --- P&L distribution: gross vs net ---
    ax4 = fig.add_subplot(gs[1, 2])
    gross_pnl.hist(ax=ax4, bins=80, color="#93c5fd", alpha=0.6, edgecolor="none", label="Gross")
    net_pnl.hist(ax=ax4, bins=80, color="#2563eb", alpha=0.6, edgecolor="none", label="Net")
    ax4.axvline(0, color="black", lw=0.8)
    ax4.set_title("Daily P&L Distribution")
    ax4.legend(fontsize=8)

    # --- Position counts ---
    ax5 = fig.add_subplot(gs[2, 0])
    n_long.plot(ax=ax5, color="#16a34a", lw=0.8, alpha=0.8, label="Longs")
    n_short.plot(ax=ax5, color="#dc2626", lw=0.8, alpha=0.8, label="Shorts")
    ax5.set_title("Active Positions")
    ax5.legend(fontsize=8)

    # --- Mean kappa ---
    ax6 = fig.add_subplot(gs[2, 1])
    kappa_df.mean(axis=1).rolling(20).mean().plot(ax=ax6, color="#7c3aed", lw=1.0)
    ax6.axhline(8.4, color="#94a3b8", lw=0.6, ls="--", label="min kappa")
    ax6.axhline(40.0, color="#f97316", lw=0.6, ls="--", label="max kappa")
    ax6.set_title("Mean Kappa (mean-reversion speed)")
    ax6.legend(fontsize=8)

    # --- Avg |s-score| with entry threshold reference line ---
    ax7 = fig.add_subplot(gs[2, 2])
    sscore_df.abs().mean(axis=1).rolling(20).mean().plot(ax=ax7, color="#ea580c", lw=1.0)
    ax7.axhline(ENTRY_THRESH, color="#64748b", lw=0.8, ls="--", label=f"entry={ENTRY_THRESH}")
    ax7.set_title("Avg |s-score| (cross-sectional)")
    ax7.legend(fontsize=8)

    plt.suptitle(
        f"PCA Statistical Arbitrage — Avellaneda & Lee (2010) | {TEST_CASE}",
        fontsize=11,
        y=1.01,
    )

    # File name includes test case so runs don't overwrite each other
    outfile = f"results_{TEST_CASE}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outfile}")
    plt.show()


if __name__ == "__main__":
    main()
