Statistical Arbitrage — Avellaneda & Lee (2010

Here’s a Python implementation of a mean-reversion strategy, inspired by the framework in Avellaneda and Lee’s 2010 paper on statistical arbitrage in US equities.

Overview

This project walks through how to build a market-neutral, mean-reversion trading system using factor models. Basically, you’re stripping out broad market effects (the factors), modeling the leftover returns as mean-reverting, and trading whenever stocks stray too far from their “normal” range.

The system handles everything—from factor modeling (with PCA or sector ETFs), constructing residuals, fitting an Ornstein–Uhlenbeck model, generating signals (s-scores), building portfolios, and running backtests with performance tracking.

Just to be clear: this is for research and learning. It’s not a production trading bot.

How It Works

1. Factor Model

First off, you get rid of the big-picture movements—the stuff driven by the overall market or big sectors. There are two ways:

- PCA: Find statistical factors in returns.
- ETF model: Regress each stock against sector ETFs.

What you have left are “residual returns”—the idiosyncratic moves of each stock.

2. OU / AR(1) Model

Once you have those residuals, you treat them as mean-reverting using the Ornstein-Uhlenbeck process. Here's the math:

dX = kappa * (mu - X) * dt + sigma * dW

Or, discretely:
X_{t+1} = a + b * X_t + noise

You estimate three things:
- kappa: How fast the mean-reversion happens.
- mu: Where the process settles over time.
- sigma_eq: The equilibrium volatility.

3. S-score (Signal)

The s-score tells you how far a stock is from its “normal” value:

s = (X_t - mu_adj) / sigma_eq

- X_t is the running sum of residuals.
- mu_adj adjusts for the cross-sectional average.

If s < 0, the stock’s cheap—go long. If s > 0, it’s pricey—go short.

4. Trading Logic

The strategy is simple: jump in when a stock’s s-score is way off; get out when it drifts back to normal.

Positions are either:
- Long
- Short
- Flat (no position)

5. Portfolio Construction

Positions are dollar-neutral and equally weighted. You can go a step further and adjust for factor neutrality if you want.

6. Backtesting

You rebalance every day, factor in transaction costs, and track key stats like Sharpe ratio and drawdown.

Project Structure

stat_arb/
├── data.py        # gets price and volume data
├── factors.py     # builds PCA / ETF models
├── ou_model.py    # fits OU model & calculates s-scores
├── signals.py     # triggers trades
├── portfolio.py   # sizes positions
├── backtest.py    # simulates PnL
├── metrics.py     # tracks performance stats
├── main.py        # runs the whole pipeline
└── README.md

How to Run

Install dependencies:

pip install pandas numpy yfinance scikit-learn matplotlib
python main.py

On the first run, data downloads automatically. Later runs use stored data.

Key Parameters

| Parameter            | Description                       |
| -------------------- | --------------------------------- |
| FACTOR_MODEL         | "PCA" or "ETF"                    |
| PCA_WINDOW           | Rolling window for PCA (252)       |
| OU_WINDOW            | OU estimation window (60)          |
| ENTRY_THRESH         | Threshold for trade entry          |
| EXIT_THRESH          | Threshold for trade exit           |
| TC_BPS               | Transaction cost (bps)             |
| USE_VOLUME_ADJ       | Optional volume-based scaling      |
| USE_FACTOR_NEUTRAL   | Reduce factor exposure             |

Limitations

A few caveats:
- There’s survivorship bias—you’re using today’s stocks, not the historical universe.
- The universe is a lot smaller than the original paper.
- Execution is simplified; there's no slippage model.
- Flat transaction costs, no fancy structure for factors.

So, your results won’t match the original study.

What This Project Focuses On

The main goals here:
- Get hands-on with factor models
- See how mean reversion works
- Build a full research workflow
- Debug trading signals, see what works

References

- Avellaneda, M. & Lee, J.H. (2010). Statistical Arbitrage in the US Equities Market. Quantitative Finance.
- Khandani, A.E. & Lo, A.W. (2007). What happened to the quants in August 2007?

Final Note

There have been a bunch of iterations—fixing how signals are defined, tweaking residuals, sorting out scaling and turnover.

What you see now works pretty well for research and learning.
