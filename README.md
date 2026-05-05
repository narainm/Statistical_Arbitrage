# **Statistical Arbitrage — Avellaneda & Lee (2010)**

Implementation of a statistical arbitrage strategy, inspired by the framework in Avellaneda and Lee’s 2010 paper on U.S. equities.

---

## **Overview**

This project walks through how to build a market-neutral, mean-reversion trading system using factor models.

The idea is pretty simple:

* Strip out broad market and sector effects (factors)
* Model the leftover (residual) returns as mean-reverting
* Trade when stocks deviate too far from their “normal” range

The system covers everything end-to-end:

* factor modeling (PCA or sector ETFs)
* residual construction
* Ornstein–Uhlenbeck modeling
* signal generation (s-scores)
* portfolio construction
* backtesting and performance tracking

> This is a research/learning project — not a production trading system.

---

## **How It Works**

### **1. Factor Model**

First, remove the big-picture movements driven by the market or sectors.

Two approaches:

* **PCA** → extract statistical factors from returns
* **ETF model** → regress each stock on its sector ETF

What’s left are **residual returns** — the idiosyncratic moves of each stock.

---

### **2. OU / AR(1) Model**

Those residuals are modeled as mean-reverting using an Ornstein–Uhlenbeck process:

```
dX = kappa * (mu - X) * dt + sigma * dW
```

In discrete form:

```
X_{t+1} = a + b * X_t + noise
```

From this, we estimate:

* **kappa** → speed of mean reversion
* **mu** → long-run equilibrium level
* **sigma_eq** → equilibrium volatility

---

### **3. S-score (Signal)**

The signal is based on how far a stock is from its equilibrium:

```
s = (X_t - mu_adj) / sigma_eq
```

* `X_t` = cumulative residual for the stock
* `mu_adj` = cross-sectional adjusted mean

Interpretation:

* **s < 0** → relatively cheap → go long
* **s > 0** → relatively expensive → go short

---

### **4. Trading Logic**

The strategy is simple:

* Enter when the deviation is large
* Exit when it mean-reverts

Positions are:

* Long
* Short
* Flat

---

### **5. Portfolio Construction**

* Dollar-neutral long/short portfolio
* Equal-weight positions
* Optional factor-neutral adjustment

---

### **6. Backtesting**

The backtest includes:

* daily rebalancing
* transaction costs
* performance metrics (Sharpe, drawdown, etc.)

---

## **Project Structure**

```
stat_arb/
├── data.py        # gets price and volume data
├── factors.py     # PCA / ETF models
├── ou_model.py    # OU model + s-scores
├── signals.py     # trading logic
├── portfolio.py   # position sizing
├── backtest.py    # PnL simulation
├── metrics.py     # performance stats
├── main.py        # runs everything
└── README.md
```

---

## **How to Run**

Install dependencies:

```bash
pip install pandas numpy yfinance scikit-learn matplotlib
python main.py
```

On the first run, data downloads automatically. Later runs use cached data.

---

## **Key Parameters**

| Parameter          | Description                     |
| ------------------ | ------------------------------- |
| FACTOR_MODEL       | "PCA" or "ETF"                  |
| PCA_WINDOW         | Rolling window for PCA (252)    |
| OU_WINDOW          | OU estimation window (60)       |
| ENTRY_THRESH       | Entry threshold                 |
| EXIT_THRESH        | Exit threshold                  |
| TC_BPS             | Transaction cost (bps)          |
| USE_VOLUME_ADJ     | Volume-based scaling (optional) |
| USE_FACTOR_NEUTRAL | Reduce factor exposure          |

---

## **Limitations**

A few things to keep in mind:

* Survivorship bias — using current stocks, not historical membership
* Smaller universe than the original paper
* Simplified execution (no slippage model)
* Flat transaction cost assumption
* Simplified factor setup

Because of this, results won’t match the original paper exactly.

---

## **What This Project Focuses On**

This is mainly about:

* understanding factor models
* modeling mean reversion properly
* building a full research pipeline
* debugging trading signals

---

## **References**

* Avellaneda, M. & Lee, J.H. (2010). *Statistical Arbitrage in the US Equities Market*. Quantitative Finance.
* Khandani, A.E. & Lo, A.W. (2007). *What happened to the quants in August 2007?*

---

## **Final Note**

This project went through a lot of iteration — fixing signal definitions, residual construction, scaling issues, and turnover.

The current version is stable and works well for research and learning.
