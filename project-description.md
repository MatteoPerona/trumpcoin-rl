# Modeling Rational Investor Behavior in TRUMP Coin Using Reinforcement Learning

## Overview

This project applies reinforcement learning — specifically Temporal-Difference
methods and Deep Q-Networks — to model how a rational, risk-aware investor
holding TRUMP coin (TRUMP35336-USD) should optimally manage their position. The
central question is: given a portfolio that includes this volatile meme-coin,
what does optimal behavior look like across different market conditions?

We examine three distinct regimes: periods of relative price stability, sharp
sell-offs, and rapid price appreciation. In each case, the RL agent learns a
policy that balances expected returns against risk, effectively revealing what a
disciplined, utility-maximizing investor "should" do — and by extension, what
aggregate market behavior might look like if participants acted accordingly.

## Approach

The project proceeds in three stages:

**Training.** We build a custom trading environment and train RL agents on
hourly TRUMP coin price data. The state space incorporates price momentum,
volatility, volume, and intraday timing features. Where useful, we augment the
agent's observations with data from correlated assets — major cryptocurrencies
(BTC, ETH), gold and silver futures, and U.S. equity index futures — to capture
cross-market signals and diversification dynamics.

**Backtesting.** Trained agents are evaluated on held-out historical data and
benchmarked against standard strategies (buy-and-hold, moving-average
crossovers, RSI-based rules). We report risk-adjusted performance metrics
including Sharpe ratio, maximum drawdown, and trade efficiency.

**Analysis and reporting.** The final deliverable is a written report with
supporting tables and figures that addresses three questions beyond raw
performance. First, how do cross-asset correlations behave — and do they shift
during periods of market stress in ways that undermine diversification? Second,
does intraday timing matter — are there hours where the optimal policy
consistently differs? Third, and most importantly, what are the market-level
implications? If a meaningful share of participants adopted the RL-derived
strategy, would it stabilize or destabilize prices during crashes and rallies?

## Data

The primary dataset consists of hourly OHLCV (open, high, low, close, volume)
data for TRUMP coin against USD. The working prototype uses a 6-day, 144-hour
sample; the full analysis will use approximately two years of hourly data
(~17,500 observations). Supplementary data on major crypto pairs, precious
metals futures, and equity index futures will be incorporated as reference
prices and correlation inputs.
