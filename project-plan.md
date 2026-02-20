# RL-Driven Rational Investor Model for TRUMP Coin

## Project Scope & Roadmap

---

## 1. Problem Statement

Model how a **rational, utility-maximizing investor** holding TRUMP coin should
optimally act under three market regimes — steady-state, rapid decline, and
rapid rise — using Reinforcement Learning (Temporal-Difference learning and Deep
Q-Networks). Then backtest the learned policies on historical data and analyze
implications for broader market behavior.

---

## 2. Core Design Decisions (Keep It Simple)

### 2.1 Single-Asset First, Multi-Asset Later

Start with TRUMP coin only. The agent decides how much of its portfolio to hold
in TRUMP vs. cash (USD). Once this works, extend to include correlated assets
(BTC, ETH, gold futures, S&P futures) as additional state features and/or
tradeable instruments.

### 2.2 Environment Framing

| Element          | Choice                                                                     | Rationale                                                        |
| ---------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **Timestep**     | 1 hour                                                                     | Matches your data granularity                                    |
| **Episode**      | Rolling window of N hours (e.g., 168 = 1 week)                             | Lets the agent experience many regime transitions                |
| **Action space** | Discrete: {Heavy Sell, Light Sell, Hold, Light Buy, Heavy Buy} → 5 actions | Simple enough for tabular TD; rich enough for interesting policy |
| **State space**  | Feature vector (see §3)                                                    | Continuous → needs discretization for TD, native for DQN         |
| **Reward**       | Risk-adjusted hourly portfolio return (Sharpe-like)                        | Encourages rational, not just greedy, behavior                   |

### 2.3 Why Not Continuous Actions?

Actor-Critic / PPO with continuous actions (exact % allocation) is more
realistic but significantly harder to debug and tune. Start discrete, graduate
to continuous in a Phase 2 if results warrant it.

---

## 3. State Representation

The agent observes a feature vector at each hour `t`:

### 3.1 Core Features (TRUMP only — Phase 1)

| #  | Feature          | Description                                               |
| -- | ---------------- | --------------------------------------------------------- |
| 1  | `return_1h`      | Log return over the last 1 hour                           |
| 2  | `return_4h`      | Log return over the last 4 hours                          |
| 3  | `return_24h`     | Log return over the last 24 hours                         |
| 4  | `volatility_24h` | Rolling 24h standard deviation of hourly returns          |
| 5  | `volume_ratio`   | Current hour volume / 24h average volume                  |
| 6  | `hour_of_day`    | Cyclically encoded (sin/cos) — captures intraday patterns |
| 7  | `day_of_week`    | Cyclically encoded (sin/cos) — captures weekly patterns   |
| 8  | `rsi_14`         | 14-period RSI (momentum signal)                           |
| 9  | `price_vs_sma24` | Price / 24h SMA — mean-reversion signal                   |
| 10 | `position`       | Agent's current TRUMP holding as fraction of portfolio    |

### 3.2 Extended Features (Phase 2 — Multi-Asset)

| #  | Feature              | Description                                    |
| -- | -------------------- | ---------------------------------------------- |
| 11 | `btc_return_1h`      | BTC hourly return (crypto market beta)         |
| 12 | `eth_return_1h`      | ETH hourly return                              |
| 13 | `gold_return_1h`     | Gold futures hourly return (safe haven signal) |
| 14 | `sp500_return_1h`    | S&P 500 futures hourly return (risk appetite)  |
| 15 | `trump_btc_corr_24h` | Rolling 24h correlation TRUMP vs. BTC          |

### 3.3 Regime Detection Features (Phase 2)

| #  | Feature                 | Description                                                                                   |
| -- | ----------------------- | --------------------------------------------------------------------------------------------- |
| 16 | `drawdown_from_peak`    | Current drawdown from rolling 72h high                                                        |
| 17 | `rally_from_trough`     | Current rally from rolling 72h low                                                            |
| 18 | `cross_asset_corr_mean` | Mean pairwise correlation across assets (stress indicator — correlations spike during stress) |

---

## 4. Action Space & Transaction Costs

### Actions

| Action         | Effect                                     |
| -------------- | ------------------------------------------ |
| **Heavy Sell** | Move to 0% TRUMP (100% cash)               |
| **Light Sell** | Reduce position by 25% of current holding  |
| **Hold**       | No trade                                   |
| **Light Buy**  | Increase position by 25% of available cash |
| **Heavy Buy**  | Move to 100% TRUMP (0% cash)               |

### Transaction Costs

Apply a **0.1% proportional cost** per trade (realistic for major crypto
exchanges). This is critical — without it the agent will overtrade. The cost is
deducted from the portfolio value at each trade and embedded in the reward
signal.

---

## 5. Reward Design

```
r_t = portfolio_return_t - λ * (portfolio_return_t)² - transaction_cost_t
```

Where `λ` is a risk-aversion parameter (start with `λ = 1.0`). This is a
**quadratic utility** reward — it penalizes variance, encouraging the agent to
behave like a risk-averse rational investor rather than a pure return maximizer.
You can sweep `λ ∈ {0.0, 0.5, 1.0, 2.0, 5.0}` to model investors with different
risk tolerances.

**Why not raw returns?** A risk-neutral agent would just go 100% TRUMP always
(positive expected return) or 0% always (negative). The risk penalty forces
nuanced position sizing.

---

## 6. Models

### 6.1 Model 1: Tabular TD (SARSA / Q-Learning) — Baseline

- **Purpose:** Interpretable baseline. You can literally inspect the Q-table to
  see what the agent "thinks" in each state.
- **State discretization:** Bin each continuous feature into 3–5 buckets. Use a
  modest subset of features (e.g., `return_1h`, `volatility_24h`, `position`,
  `hour_of_day`) to keep the table tractable.
- **Algorithm:** Q-Learning (off-policy) with ε-greedy exploration.
- **Hyperparameters:** `α = 0.1`, `γ = 0.99`, `ε` decaying from 1.0 → 0.01 over
  training.
- **Value:** Shows whether RL can learn anything useful here at all before
  scaling up.

### 6.2 Model 2: DQN — Main Model

- **Purpose:** Handle full continuous state space without discretization.
- **Architecture:** Simple feedforward network:
  - Input: state vector (10–18 features depending on phase)
  - Hidden: 2 layers × 128 units, ReLU
  - Output: 5 Q-values (one per action)
- **Key techniques:**
  - Experience replay buffer (size 50,000)
  - Target network (updated every 500 steps)
  - Gradient clipping
  - Double DQN (reduces overestimation bias)
- **Hyperparameters:** `γ = 0.99`, `batch_size = 64`, `lr = 1e-4`, `ε` decaying
  from 1.0 → 0.01.

### 6.3 Model 3 (Optional, Phase 2): Dueling DQN or PPO

Only if DQN results suggest continuous actions or better value estimation would
help.

---

## 7. Technology Stack

| Component                | Tool                                                            | Why                                                                             |
| ------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **RL Environment**       | [Gymnasium](https://gymnasium.farama.org/)(formerly OpenAI Gym) | Standard API, easy to customize, well-documented                                |
| **Neural Networks**      | **PyTorch**                                                     | Most flexible for custom DQN variants                                           |
| **RL Algorithms**        | **Stable-Baselines3**(optional, for comparison)                 | Production-grade implementations of DQN, PPO, etc. Built on PyTorch + Gymnasium |
| **Data Handling**        | **Pandas + NumPy**                                              | Standard                                                                        |
| **Technical Indicators** | **ta-lib**or**pandas-ta**                                       | RSI, SMA, Bollinger Bands, etc.                                                 |
| **Backtesting**          | **Custom**(simple loop over historical data)                    | Full control; RL backtesting doesn't fit standard backtest frameworks well      |
| **Visualization**        | **Matplotlib + Seaborn**                                        | Plotting equity curves, heatmaps, action distributions                          |
| **Report**               | **Jupyter Notebook → PDF/HTML**                                 | Reproducible, inline plots                                                      |

### Why Not Stable-Baselines3 Exclusively?

SB3 is great for standard benchmarks, but for this project you want to (a)
implement tabular TD by hand (it's 30 lines of code), (b) understand every
component of DQN, and (c) customize the replay buffer and reward shaping. Use
SB3 as a **sanity check** — if your custom DQN and SB3's DQN learn similar
policies, you're on solid ground.

---

## 8. Custom Gymnasium Environment Sketch

```python
import gymnasium as gym
import numpy as np

class TrumpCoinTradingEnv(gym.Env):
    """
    Single-asset TRUMP/USD trading environment.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, risk_aversion=1.0, tx_cost=0.001):
        super().__init__()
        self.df = df                    # Preprocessed DataFrame with features
        self.risk_aversion = risk_aversion
        self.tx_cost = tx_cost

        # 5 discrete actions
        self.action_space = gym.spaces.Discrete(5)

        # State: feature vector
        n_features = 10  # Phase 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features,), dtype=np.float32
        )

        self.current_step = None
        self.position = None        # Fraction in TRUMP [0, 1]
        self.portfolio_value = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 24  # Need lookback for features
        self.position = 0.0
        self.portfolio_value = 10000.0
        return self._get_obs(), {}

    def step(self, action):
        # 1. Map action to target position
        target = self._action_to_position(action)

        # 2. Calculate transaction cost
        trade_size = abs(target - self.position)
        cost = trade_size * self.tx_cost * self.portfolio_value

        # 3. Update position
        self.position = target

        # 4. Calculate portfolio return for this hour
        price_return = self.df.iloc[self.current_step]["return_1h"]
        portfolio_return = self.position * price_return

        # 5. Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_value -= cost

        # 6. Risk-adjusted reward
        reward = (portfolio_return
                  - self.risk_aversion * portfolio_return**2
                  - cost / self.portfolio_value)

        # 7. Advance
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _action_to_position(self, action):
        # {0: sell all, 1: sell 25%, 2: hold, 3: buy 25%, 4: buy all}
        if action == 0: return 0.0
        if action == 1: return self.position * 0.75
        if action == 2: return self.position
        if action == 3: return self.position + (1 - self.position) * 0.25
        if action == 4: return 1.0

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            row["return_1h"], row["return_4h"], row["return_24h"],
            row["volatility_24h"], row["volume_ratio"],
            row["hour_sin"], row["hour_cos"],
            row["rsi_14"], row["price_vs_sma24"],
            self.position
        ], dtype=np.float32)
```

---

## 9. Backtesting Plan

### 9.1 Data Split

With 2 years of hourly data (~17,520 rows):

| Set            | Period                  | Purpose                                          |
| -------------- | ----------------------- | ------------------------------------------------ |
| **Train**      | First 70% (~1.4 years)  | Learn policy                                     |
| **Validation** | Next 15% (~3.5 months)  | Tune hyperparameters, early stopping             |
| **Test**       | Final 15% (~3.5 months) | Final evaluation — never touched during training |

### 9.2 Baselines to Compare Against

| Baseline               | Description                                       |
| ---------------------- | ------------------------------------------------- |
| **Buy & Hold**         | 100% TRUMP at t=0, hold forever                   |
| **Cash**               | 0% TRUMP (trivial lower bound)                    |
| **SMA Crossover**      | Classic: buy when price > SMA-24, sell when below |
| **RSI Mean-Reversion** | Buy when RSI < 30, sell when RSI > 70             |
| **Random Agent**       | Uniform random actions (sanity check)             |

### 9.3 Metrics

| Metric                            | What It Tells You                                  |
| --------------------------------- | -------------------------------------------------- |
| **Cumulative Return**             | Raw performance                                    |
| **Sharpe Ratio**(annualized)      | Risk-adjusted performance                          |
| **Max Drawdown**                  | Worst peak-to-trough loss                          |
| **Sortino Ratio**                 | Downside-risk-adjusted return                      |
| **Win Rate**                      | Fraction of profitable trades                      |
| **Trade Frequency**               | How often the agent trades (overtrade check)       |
| **Action Distribution by Regime** | What does the agent do during crashes vs. rallies? |

---

## 10. Analysis & Report Deliverables

### 10.1 Core Tables

| Table  | Content                                                                            |
| ------ | ---------------------------------------------------------------------------------- |
| **T1** | Performance summary: all strategies × all metrics (train / val / test)             |
| **T2** | Action distribution heatmap: action × hour-of-day                                  |
| **T3** | Action distribution by regime (steady / decline / rise)                            |
| **T4** | Correlation matrix: TRUMP vs. BTC, ETH, Gold, S&P (full period vs. stress periods) |
| **T5** | Risk-aversion sensitivity: metrics across λ values                                 |

### 10.2 Core Figures

| Figure | Content                                                                            |
| ------ | ---------------------------------------------------------------------------------- |
| **F1** | Equity curves: RL agent vs. all baselines on test set                              |
| **F2** | Agent position over time overlaid on TRUMP price                                   |
| **F3** | Rolling 24h correlation: TRUMP vs. BTC during stress vs. calm                      |
| **F4** | Training curves: episode reward over training                                      |
| **F5** | Q-value surface (tabular TD): heatmap of Q(state, action) for key state dimensions |

### 10.3 Market Implications Analysis

The report should address:

1. **Steady-state behavior:** Does the RL agent prefer a partial allocation or
   full allocation? How does this change with risk aversion λ?
2. **Crash response:** How quickly does the agent de-risk? Does it front-run the
   decline or react to it? If many rational agents followed this policy, would
   it **amplify** the crash (herding into sells)?
3. **Rally response:** Does the agent chase momentum or take profits? Would
   collective adoption of this policy **dampen** rallies?
4. **Correlation dynamics:** Do TRUMP-BTC correlations increase during stress?
   If so, the RL agent's diversification benefits are _illusory_ precisely when
   they matter most.
5. **Intraday effects:** Are there hours where the agent consistently avoids
   trading? This could reflect thin liquidity or heightened volatility.
6. **Reflexivity:** If the market were populated by such RL agents, would the
   strategies remain profitable, or would they self-defeat through crowding?

---

## 11. Project Roadmap

### Phase 1: Foundation

| Task                 | Details                                                                               |
| -------------------- | ------------------------------------------------------------------------------------- |
| **1.1**Data pipeline | Load CSV (handle the MultiIndex header), clean zero-volume rows, compute all features |
| **1.2**Gymnasium env | Implement `TrumpCoinTradingEnv`as sketched above                                      |
| **1.3**Tabular TD    | Implement Q-Learning with discretized states. Train, inspect Q-table.                 |
| **1.4**Basic DQN     | Implement vanilla DQN in PyTorch. Train on TRUMP-only features.                       |
| **1.5**Backtest v1   | Run trained agents on held-out data, compare to Buy & Hold and SMA baselines.         |

**Milestone:** A working RL agent that demonstrably learns _something_
non-trivial on the training data and doesn't completely fail on the test set.

### Phase 2: Refinement

| Task                          | Details                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------- |
| **2.1**Double DQN             | Upgrade to reduce Q-value overestimation                                        |
| **2.2**Hyperparameter sweep   | Grid search over `lr`,`γ`,`ε`-schedule,`λ`, network size                        |
| **2.3**Risk-aversion analysis | Train separate agents at different λ values                                     |
| **2.4**Multi-asset features   | Add BTC, ETH, Gold, S&P returns as state features                               |
| **2.5**Regime analysis        | Label historical data as steady/decline/rise, analyze agent behavior per regime |

**Milestone:** A well-tuned DQN that outperforms simple baselines on the test
set, with clear behavioral differences across regimes and risk-aversion levels.

### Phase 3: Analysis & Report

| Task                                   | Details                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------ |
| **3.1**Correlation analysis            | Compute and visualize cross-asset correlations, especially during stress |
| **3.2**Hour-of-day analysis            | Does the agent's policy vary by time of day?                             |
| **3.3**Generate all tables and figures | T1–T5, F1–F5 as listed above                                             |
| **3.4**Write market implications       | The "so what" — what does this mean for how markets would react?         |
| **3.5**Final report                    | Compile into Jupyter notebook / PDF                                      |

**Milestone:** Complete, polished report with actionable insights.

---

## 12. File Structure

```
trump-coin-rl/
├── data/
│   ├── raw/                    # Original CSVs
│   └── processed/              # Cleaned, feature-engineered DataFrames
├── src/
│   ├── data_pipeline.py        # Loading, cleaning, feature engineering
│   ├── env.py                  # TrumpCoinTradingEnv (Gymnasium)
│   ├── td_agent.py             # Tabular Q-Learning
│   ├── dqn_agent.py            # DQN (PyTorch)
│   ├── baselines.py            # Buy & Hold, SMA, RSI, Random
│   ├── backtest.py             # Run agents on historical data, compute metrics
│   └── analysis.py             # Correlation, regime detection, plotting
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_training.ipynb       # Train and evaluate agents
│   └── 03_report.ipynb         # Final report with all tables and figures
├── requirements.txt
└── README.md
```

---

## 13. `requirements.txt`

```
gymnasium>=0.29
torch>=2.0
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
pandas-ta>=0.3.14
stable-baselines3>=2.1    # Optional: for sanity-check comparisons
scikit-learn>=1.3          # For discretization, metrics
jupyter>=1.0
```

---

## 14. Known Risks & Mitigations

| Risk                               | Mitigation                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Overfitting**(144 hours is tiny) | Start with this data for pipeline testing only. All real results on 2-year dataset. Use train/val/test split. |
| **Non-stationarity**               | Crypto markets are non-stationary. Use rolling-window training and periodic retraining.                       |
| **Reward hacking**                 | Agent finds degenerate policy (e.g., never trade). Monitor action distributions; ensure diverse behavior.     |
| **Sparse signal**                  | Hourly returns in crypto are noisy. Risk-adjusted reward and enough training episodes are essential.          |
| **Look-ahead bias**                | Features must use only past data. Enforce strict `t-1`convention in feature engineering.                      |

---

## 15. Stretch Goals (If Time Permits)

- **PPO with continuous actions** (exact portfolio weight from 0% to 100%)
- **Multi-asset trading** (agent can allocate across TRUMP, BTC, ETH, cash)
- **Market impact model** (simulate what happens if N agents follow the same
  policy)
- **Regime-switching environment** (explicitly model Markov regime changes)
- **Online learning** (agent continues to learn during test period — simulates
  real deployment)
