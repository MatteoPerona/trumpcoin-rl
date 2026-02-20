# Modeling Rational Investor Behavior in TRUMP Coin Using RL

This repository applies Reinforcement Learning — specifically Tabular Q-Learning
and Double Deep Q-Networks (DDQN) — to model how a rational, risk-aware investor
should optimally manage a position in TRUMP coin (TRUMP35336-USD). The project
includes a full Streamlit dashboard for running experiments, comparing results,
and inspecting learned policies interactively.

---

## Project Structure

```
trumpcoin-rl/
├── data/
│   ├── raw/                        # Raw hourly yfinance CSV pulls
│   ├── processed/                  # Feature-engineered RL datasets
│   │   ├── rl_dataset_hourly.csv   #   └─ TRUMP (primary)
│   │   └── rl_dataset_btc_hourly.csv #  └─ BTC (secondary)
│   └── runs/                       # Saved experiment results (JSON + model artifacts)
├── dataloader.py                   # Fetches & preprocesses Yahoo Finance data
├── env.py                          # Custom Gymnasium trading environment
├── td_agent.py                     # Tabular Q-Learning agent (Phase 1)
├── dqn_agent.py                    # Double DQN agent (Phase 2)
├── baselines.py                    # Non-learning baseline strategies
├── dashboard.py                    # Streamlit dashboard (experiment runner + analysis)
├── project-description.md          # Formal project description
├── project-plan.md                 # Detailed project plan
├── requirements.txt                # Python dependencies (pip install -r)
└── README.md
```

---

## Setup & Dependencies

It's recommended to run this project inside a Python virtual environment.

```zsh
# 1. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 1. Getting the Data (`dataloader.py`)

The `dataloader.py` script connects to Yahoo Finance (`yfinance`) and downloads
up to 730 days of hourly OHLCV data for TRUMP coin and correlated assets
(`BTC-USD`, `ETH-USD`, `GC=F` [Gold], `SI=F` [Silver], `ES=F` [S&P 500]).

It parses raw data into state features for an RL agent:

- **Momentum**: 1h, 4h, and 24h log returns.
- **Volatility**: 24h rolling standard deviation of returns.
- **Volume**: Log-transformed volume.
- **Cyclic Time**: Sine and Cosine encodings of Hour of Day and Day of Week.

### Usage

```zsh
python dataloader.py
```

**Output:** Creates `data/processed/rl_dataset_hourly.csv` and
`data/processed/rl_dataset_btc_hourly.csv`.

---

## 2. Trading Environment (`env.py`)

The `TradingEnv` class is a custom `gymnasium.Env` that steps an agent hourly
through historical price data.

| Property              | Details                                                                                      |
| --------------------- | -------------------------------------------------------------------------------------------- |
| **Action Space**      | `Discrete(5)` — Heavy Sell (0%), Light Sell (−25%), Hold, Light Buy (+25%), Heavy Buy (100%) |
| **Observation Space** | 10-dim continuous vector: 9 market features + portfolio position                             |
| **Reward**            | Quadratic utility:`R - λ·R² - tx_cost`, balancing return against risk                        |
| **Transaction Cost**  | 0.1% of trade size to prevent churning                                                       |

### Usage

```zsh
python env.py          # Sanity-check with 1,000 random-action steps
```

---

## 3. Tabular Q-Learning Agent (`td_agent.py`)

Phase 1 baseline using off-policy Temporal-Difference learning. Because tabular
Q-Learning requires discrete states, the continuous space is bucketed into **54
states**:

| Dimension            | Bins                                    |
| -------------------- | --------------------------------------- |
| Momentum (1h Return) | 3 — Negative / Flat / Positive          |
| Volatility (24h)     | 3 — Low / Medium / High                 |
| Position             | 3 — Cash-heavy / Balanced / TRUMP-heavy |
| Time                 | 2 — Night / Day                         |

Training uses a strict **70/30 temporal split** with ε-greedy exploration on the
train set and deterministic evaluation on the unseen test set.

### Usage

```zsh
python td_agent.py     # Train 20 epochs, evaluate, print Q-table highlights
```

**Output:** Saves training reward plot to `data/q_learning_training.png`.

---

## 4. Double DQN Agent (`dqn_agent.py`)

Phase 2 agent built with PyTorch. Accepts the full 10-dimensional continuous
state vector.

**Architecture:** Input(10) → Hidden(HIDDEN_DIM) × 2 → Output(5). Uses a
disjoint target network (updated every 500 steps) and experience replay (buffer
of 50k, batch size 64).

### Regularization Toggles

| Toggle         | Default | Overfit | Aggressive | Purpose                     |
| -------------- | ------- | ------- | ---------- | --------------------------- |
| `HIDDEN_DIM`   | 64      | 128     | 32         | Network capacity            |
| `DROPOUT_RATE` | 0.1     | 0.0     | 0.2        | Random neuron dropout       |
| `WEIGHT_DECAY` | 1e-5    | 0.0     | 1e-4       | L2 penalty                  |
| `STATE_NOISE`  | 0.01    | 0.0     | 0.05       | Gaussian state augmentation |

### Usage

```zsh
python dqn_agent.py    # Train 50 episodes, evaluate on holdout
```

**Output:** Saves training reward plot to `data/dqn_training.png`.

---

## 5. Baseline Strategies (`baselines.py`)

Three non-learning strategies for benchmarking against the RL agents:

| Strategy          | Description                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Buy & Hold**    | Buys 100% on step 0, then holds for the remainder                                                                        |
| **SMA Crossover** | Goes long/short based on short-window vs. long-window moving-average crossover (configurable windows, default 12h / 48h) |
| **Random Agent**  | Samples a uniformly random action every step                                                                             |

### Usage

```zsh
python baselines.py    # Runs all three strategies and prints results
```

---

## 6. Streamlit Dashboard (`dashboard.py`)

The interactive dashboard is the primary interface for running and analyzing
experiments. Launch it with:

```zsh
streamlit run dashboard.py
```

The dashboard has **three pages**, accessed via the sidebar:

### 🚀 Run Experiment

Configure and kick off training (or evaluation for baselines) from the browser.

- **Model selector** — choose from Tabular Q-Learning, Double DQN, Buy & Hold,
  SMA Crossover, or Random Agent.
- **Environment controls** — dataset (TRUMP / BTC), risk aversion (λ),
  transaction cost, and train/test split ratio.
- **Per-model hyperparameter panels** — all relevant knobs are exposed as
  sliders and inputs (learning rate, γ, ε-decay, hidden dim, dropout, etc.).
- **Live progress bar** during training with per-episode reward, portfolio, and
  ε readout.
- **Post-run results**: test portfolio value, total reward, dominant action,
  training reward curve, action distribution bar chart, and portfolio equity
  curve.
- Results are **automatically persisted** to `data/runs/` as JSON (plus `.npy`
  for Q-tables, `.pt` for DQN weights).

### 📊 Explore Past Runs

Browse, filter, and compare all saved experiments.

- **Runs table** with model, dataset, timestamp, hyperparameters, and headline
  metrics.
- **Multi-select** any subset of runs for side-by-side comparison.
- **Overlay training reward curves** from multiple runs on one chart.
- **Per-run detail expanders** showing full hyperparameter JSON, environment
  config, and portfolio equity curves.

### 🔍 Inspect Q-Values

Interactively explore the learned value functions of saved models.

**Tabular Q-Learning:**

- Select any saved Q-Learning run.
- Pick a discrete state (momentum × volatility × position × time) from dropdown
  menus.
- View a bar chart of Q-values for all 5 actions with the greedy action
  highlighted.
- Browse the **full 54-state policy table** showing the greedy action and
  Q-values for every state.

**Double DQN:**

- Select any saved DQN run (loads the `.pt` weights).
- Adjust each of the 10 continuous state features via sliders.
- View real-time Q-value bar chart as sliders move.
- **Policy surface heatmap**: sweeps position × 1h-return to visualize how the
  greedy action changes across the two most important dimensions while the other
  features are held at slider values.

---

## Data Split Convention

All agents use a **70/30 temporal split** by default (configurable via the
dashboard). The train set covers roughly the first ~6,500 hours and the test set
covers the final ~2,800 hours. Baselines are evaluated on the test set only.
