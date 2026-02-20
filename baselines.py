import numpy as np
import pandas as pd
import os
from env import TradingEnv

ACTION_NAMES = ["Heavy Sell", "Light Sell", "Hold", "Light Buy", "Heavy Buy"]


def split_dataset(csv_path, train_ratio=0.7):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def _resolve_data_path(config):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
    dataset = config.get('dataset', 'trump')
    csv_name = 'rl_dataset_btc_hourly.csv' if dataset == 'btc' else 'rl_dataset_hourly.csv'
    return os.path.join(data_dir, csv_name)


def _evaluate(env, action_fn):
    """Run a full evaluation episode using an action selection function."""
    obs, _ = env.reset()
    done, total_reward, step = False, 0.0, 0
    action_counts = {i: 0 for i in range(5)}
    portfolios = []

    while not done:
        action = action_fn(obs, step, env)
        action_counts[action] += 1
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        portfolios.append(env.portfolio_value)
        step += 1

    return {
        'final_portfolio': env.portfolio_value,
        'total_reward': total_reward,
        'action_distribution': {ACTION_NAMES[a]: c for a, c in action_counts.items()},
        'portfolio_curve': portfolios,
    }


# ──────────────────────────────────────────────────────────────────────
# Strategy 1: Buy & Hold
# ──────────────────────────────────────────────────────────────────────
def _buy_and_hold(config):
    data_path = _resolve_data_path(config)
    _, test_df = split_dataset(data_path, config.get('train_ratio', 0.7))
    env = TradingEnv(data_path_or_df=test_df,
                     risk_aversion=config.get('risk_aversion', 1.0),
                     tx_cost=config.get('tx_cost', 0.001))

    def action_fn(obs, step, env):
        return 4 if step == 0 else 2  # Heavy Buy on step 0, then Hold

    return _evaluate(env, action_fn)


# ──────────────────────────────────────────────────────────────────────
# Strategy 2: SMA Crossover
# ──────────────────────────────────────────────────────────────────────
def _sma_crossover(config):
    data_path = _resolve_data_path(config)
    _, test_df = split_dataset(data_path, config.get('train_ratio', 0.7))
    env = TradingEnv(data_path_or_df=test_df,
                     risk_aversion=config.get('risk_aversion', 1.0),
                     tx_cost=config.get('tx_cost', 0.001))

    short_window = config.get('sma_short', 12)
    long_window = config.get('sma_long', 48)

    # Pre-compute MAs on the 1h log return column from the test set
    returns = test_df['feature_log_return_1h'].values
    sma_short = pd.Series(returns).rolling(short_window, min_periods=1).mean().values
    sma_long = pd.Series(returns).rolling(long_window, min_periods=1).mean().values

    def action_fn(obs, step, env):
        if step < long_window:
            return 2  # Hold during warm-up
        if sma_short[step] > sma_long[step]:
            return 4  # Heavy Buy (bullish crossover)
        elif sma_short[step] < sma_long[step]:
            return 0  # Heavy Sell (bearish crossover)
        return 2  # Hold

    return _evaluate(env, action_fn)


# ──────────────────────────────────────────────────────────────────────
# Strategy 3: Random Agent
# ──────────────────────────────────────────────────────────────────────
def _random_agent(config):
    data_path = _resolve_data_path(config)
    _, test_df = split_dataset(data_path, config.get('train_ratio', 0.7))
    env = TradingEnv(data_path_or_df=test_df,
                     risk_aversion=config.get('risk_aversion', 1.0),
                     tx_cost=config.get('tx_cost', 0.001))

    def action_fn(obs, step, env):
        return env.action_space.sample()

    return _evaluate(env, action_fn)


# ──────────────────────────────────────────────────────────────────────
# Unified entry point
# ──────────────────────────────────────────────────────────────────────
STRATEGIES = {
    'buy_hold': _buy_and_hold,
    'sma_crossover': _sma_crossover,
    'random': _random_agent,
}


def run_experiment(config, progress_cb=None):
    """
    Run a baseline strategy evaluation (no training phase).

    config must include 'strategy': 'buy_hold' | 'sma_crossover' | 'random'
    """
    strategy = config.get('strategy', 'buy_hold')
    fn = STRATEGIES[strategy]
    evaluation = fn(config)

    # Build hyperparameters dict
    hp = {'strategy': strategy}
    if strategy == 'sma_crossover':
        hp['sma_short'] = config.get('sma_short', 12)
        hp['sma_long'] = config.get('sma_long', 48)

    dataset = config.get('dataset', 'trump')

    return {
        'hyperparameters': hp,
        'env_config': {
            'risk_aversion': config.get('risk_aversion', 1.0),
            'tx_cost': config.get('tx_cost', 0.001),
            'train_ratio': config.get('train_ratio', 0.7),
            'dataset': dataset,
        },
        'training': {'rewards': [], 'portfolios': []},
        'evaluation': evaluation,
    }


if __name__ == "__main__":
    for name in STRATEGIES:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        result = run_experiment({'strategy': name, 'dataset': 'trump'})
        ev = result['evaluation']
        print(f"Portfolio: ${ev['final_portfolio']:.2f}")
        print(f"Reward: {ev['total_reward']:.4f}")
        print(f"Actions: {ev['action_distribution']}")
