import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from env import TradingEnv


class QLearningAgent:
    def __init__(self, action_space_size=5, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # State Discretization: 3 * 3 * 3 * 2 = 54 states
        self.state_dims = (3, 3, 3, 2)
        self.q_table = np.zeros(self.state_dims + (self.action_space_size,))

    def _discretize_state(self, obs, env):
        return_1h = obs[0]
        volatility_24h = obs[3]
        hour_sin = obs[5]
        position = obs[9]

        if return_1h < -0.005:    return_bin = 0
        elif return_1h > 0.005:   return_bin = 2
        else:                     return_bin = 1

        if volatility_24h < 0.015:  vol_bin = 0
        elif volatility_24h > 0.030: vol_bin = 2
        else:                        vol_bin = 1

        if position < 0.33:   pos_bin = 0
        elif position > 0.66: pos_bin = 2
        else:                 pos_bin = 1

        time_bin = 0 if hour_sin < 0 else 1
        return (return_bin, vol_bin, pos_bin, time_bin)

    def choose_action(self, obs, env, evaluation=False):
        state = self._discretize_state(obs, env)
        if not evaluation and np.random.rand() < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, obs, action, reward, next_obs, env, done):
        state = self._discretize_state(obs, env)
        next_state = self._discretize_state(next_obs, env)
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def print_q_table_highlights(self):
        action_map = {0: "Heavy Sell", 1: "Light Sell", 2: "Hold", 3: "Light Buy", 4: "Heavy Buy"}
        ret_map = {0: "Negative Momentum", 1: "Flat", 2: "Positive Momentum"}
        vol_map = {0: "Low Vol", 1: "Med Vol", 2: "High Vol"}
        pos_map = {0: "Mostly Cash", 1: "Balanced", 2: "Mostly TRUMP"}

        print("\n=== Interpreting Learned Baseline Policy ===")
        for label, state in [
            (f"{ret_map[0]}, {vol_map[2]}, {pos_map[2]}", (0, 2, 2, 1)),
            (f"{ret_map[1]}, {vol_map[0]}, {pos_map[0]}", (1, 0, 0, 1)),
            (f"{ret_map[2]}, {vol_map[0]}, {pos_map[1]}", (2, 0, 1, 1)),
        ]:
            q_vals = self.q_table[state]
            best = np.argmax(q_vals)
            print(f"  {label}")
            print(f"    -> Preferred: {action_map[best]}")
            print(f"    -> Q-Values: {[f'{action_map[i]}: {v:.4f}' for i, v in enumerate(q_vals)]}")


# ──────────────────────────────────────────────────────────────────────
# Dashboard-consumable API
# ──────────────────────────────────────────────────────────────────────
ACTION_NAMES = ["Heavy Sell", "Light Sell", "Hold", "Light Buy", "Heavy Buy"]


def split_dataset(csv_path, train_ratio=0.7):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def run_experiment(config, progress_cb=None):
    """
    Run a full Q-Learning experiment.

    Parameters
    ----------
    config : dict
        alpha, gamma, epsilon_start, epsilon_min, epsilon_decay,
        episodes, risk_aversion, tx_cost, train_ratio
    progress_cb : callable(episode, total, metrics_dict) or None
        Optional callback invoked after each episode.

    Returns
    -------
    dict  with keys: training, evaluation, hyperparameters, env_config
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
    dataset = config.get('dataset', 'trump')
    csv_name = 'rl_dataset_btc_hourly.csv' if dataset == 'btc' else 'rl_dataset_hourly.csv'
    data_path = os.path.join(data_dir, csv_name)

    train_ratio = config.get('train_ratio', 0.7)
    train_df, test_df = split_dataset(data_path, train_ratio)

    risk_aversion = config.get('risk_aversion', 1.0)
    tx_cost = config.get('tx_cost', 0.001)
    train_env = TradingEnv(data_path_or_df=train_df, risk_aversion=risk_aversion, tx_cost=tx_cost)
    test_env  = TradingEnv(data_path_or_df=test_df,  risk_aversion=risk_aversion, tx_cost=tx_cost)

    agent = QLearningAgent(
        alpha=config.get('alpha', 0.1),
        gamma=config.get('gamma', 0.99),
        epsilon_start=config.get('epsilon_start', 1.0),
        epsilon_min=config.get('epsilon_min', 0.01),
        epsilon_decay=config.get('epsilon_decay', 0.995),
    )

    episodes = config.get('episodes', 20)
    rewards_hist, portfolios_hist = [], []

    # ── Training ──
    for ep in range(episodes):
        obs, _ = train_env.reset()
        done, total_reward = False, 0.0
        while not done:
            action = agent.choose_action(obs, train_env)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            agent.learn(obs, action, reward, next_obs, train_env, done)
            obs = next_obs
            total_reward += reward
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        rewards_hist.append(total_reward)
        portfolios_hist.append(train_env.portfolio_value)
        if progress_cb:
            progress_cb(ep + 1, episodes, {
                'reward': total_reward,
                'portfolio': train_env.portfolio_value,
                'epsilon': agent.epsilon,
            })

    # ── Evaluation ──
    obs, _ = test_env.reset()
    done, total_reward = False, 0.0
    action_counts = {i: 0 for i in range(5)}
    eval_portfolios = []
    while not done:
        action = agent.choose_action(obs, test_env, evaluation=True)
        action_counts[action] += 1
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        total_reward += reward
        eval_portfolios.append(test_env.portfolio_value)

    action_dist = {ACTION_NAMES[a]: c for a, c in action_counts.items()}

    return {
        'hyperparameters': {
            'alpha': agent.alpha, 'gamma': agent.gamma,
            'epsilon_start': config.get('epsilon_start', 1.0),
            'epsilon_decay': agent.epsilon_decay, 'episodes': episodes,
        },
        'env_config': {'risk_aversion': risk_aversion, 'tx_cost': tx_cost,
                       'train_ratio': train_ratio, 'dataset': dataset},
        'training': {
            'rewards': rewards_hist,
            'portfolios': portfolios_hist,
        },
        'evaluation': {
            'final_portfolio': test_env.portfolio_value,
            'total_reward': total_reward,
            'action_distribution': action_dist,
            'portfolio_curve': eval_portfolios,
        },
        'q_table': agent.q_table,  # numpy array, saved separately by dashboard
    }


# ──────────────────────────────────────────────────────────────────────
# CLI entry point (preserved for standalone use)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def _print_progress(ep, total, m):
        print(f"Episode {ep}/{total} - Reward: {m['reward']:.2f} "
              f"- Epsilon: {m['epsilon']:.3f} - Portfolio: ${m['portfolio']:.2f}")

    result = run_experiment({
        'alpha': 0.1, 'gamma': 0.99, 'epsilon_decay': 0.8, 'episodes': 20,
        'risk_aversion': 1.0, 'tx_cost': 0.001,
    }, progress_cb=_print_progress)

    print(f"\n[UNSEEN DATA TEST]")
    ev = result['evaluation']
    print(f"Final Portfolio: ${ev['final_portfolio']:.2f}")
    print(f"Reward: {ev['total_reward']:.4f}")
    print(f"Actions: {ev['action_distribution']}")

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(result['training']['rewards'], label="Episode Reward")
        plt.title("Tabular Q-Learning Training Rewards (Train Set)")
        plt.xlabel("Episode"); plt.ylabel("Reward"); plt.legend()
        plt.tight_layout(); plt.savefig("data/q_learning_training.png")
        print("\nSaved plot to data/q_learning_training.png")
    except Exception as e:
        print(f"Could not save plot: {e}")
