import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import random
import os
import pandas as pd
import matplotlib.pyplot as plt

from env import TradingEnv

# ── Default Hyperparameters (used by CLI; dashboard overrides via config) ──
DEFAULTS = {
    'lr': 1e-4,
    'gamma': 0.99,
    'batch_size': 64,
    'replay_size': 50000,
    'target_update': 500,
    'hidden_dim': 64,
    'dropout_rate': 0.1,
    'weight_decay': 1e-5,
    'state_noise': 0.01,
    'episodes': 50,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.95,
    'risk_aversion': 0.5,
    'tx_cost': 0.001,
    'train_ratio': 0.7,
}


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.epsilon = cfg.get('epsilon_start', 1.0)
        self.epsilon_end = cfg.get('epsilon_end', 0.01)
        self.epsilon_decay = cfg.get('epsilon_decay', 0.95)
        self.gamma = cfg.get('gamma', 0.99)
        self.batch_size = cfg.get('batch_size', 64)
        self.target_update = cfg.get('target_update', 500)
        self.state_noise = cfg.get('state_noise', 0.01)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        hidden = cfg.get('hidden_dim', 64)
        drop = cfg.get('dropout_rate', 0.1)
        self.q_net = QNetwork(obs_dim, action_dim, hidden, drop).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, hidden, drop).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=cfg.get('lr', 1e-4),
            weight_decay=cfg.get('weight_decay', 1e-5),
        )
        self.memory = ReplayBuffer(cfg.get('replay_size', 50000))
        self.steps_done = 0

    def choose_action(self, state, evaluation=False):
        if not evaluation and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(state_t)
        self.q_net.train()
        return q.argmax().item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


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
    Run a full Double-DQN experiment.

    Parameters
    ----------
    config : dict   (keys match DEFAULTS above)
    progress_cb : callable(episode, total, metrics_dict) or None

    Returns
    -------
    dict  with keys: training, evaluation, hyperparameters, env_config
    """
    cfg = {**DEFAULTS, **config}  # merge with defaults

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
    dataset = cfg.get('dataset', 'trump')
    csv_name = 'rl_dataset_btc_hourly.csv' if dataset == 'btc' else 'rl_dataset_hourly.csv'
    data_path = os.path.join(data_dir, csv_name)
    train_df, test_df = split_dataset(data_path, cfg['train_ratio'])

    train_env = TradingEnv(data_path_or_df=train_df,
                           risk_aversion=cfg['risk_aversion'], tx_cost=cfg['tx_cost'],
                           normalize=True)
    test_env = TradingEnv(data_path_or_df=test_df,
                          risk_aversion=cfg['risk_aversion'], tx_cost=cfg['tx_cost'],
                          normalize=True, norm_stats=train_env.norm_stats)

    obs_dim, action_dim = 10, 5
    agent = DQNAgent(obs_dim, action_dim, cfg)
    episodes = cfg['episodes']
    noise = cfg['state_noise']

    rewards_hist, portfolios_hist = [], []

    # ── Training ──
    for ep in range(episodes):
        obs, _ = train_env.reset()
        # Randomize starting position to avoid degenerate never-buy policy
        rp = np.random.uniform(0.0, 1.0)
        train_env.position = rp
        obs[-1] = rp

        done, total_reward = False, 0.0
        while not done:
            noisy = obs + np.random.normal(0, noise, obs.shape) if noise > 0 else obs
            action = agent.choose_action(noisy)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            noisy_next = next_obs + np.random.normal(0, noise, next_obs.shape) if noise > 0 else next_obs
            agent.memory.push(noisy, action, reward, noisy_next, done)
            agent.learn()
            obs = next_obs
            total_reward += reward

        if agent.epsilon > agent.epsilon_end:
            agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
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
        action = agent.choose_action(obs, evaluation=True)
        action_counts[action] += 1
        obs, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        total_reward += reward
        eval_portfolios.append(test_env.portfolio_value)

    action_dist = {ACTION_NAMES[a]: c for a, c in action_counts.items()}

    return {
        'hyperparameters': {k: cfg[k] for k in [
            'lr', 'gamma', 'batch_size', 'replay_size', 'target_update',
            'hidden_dim', 'dropout_rate', 'weight_decay', 'state_noise',
            'epsilon_start', 'epsilon_end', 'epsilon_decay', 'episodes',
        ]},
        'env_config': {k: cfg[k] for k in ['risk_aversion', 'tx_cost', 'train_ratio', 'dataset']},
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
        'model_state': agent.q_net.state_dict(),  # saved separately by dashboard
    }


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def _print_progress(ep, total, m):
        print(f"Episode {ep}/{total} - Reward: {m['reward']:.2f} "
              f"- Epsilon: {m['epsilon']:.3f} - Portfolio: ${m['portfolio']:.2f}")

    result = run_experiment({}, progress_cb=_print_progress)

    print(f"\n[UNSEEN DATA TEST]")
    ev = result['evaluation']
    print(f"Final Portfolio: ${ev['final_portfolio']:.2f}")
    print(f"Reward: {ev['total_reward']:.4f}")
    print(f"Actions: {ev['action_distribution']}")

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(result['training']['rewards'], label="Episode Reward (DDQN)")
        plt.title("Double DQN Training Rewards (Train Set)")
        plt.xlabel("Episode"); plt.ylabel("Reward"); plt.legend()
        plt.tight_layout(); plt.savefig("data/dqn_training.png")
        print("\nSaved plot to data/dqn_training.png")
    except Exception as e:
        print(f"Could not save plot: {e}")
