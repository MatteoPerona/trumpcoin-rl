import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from env import TradingEnv

DEFAULTS = {
    "policy_lr": 1e-4,
    "value_lr": 1e-3,
    "gamma": 0.99,
    "hidden_dim": 64,
    "episodes": 50,
    "episode_length": 168,
    "risk_aversion": 0.5,
    "tx_cost": 0.001,
    "train_ratio": 0.7,
    "entropy_coef": 0.0,
}

ACTION_NAMES = ["Heavy Sell", "Light Sell", "Hold", "Light Buy", "Heavy Buy"]


def split_dataset(csv_path, train_ratio=0.7):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def sample_train_window(df, episode_length):
    """Sample a contiguous training window for a single on-policy episode."""
    if episode_length >= len(df):
        return df.copy()

    start_idx = np.random.randint(0, len(df) - episode_length + 1)
    return df.iloc[start_idx:start_idx + episode_length].copy()


def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class REINFORCEAgent:
    def __init__(self, obs_dim, action_dim, cfg):
        self.gamma = cfg.get("gamma", 0.99)
        self.entropy_coef = cfg.get("entropy_coef", 0.0)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        hidden_dim = cfg.get("hidden_dim", 64)
        self.policy_net = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.value_net = ValueNetwork(obs_dim, hidden_dim).to(self.device)

        self.policy_opt = optim.Adam(
            self.policy_net.parameters(),
            lr=cfg.get("policy_lr", 1e-4),
        )
        self.value_opt = optim.Adam(
            self.value_net.parameters(),
            lr=cfg.get("value_lr", 1e-3),
        )

    def choose_action(self, state, evaluation=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy_net(state_t)

        if evaluation:
            return int(torch.argmax(logits, dim=-1).item())

        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def update_from_episode(self, states, actions, rewards):
        states_t = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        returns_t = compute_returns(rewards, self.gamma).to(self.device)

        logits = self.policy_net(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        values = self.value_net(states_t).squeeze(-1)
        advantages = returns_t - values

        adv = advantages.detach()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        policy_loss = -(log_probs * adv).mean() - self.entropy_coef * entropy
        value_loss = F.mse_loss(values, returns_t)

        self.policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_opt.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "mean_return": float(returns_t.mean().item()),
        }


def run_experiment(config, progress_cb=None):
    """
    Run a full REINFORCE-with-baseline experiment.

    Parameters
    ----------
    config : dict
        policy_lr, value_lr, gamma, hidden_dim, episodes, episode_length,
        risk_aversion, tx_cost, train_ratio, entropy_coef, dataset
    progress_cb : callable(episode, total, metrics_dict) or None

    Returns
    -------
    dict with keys: training, evaluation, hyperparameters, env_config
    """
    cfg = {**DEFAULTS, **config}

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
    dataset = cfg.get("dataset", "trump")
    csv_name = "rl_dataset_btc_hourly.csv" if dataset == "btc" else "rl_dataset_hourly.csv"
    data_path = os.path.join(data_dir, csv_name)

    train_df, test_df = split_dataset(data_path, cfg["train_ratio"])

    train_reference_env = TradingEnv(
        data_path_or_df=train_df,
        risk_aversion=cfg["risk_aversion"],
        tx_cost=cfg["tx_cost"],
        normalize=True,
    )
    test_env = TradingEnv(
        data_path_or_df=test_df,
        risk_aversion=cfg["risk_aversion"],
        tx_cost=cfg["tx_cost"],
        normalize=True,
        norm_stats=train_reference_env.norm_stats,
    )

    obs_dim, action_dim = 10, 5
    agent = REINFORCEAgent(obs_dim, action_dim, cfg)
    episodes = cfg["episodes"]
    episode_length = cfg["episode_length"]

    rewards_hist, portfolios_hist = [], []
    policy_losses_hist, value_losses_hist = [], []

    # Training
    for ep in range(episodes):
        episode_df = sample_train_window(train_df, episode_length)
        train_env = TradingEnv(
            data_path_or_df=episode_df,
            risk_aversion=cfg["risk_aversion"],
            tx_cost=cfg["tx_cost"],
            normalize=True,
            norm_stats=train_reference_env.norm_stats,
        )
        obs, _ = train_env.reset()

        # Break the zero-position symmetry at reset so the policy sees all trade types.
        rp = np.random.uniform(0.0, 1.0)
        train_env.position = rp
        obs[-1] = rp

        done, total_reward = False, 0.0

        states, actions, rewards = [], [], []

        while not done:
            action = agent.choose_action(obs, evaluation=False)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated

            states.append(obs.copy())
            actions.append(action)
            rewards.append(reward)

            obs = next_obs
            total_reward += reward

        train_stats = agent.update_from_episode(states, actions, rewards)

        rewards_hist.append(total_reward)
        portfolios_hist.append(train_env.portfolio_value)
        policy_losses_hist.append(train_stats["policy_loss"])
        value_losses_hist.append(train_stats["value_loss"])

        if progress_cb:
            progress_cb(ep + 1, episodes, {
                "reward": total_reward,
                "portfolio": train_env.portfolio_value,
                "policy_loss": train_stats["policy_loss"],
                "value_loss": train_stats["value_loss"],
                "epsilon": 0.0,  # kept for backward compatibility with dashboard formatting
            })

    # Evaluation
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
        "hyperparameters": {
            "policy_lr": cfg["policy_lr"],
            "value_lr": cfg["value_lr"],
            "gamma": cfg["gamma"],
            "hidden_dim": cfg["hidden_dim"],
            "entropy_coef": cfg["entropy_coef"],
            "episodes": episodes,
            "episode_length": episode_length,
        },
        "env_config": {
            "risk_aversion": cfg["risk_aversion"],
            "tx_cost": cfg["tx_cost"],
            "train_ratio": cfg["train_ratio"],
            "dataset": dataset,
        },
        "training": {
            "rewards": rewards_hist,
            "portfolios": portfolios_hist,
            "policy_losses": policy_losses_hist,
            "value_losses": value_losses_hist,
        },
        "evaluation": {
            "final_portfolio": test_env.portfolio_value,
            "total_reward": total_reward,
            "action_distribution": action_dist,
            "portfolio_curve": eval_portfolios,
        },
        "model_state": {
            "policy_state_dict": agent.policy_net.state_dict(),
            "value_state_dict": agent.value_net.state_dict(),
        },
    }


if __name__ == "__main__":
    def _print_progress(ep, total, m):
        print(
            f"Episode {ep}/{total} - Reward: {m['reward']:.2f} "
            f"- Policy Loss: {m['policy_loss']:.4f} "
            f"- Value Loss: {m['value_loss']:.4f} "
            f"- Portfolio: ${m['portfolio']:.2f}"
        )

    result = run_experiment({}, progress_cb=_print_progress)

    print("\n[UNSEEN DATA TEST]")
    ev = result["evaluation"]
    print(f"Final Portfolio: ${ev['final_portfolio']:.2f}")
    print(f"Reward: {ev['total_reward']:.4f}")
    print(f"Actions: {ev['action_distribution']}")

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(result["training"]["rewards"], label="Episode Reward")
        plt.title("REINFORCE + Baseline Training Rewards (Train Set)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig("data/reinforce_training.png")
        print("\nSaved plot to data/reinforce_training.png")
    except Exception as e:
        print(f"Could not save plot: {e}")
