import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class TradingEnv(gym.Env):
    """
    Single-asset TRUMP/USD trading environment utilizing discrete actions.
    Modeled after the requirements in Section 4 & 8 of project-plan.md
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_path_or_df=None, risk_aversion=1.0, tx_cost=0.001, initial_capital=10000.0,
                 normalize=False, norm_stats=None):
        super(TradingEnv, self).__init__()
        
        # Load the dataframe
        if isinstance(data_path_or_df, pd.DataFrame):
            self.df = data_path_or_df.copy()
        elif isinstance(data_path_or_df, str):
            self.df = pd.read_csv(data_path_or_df, index_col=0, parse_dates=True)
        else:
            # Default fallback for testing
            default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed', 'rl_dataset_hourly.csv')
            self.df = pd.read_csv(default_path, index_col=0, parse_dates=True)
            
        self.risk_aversion = risk_aversion
        self.tx_cost = tx_cost
        self.initial_capital = initial_capital
        self.normalize = normalize
        
        # Discrete Action Space: 5 Actions
        # 0: Heavy Sell (0% TRUMP)
        # 1: Light Sell (Reduce TRUMP by 25%)
        # 2: Hold (No change)
        # 3: Light Buy (Increase TRUMP by 25% of avail cash)
        # 4: Heavy Buy (100% TRUMP)
        self.action_space = spaces.Discrete(5)
        
        # State Space Feature columns (as defined in our preprocessing)
        self.feature_cols = [
            'feature_log_return_1h',
            'feature_log_return_4h',
            'feature_log_return_24h',
            'feature_volatility_24h',
            'feature_log_volume',
            'feature_hour_sin',
            'feature_hour_cos',
            'feature_day_sin',
            'feature_day_cos',
        ]
        
        # Ensure all required columns are in the dataframe
        missing_cols = [c for c in self.feature_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required features in dataframe: {missing_cols}")
            
        # The agent's full observation includes the features + current portfolio position
        self.n_features = len(self.feature_cols) + 1  # +1 for self.position
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.n_features,), 
            dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.position = 0.0          # Fraction of portfolio in TRUMP [0, 1]
        self.portfolio_value = self.initial_capital
        
        # Cache values to speed up step()
        self.returns_1h = self.df['feature_log_return_1h'].values
        self.features_matrix = self.df[self.feature_cols].values

        # Normalization: compute or reuse z-score stats
        if self.normalize:
            if norm_stats is not None:
                self.norm_stats = norm_stats
            else:
                # Compute from this dataset (should be training set)
                self.norm_stats = {
                    'mean': self.features_matrix.mean(axis=0),
                    'std': self.features_matrix.std(axis=0),
                }
                # Prevent division by zero for constant features
                self.norm_stats['std'][self.norm_stats['std'] < 1e-8] = 1.0
        else:
            self.norm_stats = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start at index 0 because our df should already have the 24h warmup removed
        self.current_step = 0
        self.position = 0.0
        self.portfolio_value = self.initial_capital
        
        return self._get_obs(), {}

    def _action_to_position(self, action):
        """Maps discrete actions to target portfolio allocation."""
        if action == 0:
            # Heavy Sell -> 0% TRUMP
            return 0.0
        elif action == 1:
            # Light Sell -> Reduce TRUMP position by 25%
            return self.position * 0.75
        elif action == 2:
            # Hold -> No change
            return self.position
        elif action == 3:
            # Light Buy -> Buy with 25% of available cash
            # Available cash fraction = 1.0 - self.position
            return self.position + (1.0 - self.position) * 0.25
        elif action == 4:
            # Heavy Buy -> 100% TRUMP
            return 1.0
        else:
            raise ValueError(f"Invalid action: {action}")

    def step(self, action):
        # Prevent stepping past the end
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0.0, True, False, {}

        # 1. Map discrete action to a target position percentage
        target_position = self._action_to_position(action)
        
        # 2. Calculate trade size and transaction cost
        trade_size = abs(target_position - self.position)
        cost = trade_size * self.tx_cost * self.portfolio_value
        
        # Update position post-trade
        self.position = target_position
        
        # 3. Time steps forward by 1 hour (the return applies to our new position)
        self.current_step += 1
        
        # 4. Calculate Portfolio Return for the new hour
        # Note: We use the log returns (from dataloader) directly as an approx for simple returns 
        # for small increments, or we could exp() it if we wanted simple returns.
        # Log return is standard and robust.
        price_return = self.returns_1h[self.current_step]
        portfolio_return = self.position * price_return
        
        # 5. Update Portfolio Value
        # Update by the return, then subtract the absolute cash cost of the trade
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_value -= cost
        
        # 6. Calculate Risk-Adjusted Reward (Quadratic Utility)
        # r_t = R_t - λ * (R_t)^2 - transaction_costs
        # We model the cost as a percentage of portfolio to keep the reward scale consistent
        cost_pct = cost / max(self.portfolio_value, 1e-8)
        reward = portfolio_return - (self.risk_aversion * (portfolio_return ** 2)) - cost_pct

        # 7. Check terminal conditions
        terminated = self.current_step >= len(self.df) - 1
        
        # Bankruptcy condition
        if self.portfolio_value <= 0:
            self.portfolio_value = 0
            reward -= 1.0  # Heavy penalty for bankruptcy
            terminated = True
            
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'action': action,
            'reward': reward,
            'cost': cost
        }

        return self._get_obs(), float(reward), terminated, False, info

    def _get_obs(self):
        """Constructs the observation vector."""
        row_features = self.features_matrix[self.current_step]
        # Apply z-score normalization if enabled
        if self.normalize and self.norm_stats is not None:
            row_features = (row_features - self.norm_stats['mean']) / self.norm_stats['std']
        # Append the current position to the feature array (position stays in [0, 1])
        obs = np.append(row_features, self.position)
        return obs.astype(np.float32)

def main():
    """Simple test loop using a random agent."""
    print("Testing TradingEnv with random actions...")
    
    try:
        env = TradingEnv()
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return
        
    obs, _ = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial Obs shape: {obs.shape}")
    
    total_reward = 0
    terminated = False
    truncated = False
    
    # Run 1000 steps
    for step in range(1000):
        # Random action [0..4]
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        if terminated or truncated:
            print(f"Terminated early at step {step}")
            break
            
    print(f"\nFinal Test Results after {env.current_step} steps:")
    print(f"  Final Portfolio Value: ${info['portfolio_value']:.2f}")
    print(f"  Total Accumulated Reward: {total_reward:.4f}")
    print(f"  Final Position: {info['position'] * 100:.1f}% TRUMP")
    print("Environment test completed successfully!")

if __name__ == "__main__":
    main()
