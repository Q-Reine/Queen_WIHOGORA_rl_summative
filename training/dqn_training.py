"""
DQN Training Script
====================
Trains a Deep Q-Network agent on the AssistiveTechRehabEnv using Stable Baselines3.
Runs 10 hyperparameter configurations and logs results for comparison.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environment.custom_env import AssistiveTechRehabEnv

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "dqn"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "dqn"


class RewardTrackingCallback(BaseCallback):
    """Tracks episode rewards and lengths during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.cumulative_rewards = []
        self._current_sum = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(float(ep_reward))
                self.episode_lengths.append(int(ep_length))
                self._current_sum += ep_reward
                self.cumulative_rewards.append(float(self._current_sum))
        return True


# 10 hyperparameter configurations for DQN
DQN_CONFIGS = [
    {
        "name": "baseline",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "buffer_size": 10000,
        "batch_size": 32,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "train_freq": 4,
        "total_timesteps": 80000,
    },
    {
        "name": "high_lr",
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "buffer_size": 10000,
        "batch_size": 64,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "train_freq": 4,
        "total_timesteps": 80000,
    },
    {
        "name": "very_high_lr",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "buffer_size": 10000,
        "batch_size": 64,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "train_freq": 4,
        "total_timesteps": 80000,
    },
    {
        "name": "aggressive_lr",
        "learning_rate": 3e-3,
        "gamma": 0.99,
        "buffer_size": 10000,
        "batch_size": 64,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "train_freq": 4,
        "total_timesteps": 80000,
    },
    {
        "name": "large_buffer",
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "buffer_size": 50000,
        "batch_size": 64,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "train_freq": 4,
        "total_timesteps": 80000,
    },
    {
        "name": "large_batch",
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "buffer_size": 10000,
        "batch_size": 128,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "train_freq": 4,
        "total_timesteps": 80000,
    },
    {
        "name": "low_gamma",
        "learning_rate": 5e-4,
        "gamma": 0.95,
        "buffer_size": 10000,
        "batch_size": 64,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "train_freq": 4,
        "total_timesteps": 80000,
    },
    {
        "name": "high_gamma",
        "learning_rate": 5e-4,
        "gamma": 0.999,
        "buffer_size": 10000,
        "batch_size": 64,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "train_freq": 4,
        "total_timesteps": 80000,
    },
    {
        "name": "high_explore",
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "buffer_size": 10000,
        "batch_size": 64,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 500,
        "train_freq": 4,
        "total_timesteps": 80000,
    },
    {
        "name": "freq_target_update",
        "learning_rate": 5e-4,
        "gamma": 0.995,
        "buffer_size": 5000,
        "batch_size": 32,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.1,
        "target_update_interval": 2500,
        "train_freq": 2,
        "total_timesteps": 80000,
    },
]


def evaluate_model(model, env, n_episodes=20):
    """Evaluate a trained model over n episodes."""
    rewards = []
    lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        lengths.append(steps)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
    }


def train_single_config(config, run_idx):
    """Train DQN with a single hyperparameter configuration."""
    print(f"\n{'='*60}")
    print(f"DQN Run {run_idx + 1}/10: {config['name']}")
    print(f"{'='*60}")

    env = Monitor(AssistiveTechRehabEnv())
    eval_env = Monitor(AssistiveTechRehabEnv())

    model_params = {k: v for k, v in config.items()
                    if k not in ("name", "total_timesteps")}

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(RESULTS_DIR / "tb_logs"),
        **model_params,
    )

    reward_cb = RewardTrackingCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / config["name"]),
        log_path=str(RESULTS_DIR / config["name"]),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[reward_cb, eval_cb],
        progress_bar=True,
    )

    # Save final model
    model_path = MODELS_DIR / config["name"] / "final_model"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    # Evaluate
    eval_results = evaluate_model(model, eval_env)

    # Save training curves
    curves = {
        "episode_rewards": reward_cb.episode_rewards,
        "episode_lengths": reward_cb.episode_lengths,
        "cumulative_rewards": reward_cb.cumulative_rewards,
    }
    curves_path = RESULTS_DIR / config["name"] / "training_curves.json"
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    with open(curves_path, "w") as f:
        json.dump(curves, f)

    result = {
        "run": run_idx + 1,
        "name": config["name"],
        "learning_rate": config["learning_rate"],
        "gamma": config["gamma"],
        "buffer_size": config["buffer_size"],
        "batch_size": config["batch_size"],
        "exploration_strategy": (
            f"frac={config['exploration_fraction']}, "
            f"eps={config['exploration_final_eps']}"
        ),
        **eval_results,
    }

    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} "
          f"(+/- {eval_results['std_reward']:.2f})")

    env.close()
    eval_env.close()

    return result


def run_all_experiments():
    """Run all 10 DQN hyperparameter experiments."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for idx, config in enumerate(DQN_CONFIGS):
        result = train_single_config(config, idx)
        all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "dqn_summary.csv", index=False)
    print("\n" + "=" * 60)
    print("DQN EXPERIMENT SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))

    best_idx = df["mean_reward"].idxmax()
    best = df.iloc[best_idx]
    print(f"\nBest DQN Config: {best['name']} "
          f"(Mean Reward: {best['mean_reward']:.2f})")

    return df


if __name__ == "__main__":
    run_all_experiments()
