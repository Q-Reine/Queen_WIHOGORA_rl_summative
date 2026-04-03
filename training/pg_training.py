"""
Policy Gradient Training Script
================================
Trains REINFORCE and PPO agents on the AssistiveTechRehabEnv.
REINFORCE is implemented from scratch using PyTorch.
PPO uses Stable Baselines3.
Each algorithm runs 10 hyperparameter configurations.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environment.custom_env import AssistiveTechRehabEnv

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "pg"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "pg"


class RewardTrackingCallback(BaseCallback):
    """Tracks episode rewards during SB3 training."""

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
                ep_reward = float(info["episode"]["r"])
                ep_length = int(info["episode"]["l"])
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self._current_sum += ep_reward
                self.cumulative_rewards.append(float(self._current_sum))
        return True


# ============================================================
# REINFORCE Implementation (from scratch with PyTorch)
# ============================================================

class REINFORCEPolicy(nn.Module):
    """Policy network for REINFORCE algorithm.
    Outputs raw logits (no softmax) for numerical stability.
    """

    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim),
        )

    def forward(self, x):
        """Return action probabilities (softmax of logits)."""
        logits = self.network(x)
        return torch.softmax(logits, dim=-1)

    def get_logits(self, obs):
        """Return raw logits for Categorical distribution."""
        return self.network(obs)

    def get_action(self, obs):
        """Sample action from policy distribution using logits."""
        logits = self.get_logits(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()


class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) agent.
    Implemented from scratch using PyTorch.

    The REINFORCE algorithm computes policy gradients using complete
    episode returns. It uses the score function estimator:

        nabla J(theta) = E[sum_t nabla log pi_theta(a_t|s_t) * G_t]

    where G_t is the discounted return from time t.
    Optional baseline subtraction reduces variance.
    """

    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99,
                 hidden_size=128, use_baseline=True, entropy_coeff=0.01):
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.entropy_coeff = entropy_coeff
        self.policy = REINFORCEPolicy(obs_dim, act_dim, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.baseline_val = 0.0

    def compute_returns(self, rewards):
        """Compute discounted returns G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Baseline subtraction (exponential moving average of returns)
        if self.use_baseline:
            self.baseline_val = 0.9 * self.baseline_val + 0.1 * returns.mean().item()
            returns = returns - self.baseline_val

        # Normalize returns for stability
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, log_probs, rewards, entropies):
        """
        Update policy using REINFORCE gradient estimate.

        Loss = -sum_t log pi(a_t|s_t) * G_t - beta * H(pi)

        where H(pi) is the entropy bonus for exploration.
        """
        returns = self.compute_returns(rewards)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        # Policy gradient loss
        policy_loss = -(log_probs_t * returns).mean()
        # Entropy bonus (encourages exploration)
        entropy_loss = -self.entropy_coeff * entropies_t.mean()

        loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        return loss.item(), entropies_t.mean().item()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, weights_only=True))


# ============================================================
# REINFORCE Hyperparameter Configurations (10 runs)
# ============================================================

REINFORCE_CONFIGS = [
    {"name": "baseline", "learning_rate": 1e-3, "gamma": 0.99,
     "hidden_size": 128, "entropy_coeff": 0.01, "use_baseline": True,
     "num_episodes": 1500},
    {"name": "high_lr", "learning_rate": 5e-3, "gamma": 0.99,
     "hidden_size": 128, "entropy_coeff": 0.01, "use_baseline": True,
     "num_episodes": 1500},
    {"name": "low_lr", "learning_rate": 1e-4, "gamma": 0.99,
     "hidden_size": 128, "entropy_coeff": 0.01, "use_baseline": True,
     "num_episodes": 1500},
    {"name": "no_baseline", "learning_rate": 1e-3, "gamma": 0.99,
     "hidden_size": 128, "entropy_coeff": 0.01, "use_baseline": False,
     "num_episodes": 1500},
    {"name": "high_gamma", "learning_rate": 1e-3, "gamma": 0.999,
     "hidden_size": 128, "entropy_coeff": 0.01, "use_baseline": True,
     "num_episodes": 1500},
    {"name": "low_gamma", "learning_rate": 1e-3, "gamma": 0.9,
     "hidden_size": 128, "entropy_coeff": 0.01, "use_baseline": True,
     "num_episodes": 1500},
    {"name": "large_net", "learning_rate": 1e-3, "gamma": 0.99,
     "hidden_size": 256, "entropy_coeff": 0.01, "use_baseline": True,
     "num_episodes": 1500},
    {"name": "small_net", "learning_rate": 1e-3, "gamma": 0.99,
     "hidden_size": 64, "entropy_coeff": 0.01, "use_baseline": True,
     "num_episodes": 1500},
    {"name": "high_entropy", "learning_rate": 1e-3, "gamma": 0.99,
     "hidden_size": 128, "entropy_coeff": 0.05, "use_baseline": True,
     "num_episodes": 1500},
    {"name": "low_entropy", "learning_rate": 1e-3, "gamma": 0.99,
     "hidden_size": 128, "entropy_coeff": 0.001, "use_baseline": True,
     "num_episodes": 1500},
]

# ============================================================
# PPO Hyperparameter Configurations (10 runs)
# ============================================================

PPO_CONFIGS = [
    {"name": "baseline", "learning_rate": 3e-4, "gamma": 0.99,
     "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
     "clip_range": 0.2, "ent_coef": 0.01, "total_timesteps": 80000},
    {"name": "high_lr", "learning_rate": 1e-3, "gamma": 0.99,
     "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
     "clip_range": 0.2, "ent_coef": 0.01, "total_timesteps": 80000},
    {"name": "low_lr", "learning_rate": 1e-4, "gamma": 0.99,
     "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
     "clip_range": 0.2, "ent_coef": 0.01, "total_timesteps": 80000},
    {"name": "wide_clip", "learning_rate": 3e-4, "gamma": 0.99,
     "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
     "clip_range": 0.3, "ent_coef": 0.01, "total_timesteps": 80000},
    {"name": "narrow_clip", "learning_rate": 3e-4, "gamma": 0.99,
     "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
     "clip_range": 0.1, "ent_coef": 0.01, "total_timesteps": 80000},
    {"name": "high_entropy", "learning_rate": 3e-4, "gamma": 0.99,
     "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
     "clip_range": 0.2, "ent_coef": 0.05, "total_timesteps": 80000},
    {"name": "low_entropy", "learning_rate": 3e-4, "gamma": 0.99,
     "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
     "clip_range": 0.2, "ent_coef": 0.001, "total_timesteps": 80000},
    {"name": "short_rollout", "learning_rate": 3e-4, "gamma": 0.99,
     "n_steps": 512, "batch_size": 64, "n_epochs": 10,
     "clip_range": 0.2, "ent_coef": 0.01, "total_timesteps": 80000},
    {"name": "many_epochs", "learning_rate": 3e-4, "gamma": 0.99,
     "n_steps": 2048, "batch_size": 64, "n_epochs": 20,
     "clip_range": 0.2, "ent_coef": 0.01, "total_timesteps": 80000},
    {"name": "high_gamma", "learning_rate": 3e-4, "gamma": 0.999,
     "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
     "clip_range": 0.2, "ent_coef": 0.01, "total_timesteps": 80000},
]

# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_model(model, env, n_episodes=20):
    """Evaluate a trained SB3 model."""
    rewards, lengths = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward, steps, done = 0, 0, False
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


def evaluate_reinforce(agent, env, n_episodes=20):
    """Evaluate REINFORCE agent deterministically."""
    rewards, lengths = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward, steps, done = 0, 0, False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                probs = agent.policy(obs_t)
                action = probs.argmax(dim=-1).item()
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


# ============================================================
# Training functions
# ============================================================

def train_reinforce(config, run_idx):
    """Train REINFORCE with given config."""
    print(f"\n{'='*60}")
    print(f"REINFORCE Run {run_idx + 1}/10: {config['name']}")
    print(f"{'='*60}")

    env = AssistiveTechRehabEnv()
    eval_env = AssistiveTechRehabEnv()

    agent = REINFORCEAgent(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.n,
        lr=config["learning_rate"],
        gamma=config["gamma"],
        hidden_size=config["hidden_size"],
        use_baseline=config["use_baseline"],
        entropy_coeff=config["entropy_coeff"],
    )

    episode_rewards = []
    episode_lengths = []
    cumulative_rewards = []
    entropy_history = []
    cum_sum = 0
    best_mean_reward = -float("inf")

    num_episodes = config["num_episodes"]

    for ep in range(num_episodes):
        obs, _ = env.reset()
        log_probs, rewards, entropies = [], [], []
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob, entropy = agent.policy.get_action(obs_t)
            obs, reward, terminated, truncated, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            done = terminated or truncated

        ep_reward = sum(rewards)
        episode_rewards.append(float(ep_reward))
        episode_lengths.append(len(rewards))
        cum_sum += ep_reward
        cumulative_rewards.append(float(cum_sum))

        loss, mean_entropy = agent.update(log_probs, rewards, entropies)
        entropy_history.append(float(mean_entropy))

        if (ep + 1) % 100 == 0:
            recent = np.mean(episode_rewards[-100:])
            print(f"  Episode {ep + 1}/{num_episodes} | "
                  f"Avg Reward (100): {recent:.2f} | "
                  f"Entropy: {mean_entropy:.4f}")

            if recent > best_mean_reward:
                best_mean_reward = recent
                save_dir = MODELS_DIR / "reinforce" / config["name"]
                save_dir.mkdir(parents=True, exist_ok=True)
                agent.save(str(save_dir / "best_model.pt"))

    # Save final
    save_dir = MODELS_DIR / "reinforce" / config["name"]
    save_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(save_dir / "final_model.pt"))

    # Evaluate
    eval_results = evaluate_reinforce(agent, eval_env)

    # Save curves
    curves = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "cumulative_rewards": cumulative_rewards,
        "entropy_history": entropy_history,
    }
    curves_dir = RESULTS_DIR / "reinforce" / config["name"]
    curves_dir.mkdir(parents=True, exist_ok=True)
    with open(curves_dir / "training_curves.json", "w") as f:
        json.dump(curves, f)

    result = {
        "run": run_idx + 1,
        "name": config["name"],
        "learning_rate": config["learning_rate"],
        "gamma": config["gamma"],
        "hidden_size": config["hidden_size"],
        "entropy_coeff": config["entropy_coeff"],
        "use_baseline": config["use_baseline"],
        **eval_results,
    }

    print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
    return result


def train_sb3_model(algo_class, algo_name, config, run_idx):
    """Train a Stable Baselines3 model (PPO)."""
    print(f"\n{'='*60}")
    print(f"{algo_name} Run {run_idx + 1}/10: {config['name']}")
    print(f"{'='*60}")

    env = Monitor(AssistiveTechRehabEnv())
    eval_env = Monitor(AssistiveTechRehabEnv())

    model_params = {k: v for k, v in config.items()
                    if k not in ("name", "total_timesteps")}

    model = algo_class(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(RESULTS_DIR / algo_name.lower() / "tb_logs"),
        **model_params,
    )

    reward_cb = RewardTrackingCallback()
    save_dir = MODELS_DIR / algo_name.lower() / config["name"]
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(RESULTS_DIR / algo_name.lower() / config["name"]),
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

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / "final_model"))

    eval_results = evaluate_model(model, eval_env)

    curves = {
        "episode_rewards": reward_cb.episode_rewards,
        "episode_lengths": reward_cb.episode_lengths,
        "cumulative_rewards": reward_cb.cumulative_rewards,
    }
    curves_dir = RESULTS_DIR / algo_name.lower() / config["name"]
    curves_dir.mkdir(parents=True, exist_ok=True)
    with open(curves_dir / "training_curves.json", "w") as f:
        json.dump(curves, f)

    print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
    env.close()
    eval_env.close()

    return eval_results


def run_reinforce_experiments():
    """Run all 10 REINFORCE experiments."""
    print("\n" + "#" * 60)
    print("# REINFORCE EXPERIMENTS")
    print("#" * 60)

    results = []
    for idx, config in enumerate(REINFORCE_CONFIGS):
        result = train_reinforce(config, idx)
        results.append(result)

    df = pd.DataFrame(results)
    (RESULTS_DIR / "reinforce").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "reinforce" / "reinforce_summary.csv", index=False)
    print("\nREINFORCE Summary:")
    print(df.to_string(index=False))
    return df


def run_ppo_experiments():
    """Run all 10 PPO experiments."""
    print("\n" + "#" * 60)
    print("# PPO EXPERIMENTS")
    print("#" * 60)

    results = []
    for idx, config in enumerate(PPO_CONFIGS):
        eval_results = train_sb3_model(PPO, "PPO", config, idx)
        result = {
            "run": idx + 1,
            "name": config["name"],
            "learning_rate": config["learning_rate"],
            "gamma": config["gamma"],
            "n_steps": config["n_steps"],
            "clip_range": config["clip_range"],
            "ent_coef": config["ent_coef"],
            **eval_results,
        }
        results.append(result)

    df = pd.DataFrame(results)
    (RESULTS_DIR / "ppo").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "ppo" / "ppo_summary.csv", index=False)
    print("\nPPO Summary:")
    print(df.to_string(index=False))
    return df


def run_all_pg_experiments():
    """Run all policy gradient experiments."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    run_reinforce_experiments()
    run_ppo_experiments()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Policy Gradient Training")
    parser.add_argument("--algo", type=str, default="all",
                        choices=["reinforce", "ppo", "all"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.algo == "reinforce":
        run_reinforce_experiments()
    elif args.algo == "ppo":
        run_ppo_experiments()
    else:
        run_all_pg_experiments()
