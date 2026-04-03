"""
Plot Results - Generates all visualizations for the report.
=============================================================
Creates:
1. Cumulative reward curves (all methods in subplots)
2. DQN objective/loss curves
3. Policy gradient entropy curves
4. Convergence comparison plots
5. Generalization test results
6. Hyperparameter comparison bar charts
"""

import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"


def smooth(data, window=20):
    """Moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def load_training_curves(algo, config_name):
    """Load training curves JSON for a specific run."""
    if algo == "dqn":
        path = RESULTS_DIR / "dqn" / config_name / "training_curves.json"
    else:
        path = RESULTS_DIR / "pg" / algo / config_name / "training_curves.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def find_best_config(algo):
    """Find the best config name for an algorithm."""
    if algo == "dqn":
        summary_path = RESULTS_DIR / "dqn" / "dqn_summary.csv"
    else:
        summary_path = RESULTS_DIR / "pg" / algo / f"{algo}_summary.csv"
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    return df.loc[df["mean_reward"].idxmax(), "name"]


def plot_cumulative_rewards():
    """Plot cumulative rewards for all methods' best configs in subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cumulative Rewards Over Training Episodes",
                 fontsize=16, fontweight="bold")

    algos = [("dqn", "DQN"), ("reinforce", "REINFORCE"),
             ("ppo", "PPO")]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, (algo, title), color in zip(axes.flat, algos, colors):
        best_config = find_best_config(algo)
        if best_config is None:
            ax.set_title(f"{title} - No Data")
            ax.text(0.5, 0.5, "No training data found",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        curves = load_training_curves(algo, best_config)
        if curves is None:
            ax.set_title(f"{title} - No Curves")
            continue

        rewards = curves["episode_rewards"]
        smoothed = smooth(rewards, window=20)
        ax.plot(rewards, alpha=0.15, color=color)
        ax.plot(smoothed, alpha=0.9, label="Smoothed (w=20)", color=color,
                linewidth=2)
        ax.set_title(f"{title} (config: {best_config})", fontsize=12)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cumulative_rewards.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved: cumulative_rewards.png")


def plot_rewards_comparison():
    """Plot all methods overlaid on one plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Training Reward Comparison - All Methods",
                 fontsize=14, fontweight="bold")

    colors = {"dqn": "#1f77b4", "reinforce": "#ff7f0e",
              "ppo": "#2ca02c"}
    labels = {"dqn": "DQN", "reinforce": "REINFORCE",
              "ppo": "PPO"}

    for algo in ["dqn", "reinforce", "ppo"]:
        best_config = find_best_config(algo)
        if best_config is None:
            continue
        curves = load_training_curves(algo, best_config)
        if curves is None:
            continue
        rewards = curves["episode_rewards"]
        smoothed = smooth(rewards, window=30)
        ax.plot(smoothed, label=f"{labels[algo]} ({best_config})",
                color=colors[algo], linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward (smoothed)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rewards_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved: rewards_comparison.png")


def plot_dqn_objective():
    """Plot DQN training curves for all 10 configs."""
    summary_path = RESULTS_DIR / "dqn" / "dqn_summary.csv"
    if not summary_path.exists():
        print("  Skipping DQN objective plot - no data")
        return

    df = pd.read_csv(summary_path)
    n = len(df)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.suptitle("DQN Training Curves - All Configurations",
                 fontsize=14, fontweight="bold")

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes.flat[idx] if n > cols else axes[idx]
        config_name = row["name"]
        curves = load_training_curves("dqn", config_name)
        if curves is None:
            ax.set_title(config_name, fontsize=9)
            continue
        rewards = curves["episode_rewards"]
        smoothed = smooth(rewards, window=15)
        ax.plot(smoothed, color="steelblue", linewidth=1.5)
        ax.set_title(f"{config_name}\nR={row['mean_reward']:.1f}", fontsize=9)
        ax.set_xlabel("Episode", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide empty axes
    for idx in range(n, rows * cols):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dqn_objective_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved: dqn_objective_curves.png")


def plot_pg_entropy():
    """Plot policy entropy curves for REINFORCE configs."""
    summary_path = RESULTS_DIR / "pg" / "reinforce" / "reinforce_summary.csv"
    if not summary_path.exists():
        print("  Skipping entropy plot - no data")
        return

    df = pd.read_csv(summary_path)
    n = min(len(df), 10)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.suptitle("REINFORCE Policy Entropy - All Configurations",
                 fontsize=14, fontweight="bold")

    for idx in range(n):
        row = df.iloc[idx]
        ax = axes.flat[idx] if n > cols else axes[idx]
        config_name = row["name"]
        curves = load_training_curves("reinforce", config_name)
        if curves is None or "entropy_history" not in curves:
            ax.set_title(config_name, fontsize=9)
            continue
        entropy = curves["entropy_history"]
        smoothed = smooth(entropy, window=20)
        ax.plot(smoothed, color="darkorange", linewidth=1.5)
        ax.set_title(f"{config_name}\nR={row['mean_reward']:.1f}", fontsize=9)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel("Entropy", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pg_entropy_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved: pg_entropy_curves.png")


def plot_convergence():
    """Plot convergence comparison across all methods."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Convergence Comparison - Episodes to Stable Performance",
                 fontsize=14, fontweight="bold")

    colors = {"dqn": "#1f77b4", "reinforce": "#ff7f0e",
              "ppo": "#2ca02c"}
    labels = {"dqn": "DQN", "reinforce": "REINFORCE",
              "ppo": "PPO"}
    convergence_data = {}

    for algo in ["dqn", "reinforce", "ppo"]:
        best_config = find_best_config(algo)
        if best_config is None:
            continue
        curves = load_training_curves(algo, best_config)
        if curves is None:
            continue

        rewards = curves["episode_rewards"]
        window = 50
        if len(rewards) > window:
            running_mean = [np.mean(rewards[max(0, i - window):i + 1])
                            for i in range(len(rewards))]
        else:
            running_mean = rewards

        ax.plot(running_mean, label=labels[algo], color=colors[algo],
                linewidth=2)

        # Find convergence point
        if len(running_mean) > 100:
            final_mean = np.mean(running_mean[-50:])
            threshold = final_mean * 0.9
            convergence_ep = next(
                (i for i, v in enumerate(running_mean)
                 if v >= threshold and i > 50),
                len(running_mean))
            convergence_data[algo] = convergence_ep
            ax.axvline(x=convergence_ep, color=colors[algo],
                        linestyle="--", alpha=0.5)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Running Mean Reward (window=50)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "convergence_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved: convergence_comparison.png")

    if convergence_data:
        with open(PLOTS_DIR / "convergence_data.json", "w") as f:
            json.dump(convergence_data, f, indent=2)


def plot_hyperparameter_comparison():
    """Bar charts comparing all hyperparameter configs per algorithm."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Hyperparameter Tuning Results - Mean Reward by Configuration",
                 fontsize=16, fontweight="bold")

    algo_info = [
        ("dqn", "DQN", RESULTS_DIR / "dqn" / "dqn_summary.csv", "#1f77b4"),
        ("reinforce", "REINFORCE",
         RESULTS_DIR / "pg" / "reinforce" / "reinforce_summary.csv", "#ff7f0e"),
        ("ppo", "PPO",
         RESULTS_DIR / "pg" / "ppo" / "ppo_summary.csv", "#2ca02c"),
    ]

    for ax, (algo, title, csv_path, color) in zip(axes.flat, algo_info):
        if not csv_path.exists():
            ax.set_title(f"{title} - No Data")
            continue

        df = pd.read_csv(csv_path)
        names = df["name"].tolist()
        rewards = df["mean_reward"].tolist()
        stds = df["std_reward"].tolist() if "std_reward" in df else [0] * len(names)

        bars = ax.barh(range(len(names)), rewards, xerr=stds,
                       capsize=3, color=color, alpha=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Mean Reward", fontsize=10)
        ax.set_title(f"{title} Hyperparameter Comparison", fontsize=12)
        ax.grid(True, alpha=0.3, axis="x")

        # Highlight best
        best_idx = int(np.argmax(rewards))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hyperparameter_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved: hyperparameter_comparison.png")


def plot_generalization():
    """Test trained models on unseen initial states (different seeds)."""
    from environment.custom_env import AssistiveTechRehabEnv

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Generalization Test - Performance on Unseen Initial States",
                 fontsize=14, fontweight="bold")

    algos = ["dqn", "reinforce", "ppo"]
    labels = {"dqn": "DQN", "reinforce": "REINFORCE",
              "ppo": "PPO"}
    colors = {"dqn": "#1f77b4", "reinforce": "#ff7f0e",
              "ppo": "#2ca02c"}

    all_rewards = {}
    all_lengths = {}

    for algo in algos:
        best_config = find_best_config(algo)
        if best_config is None:
            continue

        try:
            if algo == "reinforce":
                import torch
                from training.pg_training import REINFORCEAgent
                env = AssistiveTechRehabEnv()
                agent = REINFORCEAgent(
                    obs_dim=env.observation_space.shape[0],
                    act_dim=env.action_space.n,
                )
                model_path = (Path("models/pg/reinforce") /
                              best_config / "best_model.pt")
                if not model_path.exists():
                    model_path = (Path("models/pg/reinforce") /
                                  best_config / "final_model.pt")
                agent.load(str(model_path))

                rewards, lengths = [], []
                for seed in range(50):
                    obs, _ = env.reset(seed=seed + 1000)
                    total_r, done = 0, False
                    steps = 0
                    while not done:
                        obs_t = torch.FloatTensor(obs).unsqueeze(0)
                        with torch.no_grad():
                            probs = agent.policy(obs_t)
                            action = probs.argmax(dim=-1).item()
                        obs, r, term, trunc, info = env.step(action)
                        total_r += r
                        steps += 1
                        done = term or trunc
                    rewards.append(total_r)
                    lengths.append(steps)
                env.close()
            else:
                from stable_baselines3 import DQN, PPO
                algo_map = {"dqn": DQN, "ppo": PPO}
                if algo == "dqn":
                    model_dir = Path("models/dqn") / best_config
                else:
                    model_dir = Path("models/pg") / algo / best_config
                model_path = model_dir / "best_model.zip"
                if not model_path.exists():
                    model_path = model_dir / "final_model.zip"
                model = algo_map[algo].load(str(model_path))

                env = AssistiveTechRehabEnv()
                rewards, lengths = [], []
                for seed in range(50):
                    obs, _ = env.reset(seed=seed + 1000)
                    total_r, done, steps = 0, False, 0
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, r, term, trunc, info = env.step(action)
                        total_r += r
                        steps += 1
                        done = term or trunc
                    rewards.append(total_r)
                    lengths.append(steps)
                env.close()

            all_rewards[algo] = rewards
            all_lengths[algo] = lengths
        except Exception as e:
            print(f"  Warning: Could not load {algo} model: {e}")

    if not all_rewards:
        print("  No models found for generalization test")
        plt.close()
        return

    # Box plot of rewards
    ax1 = axes[0]
    data = [all_rewards[a] for a in algos if a in all_rewards]
    bp_labels = [labels[a] for a in algos if a in all_rewards]
    bp_colors = [colors[a] for a in algos if a in all_rewards]

    bp = ax1.boxplot(data, labels=bp_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_ylabel("Total Reward", fontsize=12)
    ax1.set_title("Reward Distribution on Unseen States", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Bar plot of mean episode lengths
    ax2 = axes[1]
    means = [np.mean(all_lengths[a]) for a in algos if a in all_lengths]
    stds = [np.std(all_lengths[a]) for a in algos if a in all_lengths]
    x = range(len(means))
    ax2.bar(x, means, yerr=stds, capsize=5,
            color=[colors[a] for a in algos if a in all_lengths], alpha=0.7)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([labels[a] for a in algos if a in all_lengths])
    ax2.set_ylabel("Mean Episode Length", fontsize=12)
    ax2.set_title("Episode Length on Unseen States", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "generalization_test.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved: generalization_test.png")


def generate_all_plots():
    """Generate all report plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating report plots...")

    plot_cumulative_rewards()
    plot_rewards_comparison()
    plot_dqn_objective()
    plot_pg_entropy()
    plot_convergence()
    plot_hyperparameter_comparison()
    plot_generalization()

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    generate_all_plots()
