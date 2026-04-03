"""
Main Entry Point - Assistive Technology Rehabilitation Center RL
================================================================
Runs the best-performing trained model with Pygame visualization,
or demonstrates random actions if no trained model is found.

Usage:
    python main.py                     # Run best model (auto-detect)
    python main.py --model dqn         # Run specific algorithm's best model
    python main.py --random            # Run random agent demo
    python main.py --train             # Train all models
    python main.py --train --algo dqn  # Train specific algorithm
    python main.py --plot              # Generate all report plots
    python main.py --api-demo          # Show JSON API serialization for SMS/USSD
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from environment.custom_env import AssistiveTechRehabEnv, ACTION_NAMES


# ============================================================
# JSON API Serialization for SMS/USSD Integration
# ============================================================

class RehabAdvisoryAPI:
    """
    JSON API interface that serializes the trained RL model's recommendations
    for use as a backend service in the SMS/USSD-based advisory system.

    This demonstrates how the trained agent can be integrated into a
    production pipeline — a USSD frontend sends patient/center data, this API
    returns actionable coordination advice formatted for SMS-length text menus.
    Community health workers use basic phones to receive guidance.
    """

    ACTION_ADVICE = {
        0: "No action needed today. Continue monitoring patients and center status.",
        1: "Assess the next patient in the waiting queue to determine their needs.",
        2: "Assign a wheelchair to the assessed patient for mobility support.",
        3: "Assign a prosthetic limb to the assessed patient for rehabilitation.",
        4: "Assign a hearing aid to the assessed patient for hearing support.",
        5: "Schedule a therapy session to boost rehabilitation progress.",
        6: "Perform maintenance on deployed devices to prevent deterioration.",
        7: "Discharge the most progressed patient with independence support plan.",
    }

    ACTION_KINYARWANDA = {
        0: "Nta gikorwa gikenewe uyu munsi. Komeza gukurikirana abarwayi.",
        1: "Suzuma umurwayi ukurikira mu murongo w'abaretse.",
        2: "Ha umurwayi intebe y'amapine yo kumufasha kugenda.",
        3: "Ha umurwayi igihimba cy'umubiri gisimbuye icyatakaye.",
        4: "Ha umurwayi igikoresho cyo kumufasha kumva neza.",
        5: "Shiraho gahunda yo kwivuza kugira ngo abarwayi barusheho kuvura.",
        6: "Kora isuku n'isanwa ry'ibikoresho byakoreshejwe.",
        7: "Sezera umurwayi wamaze gukira amufashe gushinga ubuzima bushya.",
    }

    def __init__(self, model, model_type="sb3"):
        self.model = model
        self.model_type = model_type

    def get_recommendation(self, center_data):
        """
        Get coordination recommendation from trained RL model.

        Args:
            center_data: dict with current center conditions

        Returns:
            dict: JSON-serializable recommendation for USSD frontend
        """
        obs = self._center_data_to_obs(center_data)

        if self.model_type == "reinforce":
            import torch
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                probs = self.model.policy(obs_t).numpy()[0]
            action = int(np.argmax(probs))
            confidence = float(probs[action])
            all_probs = {ACTION_NAMES[i]: float(p) for i, p in enumerate(probs)}
        else:
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)
            confidence = 0.85
            all_probs = {}

        # Build alerts
        alerts = []
        if center_data.get("patients_waiting", 0) > 8:
            alerts.append("HIGH QUEUE: Many patients waiting for assessment.")
        if center_data.get("device_condition", 1.0) < 0.3:
            alerts.append("DEVICE ALERT: Devices need urgent maintenance.")
        if center_data.get("budget_remaining", 200) < 30:
            alerts.append("LOW BUDGET: Prioritize essential actions only.")
        if center_data.get("patient_satisfaction", 1.0) < 0.4:
            alerts.append("LOW SATISFACTION: Immediate care improvements needed.")

        return {
            "status": "success",
            "center": center_data.get("center", "Kigali Rehab Center"),
            "recommended_action": ACTION_NAMES[action],
            "action_id": action,
            "confidence": round(confidence, 2),
            "advice_en": self.ACTION_ADVICE[action],
            "advice_rw": self.ACTION_KINYARWANDA[action],
            "alerts": alerts,
            "action_probabilities": all_probs,
            "sms_message": self._format_sms(action, alerts),
            "center_input": center_data,
        }

    def _center_data_to_obs(self, data):
        """Convert center-reported data to observation vector."""
        disability_map = {"none": 0.0, "mobility": 0.33, "amputation": 0.5,
                          "hearing": 0.67, "multiple": 1.0}

        return np.array([
            data.get("day_of_period", 30) / 90.0,
            1.0 if data.get("center_type", "urban") == "rural" else 0.0,
            min(data.get("patients_waiting", 5) / 20.0, 1.0),
            min(data.get("patients_in_rehab", 3) / 10.0, 1.0),
            min(data.get("wheelchair_stock", 5) / 15.0, 1.0),
            min(data.get("prosthetic_stock", 3) / 10.0, 1.0),
            min(data.get("hearing_aid_stock", 4) / 12.0, 1.0),
            min(data.get("therapist_availability", 3) / 5.0, 1.0),
            data.get("avg_patient_progress", 0.3),
            min(data.get("community_impact", 20) / 100.0, 1.0),
            data.get("budget_remaining", 150) / 200.0,
            data.get("device_condition", 0.8),
            data.get("patient_satisfaction", 0.7),
            min(data.get("referral_backlog", 2) / 10.0, 1.0),
            min(data.get("days_since_maintenance", 5) / 14.0, 1.0),
            disability_map.get(data.get("current_disability", "mobility"), 0.33),
            data.get("urgency_level", 0.5),
        ], dtype=np.float32)

    def _format_sms(self, action, alerts):
        """Format as SMS-compatible short text message."""
        msg = f"UBUFASHA BW'IKORANABUHANGA:\n{self.ACTION_KINYARWANDA[action]}"
        if alerts:
            msg += f"\n! {alerts[0]}"
        return msg

    def to_json(self, recommendation):
        return json.dumps(recommendation, indent=2, ensure_ascii=False)


def demo_api():
    """Demonstrate the JSON API for SMS/USSD integration."""
    print("=" * 60)
    print("REHAB ADVISORY API - SMS/USSD Integration Demo")
    print("=" * 60)

    sample_requests = [
        {
            "center": "Kigali Rehab Center",
            "center_type": "urban",
            "day_of_period": 10,
            "patients_waiting": 8,
            "patients_in_rehab": 2,
            "wheelchair_stock": 6,
            "prosthetic_stock": 4,
            "hearing_aid_stock": 5,
            "therapist_availability": 3,
            "avg_patient_progress": 0.2,
            "community_impact": 10,
            "budget_remaining": 180,
            "device_condition": 0.9,
            "patient_satisfaction": 0.85,
            "referral_backlog": 1,
            "days_since_maintenance": 3,
            "current_disability": "mobility",
            "urgency_level": 0.7,
        },
        {
            "center": "Huye District Outreach",
            "center_type": "rural",
            "day_of_period": 45,
            "patients_waiting": 12,
            "patients_in_rehab": 5,
            "wheelchair_stock": 2,
            "prosthetic_stock": 1,
            "hearing_aid_stock": 3,
            "therapist_availability": 1,
            "avg_patient_progress": 0.55,
            "community_impact": 40,
            "budget_remaining": 60,
            "device_condition": 0.25,
            "patient_satisfaction": 0.45,
            "referral_backlog": 6,
            "days_since_maintenance": 12,
            "current_disability": "amputation",
            "urgency_level": 0.9,
        },
        {
            "center": "Musanze Rehab Center",
            "center_type": "urban",
            "day_of_period": 80,
            "patients_waiting": 2,
            "patients_in_rehab": 3,
            "wheelchair_stock": 1,
            "prosthetic_stock": 0,
            "hearing_aid_stock": 2,
            "therapist_availability": 4,
            "avg_patient_progress": 0.82,
            "community_impact": 75,
            "budget_remaining": 35,
            "device_condition": 0.65,
            "patient_satisfaction": 0.9,
            "referral_backlog": 0,
            "days_since_maintenance": 7,
            "current_disability": "hearing",
            "urgency_level": 0.4,
        },
    ]

    algo, config_name, _ = find_best_model()
    if algo:
        model = _load_model(algo, config_name)
        model_type = "reinforce" if algo == "reinforce" else "sb3"
        api = RehabAdvisoryAPI(model, model_type)
    else:
        print("No trained model found. Using random policy for demo.\n")
        class RandomModel:
            def predict(self, obs, deterministic=False):
                return np.random.randint(0, 8), None
        api = RehabAdvisoryAPI(RandomModel(), "sb3")

    for i, center_data in enumerate(sample_requests):
        print(f"\n{'─'*60}")
        print(f"SMS Request {i+1}: {center_data['center']}")
        print(f"  Type: {center_data['center_type']}, Day: {center_data['day_of_period']}")
        print(f"  Waiting: {center_data['patients_waiting']}, "
              f"In Rehab: {center_data['patients_in_rehab']}")
        print(f"{'─'*60}")

        recommendation = api.get_recommendation(center_data)
        print(api.to_json(recommendation))


# ============================================================
# Model loading and running
# ============================================================

def find_best_model():
    """Find the best performing model across all algorithms."""
    import pandas as pd

    best_reward = -float("inf")
    best_algo = None
    best_config = None

    for algo, path in [
        ("dqn", Path("results/dqn/dqn_summary.csv")),
        ("reinforce", Path("results/pg/reinforce/reinforce_summary.csv")),
        ("ppo", Path("results/pg/ppo/ppo_summary.csv")),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            idx = df["mean_reward"].idxmax()
            if df.loc[idx, "mean_reward"] > best_reward:
                best_reward = df.loc[idx, "mean_reward"]
                best_algo = algo
                best_config = df.loc[idx, "name"]

    if best_algo:
        print(f"Best model: {best_algo.upper()} ({best_config}) "
              f"with mean reward: {best_reward:.2f}")

    return best_algo, best_config, best_reward


def _load_model(algo, config_name):
    """Load a trained model."""
    if algo == "reinforce":
        import torch
        from training.pg_training import REINFORCEAgent
        env = AssistiveTechRehabEnv()
        agent = REINFORCEAgent(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.n,
        )
        model_path = Path("models/pg/reinforce") / config_name / "best_model.pt"
        if not model_path.exists():
            model_path = Path("models/pg/reinforce") / config_name / "final_model.pt"
        agent.load(str(model_path))
        env.close()
        return agent
    else:
        from stable_baselines3 import DQN, PPO
        algo_map = {"dqn": DQN, "ppo": PPO}
        if algo == "dqn":
            model_dir = Path("models/dqn") / config_name
        else:
            model_dir = Path("models/pg") / algo / config_name
        model_path = model_dir / "best_model.zip"
        if not model_path.exists():
            model_path = model_dir / "final_model.zip"
        return algo_map[algo].load(str(model_path))


def run_agent(algo, config_name, num_episodes=3, render=True):
    """Run a trained agent in the environment."""
    print("=" * 60)
    print(f"RUNNING {algo.upper()} AGENT: {config_name}")
    print("=" * 60)

    render_mode = "human" if render else None
    env = AssistiveTechRehabEnv(render_mode=render_mode)

    model = _load_model(algo, config_name)
    is_reinforce = (algo == "reinforce")

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        step = 0

        print(f"\n--- Episode {ep + 1} | Center: {info['center']} ---")

        while not done:
            if is_reinforce:
                import torch
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    probs = model.policy(obs_t)
                    action = probs.argmax(dim=-1).item()
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            if render:
                env.render()

            if step % 10 == 0:
                print(f"  Day {info['day']:3d} | Action: {ACTION_NAMES[action]:<18s} | "
                      f"Queue: {info['patients_waiting']:<3d} Rehab: {info['patients_in_rehab']:<3d} | "
                      f"Satisfaction: {info['patient_satisfaction']:.0%} | "
                      f"Budget: {info['budget']:.0f} | "
                      f"Reward: {total_reward:+.1f}")

        status = ("ALL SERVED" if info['patients_waiting'] == 0 and info['patients_in_rehab'] == 0
                  else "SATISFACTION FAILED" if info['patient_satisfaction'] <= 0
                  else "BUDGET DEPLETED" if info['budget'] <= 0
                  else "PERIOD ENDED")
        print(f"  Result: {status} | Days: {step} | "
              f"Served: {info['patients_served']} | "
              f"Total Reward: {total_reward:+.1f}")

    env.close()


def run_random_agent(num_episodes=3, render=True):
    """Demo with random agent."""
    print("=" * 60)
    print("RANDOM AGENT DEMONSTRATION")
    print("=" * 60)

    render_mode = "human" if render else None
    env = AssistiveTechRehabEnv(render_mode=render_mode)

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        step = 0

        print(f"\n--- Episode {ep + 1} | Center: {info['center']} ---")

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            if render:
                env.render()

            if step % 20 == 0:
                print(f"  Day {info['day']:3d} | Action: {ACTION_NAMES[action]:<18s} | "
                      f"Satisfaction: {info['patient_satisfaction']:.0%} | "
                      f"Reward: {total_reward:+.1f}")

        print(f"  Finished: Days={step}, Served={info['patients_served']}, "
              f"Reward={total_reward:+.1f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Assistive Tech Rehab Center RL - Kigali, Rwanda")
    parser.add_argument("--random", action="store_true",
                        help="Run random agent demo")
    parser.add_argument("--model", type=str, default=None,
                        choices=["dqn", "reinforce", "ppo"],
                        help="Specific model to run")
    parser.add_argument("--config", type=str, default=None,
                        help="Specific config name")
    parser.add_argument("--train", action="store_true",
                        help="Train models")
    parser.add_argument("--algo", type=str, default="all",
                        choices=["dqn", "reinforce", "ppo", "all"],
                        help="Algorithm to train")
    parser.add_argument("--plot", action="store_true",
                        help="Generate report plots")
    parser.add_argument("--api-demo", action="store_true",
                        help="Demonstrate JSON API for SMS/USSD")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable Pygame rendering")
    args = parser.parse_args()

    if args.train:
        if args.algo in ("dqn", "all"):
            from training.dqn_training import run_all_experiments
            run_all_experiments()
        if args.algo in ("reinforce", "all"):
            from training.pg_training import run_reinforce_experiments
            run_reinforce_experiments()
        if args.algo in ("ppo", "all"):
            from training.pg_training import run_ppo_experiments
            run_ppo_experiments()
        return

    if args.plot:
        from training.plot_results import generate_all_plots
        generate_all_plots()
        return

    if args.api_demo:
        demo_api()
        return

    if args.random:
        run_random_agent(num_episodes=args.episodes,
                         render=not args.no_render)
        return

    if args.model:
        algo = args.model
        config_name = args.config
        if not config_name:
            _, config_name, _ = find_best_model()
        if config_name:
            run_agent(algo, config_name, num_episodes=args.episodes,
                      render=not args.no_render)
        else:
            print(f"No trained {algo} model found. Run --train first.")
        return

    # Auto-detect best model
    algo, config_name, reward = find_best_model()
    if algo:
        run_agent(algo, config_name, num_episodes=args.episodes,
                  render=not args.no_render)
    else:
        print("No trained models found. Running random agent demo.")
        print("To train: python main.py --train\n")
        run_random_agent(num_episodes=args.episodes,
                         render=not args.no_render)


if __name__ == "__main__":
    main()
