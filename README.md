# Assistive Technology Rehabilitation Center - Reinforcement Learning

A reinforcement learning system for optimizing the management of a disability rehabilitation center in Kigali, Rwanda. The RL agent learns to coordinate patient assessment, assistive device allocation, therapy scheduling, and patient discharge to maximize independence outcomes for people with disabilities.

**Mission:** Improve the lives of people with disabilities in Africa by using technology skills to make assistive technology and devices accessible.

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment (AssistiveTechRehabEnv)
│   ├── rendering.py             # Pygame visualization (900x650 window)
│   └── __init__.py
├── training/
│   ├── dqn_training.py          # DQN training with 10 hyperparameter configs
│   ├── pg_training.py           # REINFORCE (from scratch) + PPO training (10 configs each)
│   └── plot_results.py          # Generate all report visualizations
├── models/
│   ├── dqn/                     # Saved DQN models (10 configs)
│   └── pg/                      # Saved REINFORCE + PPO models
├── results/
│   ├── dqn/                     # DQN training curves + summary CSV
│   ├── pg/                      # Policy gradient results
│   └── plots/                   # Generated report figures
├── main.py                      # Entry point (run, train, plot, API demo)
├── random_agent_demo.py         # Static random agent visualization demo
├── capture_screenshot.py        # Capture Pygame screenshot for report
├── generate_report_pdf.py       # Generate PDF report with plots and tables
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run random agent demo (no training needed)
python random_agent_demo.py

# Train all models (DQN, REINFORCE, PPO - 30 total runs)
python main.py --train

# Train specific algorithm
python main.py --train --algo dqn
python main.py --train --algo reinforce
python main.py --train --algo ppo

# Run best trained model with Pygame visualization
python main.py

# Run specific algorithm's best model
python main.py --model dqn

# Generate report plots
python main.py --plot

# Demonstrate JSON API for SMS/USSD integration
python main.py --api-demo

# Run without Pygame rendering
python main.py --no-render
```

## Environment Details

### Agent
A virtual rehabilitation center coordinator making daily management decisions over a 90-day planning period.

### Observation Space (17 dimensions, normalized [0, 1])
| Feature | Description |
|---------|-------------|
| Day of period | Current day / 90 |
| Center type | Urban (Kigali) or Rural Outreach |
| Patients waiting | Queue size |
| Patients in rehab | Active patients |
| Wheelchair stock | Available wheelchairs |
| Prosthetic stock | Available prosthetics |
| Hearing aid stock | Available hearing aids |
| Therapist availability | Free therapy slots |
| Avg patient progress | Mean rehabilitation progress |
| Community impact | Cumulative impact score |
| Budget remaining | Remaining budget |
| Device condition | Average deployed device condition |
| Patient satisfaction | Current satisfaction level |
| Referral backlog | Pending specialist referrals |
| Days since maintenance | Time since last device service |
| Current disability | Next patient's disability type |
| Urgency level | Case severity |

### Action Space (8 discrete actions)
| Action | Cost | Description |
|--------|------|-------------|
| Wait/Monitor | 0 | Observe conditions |
| Assess Patient | 5 | Evaluate next patient from queue |
| Assign Wheelchair | 20 | Provide wheelchair (mobility/multiple) |
| Assign Prosthetic | 35 | Provide prosthetic (amputation/multiple) |
| Assign Hearing Aid | 15 | Provide hearing aid (hearing/multiple) |
| Schedule Therapy | 10 | Book rehab session for patients |
| Maintain Devices | 8 | Service deployed devices |
| Discharge Patient | 3 | Graduate most progressed patient |

### Reward Structure
- **Correct device assignment:** +15 (matching disability type)
- **Therapy scheduled:** +5
- **High-independence discharge:** +25
- **Wrong device:** -2
- **Device deterioration:** -2/step when condition < 0.3
- **Satisfaction collapse:** -15

## Algorithms
- **DQN** (Stable Baselines3) - Value-based with experience replay
- **REINFORCE** (PyTorch from scratch) - Monte Carlo policy gradient with baseline
- **PPO** (Stable Baselines3) - Clipped surrogate objective

## SMS/USSD Integration
The `RehabAdvisoryAPI` class in `main.py` serializes the trained model as a JSON API for deployment behind a USSD/SMS interface, enabling community health workers to receive AI-driven rehabilitation coordination advice on basic feature phones, in both English and Kinyarwanda.
