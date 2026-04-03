"""
Assistive Technology Rehabilitation Center Environment
=======================================================
A custom Gymnasium environment simulating a 90-day planning period at a
disability rehabilitation center in Kigali, Rwanda. The agent acts as a
virtual center coordinator, making daily decisions about patient assessment,
assistive device allocation, therapy scheduling, device maintenance, and
patient discharge — optimizing independence outcomes and community impact
for people with disabilities.

This environment models realistic rehabilitation dynamics calibrated to
the East African context, where assistive technology access is limited
(only ~5-15% of those who need assistive devices have them in Sub-Saharan
Africa — WHO). It is designed to power a USSD/SMS-based system that helps
community health workers coordinate rehabilitation services via basic phones.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Action constants
WAIT = 0
ASSESS_PATIENT = 1
ASSIGN_WHEELCHAIR = 2
ASSIGN_PROSTHETIC = 3
ASSIGN_HEARING_AID = 4
SCHEDULE_THERAPY = 5
MAINTAIN_DEVICES = 6
DISCHARGE_PATIENT = 7

ACTION_NAMES = [
    "Wait/Monitor", "Assess Patient", "Assign Wheelchair", "Assign Prosthetic",
    "Assign Hearing Aid", "Schedule Therapy", "Maintain Devices", "Discharge Patient"
]

# Disability type encoding
DISABILITY_NONE = 0.0
DISABILITY_MOBILITY = 0.33
DISABILITY_AMPUTATION = 0.5
DISABILITY_HEARING = 0.67
DISABILITY_MULTIPLE = 1.0


class AssistiveTechRehabEnv(gym.Env):
    """
    Assistive Technology Rehabilitation Center for Kigali, Rwanda.

    Observation Space (17 dimensions, all normalized [0, 1]):
        0  day_of_period         - Current day / 90
        1  center_type           - 0 = urban (Kigali), 1 = rural outreach
        2  patients_waiting      - Patients in queue / 20
        3  patients_in_rehab     - Active patients / 10
        4  wheelchair_stock      - Available wheelchairs / 15
        5  prosthetic_stock      - Available prosthetics / 10
        6  hearing_aid_stock     - Available hearing aids / 12
        7  therapist_availability - Free therapy slots / 5
        8  avg_patient_progress  - Mean rehabilitation progress [0, 1]
        9  community_impact      - Cumulative impact score / 100
        10 budget_remaining      - Remaining budget (normalized)
        11 device_condition      - Average condition of deployed devices
        12 patient_satisfaction  - Current satisfaction level
        13 referral_backlog      - Pending specialist referrals / 10
        14 days_since_maintenance - Days since device maintenance / 14
        15 current_patient_disability - Type of next patient to assess
        16 urgency_level         - Current most urgent case severity

    Action Space (8 discrete actions):
        0: Wait / Monitor       - Observe and gather data
        1: Assess Patient       - Evaluate new patient from queue (cost: 5)
        2: Assign Wheelchair    - Provide wheelchair to assessed patient (cost: 20)
        3: Assign Prosthetic    - Provide prosthetic limb to patient (cost: 35)
        4: Assign Hearing Aid   - Provide hearing aid to patient (cost: 15)
        5: Schedule Therapy     - Book rehabilitation session (cost: 10)
        6: Maintain Devices     - Service deployed devices (cost: 8)
        7: Discharge Patient    - Graduate patient from program (cost: 3)

    Reward Structure:
        +15    : Correct device assignment (matching disability)
        +5     : Therapy session scheduled for active patient
        +25    : Patient discharged with high independence score
        +3     : Device maintenance (prevents deterioration)
        +0.5   : Per-step patient progress
        -2     : Wrong device for patient's disability
        -3     : Action when no patients available
        -1     : Invalid action (wrong timing/state)
        -10    : Device breakdown from neglected maintenance
        -15    : Patient deterioration from delayed care
        -20    : Budget fully depleted

    Terminal Conditions:
        - Planning period ends (90 days) — truncated
        - All patients successfully served — terminated (success)
        - Patient satisfaction drops to 0 — terminated (failure)
        - Budget depleted with patients still waiting — terminated
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Action costs in budget units
    ACTION_COSTS = {
        WAIT: 0, ASSESS_PATIENT: 5, ASSIGN_WHEELCHAIR: 20,
        ASSIGN_PROSTHETIC: 35, ASSIGN_HEARING_AID: 15,
        SCHEDULE_THERAPY: 10, MAINTAIN_DEVICES: 8, DISCHARGE_PATIENT: 3,
    }

    def __init__(self, render_mode=None, max_days=90):
        super().__init__()

        self.render_mode = render_mode
        self.max_days = max_days

        # 8 discrete actions
        self.action_space = spaces.Discrete(8)

        # 17-dimensional continuous observation, all normalized [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(17,), dtype=np.float32
        )

        # Rendering state
        self._renderer = None
        self._screen = None
        self._clock = None
        self._action_log = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Center type: randomly urban (Kigali) or rural outreach
        self.center_type = self.np_random.integers(0, 2)

        # Day counter
        self.day = 0

        # Patient queue
        self.patients_waiting = self.np_random.integers(5, 12)
        self.patients_in_rehab = 0
        self.patients_served = 0
        self.total_patients = self.patients_waiting + self.np_random.integers(5, 10)
        self._new_patient_rate = 0.3 if self.center_type == 0 else 0.2

        # Current patient being assessed
        self.current_patient_assessed = False
        self.current_patient_disability = self._generate_disability()
        self.current_patient_urgency = self.np_random.uniform(0.3, 1.0)

        # Device inventory
        self.wheelchair_stock = self.np_random.integers(5, 10)
        self.prosthetic_stock = self.np_random.integers(3, 7)
        self.hearing_aid_stock = self.np_random.integers(4, 8)
        self.initial_wheelchairs = self.wheelchair_stock
        self.initial_prosthetics = self.prosthetic_stock
        self.initial_hearing_aids = self.hearing_aid_stock

        # Therapy slots
        self.therapist_availability = self.np_random.integers(2, 5)
        self.max_therapists = 5

        # Patient outcomes
        self.avg_patient_progress = 0.0
        self.community_impact = 0.0
        self.patient_satisfaction = 0.8

        # Budget
        self.budget = 200.0
        self.initial_budget = 200.0

        # Device condition & maintenance
        self.device_condition = 1.0
        self.days_since_maintenance = 0

        # Referral backlog
        self.referral_backlog = self.np_random.integers(0, 4)

        # Internal tracking
        self._active_patients = []  # list of dicts with progress info
        self.total_reward = 0.0
        self._action_log = []

        return self._get_obs(), self._get_info()

    def _generate_disability(self):
        """Generate a random disability type weighted by prevalence."""
        # In East Africa: mobility ~40%, amputation ~15%, hearing ~25%, multiple ~20%
        r = self.np_random.random()
        if r < 0.40:
            return DISABILITY_MOBILITY
        elif r < 0.55:
            return DISABILITY_AMPUTATION
        elif r < 0.80:
            return DISABILITY_HEARING
        else:
            return DISABILITY_MULTIPLE

    def step(self, action):
        action = int(action)
        self.day += 1
        reward = 0.0

        # Check affordability
        cost = self.ACTION_COSTS[action]
        can_afford = self.budget >= cost

        prev_progress = self.avg_patient_progress

        # ---- Process action ----
        if action == WAIT:
            # Small penalty for waiting when patients are in queue
            if self.patients_waiting > 0 and self.day <= 60:
                reward -= 1.0

        elif action == ASSESS_PATIENT:
            if self.patients_waiting > 0 and can_afford:
                self.budget -= cost
                self.patients_waiting -= 1
                self.current_patient_assessed = True
                self.current_patient_disability = self._generate_disability()
                self.current_patient_urgency = self.np_random.uniform(0.3, 1.0)
                reward += 5.0  # reward for progressing patient through pipeline
            else:
                reward -= 1.0

        elif action == ASSIGN_WHEELCHAIR:
            if (self.current_patient_assessed and can_afford
                    and self.wheelchair_stock > 0):
                self.budget -= cost
                self.wheelchair_stock -= 1
                self.current_patient_assessed = False

                # Check if correct device for disability
                if self.current_patient_disability in (DISABILITY_MOBILITY, DISABILITY_MULTIPLE):
                    reward += 15.0  # correct match
                    progress = 0.4 + self.np_random.uniform(0, 0.2)
                else:
                    reward -= 2.0  # wrong device
                    progress = 0.1

                self.patients_in_rehab += 1
                self._active_patients.append({
                    "progress": progress,
                    "disability": self.current_patient_disability,
                    "device": "wheelchair",
                    "days_in_rehab": 0,
                })
                self.community_impact += 2.0
            else:
                reward -= 1.0

        elif action == ASSIGN_PROSTHETIC:
            if (self.current_patient_assessed and can_afford
                    and self.prosthetic_stock > 0):
                self.budget -= cost
                self.prosthetic_stock -= 1
                self.current_patient_assessed = False

                if self.current_patient_disability in (DISABILITY_AMPUTATION, DISABILITY_MULTIPLE):
                    reward += 15.0
                    progress = 0.35 + self.np_random.uniform(0, 0.15)
                else:
                    reward -= 2.0
                    progress = 0.1

                self.patients_in_rehab += 1
                self._active_patients.append({
                    "progress": progress,
                    "disability": self.current_patient_disability,
                    "device": "prosthetic",
                    "days_in_rehab": 0,
                })
                self.community_impact += 3.0
            else:
                reward -= 1.0

        elif action == ASSIGN_HEARING_AID:
            if (self.current_patient_assessed and can_afford
                    and self.hearing_aid_stock > 0):
                self.budget -= cost
                self.hearing_aid_stock -= 1
                self.current_patient_assessed = False

                if self.current_patient_disability in (DISABILITY_HEARING, DISABILITY_MULTIPLE):
                    reward += 15.0
                    progress = 0.45 + self.np_random.uniform(0, 0.2)
                else:
                    reward -= 2.0
                    progress = 0.1

                self.patients_in_rehab += 1
                self._active_patients.append({
                    "progress": progress,
                    "disability": self.current_patient_disability,
                    "device": "hearing_aid",
                    "days_in_rehab": 0,
                })
                self.community_impact += 2.0
            else:
                reward -= 1.0

        elif action == SCHEDULE_THERAPY:
            if (self.patients_in_rehab > 0 and can_afford
                    and self.therapist_availability > 0):
                self.budget -= cost
                self.therapist_availability -= 1

                # Therapy boosts progress for all active patients
                for p in self._active_patients:
                    boost = 0.08 + self.np_random.uniform(0, 0.05)
                    p["progress"] = min(1.0, p["progress"] + boost)
                reward += 5.0
                self.patient_satisfaction = min(1.0, self.patient_satisfaction + 0.05)
            else:
                reward -= 1.0

        elif action == MAINTAIN_DEVICES:
            if can_afford:
                self.budget -= cost
                self.days_since_maintenance = 0
                if self.device_condition < 0.5:
                    reward += 5.0  # well-timed maintenance
                elif self.device_condition < 0.8:
                    reward += 3.0
                else:
                    reward += 1.0  # preventive but less needed
                self.device_condition = min(1.0, self.device_condition + 0.4)
            else:
                reward -= 1.0

        elif action == DISCHARGE_PATIENT:
            if self.patients_in_rehab > 0 and can_afford and len(self._active_patients) > 0:
                self.budget -= cost

                # Discharge the most progressed patient
                best_idx = max(range(len(self._active_patients)),
                               key=lambda i: self._active_patients[i]["progress"])
                patient = self._active_patients.pop(best_idx)
                self.patients_in_rehab -= 1
                self.patients_served += 1

                independence = patient["progress"]
                if independence >= 0.8:
                    reward += 25.0  # excellent outcome
                    self.community_impact += 5.0
                elif independence >= 0.6:
                    reward += 15.0  # good outcome
                    self.community_impact += 3.0
                elif independence >= 0.4:
                    reward += 5.0  # adequate
                    self.community_impact += 1.0
                else:
                    reward -= 5.0  # discharged too early
                    self.patient_satisfaction -= 0.1
            else:
                reward -= 1.0

        # ---- Update environment dynamics ----
        self._update_patient_progress()
        self._update_device_condition()
        self._update_satisfaction()
        self._update_referrals()
        self._update_new_arrivals()
        self._replenish_therapists()

        # ---- Dense reward shaping ----
        if self.patients_in_rehab > 0:
            # Progress reward
            progress_delta = self.avg_patient_progress - prev_progress
            if progress_delta > 0:
                reward += progress_delta * 20.0

            # Patient progress milestone
            for p in self._active_patients:
                if p["progress"] >= 0.8 and p.get("milestone_80") is None:
                    reward += 3.0
                    p["milestone_80"] = True

        # Community impact milestone
        if self.community_impact >= 50 and not hasattr(self, '_impact_50'):
            reward += 10.0
            self._impact_50 = True

        # Device deterioration penalty
        if self.device_condition < 0.3:
            reward -= 2.0

        # Satisfaction bonus
        if self.patient_satisfaction > 0.8:
            reward += 0.2

        # ---- Terminal conditions ----
        terminated = False
        truncated = False

        # Success: all patients served
        if (self.patients_waiting == 0 and self.patients_in_rehab == 0
                and self.patients_served > 0):
            reward += 20.0
            terminated = True

        # Failure: satisfaction collapsed
        if self.patient_satisfaction <= 0.0:
            reward -= 15.0
            terminated = True

        # Failure: budget depleted with patients still waiting
        if self.budget <= 0 and self.patients_waiting > 0:
            reward -= 10.0
            terminated = True

        # Season end
        if self.day >= self.max_days:
            truncated = True
            if self.patients_waiting > 0:
                reward -= 5.0  # unserved patients penalty

        # Log action
        self._action_log.append({
            "day": self.day,
            "action": ACTION_NAMES[action],
            "reward": round(reward, 1),
        })
        if len(self._action_log) > 10:
            self._action_log.pop(0)

        self.total_reward += reward

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _update_patient_progress(self):
        """Update rehabilitation progress for active patients."""
        if not self._active_patients:
            self.avg_patient_progress = 0.0
            return

        for p in self._active_patients:
            p["days_in_rehab"] += 1

            # Natural progress (slow without therapy)
            base_rate = 0.005 * self.device_condition
            p["progress"] = min(1.0, p["progress"] + base_rate)

            # Deterioration if device condition is poor
            if self.device_condition < 0.3:
                p["progress"] = max(0.0, p["progress"] - 0.01)

        self.avg_patient_progress = np.mean([p["progress"] for p in self._active_patients])

    def _update_device_condition(self):
        """Deployed devices degrade over time without maintenance."""
        self.days_since_maintenance += 1

        # Faster degradation with more deployed devices
        deployed = (self.initial_wheelchairs - self.wheelchair_stock +
                    self.initial_prosthetics - self.prosthetic_stock +
                    self.initial_hearing_aids - self.hearing_aid_stock)
        degradation = 0.01 + 0.005 * max(0, deployed - 3)

        self.device_condition -= degradation
        self.device_condition = np.clip(self.device_condition, 0.0, 1.0)

    def _update_satisfaction(self):
        """Update patient satisfaction based on care quality."""
        # Waiting patients reduce satisfaction (gradual)
        if self.patients_waiting > 8:
            self.patient_satisfaction -= 0.005 * (self.patients_waiting - 8)

        # Poor device condition reduces satisfaction
        if self.device_condition < 0.3:
            self.patient_satisfaction -= 0.008

        # High referral backlog reduces satisfaction
        if self.referral_backlog > 7:
            self.patient_satisfaction -= 0.005

        # Good progress improves satisfaction
        if self.avg_patient_progress > 0.5:
            self.patient_satisfaction += 0.008

        # Serving patients improves satisfaction
        if self.patients_served > 0:
            self.patient_satisfaction += 0.003

        self.patient_satisfaction = np.clip(self.patient_satisfaction, 0.0, 1.0)

    def _update_referrals(self):
        """Specialist referrals accumulate randomly."""
        if self.np_random.random() < 0.15:
            self.referral_backlog = min(10, self.referral_backlog + 1)

        # Referrals resolve slowly on their own
        if self.np_random.random() < 0.08:
            self.referral_backlog = max(0, self.referral_backlog - 1)

    def _update_new_arrivals(self):
        """New patients arrive stochastically."""
        if self.np_random.random() < self._new_patient_rate:
            self.patients_waiting += 1

    def _replenish_therapists(self):
        """Therapists become available again each day."""
        if self.therapist_availability < self.max_therapists:
            if self.np_random.random() < 0.4:
                self.therapist_availability = min(
                    self.max_therapists, self.therapist_availability + 1)

    def _get_obs(self):
        """Return normalized observation vector."""
        obs = np.array([
            self.day / self.max_days,                                # 0: day
            float(self.center_type),                                 # 1: center type
            min(self.patients_waiting / 20.0, 1.0),                  # 2: queue
            min(self.patients_in_rehab / 10.0, 1.0),                 # 3: active patients
            min(self.wheelchair_stock / 15.0, 1.0),                  # 4: wheelchairs
            min(self.prosthetic_stock / 10.0, 1.0),                  # 5: prosthetics
            min(self.hearing_aid_stock / 12.0, 1.0),                 # 6: hearing aids
            min(self.therapist_availability / 5.0, 1.0),             # 7: therapists
            self.avg_patient_progress,                               # 8: progress
            min(self.community_impact / 100.0, 1.0),                 # 9: impact
            self.budget / self.initial_budget,                       # 10: budget
            self.device_condition,                                   # 11: device condition
            self.patient_satisfaction,                                # 12: satisfaction
            min(self.referral_backlog / 10.0, 1.0),                  # 13: referrals
            min(self.days_since_maintenance / 14.0, 1.0),            # 14: maintenance
            self.current_patient_disability,                         # 15: disability type
            self.current_patient_urgency,                            # 16: urgency
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def _get_info(self):
        """Return human-readable state information."""
        disability_names = {
            DISABILITY_NONE: "None",
            DISABILITY_MOBILITY: "Mobility",
            DISABILITY_AMPUTATION: "Amputation",
            DISABILITY_HEARING: "Hearing",
            DISABILITY_MULTIPLE: "Multiple",
        }

        return {
            "day": self.day,
            "max_days": self.max_days,
            "center": "Urban (Kigali)" if self.center_type == 0 else "Rural Outreach",
            "patients_waiting": self.patients_waiting,
            "patients_in_rehab": self.patients_in_rehab,
            "patients_served": self.patients_served,
            "wheelchair_stock": self.wheelchair_stock,
            "prosthetic_stock": self.prosthetic_stock,
            "hearing_aid_stock": self.hearing_aid_stock,
            "therapist_availability": self.therapist_availability,
            "avg_progress": round(self.avg_patient_progress, 3),
            "community_impact": round(self.community_impact, 1),
            "budget": round(self.budget, 1),
            "device_condition": round(self.device_condition, 2),
            "patient_satisfaction": round(self.patient_satisfaction, 2),
            "referral_backlog": self.referral_backlog,
            "current_disability": disability_names.get(
                self.current_patient_disability, "Unknown"),
            "urgency": round(self.current_patient_urgency, 2),
            "patient_assessed": self.current_patient_assessed,
            "total_reward": round(self.total_reward, 1),
            "action_log": list(self._action_log),
        }

    def render(self):
        if self.render_mode == "human":
            from environment.rendering import render_frame
            return render_frame(self)
        elif self.render_mode == "rgb_array":
            from environment.rendering import render_to_array
            return render_to_array(self)

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
