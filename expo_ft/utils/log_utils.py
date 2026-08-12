"""Dataclasses for organizing training loop state."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class EpisodeState:
    ep_return: float = 0.0
    ep_len: int = 1
    observations: list = field(default_factory=list)
    n_remaining: list = field(default_factory=list)
    human_controlled: list = field(default_factory=list)
    human_actions: list = field(default_factory=list)
    sample_info_history: list = field(default_factory=list)
    step_data_history: list = field(default_factory=list)
    policy_steps: int = 0
    human_steps: int = 0
    had_intervention: bool = False

    def reset(self):
        self.ep_return = 0.0
        self.ep_len = 0
        self.observations.clear()
        self.n_remaining.clear()
        self.human_controlled.clear()
        self.human_actions.clear()
        self.sample_info_history.clear()
        self.step_data_history.clear()
        self.policy_steps = 0
        self.human_steps = 0
        self.had_intervention = False

    def record_step(self, observation, n_remaining, action_type, real_action, reward):
        self.observations.append(observation)
        self.n_remaining.append(n_remaining)
        self.human_controlled.append(action_type == "human")
        self.human_actions.append(real_action if action_type == "human" else None)
        self.ep_return += reward
        self.ep_len += 1
        if action_type == "policy":
            self.policy_steps += 1
        else:
            self.human_steps += 1
            self.had_intervention = True


@dataclass
class TrainingStats:
    sample_time: deque = field(default_factory=lambda: deque(maxlen=10))
    update_time: deque = field(default_factory=lambda: deque(maxlen=10))
    sample_count: int = 0
    update_count: int = 0
    ep_successes: deque = field(default_factory=lambda: deque(maxlen=10))
    ep_count: int = 0
    intervention_count: int = 0
    total_intervention_transitions: int = 0

    def on_episode_done(self, ep: EpisodeState, success: bool, metrics: dict):
        self.ep_count += 1
        self.ep_successes.append(float(success))
        if ep.had_intervention:
            self.intervention_count += 1
        self.total_intervention_transitions += ep.human_steps

        total_actions = ep.policy_steps + ep.human_steps
        intervention_rate = float(ep.human_steps) / total_actions if total_actions else 0.0
        metrics.update({
            "training/return": float(ep.ep_return),
            "training/length": float(ep.ep_len),
            "training/success": float(success),
            "training/intervention_rate": intervention_rate,
            "training/episodes_with_intervention": float(self.intervention_count),
            "training/total_intervention_transitions": float(self.total_intervention_transitions),
        })
        if len(self.ep_successes) == self.ep_successes.maxlen:
            metrics["training/success_rate"] = float(np.mean(self.ep_successes))

    def record_sample_time(self, duration: float, metrics: dict):
        self.sample_time.append(duration)
        self.sample_count += 1
        if self.sample_count % 10 == 0 and len(self.sample_time) == self.sample_time.maxlen:
            metrics["training/sample_time_avg_ms"] = float(np.mean(self.sample_time)) * 1000.0

    def record_update_time(self, duration: float, metrics: dict):
        self.update_time.append(duration)
        self.update_count += 1
        if self.update_count % 10 == 0 and len(self.update_time) == self.update_time.maxlen:
            metrics["training/update_time_avg_ms"] = float(np.mean(self.update_time)) * 1000.0
