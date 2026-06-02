"""
Tabular Q-learning agent for Peg Solitaire.

No external RL frameworks are used. The Q-table is a normal Python dictionary:

    q_table[state][action] = expected future value

Both states and actions are tuples, so they are hashable and can be used as
dictionary keys.
"""

from __future__ import annotations

import os
import pickle
import random
from typing import Dict, Iterable, Optional, Tuple


State = Tuple[int, ...]
Action = Tuple[int, int, int, int]
QTable = Dict[State, Dict[Action, float]]


class QLearningAgent:
    """A small, educational Q-learning implementation."""

    def __init__(
        self,
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9995,
    ) -> None:
        self.q_table: QTable = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(
        self,
        state: State,
        valid_actions: Iterable[Action],
        explore: bool = True,
    ) -> Optional[Action]:
        """Choose an action using epsilon-greedy exploration."""
        actions = list(valid_actions)
        if not actions:
            return None

        if explore and random.random() < self.epsilon:
            return random.choice(actions)

        state_actions = self.q_table.get(state, {})
        return max(actions, key=lambda action: state_actions.get(action, 0.0))

    def update_q_table(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_valid_actions: Iterable[Action],
    ) -> None:
        """Apply the Bellman update equation."""
        self.q_table.setdefault(state, {})
        old_value = self.q_table[state].get(action, 0.0)

        future_values = self.q_table.get(next_state, {})
        next_actions = list(next_valid_actions)
        best_future_value = 0.0
        if next_actions:
            best_future_value = max(
                future_values.get(next_action, 0.0)
                for next_action in next_actions
            )

        target = reward + (self.discount_factor * best_future_value)
        new_value = old_value + self.learning_rate * (target - old_value)
        self.q_table[state][action] = new_value

    def decay_epsilon(self) -> None:
        """Reduce exploration over time while keeping a minimum amount."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_qtable(self, path: str = "qtable.pkl") -> None:
        """Save Q-table and agent parameters with pickle."""
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "epsilon": self.epsilon,
                    "learning_rate": self.learning_rate,
                    "discount_factor": self.discount_factor,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                },
                file,
            )

    def load_qtable(self, path: str = "qtable.pkl") -> bool:
        """Load a saved Q-table. Returns False when no file exists."""
        if not os.path.exists(path):
            return False

        with open(path, "rb") as file:
            data = pickle.load(file)

        if isinstance(data, dict) and "q_table" in data:
            self.q_table = data["q_table"]
            self.epsilon = data.get("epsilon", self.epsilon)
            self.learning_rate = data.get("learning_rate", self.learning_rate)
            self.discount_factor = data.get("discount_factor", self.discount_factor)
            self.epsilon_min = data.get("epsilon_min", self.epsilon_min)
            self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        else:
            # Backwards-compatible path if someone saved only the raw table.
            self.q_table = data

        return True
