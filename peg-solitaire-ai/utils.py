"""Shared helpers for logging, plotting, and file management."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


TRAINING_LOG_DIR = "training_logs"
SOLUTION_DIR = "saved_solutions"


def ensure_project_dirs() -> None:
    """Create output directories used by training and replay."""
    os.makedirs(TRAINING_LOG_DIR, exist_ok=True)
    os.makedirs(SOLUTION_DIR, exist_ok=True)


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def move_to_json(move: tuple[int, int, int, int]) -> List[List[int]]:
    from_row, from_col, to_row, to_col = move
    return [[from_row, from_col], [to_row, to_col]]


def move_from_json(move: List[List[int]]) -> tuple[int, int, int, int]:
    return (move[0][0], move[0][1], move[1][0], move[1][1])


def save_solution(
    episode: int,
    moves: Iterable[tuple[int, int, int, int]],
    final_reward: float,
    epsilon: float,
    total_episodes: int,
    remaining_pegs: int = 1,
) -> str:
    """Save an exact successful move sequence as JSON."""
    ensure_project_dirs()
    move_list = list(moves)
    filename = f"solution_ep{episode}_{timestamp_string()}.json"
    path = os.path.join(SOLUTION_DIR, filename)

    data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "episode": episode,
        "total_episodes_trained": total_episodes,
        "epsilon": epsilon,
        "remaining_pegs": remaining_pegs,
        "moves": [move_to_json(move) for move in move_list],
        "total_moves": len(move_list),
        "final_reward": final_reward,
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return path


def save_training_log(history: Dict[str, list]) -> str:
    """Save raw training metrics to JSON for later inspection."""
    ensure_project_dirs()
    path = os.path.join(TRAINING_LOG_DIR, f"training_log_{timestamp_string()}.json")

    with open(path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    return path


def moving_average(values: List[float], window: int = 100) -> np.ndarray:
    """Return a same-length moving average for smoother plots."""
    if not values:
        return np.array([])

    window = max(1, min(window, len(values)))
    weights = np.ones(window) / window
    averaged = np.convolve(np.array(values, dtype=float), weights, mode="valid")

    if len(averaged) == len(values):
        return averaged

    padding = np.full(len(values) - len(averaged), averaged[0])
    return np.concatenate([padding, averaged])


def plot_training_graphs(history: Dict[str, list], window: int = 100) -> str:
    """Generate matplotlib graphs for the training run."""
    ensure_project_dirs()
    episodes = history["episodes"]
    solved_percent = history["solved_percent"]
    remaining = history["remaining_pegs"]
    rewards = history["rewards"]
    best_remaining = history["best_remaining"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Peg Solitaire Q-Learning Training Progress")

    axes[0, 0].plot(episodes, moving_average(solved_percent, window))
    axes[0, 0].set_title("Solved Percentage")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Solved %")

    axes[0, 1].plot(episodes, moving_average(remaining, window))
    axes[0, 1].set_title("Average Remaining Pegs")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Pegs")

    axes[1, 0].plot(episodes, moving_average(rewards, window))
    axes[1, 0].set_title("Cumulative Reward")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Reward")

    axes[1, 1].plot(episodes, best_remaining)
    axes[1, 1].set_title("Training Improvement")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Best Remaining Pegs")

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(TRAINING_LOG_DIR, f"training_graphs_{timestamp_string()}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def list_solution_files() -> List[str]:
    """Return saved solution JSON files newest first."""
    ensure_project_dirs()
    paths = [
        os.path.join(SOLUTION_DIR, name)
        for name in os.listdir(SOLUTION_DIR)
        if name.lower().endswith(".json")
    ]
    return sorted(paths, key=os.path.getmtime, reverse=True)
