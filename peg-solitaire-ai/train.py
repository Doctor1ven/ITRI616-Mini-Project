"""Fast terminal training for the Peg Solitaire Q-learning agent."""

from __future__ import annotations

import argparse
import random
from collections import deque
from typing import Dict, List, Optional

from agent import QLearningAgent
from game import Move, PegSolitaireGame
from solver import choose_endgame_move
from utils import (
    ensure_project_dirs,
    plot_training_graphs,
    save_solution,
    save_training_log,
)


def run_episode(
    game: PegSolitaireGame,
    agent: QLearningAgent,
    episode: int,
    total_episodes: int,
    max_steps: int = 80,
    train: bool = True,
    start_board=None,
    log_solution: bool = True,
    endgame_solver_rate: float = 0.35,
) -> dict:
    """Run one episode and optionally update the Q-table."""
    if start_board is None:
        state = game.reset()
    else:
        state = game.set_board(start_board)

    moves: List[Move] = []
    total_reward = 0

    for _ in range(max_steps):
        valid_moves = game.get_valid_moves()
        action = None

        if train and game.get_remaining_pegs() <= 12:
            if random.random() < endgame_solver_rate:
                action = choose_endgame_move(game, max_pegs=12)

        if action is None:
            action = agent.choose_action(state, valid_moves, explore=train)

        if action is None:
            break

        next_state, reward, done, info = game.step(action)
        next_valid_moves = game.get_valid_moves()

        if train:
            agent.update_q_table(state, action, reward, next_state, next_valid_moves)

        moves.append(action)
        total_reward += reward
        state = next_state

        if done:
            break

    solved = game.is_solved()
    if train:
        agent.decay_epsilon()

    if solved and log_solution:
        save_solution(
            episode=episode,
            moves=moves,
            final_reward=total_reward,
            epsilon=agent.epsilon,
            total_episodes=total_episodes,
            remaining_pegs=game.get_remaining_pegs(),
        )

    return {
        "solved": solved,
        "remaining_pegs": game.get_remaining_pegs(),
        "reward": total_reward,
        "moves": moves,
    }


def make_curriculum_board(game: PegSolitaireGame, depth: int):
    """
    Build a guaranteed-solvable practice board.

    The board starts from the solved one-peg centre state. Random reverse moves
    add pegs back onto the board. The resulting state is easier than the full
    puzzle but still teaches the same action values because every reverse move
    can be undone by a normal legal forward move.
    """
    game.reset_to_solved()

    for _ in range(depth):
        reverse_moves = game.get_reverse_moves()
        if not reverse_moves:
            break
        game.reverse_step(random.choice(reverse_moves))

    return game.clone_board()


def curriculum_depth_for_episode(episode: int, total_episodes: int) -> int:
    """
    Gradually increase curriculum difficulty.

    Depth 4 means tiny endgames. Depth 31 is close to a full 32-peg board.
    Growing the depth teaches the agent backwards from solved endings toward
    full-game positions.
    """
    progress = episode / max(1, total_episodes)
    return max(4, min(31, int(4 + progress * 27)))


def train_terminal(
    episodes: int = 10000,
    progress_interval: int = 500,
    qtable_path: str = "qtable.pkl",
    load_existing: bool = True,
    use_curriculum: bool = True,
    curriculum_rate: float = 0.35,
    endgame_solver_rate: float = 0.35,
) -> Dict[str, list]:
    """Train rapidly in the terminal without GUI rendering."""
    ensure_project_dirs()

    game = PegSolitaireGame()
    agent = QLearningAgent()
    if load_existing:
        agent.load_qtable(qtable_path)

    recent_remaining: deque[int] = deque(maxlen=progress_interval)
    recent_rewards: deque[float] = deque(maxlen=progress_interval)
    recent_solved: deque[int] = deque(maxlen=progress_interval)

    solved_count = 0
    full_board_episode_count = 0
    curriculum_solved_count = 0
    best_remaining = 32
    best_solution: Optional[List[Move]] = None

    history = {
        "episodes": [],
        "solved_percent": [],
        "remaining_pegs": [],
        "rewards": [],
        "best_remaining": [],
        "epsilon": [],
    }

    for episode in range(1, episodes + 1):
        use_practice_board = use_curriculum and random.random() < curriculum_rate
        start_board = None

        if use_practice_board:
            depth = curriculum_depth_for_episode(episode, episodes)
            start_board = make_curriculum_board(game, depth)

        result = run_episode(
            game,
            agent,
            episode,
            episodes,
            start_board=start_board,
            log_solution=not use_practice_board,
            endgame_solver_rate=endgame_solver_rate,
        )

        if use_practice_board:
            curriculum_solved_count += int(result["solved"])
        else:
            full_board_episode_count += 1
            solved_count += int(result["solved"])

        remaining = int(result["remaining_pegs"])
        reward = float(result["reward"])
        recent_remaining.append(remaining)
        recent_rewards.append(reward)
        recent_solved.append(int(result["solved"]))

        if not use_practice_board and remaining < best_remaining:
            best_remaining = remaining
            best_solution = list(result["moves"])

        history["episodes"].append(episode)
        solved_percent = 0.0
        if full_board_episode_count > 0:
            solved_percent = (solved_count / full_board_episode_count) * 100
        history["solved_percent"].append(solved_percent)
        history["remaining_pegs"].append(remaining)
        history["rewards"].append(reward)
        history["best_remaining"].append(best_remaining)
        history["epsilon"].append(agent.epsilon)

        if episode % progress_interval == 0 or episode == 1:
            avg_remaining = sum(recent_remaining) / len(recent_remaining)
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            recent_solved_count = sum(recent_solved)
            print(f"Episode: {episode}")
            print(f"Full-board solved: {solved_count}")
            print(f"Curriculum solved: {curriculum_solved_count}")
            print(f"Solved in recent window: {recent_solved_count}")
            print(f"Average Remaining Pegs: {avg_remaining:.2f}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print("-" * 36)

    agent.save_qtable(qtable_path)
    log_path = save_training_log(history)
    graph_path = plot_training_graphs(history)

    print("Training complete.")
    print(f"Q-table saved to: {qtable_path}")
    print(f"Training log saved to: {log_path}")
    print(f"Graphs saved to: {graph_path}")
    if best_solution is not None:
        print(f"Best remaining pegs: {best_remaining}")
        print(f"Best solution length so far: {len(best_solution)} moves")

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Peg Solitaire Q-learning AI.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--progress", type=int, default=500)
    parser.add_argument("--qtable", default="qtable.pkl")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing qtable.pkl")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable reverse curriculum")
    parser.add_argument("--curriculum-rate", type=float, default=0.35)
    parser.add_argument("--endgame-solver-rate", type=float, default=0.35)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_terminal(
        episodes=args.episodes,
        progress_interval=args.progress,
        qtable_path=args.qtable,
        load_existing=not args.fresh,
        use_curriculum=not args.no_curriculum,
        curriculum_rate=args.curriculum_rate,
        endgame_solver_rate=args.endgame_solver_rate,
    )
