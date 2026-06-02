"""Tkinter visualisation tools for training and watching the AI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import Optional

from agent import QLearningAgent
from game import Move, PegSolitaireGame
from solver import choose_endgame_move
from train import run_episode
from utils import ensure_project_dirs, save_solution


class BoardCanvas(tk.Canvas):
    """Reusable Peg Solitaire board renderer."""

    def __init__(self, master: tk.Misc, cell_size: int = 62) -> None:
        super().__init__(
            master,
            width=7 * cell_size,
            height=7 * cell_size,
            bg="#b7b7b7",
            highlightthickness=0,
        )
        self.cell_size = cell_size

    def draw_board(self, game: PegSolitaireGame, highlight: Optional[Move] = None) -> None:
        """Redraw the whole board. Simple full redraws are reliable and clear."""
        self.delete("all")
        board = game.clone_board()

        for row in range(7):
            for col in range(7):
                value = int(board[row, col])
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if value == PegSolitaireGame.INVALID:
                    self.create_rectangle(x1, y1, x2, y2, fill="#8f8f8f", outline="#8f8f8f")
                    continue

                self.create_rectangle(x1, y1, x2, y2, fill="#d9d9d9", outline="#c0c0c0")
                margin = 11
                fill = "white" if value == PegSolitaireGame.EMPTY else "#0b2f6b"
                outline = "#222222" if value == PegSolitaireGame.EMPTY else "#061d42"

                if highlight and (row, col) in (
                    (highlight[0], highlight[1]),
                    (highlight[2], highlight[3]),
                ):
                    outline = "#d68b00"

                self.create_oval(
                    x1 + margin,
                    y1 + margin,
                    x2 - margin,
                    y2 - margin,
                    fill=fill,
                    outline=outline,
                    width=3,
                )


class VisualTrainingWindow(tk.Toplevel):
    """Slow visual training loop for demonstrations and debugging."""

    def __init__(self, master: tk.Misc, episodes: int = 1000) -> None:
        super().__init__(master)
        self.title("Peg Solitaire AI - Visual Training")
        self.resizable(False, False)

        ensure_project_dirs()
        self.game = PegSolitaireGame()
        self.agent = QLearningAgent()
        self.agent.load_qtable("qtable.pkl")
        self.episodes = episodes
        self.episode = 1
        self.step_count = 0
        self.total_reward = 0
        self.episode_moves: list[Move] = []
        self.solved_count = 0
        self.running = True
        self.state = self.game.reset()

        self.board = BoardCanvas(self)
        self.board.grid(row=0, column=0, rowspan=2, padx=12, pady=12)

        panel = tk.Frame(self)
        panel.grid(row=0, column=1, sticky="n", padx=12, pady=12)

        self.info = tk.StringVar()
        tk.Label(panel, textvariable=self.info, justify="left", font=("Segoe UI", 11)).pack(anchor="w")

        tk.Label(panel, text="Training speed").pack(anchor="w", pady=(16, 0))
        self.speed = tk.Scale(panel, from_=20, to=1000, orient="horizontal", length=220)
        self.speed.set(160)
        self.speed.pack(anchor="w")

        tk.Button(panel, text="Pause / Resume", command=self.toggle_running, width=18).pack(pady=8)
        tk.Button(panel, text="Save Q-table", command=self.save_agent, width=18).pack(pady=4)

        self.board.draw_board(self.game)
        self.update_info()
        self.after(200, self.training_step)

    def toggle_running(self) -> None:
        self.running = not self.running

    def save_agent(self) -> None:
        self.agent.save_qtable("qtable.pkl")
        messagebox.showinfo("Saved", "Q-table saved to qtable.pkl")

    def update_info(self) -> None:
        self.info.set(
            f"Episode: {self.episode}/{self.episodes}\n"
            f"Remaining pegs: {self.game.get_remaining_pegs()}\n"
            f"Epsilon: {self.agent.epsilon:.4f}\n"
            f"Solved count: {self.solved_count}\n"
            f"Reward: {self.total_reward:.1f}"
        )

    def start_next_episode(self) -> None:
        self.episode += 1
        self.step_count = 0
        self.total_reward = 0
        self.episode_moves = []
        self.state = self.game.reset()

    def training_step(self) -> None:
        if self.running and self.episode <= self.episodes:
            valid_moves = self.game.get_valid_moves()
            action = self.agent.choose_action(self.state, valid_moves, explore=True)

            if action is None:
                self.agent.decay_epsilon()
                self.start_next_episode()
            else:
                next_state, reward, done, info = self.game.step(action)
                self.agent.update_q_table(
                    self.state,
                    action,
                    reward,
                    next_state,
                    self.game.get_valid_moves(),
                )
                self.state = next_state
                self.total_reward += reward
                self.step_count += 1
                self.episode_moves.append(action)

                self.board.draw_board(self.game, highlight=action)

                if done:
                    if info["solved"]:
                        self.solved_count += 1
                        save_solution(
                            episode=self.episode,
                            moves=self.episode_moves,
                            final_reward=self.total_reward,
                            epsilon=self.agent.epsilon,
                            total_episodes=self.episodes,
                            remaining_pegs=self.game.get_remaining_pegs(),
                        )
                    self.agent.decay_epsilon()
                    self.agent.save_qtable("qtable.pkl")
                    self.after(self.speed.get(), self.start_next_episode)

            self.update_info()

        self.after(self.speed.get(), self.training_step)


class WatchAIWindow(tk.Toplevel):
    """Load a trained Q-table and watch greedy AI play."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title("Peg Solitaire AI - Watch AI Play")
        self.resizable(False, False)

        self.game = PegSolitaireGame()
        self.agent = QLearningAgent(epsilon=0.0)
        loaded = self.agent.load_qtable("qtable.pkl")
        self.agent.epsilon = 0.0
        self.running = loaded
        self.moves_played: list[Move] = []
        self.total_reward = 0

        self.board = BoardCanvas(self)
        self.board.grid(row=0, column=0, padx=12, pady=12)

        panel = tk.Frame(self)
        panel.grid(row=0, column=1, sticky="n", padx=12, pady=12)

        self.info = tk.StringVar()
        tk.Label(panel, textvariable=self.info, justify="left", font=("Segoe UI", 11)).pack(anchor="w")
        tk.Label(panel, text="Move speed").pack(anchor="w", pady=(16, 0))
        self.speed = tk.Scale(panel, from_=80, to=1500, orient="horizontal", length=220)
        self.speed.set(400)
        self.speed.pack(anchor="w")
        tk.Button(panel, text="Restart", command=self.restart, width=18).pack(pady=8)

        if not loaded:
            messagebox.showwarning("No Q-table", "qtable.pkl was not found. Train the AI first.")

        self.board.draw_board(self.game)
        self.update_info()
        self.after(500, self.ai_step)

    def restart(self) -> None:
        self.game.reset()
        self.running = True
        self.moves_played = []
        self.total_reward = 0
        self.board.draw_board(self.game)
        self.update_info()

    def update_info(self) -> None:
        status = "Solved" if self.game.is_solved() else "Playing"
        if self.game.is_game_over() and not self.game.is_solved():
            status = "Dead end"
        self.info.set(
            f"Status: {status}\n"
            f"Remaining pegs: {self.game.get_remaining_pegs()}\n"
            f"Valid moves: {len(self.game.get_valid_moves())}"
        )

    def ai_step(self) -> None:
        if self.running and not self.game.is_game_over():
            state = self.game.get_state()
            action = choose_endgame_move(self.game, max_pegs=12)
            if action is None:
                action = self.agent.choose_action(state, self.game.get_valid_moves(), explore=False)
            if action is None:
                self.running = False
            else:
                _, reward, done, info = self.game.step(action)
                self.moves_played.append(action)
                self.total_reward += reward
                self.board.draw_board(self.game, highlight=action)
                if done:
                    self.running = False
                    if info["solved"]:
                        save_solution(
                            episode=0,
                            moves=self.moves_played,
                            final_reward=self.total_reward,
                            epsilon=0.0,
                            total_episodes=0,
                            remaining_pegs=self.game.get_remaining_pegs(),
                        )

        self.update_info()
        self.after(self.speed.get(), self.ai_step)


class TerminalTrainingLauncher(tk.Toplevel):
    """Small helper window that starts terminal training from the GUI."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title("Train AI - Terminal")
        self.resizable(False, False)

        tk.Label(self, text="Episodes").grid(row=0, column=0, padx=12, pady=12, sticky="w")
        self.episodes = tk.Entry(self, width=12)
        self.episodes.insert(0, "10000")
        self.episodes.grid(row=0, column=1, padx=12, pady=12)

        tk.Button(self, text="Start", command=self.start_training, width=16).grid(
            row=1,
            column=0,
            columnspan=2,
            pady=(0, 12),
        )

    def start_training(self) -> None:
        episodes = int(self.episodes.get())
        self.destroy()
        run_episode  # Keeps imports explicit for educational reading.
        import train

        train.train_terminal(episodes=episodes)
