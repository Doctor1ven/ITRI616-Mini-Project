"""Replay saved Peg Solitaire solution JSON files visually."""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox
from typing import List, Optional

from game import Move, PegSolitaireGame
from gui import BoardCanvas
from utils import list_solution_files, move_from_json


class ReplayWindow(tk.Toplevel):
    """Select and replay a saved successful solution."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title("Peg Solitaire AI - Replay Saved Solution")
        self.resizable(False, False)

        self.game = PegSolitaireGame()
        self.moves: List[Move] = []
        self.move_index = 0
        self.playing = False

        self.board = BoardCanvas(self)
        self.board.grid(row=0, column=0, rowspan=2, padx=12, pady=12)

        panel = tk.Frame(self)
        panel.grid(row=0, column=1, sticky="n", padx=12, pady=12)

        tk.Label(panel, text="Saved solutions").pack(anchor="w")
        self.solution_files = list_solution_files()
        self.listbox = tk.Listbox(panel, width=42, height=10)
        self.listbox.pack(pady=6)
        for path in self.solution_files:
            self.listbox.insert(tk.END, path)

        self.info = tk.StringVar(value="Select a solution to replay.")
        tk.Label(panel, textvariable=self.info, justify="left").pack(anchor="w", pady=8)

        tk.Label(panel, text="Replay speed").pack(anchor="w")
        self.speed = tk.Scale(panel, from_=100, to=2000, orient="horizontal", length=260)
        self.speed.set(500)
        self.speed.pack(anchor="w")

        tk.Button(panel, text="Load Selected", command=self.load_selected, width=18).pack(pady=4)
        tk.Button(panel, text="Play / Pause", command=self.toggle_playing, width=18).pack(pady=4)
        tk.Button(panel, text="Restart", command=self.restart, width=18).pack(pady=4)

        self.board.draw_board(self.game)
        if not self.solution_files:
            messagebox.showinfo("No solutions", "No JSON solution files were found.")

        self.after(500, self.replay_step)

    def load_selected(self) -> None:
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("No selection", "Choose a saved solution first.")
            return

        path = self.solution_files[selection[0]]
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.moves = [move_from_json(move) for move in data["moves"]]
        self.move_index = 0
        self.game.reset()
        self.playing = False
        self.info.set(
            f"Episode: {data.get('episode')}\n"
            f"Total moves: {data.get('total_moves')}\n"
            f"Move: 0/{len(self.moves)}"
        )
        self.board.draw_board(self.game)

    def toggle_playing(self) -> None:
        self.playing = not self.playing

    def restart(self) -> None:
        self.move_index = 0
        self.game.reset()
        self.board.draw_board(self.game)
        self.info.set(f"Move: 0/{len(self.moves)}")

    def replay_step(self) -> None:
        if self.playing and self.move_index < len(self.moves):
            move = self.moves[self.move_index]
            self.game.step(move)
            self.move_index += 1
            self.board.draw_board(self.game, highlight=move)
            self.info.set(
                f"Move: {self.move_index}/{len(self.moves)}\n"
                f"Remaining pegs: {self.game.get_remaining_pegs()}"
            )

            if self.move_index >= len(self.moves):
                self.playing = False

        self.after(self.speed.get(), self.replay_step)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    ReplayWindow(root)
    root.mainloop()
