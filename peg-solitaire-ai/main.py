"""Main menu for the Peg Solitaire AI project."""

from __future__ import annotations

import tkinter as tk

from gui import TerminalTrainingLauncher, VisualTrainingWindow, WatchAIWindow
from replay import ReplayWindow
from utils import ensure_project_dirs


class MainMenu(tk.Tk):
    """Simple Tkinter launcher for all project modes."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Peg Solitaire AI")
        self.resizable(False, False)
        ensure_project_dirs()

        frame = tk.Frame(self, padx=28, pady=24)
        frame.pack()

        tk.Label(frame, text="Peg Solitaire AI", font=("Segoe UI", 18, "bold")).pack(pady=(0, 18))

        buttons = [
            ("Train AI (Terminal)", self.open_terminal_training),
            ("Train AI (Visual)", self.open_visual_training),
            ("Watch AI Play", self.open_watch_ai),
            ("Replay Saved Solution", self.open_replay),
            ("Exit", self.destroy),
        ]

        for label, command in buttons:
            tk.Button(frame, text=label, command=command, width=26, height=2).pack(pady=5)

    def open_terminal_training(self) -> None:
        TerminalTrainingLauncher(self)

    def open_visual_training(self) -> None:
        VisualTrainingWindow(self)

    def open_watch_ai(self) -> None:
        WatchAIWindow(self)

    def open_replay(self) -> None:
        ReplayWindow(self)


if __name__ == "__main__":
    MainMenu().mainloop()
