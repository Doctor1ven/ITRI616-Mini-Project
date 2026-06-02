"""
Peg Solitaire environment for a standard English cross board.

This file intentionally contains no reinforcement-learning code. It only knows
the rules of Peg Solitaire: board creation, valid moves, move execution, and
win/loss checks. Keeping the environment separate makes the project easier to
study and test.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


Move = Tuple[int, int, int, int]


class PegSolitaireGame:
    """Standard 33-hole English Peg Solitaire board."""

    PEG = 1
    EMPTY = 0
    INVALID = -1

    DIRECTIONS = (
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
    )

    def __init__(self) -> None:
        self.board = self._create_start_board()

    def _create_start_board(self) -> np.ndarray:
        """Create the English cross board with the centre hole empty."""
        board = np.full((7, 7), self.INVALID, dtype=np.int8)

        for row in range(7):
            for col in range(7):
                if self._is_valid_position(row, col):
                    board[row, col] = self.PEG

        board[3, 3] = self.EMPTY
        return board

    def _create_solved_board(self) -> np.ndarray:
        """Create a solved board with one peg in the centre hole."""
        board = np.full((7, 7), self.INVALID, dtype=np.int8)

        for row in range(7):
            for col in range(7):
                if self._is_valid_position(row, col):
                    board[row, col] = self.EMPTY

        board[3, 3] = self.PEG
        return board

    def _is_valid_position(self, row: int, col: int) -> bool:
        """Return True if a row/column is part of the 33-hole cross layout."""
        if row < 0 or row >= 7 or col < 0 or col >= 7:
            return False
        return (2 <= row <= 4) or (2 <= col <= 4)

    def reset(self) -> Tuple[int, ...]:
        """Reset the game and return the starting state."""
        self.board = self._create_start_board()
        return self.get_state()

    def reset_to_solved(self) -> Tuple[int, ...]:
        """Reset to the one-peg solved state used for reverse curriculum."""
        self.board = self._create_solved_board()
        return self.get_state()

    def set_board(self, board: np.ndarray) -> Tuple[int, ...]:
        """Set the board from a safe copy and return the new state."""
        self.board = board.copy()
        return self.get_state()

    def get_valid_moves(self) -> List[Move]:
        """Return every legal jump as (from_row, from_col, to_row, to_col)."""
        moves: List[Move] = []

        for from_row in range(7):
            for from_col in range(7):
                if self.board[from_row, from_col] != self.PEG:
                    continue

                for row_step, col_step in self.DIRECTIONS:
                    jumped_row = from_row + row_step
                    jumped_col = from_col + col_step
                    to_row = from_row + (2 * row_step)
                    to_col = from_col + (2 * col_step)

                    if not self._is_valid_position(to_row, to_col):
                        continue

                    if (
                        self.board[jumped_row, jumped_col] == self.PEG
                        and self.board[to_row, to_col] == self.EMPTY
                    ):
                        moves.append((from_row, from_col, to_row, to_col))

        return moves

    def get_reverse_moves(self) -> List[Move]:
        """
        Return moves that can be applied backwards.

        If a normal move is A -> C over B, then the reverse operation requires
        C to contain a peg while A and B are empty. Applying it creates a larger
        board position that is guaranteed to be solvable by the matching
        forward move.
        """
        moves: List[Move] = []

        for to_row in range(7):
            for to_col in range(7):
                if self.board[to_row, to_col] != self.PEG:
                    continue

                for row_step, col_step in self.DIRECTIONS:
                    jumped_row = to_row - row_step
                    jumped_col = to_col - col_step
                    from_row = to_row - (2 * row_step)
                    from_col = to_col - (2 * col_step)

                    if not self._is_valid_position(from_row, from_col):
                        continue

                    if (
                        self.board[jumped_row, jumped_col] == self.EMPTY
                        and self.board[from_row, from_col] == self.EMPTY
                    ):
                        moves.append((from_row, from_col, to_row, to_col))

        return moves

    def reverse_step(self, move: Move) -> bool:
        """
        Apply one reverse move for curriculum generation.

        The move is still represented in the normal forward format. After this
        reverse operation, applying that same move normally would undo it.
        """
        from_row, from_col, to_row, to_col = move
        jumped_row = (from_row + to_row) // 2
        jumped_col = (from_col + to_col) // 2

        if (
            not self._is_valid_position(from_row, from_col)
            or not self._is_valid_position(to_row, to_col)
            or self.board[from_row, from_col] != self.EMPTY
            or self.board[jumped_row, jumped_col] != self.EMPTY
            or self.board[to_row, to_col] != self.PEG
        ):
            return False

        self.board[from_row, from_col] = self.PEG
        self.board[jumped_row, jumped_col] = self.PEG
        self.board[to_row, to_col] = self.EMPTY
        return True

    def _position_scores(self, row: int, col: int) -> Tuple[float, float]:
        """
        Return (outside_score, inside_score) for a board coordinate.

        Outside cells are far from the centre. Inside cells are close to the
        centre. The two scores move in opposite directions and are normalised
        to roughly 0..1 so the reward bonus stays small and understandable.
        """
        distance_from_center = abs(row - 3) + abs(col - 3)
        outside_score = distance_from_center / 4.0
        inside_score = 1.0 - outside_score
        return outside_score, inside_score

    def _phase_weights(self, remaining_pegs: int) -> Tuple[float, float]:
        """
        Return (early_weight, late_weight) based on game progress.

        At the start, early_weight is high and outside clearing matters more.
        Near the end, late_weight is high and centralising the final pegs matters
        more. This directly implements the intended changing reward priority.
        """
        early_weight = max(0.0, min(1.0, (remaining_pegs - 1) / 31.0))
        late_weight = 1.0 - early_weight
        return early_weight, late_weight

    def _positional_reward(
        self,
        from_row: int,
        from_col: int,
        jumped_row: int,
        jumped_col: int,
        to_row: int,
        to_col: int,
        before_pegs: int,
    ) -> float:
        """
        Reward board-shaping behaviour that changes during the episode.

        Early game:
            - reward removing outer pegs
            - give a smaller reward for moving a peg inward

        Late game:
            - reward landing near the centre
            - penalise ending jumps on the outside
        """
        early_weight, late_weight = self._phase_weights(before_pegs)

        from_outside, _ = self._position_scores(from_row, from_col)
        jumped_outside, _ = self._position_scores(jumped_row, jumped_col)
        to_outside, to_inside = self._position_scores(to_row, to_col)

        inward_progress = from_outside - to_outside

        early_bonus = early_weight * (
            2.0 * jumped_outside
            + 1.0 * max(0.0, inward_progress)
            - 0.5 * max(0.0, -inward_progress)
        )
        late_bonus = late_weight * (
            2.5 * to_inside
            - 1.0 * to_outside
        )

        return early_bonus + late_bonus

    def _late_game_progress_bonus(self, remaining_pegs: int) -> int:
        """
        Reward reaching difficult low-peg milestones.

        The previous version punished low-peg dead ends too strongly, so the
        agent learned to stay around 4-5 pegs. These milestone rewards keep
        pressure on the real objective: push from 3 pegs to 2 and eventually 1.
        """
        if remaining_pegs <= 2:
            return 60
        if remaining_pegs <= 3:
            return 35
        if remaining_pegs <= 4:
            return 20
        if remaining_pegs <= 6:
            return 10
        return 0

    def step(self, move: Move) -> Tuple[Tuple[int, ...], float, bool, dict]:
        """
        Execute one move and return (state, reward, done, info).

        Reward shaping is kept explicit so students can see how each learning
        signal contributes to the total reward.
        """
        valid_moves = self.get_valid_moves()
        if move not in valid_moves:
            return self.get_state(), -10, self.is_game_over(), {
                "valid": False,
                "solved": False,
                "remaining_pegs": self.get_remaining_pegs(),
            }

        before_pegs = self.get_remaining_pegs()
        from_row, from_col, to_row, to_col = move
        jumped_row = (from_row + to_row) // 2
        jumped_col = (from_col + to_col) // 2

        self.board[from_row, from_col] = self.EMPTY
        self.board[jumped_row, jumped_col] = self.EMPTY
        self.board[to_row, to_col] = self.PEG

        after_pegs = self.get_remaining_pegs()
        reward = 1  # valid move
        reward += 5  # successful peg removal

        if after_pegs < before_pegs:
            reward += 2

        reward += self._positional_reward(
            from_row,
            from_col,
            jumped_row,
            jumped_col,
            to_row,
            to_col,
            before_pegs,
        )

        # A small efficiency bonus nudges learning toward low-peg states.
        reward += max(0, 32 - after_pegs) // 4
        reward += self._late_game_progress_bonus(after_pegs)

        solved = self.is_solved()
        game_over = self.is_game_over()

        if solved:
            reward += 500
        elif game_over:
            # Penalise dead ends, but do not punish 2-3 peg attempts so much
            # that the learner avoids the endgame entirely.
            reward -= 20 + (after_pegs * 4)

        return self.get_state(), reward, game_over, {
            "valid": True,
            "solved": solved,
            "remaining_pegs": after_pegs,
        }

    def is_game_over(self) -> bool:
        """The game is over when solved or when no legal jumps remain."""
        return self.is_solved() or len(self.get_valid_moves()) == 0

    def is_solved(self) -> bool:
        """Return True when exactly one peg remains."""
        return self.get_remaining_pegs() == 1

    def get_remaining_pegs(self) -> int:
        """Count pegs currently on valid board cells."""
        return int(np.sum(self.board == self.PEG))

    def get_state(self) -> Tuple[int, ...]:
        """Return a hashable flattened tuple suitable for a Q-table key."""
        return tuple(int(value) for value in self.board.flatten())

    def clone_board(self) -> np.ndarray:
        """Return a copy of the board for display code."""
        return self.board.copy()
