"""Small depth-first endgame solver for Peg Solitaire.

The full English board is too large for brute force search from the start.
However, once the game reaches a small number of pegs, exact search becomes
practical and useful. This module is used as a late-game lookahead helper so
the AI does not follow Q-values into obvious isolated-peg dead ends.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from game import Move, PegSolitaireGame


BoardState = Tuple[int, ...]


def board_state(board: np.ndarray) -> BoardState:
    return tuple(int(value) for value in board.flatten())


def count_pegs(board: np.ndarray) -> int:
    return int(np.sum(board == PegSolitaireGame.PEG))


def is_valid_position(row: int, col: int) -> bool:
    if row < 0 or row >= 7 or col < 0 or col >= 7:
        return False
    return (2 <= row <= 4) or (2 <= col <= 4)


def get_valid_moves_for_board(board: np.ndarray) -> List[Move]:
    moves: List[Move] = []

    for from_row in range(7):
        for from_col in range(7):
            if board[from_row, from_col] != PegSolitaireGame.PEG:
                continue

            for row_step, col_step in PegSolitaireGame.DIRECTIONS:
                jumped_row = from_row + row_step
                jumped_col = from_col + col_step
                to_row = from_row + (2 * row_step)
                to_col = from_col + (2 * col_step)

                if not is_valid_position(to_row, to_col):
                    continue

                if (
                    board[jumped_row, jumped_col] == PegSolitaireGame.PEG
                    and board[to_row, to_col] == PegSolitaireGame.EMPTY
                ):
                    moves.append((from_row, from_col, to_row, to_col))

    return moves


def apply_move_to_board(board: np.ndarray, move: Move) -> np.ndarray:
    new_board = board.copy()
    from_row, from_col, to_row, to_col = move
    jumped_row = (from_row + to_row) // 2
    jumped_col = (from_col + to_col) // 2

    new_board[from_row, from_col] = PegSolitaireGame.EMPTY
    new_board[jumped_row, jumped_col] = PegSolitaireGame.EMPTY
    new_board[to_row, to_col] = PegSolitaireGame.PEG
    return new_board


def move_sort_score(move: Move) -> float:
    """Prefer moves that land closer to the centre during endgame search."""
    _, _, to_row, to_col = move
    return abs(to_row - 3) + abs(to_col - 3)


def find_endgame_solution(
    board: np.ndarray,
    max_pegs: int = 12,
    memo: Optional[Dict[BoardState, Optional[List[Move]]]] = None,
) -> Optional[List[Move]]:
    """
    Return a move sequence to one peg, or None if this branch cannot solve.

    The search is intentionally only allowed for small peg counts. This prevents
    accidental brute forcing of the full puzzle while still solving the exact
    kind of 2-12 peg endings that Q-learning struggles with.
    """
    peg_count = count_pegs(board)
    if peg_count == 1:
        return []
    if peg_count > max_pegs:
        return None

    if memo is None:
        memo = {}

    state = board_state(board)
    if state in memo:
        return memo[state]

    moves = sorted(get_valid_moves_for_board(board), key=move_sort_score)
    if not moves:
        memo[state] = None
        return None

    for move in moves:
        next_board = apply_move_to_board(board, move)
        tail = find_endgame_solution(next_board, max_pegs=max_pegs, memo=memo)
        if tail is not None:
            solution = [move] + tail
            memo[state] = solution
            return solution

    memo[state] = None
    return None


def choose_endgame_move(game: PegSolitaireGame, max_pegs: int = 12) -> Optional[Move]:
    """Return the first move of an exact endgame solution when one exists."""
    solution = find_endgame_solution(game.clone_board(), max_pegs=max_pegs)
    if solution:
        return solution[0]
    return None
