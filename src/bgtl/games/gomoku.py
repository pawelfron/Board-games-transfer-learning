"""Size-parameterized Gomoku (free placement, k-in-a-row, draws allowed)."""

from __future__ import annotations

import numpy as np


class Gomoku:
    """Gomoku on an ``n x n`` board.

    State channels, always from the side to move:

    * ``current`` — stones of the player to move
    * ``opponent`` — stones of the other player
    * ``to_move`` — a plane of ones
    * ``last_move`` — the cell of the previous move

    The action tensor has one channel, ``place``. A flat action index is
    ``row * board_size + col``.
    """

    game_id = "gomoku"
    state_channels = ("current", "opponent", "to_move", "last_move")
    action_channels = ("place",)

    def __init__(self, board_size: int = 15, win_length: int = 5) -> None:
        if board_size < 1:
            raise ValueError("board_size must be positive")
        if win_length < 1:
            raise ValueError("win_length must be positive")
        self.board_size = int(board_size)
        self.win_length = int(win_length)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.side = 1
        self.last: tuple[int, int] | None = None
        self.winner = 0
        self.over = False

    def clone(self) -> Gomoku:
        other = Gomoku(self.board_size, self.win_length)
        other.board = self.board.copy()
        other.side = self.side
        other.last = self.last
        other.winner = self.winner
        other.over = self.over
        return other

    def state_tensor(self) -> np.ndarray:
        current = (self.board == self.side).astype(np.float32)
        opponent = (self.board == (3 - self.side)).astype(np.float32)
        to_move = np.ones_like(current)
        last = np.zeros_like(current)
        if self.last is not None:
            last[self.last] = 1.0
        return np.stack([current, opponent, to_move, last], axis=0)

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros((1, self.board_size, self.board_size), dtype=np.float32)
        if not self.over:
            mask[0] = (self.board == 0).astype(np.float32)
        return mask

    def apply(self, action: int) -> None:
        if self.over:
            raise RuntimeError("game is over")
        action = int(action)
        n = self.board_size
        if action < 0 or action >= n * n:
            raise ValueError(f"action {action} is outside the board")
        row, col = divmod(action, n)
        if self.board[row, col] != 0:
            raise ValueError(f"illegal move at {(row, col)}")
        player = self.side
        self.board[row, col] = player
        self.last = (row, col)
        won = self._has_line(row, col, player)
        full = not bool((self.board == 0).any())
        self.side = 3 - player
        if won:
            self.winner = player
            self.over = True
        elif full:
            self.winner = 0
            self.over = True

    def terminal(self) -> bool:
        return self.over

    def outcome(self) -> float:
        """Value for the side to move. The player who just won leaves the opponent to move."""
        if not self.over:
            raise RuntimeError("outcome is only defined for a terminal position")
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == self.side else -1.0

    def _has_line(self, row: int, col: int, player: int) -> bool:
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        n = self.board_size
        for d_row, d_col in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * d_row, col + sign * d_col
                while 0 <= r < n and 0 <= c < n and self.board[r, c] == player:
                    count += 1
                    r += sign * d_row
                    c += sign * d_col
            if count >= self.win_length:
                return True
        return False
