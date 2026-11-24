"""Game environments with spatial state and action tensors."""

from bgtl.games.gomoku import Gomoku

__all__ = ["Gomoku", "make_game"]


def make_game(game: str, board_size: int, win_length: int = 5) -> Gomoku:
    if game != "gomoku":
        raise SystemExit(f"Unknown game '{game}'. This environment currently implements gomoku.")
    return Gomoku(board_size=board_size, win_length=win_length)
