"""Gomoku rules, tensors, and perspective."""

import numpy as np
import pytest

from bgtl.games.gomoku import Gomoku


def test_state_and_action_shapes_at_9_and_15():
    for size in (9, 15):
        game = Gomoku(size)
        assert game.state_tensor().shape == (4, size, size)
        assert game.legal_mask().shape == (1, size, size)
        assert game.legal_mask().sum() == size * size
        assert game.state_tensor().dtype == np.float32


def test_perspective_flips_to_the_side_to_move():
    game = Gomoku(9)
    game.apply(0)
    state = game.state_tensor()
    assert state[0].sum() == 0
    assert state[1, 0, 0] == 1
    assert state[2].min() == 1
    assert state[3, 0, 0] == 1
    assert game.side == 2


def test_illegal_move_is_rejected():
    game = Gomoku(9)
    game.apply(4)
    with pytest.raises(ValueError):
        game.apply(4)


def test_five_in_a_row_ends_the_game_for_the_player_about_to_move():
    game = Gomoku(9, win_length=5)
    for col in range(5):
        game.apply(col)
        if col < 4:
            game.apply(9 + col)
    assert game.terminal()
    assert game.winner == 1
    assert game.outcome() == -1.0


def test_full_board_without_a_line_is_a_draw():
    game = Gomoku(board_size=3, win_length=5)
    for action in range(9):
        game.apply(action)
    assert game.terminal()
    assert game.winner == 0
    assert game.outcome() == 0.0


def test_diagonal_line_wins():
    game = Gomoku(5, win_length=3)
    # (0,0), (1,1), (2,2) for player 1, with player 2 answering off the diagonal.
    game.apply(0)
    game.apply(1)
    game.apply(6)
    game.apply(2)
    game.apply(12)
    assert game.terminal()
    assert game.winner == 1
