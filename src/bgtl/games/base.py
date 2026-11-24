"""Shared game surface used by search, training, and evaluation."""

from typing import Protocol

import numpy as np


class Game(Protocol):
    """Two-player zero-sum game with convolutional state and action tensors.

    ``outcome`` is the terminal value in ``[-1, 1]`` for the player who would
    be to move. A loss for that player is ``-1``, a draw is ``0``.
    """

    board_size: int
    state_channels: tuple[str, ...]
    action_channels: tuple[str, ...]

    def clone(self) -> "Game":
        """Return an independent copy of the position."""

    def state_tensor(self) -> np.ndarray:
        """Return the state as ``(C, H, W)`` float32 from the side to move."""

    def legal_mask(self) -> np.ndarray:
        """Return a ``(C_action, H, W)`` float32 mask, 1 where the action is legal."""

    def apply(self, action: int) -> None:
        """Play a flattened action index. Illegal actions raise ``ValueError``."""

    def terminal(self) -> bool:
        """Whether the game has ended."""

    def outcome(self) -> float:
        """Terminal value for the side to move. Raises if the game is ongoing."""
