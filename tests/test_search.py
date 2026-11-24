"""Loss, search, and the custom-trainer hook."""

import numpy as np
import pytest
import torch
from torch import nn

from bgtl.eval.match import wilson_interval
from bgtl.games.gomoku import Gomoku
from bgtl.mcts.puct import run_mcts
from bgtl.train.loop import train
from bgtl.train.loss import policy_value_loss


def test_loss_is_finite_and_masks_illegal_logits():
    logits = torch.zeros(2, 1, 3, 3, requires_grad=True)
    values = torch.zeros(2, 1, requires_grad=True)
    target_v = torch.tensor([1.0, -1.0])
    target_pi = torch.zeros(2, 1, 3, 3)
    target_pi[:, 0, 0, 0] = 1
    mask = torch.ones(2, 1, 3, 3)
    mask[:, 0, 0, 1] = 0
    loss, value_loss, policy_loss = policy_value_loss(values, logits, target_v, target_pi, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(policy_loss)


def test_custom_trainer_fails_before_any_training():
    with pytest.raises(SystemExit, match="not implemented"):
        train({"trainer": "custom"})


def test_wilson_interval_covers_the_observed_rate():
    low, high = wilson_interval(50, 100)
    assert 0 < low < 0.5 < high < 1
    assert wilson_interval(0, 0) == (0.0, 0.0)


class _ZeroNet(nn.Module):
    def forward(self, x: torch.Tensor):
        batch, _, height, width = x.shape
        return torch.zeros(batch, 1), torch.zeros(batch, 1, height, width)


def test_mcts_prefers_an_immediate_win():
    game = Gomoku(board_size=3, win_length=3)
    game.board = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]], dtype=np.int8)
    game.side = 1
    policy = run_mcts(
        [game],
        _ZeroNet(),
        n_sims=16,
        device=torch.device("cpu"),
        rng=np.random.default_rng(0),
        add_noise=False,
    )[0]
    assert int(policy.reshape(-1).argmax()) == 5
