"""Self-play that batches network evaluations across simultaneous games."""

from __future__ import annotations

import numpy as np
import torch

from bgtl.games.gomoku import Gomoku
from bgtl.mcts.puct import run_mcts
from bgtl.train.replay import Sample


def self_play(
    model: torch.nn.Module,
    board_size: int,
    win_length: int,
    n_games: int,
    n_sims: int,
    device: torch.device,
    rng: np.random.Generator,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
) -> list[Sample]:
    games = [Gomoku(board_size, win_length) for _ in range(n_games)]
    records: list[list[tuple[np.ndarray, np.ndarray, int, np.ndarray]]] = [[] for _ in range(n_games)]
    pending = set(range(n_games))
    while pending:
        active = sorted(pending)
        policies = run_mcts(
            [games[index] for index in active],
            model,
            n_sims=n_sims,
            device=device,
            rng=rng,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            add_noise=True,
        )
        for index, policy in zip(active, policies):
            game = games[index]
            records[index].append(
                (
                    game.state_tensor().copy(),
                    policy.astype(np.float32, copy=True),
                    game.side,
                    game.legal_mask().copy(),
                )
            )
            flat = np.clip(policy.reshape(-1).astype(np.float64), 0.0, None)
            total = float(flat.sum())
            if total <= 0.0:
                legal = np.flatnonzero(game.legal_mask().reshape(-1))
                flat = np.zeros_like(flat)
                flat[legal] = 1.0 / legal.size
            else:
                flat /= total
            action = int(rng.choice(flat.size, p=flat))
            game.apply(action)
            if game.terminal():
                pending.remove(index)

    samples: list[Sample] = []
    for game, history in zip(games, records):
        for state, policy, side, legal_mask in history:
            if game.winner == 0:
                value = 0.0
            elif side == game.winner:
                value = 1.0
            else:
                value = -1.0
            samples.append(Sample(state=state, policy=policy, value=value, legal_mask=legal_mask))
    return samples
