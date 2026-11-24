"""Color-balanced matches between two MCTS agents."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from bgtl.checkpoint import load_checkpoint, model_from_checkpoint
from bgtl.device import require_cuda
from bgtl.games.gomoku import Gomoku
from bgtl.mcts.puct import run_mcts


def evaluate(config: dict) -> dict:
    section = config["eval"]
    device = require_cuda()
    board_size = int(section["board_size"])
    win_length = int(section.get("win_length", config.get("win_length", 5)))
    n_games = int(section["n_games"])
    n_sims = int(section["n_sims"])
    c_puct = float(section.get("c_puct", 1.25))
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)

    eval_ckpt = load_checkpoint(section["checkpoint"])
    opponent_ckpt = load_checkpoint(section["opponent_checkpoint"])
    eval_model = model_from_checkpoint(eval_ckpt, device)
    opponent_model = model_from_checkpoint(opponent_ckpt, device)
    if next(eval_model.parameters()).device.type != "cuda":
        raise SystemExit("Evaluated model is not on CUDA.")
    print(
        f"eval checkpoint board_size={eval_ckpt.get('board_size')} "
        f"opponent board_size={opponent_ckpt.get('board_size')} playing on {board_size}"
    )
    games = [Gomoku(board_size, win_length) for _ in range(n_games)]
    # Even games: the evaluated checkpoint moves first. Odd games: the opponent does.
    sides: list[dict[int, torch.nn.Module]] = []
    eval_is_first: list[bool] = []
    for index in range(n_games):
        first = index % 2 == 0
        eval_is_first.append(first)
        if first:
            sides.append({1: eval_model, 2: opponent_model})
        else:
            sides.append({1: opponent_model, 2: eval_model})

    pending = set(range(n_games))
    while pending:
        groups: dict[int, list[int]] = {}
        models: dict[int, torch.nn.Module] = {}
        for index in sorted(pending):
            model = sides[index][games[index].side]
            groups.setdefault(id(model), []).append(index)
            models[id(model)] = model
        for model_id, indices in groups.items():
            policies = run_mcts(
                [games[index] for index in indices],
                models[model_id],
                n_sims=n_sims,
                device=device,
                rng=rng,
                c_puct=c_puct,
                add_noise=False,
            )
            for index, policy in zip(indices, policies):
                action = int(policy.reshape(-1).argmax())
                games[index].apply(action)
                if games[index].terminal():
                    pending.remove(index)

    wins = losses = draws = 0
    for game, first in zip(games, eval_is_first):
        if game.winner == 0:
            draws += 1
            continue
        eval_won = (first and game.winner == 1) or ((not first) and game.winner == 2)
        if eval_won:
            wins += 1
        else:
            losses += 1
    decisive = wins + losses
    rate = (wins / decisive) if decisive else 0.0
    low, high = wilson_interval(wins, decisive)
    summary = {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "n_games": n_games,
        "decisive": decisive,
        "win_rate_excluding_draws": rate,
        "win_rate_ci95": [low, high],
        "n_sims": n_sims,
        "board_size": board_size,
        "checkpoint": section["checkpoint"],
        "opponent_checkpoint": section["opponent_checkpoint"],
        "device": "cuda:0",
    }
    print(
        f"match wins={wins} losses={losses} draws={draws} "
        f"win_rate_excluding_draws={rate:.3f} ci95=[{low:.3f}, {high:.3f}]"
    )
    out = section.get("out")
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote {path}")
    return summary


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns ``(low, high)``."""
    if n <= 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)
